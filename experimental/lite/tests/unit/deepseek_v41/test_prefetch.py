import pytest
import torch

from megatron.lite.model.deepseek_v41.lite.engram import ShardedEngramTable
from megatron.lite.model.deepseek_v41.lite.prefetch import EngramPrefetch, GradientTag
from megatron.lite.model.deepseek_v41.lite.table_state import EngramTableState
from megatron.lite.primitive.modules.engram_lookup import RowLookup


def make_state(trainable=True):
    values = torch.full((4, 32), 256.).to(torch.float8_e4m3fn)
    scales = torch.full((4, 1), 1 / 256).to(torch.float8_e8m0fnu)
    return EngramTableState(ShardedEngramTable(values, scales, RowLookup((0, 4)), trainable=trainable))


def test_batch_lookup_once_and_delayed_weighted_gradient_return(monkeypatch):
    state = make_state()
    calls = []
    original = state.table.lookup_fp8
    def counted(ids):
        calls.append(ids.clone())
        return original(ids)
    monkeypatch.setattr(state.table, 'lookup_fp8', counted)
    ids = {'a': torch.tensor([[2, 0, 2]]), 'b': torch.tensor([[1, 2]])}
    batch = EngramPrefetch(state).start(41, ids)
    assert len(calls) == 1 and calls[0].tolist() == [2, 0, 2, 1, 2]
    a, b = batch.view('a'), batch.view('b')
    # Backward order differs from prefetch order; each return has a distinct tag.
    (b(ids['b']).float() * torch.tensor([3., 5.])[None, :, None]).sum().backward()
    (a(ids['a']).float() * torch.tensor([2., 7., 11.])[None, :, None]).sum().backward()
    assert state.table.master.grad is None
    assert torch.count_nonzero(state.main_grad) == 0
    batch.flush()
    expected = torch.tensor([7., 3., 18., 0.])[:, None].expand(4, 32)
    torch.testing.assert_close(state.main_grad, expected, atol=0, rtol=0)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='closed'):
        a(ids['a'])
    with pytest.raises(RuntimeError, match='closed'):
        batch.flush()


def test_reject_missing_duplicate_stale_and_wrong_precision_returns():
    state = make_state()
    ids = {'a': torch.tensor([1]), 'b': torch.tensor([2])}
    batch = EngramPrefetch(state).start(9, ids)
    with pytest.raises(RuntimeError, match='missing'):
        batch.flush()
    with pytest.raises(RuntimeError, match='pending'):
        state.step(torch.clone, lr=0.1)
    with pytest.raises(RuntimeError, match='tag'):
        batch.return_gradient(GradientTag(8, 0, 'a'), torch.ones(1, 32))
    with pytest.raises(ValueError, match='FP32'):
        batch.return_gradient(GradientTag(9, 0, 'a'), torch.ones(1, 32).bfloat16())
    batch.return_gradient(GradientTag(9, 0, 'a'), torch.ones(1, 32))
    with pytest.raises(RuntimeError, match='duplicate'):
        batch.return_gradient(GradientTag(9, 0, 'a'), torch.ones(1, 32))
    batch.return_gradient(GradientTag(9, 0, 'b'), torch.ones(1, 32) * 3)
    batch.flush()
    state.step(torch.clone, lr=0.1)
    with pytest.raises(RuntimeError, match='step'):
        EngramPrefetch(state).start(9, ids)


def test_frozen_prefetch_and_empty_trainable_batch():
    for trainable in (False, True):
        state = make_state(trainable)
        ids = torch.empty((0, 24), dtype=torch.int64)
        batch = EngramPrefetch(state).start(0, {'empty': ids})
        output = batch.view('empty')(ids)
        assert output.shape == (0, 24, 32)
        if trainable:
            output.sum().backward()
        batch.flush()
        if trainable:
            assert torch.count_nonzero(state.main_grad) == 0
        else:
            assert state.main_grad is state.momentum is None
        state.step(torch.clone, lr=0.1)


def test_cached_provider_checks_ids_and_live_step():
    state = make_state()
    ids = torch.tensor([[1, 2]])
    batch = EngramPrefetch(state).start(0, {'a': ids})
    with pytest.raises(ValueError, match='IDs'):
        batch.view('a')(torch.tensor([[2, 1]]))
    with pytest.raises(RuntimeError, match='active'):
        EngramPrefetch(state).start(1, {'a': ids})
    with pytest.raises(RuntimeError, match='active'):
        state.state_dict()


def test_real_engram_recompute_matches_eager_gradients_and_keeps_parameter_owner(monkeypatch):
    import copy
    from torch.utils.checkpoint import checkpoint
    from megatron.lite.model.deepseek_v41.lite.engram import Engram
    state = make_state()
    projection = torch.nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        projection.weight.copy_(torch.linspace(-.1, .1, 32 * 64).reshape(32, 64))
    actual = Engram(16, 1, state.table, projection)
    reference = copy.deepcopy(actual)
    hidden = torch.linspace(-2, 2, 16).reshape(1, 1, 1, 16).requires_grad_()
    ref_hidden = hidden.detach().clone().requires_grad_()
    ids = {'first': torch.tensor([[[2, 0]]]), 'second': torch.tensor([[[1, 2]]])}
    calls = []
    original = state.table.lookup_fp8
    def count(request):
        calls.append(request)
        return original(request)
    monkeypatch.setattr(state.table, 'lookup_fp8', count)
    batch = EngramPrefetch(state).start(0, ids)
    outputs, expected = [], []
    for i, (key, request) in enumerate(ids.items()):
        view = batch.view(key)
        def run(x, request=request, view=view):
            return actual(x, request, embedding=view)
        outputs.append((i + 1) * checkpoint(run, hidden, use_reentrant=False).sum())
        expected.append((i + 1) * reference(ref_hidden, request).sum())
    sum(outputs).backward()
    sum(expected).backward()
    assert actual.embed is state.table
    assert dict(actual.named_parameters())['embed.master'] is state.table.master
    assert state.table.master.grad is None
    batch.flush()
    assert len(calls) == 1
    torch.testing.assert_close(state.main_grad, reference.embed.master.grad, atol=0, rtol=0)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=0, rtol=0)
    torch.testing.assert_close(actual.wkv.weight.grad, reference.wkv.weight.grad, atol=0, rtol=0)


def test_dependency_events_wait_before_consumption_and_gradient_return(monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.prefetch as module
    events, waits = [], []
    def record(device):
        event = len(events)
        events.append(event)
        return event
    monkeypatch.setattr(module, '_record_event', record)
    monkeypatch.setattr(module, '_wait_event', lambda event, device: waits.append(event))
    state = make_state()
    ids = torch.tensor([1])
    batch = EngramPrefetch(state).start(0, {'mb': ids})
    assert events == [0]
    batch.view('mb')(ids).sum().backward()
    assert waits == [0] and events == [0, 1]
    batch.flush()
    assert waits == [0, 0, 1]


def test_zero_row_owner_and_empty_requests():
    table = ShardedEngramTable(torch.empty(0, 32, dtype=torch.float8_e4m3fn),
                              torch.empty(0, 1, dtype=torch.float8_e8m0fnu),
                              RowLookup((0, 0)), trainable=True)
    state = EngramTableState(table)
    ids = torch.empty(0, 24, dtype=torch.int64)
    batch = EngramPrefetch(state).start(0, {'empty': ids})
    batch.view('empty')(ids).sum().backward()
    batch.flush()
    assert state.main_grad.shape == (0, 32)
    state.step(torch.clone, lr=.1)


def test_repeated_row_accumulation_is_native_fp32_not_widened_bf16():
    state = make_state()
    ids = torch.tensor([2, 2])
    batch = EngramPrefetch(state).start(0, {'mb': ids})
    weights = torch.tensor([1., 2. ** -10])[:, None]
    (batch.view('mb')(ids).float() * weights).sum().backward()
    batch.flush()
    expected = torch.full((32,), 1. + 2. ** -10)
    torch.testing.assert_close(state.main_grad[2], expected, atol=0, rtol=0)
    assert not torch.equal(state.main_grad[2], expected.bfloat16().float())


def test_stale_publication_tag_cannot_be_returned_to_next_step():
    state = make_state()
    ids = torch.tensor([1])
    prefetch = EngramPrefetch(state)
    first = prefetch.start(0, {'mb': ids})
    first.view('mb')(ids).sum().backward()
    first.flush()
    state.step(torch.clone, lr=.1)
    second = prefetch.start(1, {'mb': ids})
    with pytest.raises(RuntimeError, match='tag'):
        second.return_gradient(GradientTag(1, 0, 'mb'), torch.ones(1, 32))
    second.view('mb')(ids).sum().backward()
    second.flush()


def test_fp8_prefetch_provider_preserves_published_bytes_and_master_path():
    state = make_state()
    ids = torch.tensor([[3, 1, 3]])
    batch = EngramPrefetch(state).start(0, {'mb': ids})
    values, scales, master = batch.view('mb').lookup_fp8(ids)
    assert torch.equal(values.view(torch.uint8), state.table.weight.view(torch.uint8)[ids])
    assert torch.equal(scales.view(torch.uint8), state.table.scale.view(torch.uint8)[ids])
    master.sum().backward()
    batch.flush()
    torch.testing.assert_close(state.main_grad[3], torch.full((32,), 2.))
