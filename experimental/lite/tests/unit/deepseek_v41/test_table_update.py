import pytest
import torch

from megatron.lite.model.deepseek_v41.lite.prefetch import EngramPrefetch
from test_prefetch import make_state


def ready(state, step, ids, coefficients):
    ids = torch.tensor(ids, dtype=torch.int64)
    batch = EngramPrefetch(state).start(step, {'mb': ids})
    output = batch.view('mb')(ids)
    if state.table.master is not None:
        (output.float() * torch.tensor(coefficients)[:, None]).sum().backward()
    batch.flush()


def test_full_momentum_updates_unvisited_rows_with_independent_vectors():
    state = make_state()
    ready(state, 0, [2, 2], [1., 3.])
    assert state.step(torch.clone, lr=0.1, beta=0.95)
    # G[2]=4, M=.2, N=.39, W=1-.1*.39=.961.
    torch.testing.assert_close(state.momentum[2], torch.full((32,), .2))
    torch.testing.assert_close(state.table.master[2], torch.full((32,), .961))
    ready(state, 1, [0], [2.])
    seen = []
    def rule(nesterov):
        seen.append(nesterov.clone())
        return nesterov
    state.step(rule, lr=0.1, beta=0.95)
    assert seen[0].shape == (4, 32)
    # Row 2 was not read this step: M=.19, N=.1805, W=.94295.
    torch.testing.assert_close(state.momentum[2], torch.full((32,), .19))
    torch.testing.assert_close(state.table.master[2], torch.full((32,), .94295))
    torch.testing.assert_close(state.table.master[0], torch.full((32,), .9805))
    assert state.version == 2


def test_sub_fp8_updates_accumulate_in_master_until_publication_changes():
    state = make_state()
    initial = state.table.weight.view(torch.uint8).clone()
    for step in range(40):
        ready(state, step, [0], [1.])
        state.step(torch.clone, lr=.001, beta=0.)
        if step == 0:
            assert torch.equal(state.table.weight.view(torch.uint8), initial)
    torch.testing.assert_close(state.table.master[0], torch.full((32,), .96), atol=1e-6, rtol=0)
    assert not torch.equal(state.table.weight.view(torch.uint8)[0], initial[0])
    assert torch.equal(state.table.weight.view(torch.uint8)[1:], initial[1:])


def test_skip_and_failed_quantization_do_not_partially_publish(monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.table_state as module
    state = make_state()
    before = state.state_dict()
    ready(state, 0, [1], [3.])
    assert not state.step(torch.clone, lr=.1, skip=True)
    for key in ('weight', 'scale', 'master', 'momentum'):
        assert torch.equal(state.state_dict()[key].view(torch.uint8), before[key].view(torch.uint8))
    assert state.version == 0
    ready(state, 1, [1], [3.])
    def fail(*args, **kwargs):
        raise RuntimeError('publication failure')
    monkeypatch.setattr(module, 'quantize_block_fp8', fail)
    with pytest.raises(RuntimeError, match='publication failure'):
        state.step(torch.clone, lr=.1)
    assert state.version == 0
    torch.testing.assert_close(state.table.master, before['master'], atol=0, rtol=0)
    torch.testing.assert_close(state.momentum, before['momentum'], atol=0, rtol=0)


def test_restore_preserves_master_momentum_and_next_step_trajectory():
    state = make_state()
    ready(state, 0, [3, 3], [1., 2.])
    state.step(torch.clone, lr=.003)
    saved = state.state_dict()
    restored = make_state()
    restored.load_state_dict(saved)
    for current in (state, restored):
        ready(current, 1, [0], [4.])
        current.step(torch.clone, lr=.003)
    a, b = state.state_dict(), restored.state_dict()
    assert a['version'] == b['version'] == 2
    for key in ('weight', 'scale', 'master', 'momentum'):
        assert torch.equal(a[key].view(torch.uint8), b[key].view(torch.uint8))
    saved['master'].zero_()
    assert torch.count_nonzero(restored.table.master) > 0


def test_failed_publication_can_retry_same_ready_gradient(monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.table_state as module
    state = make_state()
    ready(state, 0, [0], [2.])
    original = module.quantize_block_fp8
    def fail(*args, **kwargs):
        raise RuntimeError('not published')
    monkeypatch.setattr(module, 'quantize_block_fp8', fail)
    with pytest.raises(RuntimeError, match='not published'):
        state.step(torch.clone, lr=.1)
    assert state.active_step == 0
    torch.testing.assert_close(state.main_grad[0], torch.full((32,), 2.))
    monkeypatch.setattr(module, 'quantize_block_fp8', original)
    assert state.step(torch.clone, lr=.1)
    torch.testing.assert_close(state.table.master[0], torch.full((32,), .9805))
    assert state.version == 1


def test_nonfinite_gradient_skips_without_changing_master_momentum_or_publication():
    state = make_state()
    before = state.state_dict()
    ready(state, 0, [0], [float('nan')])
    assert not state.step(torch.clone, lr=.1)
    assert state.version == 0 and state.last_step == 0
    for key in ('weight', 'scale', 'master', 'momentum'):
        assert torch.equal(before[key].view(torch.uint8), state.state_dict()[key].view(torch.uint8))


def test_frozen_state_never_creates_high_precision_or_requantizes(monkeypatch):
    import megatron.lite.model.deepseek_v41.lite.table_state as module
    state = make_state(False)
    before = state.state_dict()
    def fail(*args, **kwargs):
        raise AssertionError('Frozen publication must not be requantized')
    monkeypatch.setattr(module, 'quantize_block_fp8', fail)
    ready(state, 0, [1], [1.])
    assert not state.step(fail, lr=.1)
    saved = state.state_dict()
    assert 'master' not in saved and 'momentum' not in saved
    assert state.version == 0
    for key in ('weight', 'scale'):
        assert torch.equal(saved[key].view(torch.uint8), before[key].view(torch.uint8))
    restored = make_state(False)
    restored.load_state_dict(saved)
    assert restored.last_step == 0


def test_restore_rejects_wrong_precision_before_mutating_state():
    state = make_state()
    snapshot = state.state_dict()
    snapshot['momentum'] = snapshot['momentum'].bfloat16()
    snapshot['master'].zero_()
    with pytest.raises(ValueError, match='momentum'):
        state.load_state_dict(snapshot)
    torch.testing.assert_close(state.table.master, torch.ones(4, 32), atol=0, rtol=0)
