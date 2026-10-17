# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CP-A A1: one packed call over the slice equals the per-document path.

Gate: for every layer the attention output, its gradients and the whole CSA
state are bitwise equal between the packed call and the per-document
reference, at CP1 and at CP2 (gloo). Token-wise work (mHC mixes, MoE) now runs
over all rows at once; on the CPU reference those ops are not row-count
invariant (last-bit differences, independent of CP), so the full model is
compared within a float tolerance and the error distribution is reported.
"""
import multiprocessing
from contextlib import nullcontext
from dataclasses import fields, replace

import pytest
import torch
from test_redo_parity import release_config

CASES = [(5,), (7, 1, 3, 9, 2), (17, 1, 1, 30), (2, 2, 2, 2, 2, 2)]


def _bounds(lengths):
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return tuple(out)


def _config():
    """Six layers, one of each kind: SWA, Engram, CR2 owner and consumer, CR1
    owner building the candidate pool, and an index owner reusing it."""
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    source = release_config().to_hf_dict()
    source['text_config'].update(
        num_hidden_layers=6,
        compress_ratios=[0, 0, 2, 2, 1, 1, 0, 0, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5],
        candidate_source_layer_id=4,
        engram_layer_ids=[1],
        engram_num_embeddings=[18],
    )
    return DeepseekV41Config(source)


def _build():
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.manual_seed(351)
    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        token_map=list(range(64)),
        trainable_engram=True,
    )
    impl = replace(impl, quantized=False)
    return protocol.build_model(_config(), impl_cfg=impl).chunks[0]


@pytest.fixture
def model(v41_core_te):
    return _build()


def _check_layers(model, geometry, x_rows, per_document, tag):
    """Run every layer's attention packed and per document; compare bitwise."""
    from megatron.lite.primitive.modules.attention.csa import AttentionState

    packed = AttentionState()
    states = [AttentionState() for _ in range(geometry.num_documents)]
    for index, block in enumerate(model.layers):
        attn = block.attn
        params = [p for p in attn.parameters() if p.requires_grad]
        x = x_rows(index).requires_grad_()
        output, packed = attn(x, packed, geometry=geometry)
        reference_x = x.detach().clone().requires_grad_()
        outputs, offset = [], 0
        for doc in range(geometry.num_documents):
            start, stop = geometry.owned(doc)
            rows = reference_x[:, offset : offset + stop - start]
            out, states[doc] = attn(rows, states[doc], **per_document(doc))
            outputs.append(out)
            offset += stop - start
        reference = torch.cat(outputs, 1)
        assert torch.equal(output, reference), (tag, index, 'output')
        for field in fields(AttentionState):
            name = field.name
            if name in ('kv_bounds', 'query_bounds'):
                continue
            value = getattr(packed, name)
            expected = [getattr(s, name) for s in states]
            if isinstance(value, torch.Tensor):
                pad = {'indices': -1, 'candidates': False}.get(name)
                if pad is not None:  # per-document widths, padded to the widest
                    expected = [
                        torch.nn.functional.pad(
                            e, (0, value.shape[-1] - e.shape[-1]), value=pad
                        )
                        for e in expected
                    ]
                assert torch.equal(value, torch.cat(expected, 1)), (tag, index, name)
            else:
                assert all(e == value for e in expected), (tag, index, name)
        if packed.main_kv is not None:
            sizes = [s.main_kv.shape[1] for s in states]
            assert packed.kv_bounds == _bounds(sizes)
        if packed.indices is not None:
            assert packed.query_bounds[-1] == x.shape[1]
        weight = torch.randn_like(output)
        got = torch.autograd.grad(
            (output * weight).sum(), [x, *params], allow_unused=True, retain_graph=True
        )
        want = torch.autograd.grad(
            (reference * weight).sum(),
            [reference_x, *params],
            allow_unused=True,
            retain_graph=True,
        )
        for g, w in zip(got, want):
            assert (g is None) == (w is None), (tag, index, 'grad presence')
            if g is not None:
                assert torch.equal(g, w), (tag, index, 'grad')


@pytest.mark.parametrize('lengths', CASES)
def test_cp1_every_layer_attention_state_and_grads_bitwise(model, lengths):
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )

    geometry = PackedCPGeometry(_bounds(lengths))
    dim = model.layers[0].attn_norm.weight.shape[-1]
    torch.manual_seed(5)
    inputs = [torch.randn(1, geometry.total, dim) for _ in model.layers]
    _check_layers(model, geometry, lambda i: inputs[i], lambda doc: {}, 'cp1')


@pytest.mark.parametrize('lengths', CASES)
def test_engram_hashes_bitwise(model, lengths):
    from megatron.lite.model.deepseek_v41.lite.model import _hashes, sequence_hashes
    from megatron.lite.primitive.modules.attention.packed_geometry import (
        PackedCPGeometry,
    )

    bounds = _bounds(lengths)
    torch.manual_seed(9)
    ids = torch.randint(1, 64, (1, bounds[-1]))
    mask = ids != 3
    packed = _hashes(model.engram_hash, ids, mask, None, PackedCPGeometry(bounds))
    reference = torch.cat(
        [
            sequence_hashes(model.engram_hash, ids[:, b:e], mask[:, b:e], None)
            for b, e in zip(bounds, bounds[1:])
        ],
        1,
    )
    assert torch.equal(packed, reference)


def _cp2_worker(rank, path, lengths, queue):
    import torch.distributed as dist

    try:
        dist.init_process_group(
            'gloo', init_method=f'file://{path}', rank=rank, world_size=2
        )
        from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
        from megatron.lite.primitive.modules.attention.packed_geometry import (
            PackedCPGeometry,
        )

        model = _build()
        bounds = _bounds(lengths)
        group = dist.group.WORLD
        geometry = PackedCPGeometry(bounds, rank, 2, group)
        sequence = ContiguousCPSequence(bounds[-1], rank, 2, group)
        dim = model.layers[0].attn_norm.weight.shape[-1]
        torch.manual_seed(5)
        full = [torch.randn(1, bounds[-1], dim) for _ in model.layers]

        def local(index):
            parts = [
                full[index][:, slice(*geometry.owned(d))] for d in range(len(lengths))
            ]
            return torch.cat(parts, 1)

        def per_document(doc):
            return {'cp_context': sequence.document(bounds[doc], bounds[doc + 1])}

        _check_layers(model, geometry, local, per_document, f'cp2 rank{rank}')
        queue.put((rank, 'ok'))
    except BaseException as error:  # report to the parent, which asserts
        queue.put((rank, repr(error)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize('lengths', [(7, 1, 3, 9, 2), (3, 3), (17, 1, 1, 30)])
def test_cp2_every_layer_attention_state_and_grads_bitwise(
    v41_core_te, tmp_path, lengths
):
    context = multiprocessing.get_context('fork')
    queue = context.Queue()
    workers = [
        context.Process(
            target=_cp2_worker, args=(rank, tmp_path / 'pg', lengths, queue)
        )
        for rank in range(2)
    ]
    for worker in workers:
        worker.start()
    results = dict(queue.get(timeout=600) for _ in workers)
    for worker in workers:
        worker.join(timeout=60)
    assert results == {0: 'ok', 1: 'ok'}, results


def _per_document_forward(
    sequence_forward,
    hidden,
    pre_mix,
    cu_seqlens,
    *,
    input_ids=None,
    image_mask=None,
    cp_context=None,
):
    """The per-document reference: one sequence call per document (CP1)."""
    from megatron.lite.model.deepseek_v41.lite.protocol import PackedRouterReplay
    from megatron.lite.primitive.utils.packed_seq import packed_sequence_ranges

    assert cp_context is None
    replay = PackedRouterReplay(hidden.shape[1])
    outputs, mixes = [], []
    for begin, end in packed_sequence_ranges(cu_seqlens, hidden.shape[1]):
        kwargs = {}
        if input_ids is not None:
            kwargs['input_ids'] = input_ids[:, begin:end]
        if image_mask is not None:
            kwargs['image_mask'] = image_mask[:, begin:end]
        with replay.sequence(begin, end) if replay is not None else nullcontext():
            h, p = sequence_forward(
                hidden[:, begin:end], pre_mix[:, begin:end], **kwargs
            )
        outputs.append(h)
        mixes.append(p)
    replay.finish()
    return torch.cat(outputs, 1), torch.cat(mixes, 1)


def _ulps(a, b):
    ia = a.detach().float().contiguous().view(torch.int32).long()
    ib = b.detach().float().contiguous().view(torch.int32).long()
    return (ia - ib).abs()


@pytest.mark.parametrize('lengths', CASES)
def test_end_to_end_within_tolerance(model, monkeypatch, lengths):
    from megatron.lite.model.deepseek_v41.lite import protocol

    bounds = _bounds(lengths)
    torch.manual_seed(13)
    ids = torch.randint(1, 64, (1, bounds[-1]))
    cu = torch.tensor(bounds, dtype=torch.int32)
    params = [p for p in model.parameters() if p.requires_grad]

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(ids, cu_seqlens=cu)['logits']
        weight = torch.linspace(-1, 1, logits.numel()).view_as(logits)
        (logits * weight).sum().backward()
        return logits.detach(), [
            None if p.grad is None else p.grad.clone() for p in params
        ]

    packed_logits, packed_grads = run()
    monkeypatch.setattr(protocol, 'packed_paired_forward', _per_document_forward)
    reference_logits, reference_grads = run()
    ulps = _ulps(packed_logits, reference_logits)
    scale = reference_logits.abs().max().clamp_min(1)
    error = (packed_logits - reference_logits).abs().max() / scale
    grad_error = max(
        float((g - w).abs().max() / w.abs().max().clamp_min(1e-12))
        for g, w in zip(packed_grads, reference_grads)
        if w is not None
    )
    print(
        'A1_E2E',
        lengths,
        'logits ulp max/p99/zero',
        int(ulps.max()),
        float(ulps.float().quantile(0.99)),
        float((ulps == 0).float().mean()),
        'rel',
        float(error),
        'grad rel max',
        grad_error,
    )
    assert error < 1e-5 and grad_error < 1e-4


@pytest.mark.parametrize('lengths', [(7, 1, 3, 9, 2), (2, 2, 2, 2, 2, 2)])
def test_full_recompute_runs_the_packed_slice(model, lengths):
    """Recompute checks that state-consuming layers hand their state through."""
    bounds = _bounds(lengths)
    torch.manual_seed(17)
    ids = torch.randint(1, 64, (1, bounds[-1]))
    cu = torch.tensor(bounds, dtype=torch.int32)

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(ids, cu_seqlens=cu)['logits']
        weight = torch.linspace(-1, 1, logits.numel()).view_as(logits)
        (logits * weight).sum().backward()
        return logits.detach(), [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]

    plain_logits, plain_grads = run()
    model.recompute = 'full'
    try:
        logits, grads = run()
    finally:
        model.recompute = None
    assert torch.equal(logits, plain_logits)
    for got, want in zip(grads, plain_grads):
        assert (got is None) == (want is None)
        if got is not None:
            assert torch.equal(got, want)


def test_packed_state_round_trips_deployment_index_cache(v41_core_te):
    """On the deployment path index_k is an IndexCache (packed rows + scales)."""
    from megatron.lite.primitive.modules.attention.csa import (
        AttentionState,
        _document_state,
        _packed_state,
    )
    from megatron.lite.primitive.modules.deployment_math.cache import IndexCache

    torch.manual_seed(3)
    rows = [3, 0, 5]  # compressed rows per document, one empty
    queries = [4, 1, 6]
    states = [
        AttentionState(
            kv_owner=2,
            index_owner=2,
            main_kv=torch.randn(1, r, 8),
            index_k=IndexCache(
                torch.randint(0, 255, (1, r, 64), dtype=torch.uint8),
                torch.randint(0, 255, (1, r, 4), dtype=torch.uint8),
            ),
            indices=torch.randint(-1, max(r, 1), (1, q, min(2, r)), dtype=torch.int32),
        )
        for r, q in zip(rows, queries)
    ]
    packed = _packed_state(states, queries)
    assert isinstance(packed.index_k, IndexCache)
    assert packed.kv_bounds == _bounds(rows) and packed.query_bounds == _bounds(queries)
    for doc, state in enumerate(states):
        back = _document_state(packed, doc)
        assert torch.equal(back.main_kv, state.main_kv)
        assert torch.equal(back.index_k.packed, state.index_k.packed)
        assert torch.equal(back.index_k.scale, state.index_k.scale)
        assert torch.equal(back.indices, state.indices)
