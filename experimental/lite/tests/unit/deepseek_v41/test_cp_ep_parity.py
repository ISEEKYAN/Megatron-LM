# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""EP-over-CP (EP2 x CP2 on four Gloo ranks) against one unpartitioned model.

A closed 4-layer prefix of the tiny release config keeps the CSA owner layer
(KV + index source), compressed attention, the frozen Engram table (replicated
on CPU) and the MoE dispatcher in play. Every rank feeds its contiguous CP slice of the same
packed batch; per-token outputs must equal the single-rank reference slice,
and SUM-reduced gradients (dense over DP x CP, experts over expert-DP, divided
by the number of full-batch data replicas) must match the single-rank gradients
of the same global objective to FP32 rounding.
"""

import runpy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

LAYERS = 4
SEQ_LENS = (9, 7)


def _prefix_config(experts=2):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config

    release = runpy.run_path(str(Path(__file__).with_name('test_redo_parity.py')))
    hf = release['release_config']().to_hf_dict()
    text = hf['text_config']
    text.update(
        num_hidden_layers=LAYERS,
        compress_ratios=text['compress_ratios'][:LAYERS] + text['compress_ratios'][-3:],
        candidate_source_layer_id=-1,
        kv_source_layer_ids=[2],
        index_source_layer_ids=[2],
        engram_layer_ids=[1],
        engram_num_embeddings=text['engram_num_embeddings'][:1],
        n_routed_experts=experts,
    )
    return DeepseekV41Config(hf)


def _impl(parallel=None, recompute=None):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    impl = protocol.ImplConfig(
        device='cpu',
        dtype=torch.float32,
        token_map=list(range(64)),
        trainable_engram=False,
        # Sharded Engram lookups are CUDA-only (engram_lookup.py `_check`);
        # this CPU test replicates the frozen table, GPU C8 covers sharding.
        shard_engram=False,
        parallel=parallel or ParallelConfig(),
        recompute=recompute,
    )
    return replace(impl, quantized=False)


def _inputs():
    generator = torch.Generator().manual_seed(7)
    total = sum(SEQ_LENS)
    ids = torch.randint(1, 64, (1, total), generator=generator)
    cu = torch.tensor([0, SEQ_LENS[0], total], dtype=torch.int32)
    return ids, cu


def _objective(head_hidden, generator_seed=11):
    # A fixed per-token projection; the global objective is a sum over tokens,
    # so CP slices contribute disjoint, summable terms.
    width = head_hidden.shape[-1]
    weights = torch.randn(
        1, sum(SEQ_LENS), width, generator=torch.Generator().manual_seed(generator_seed)
    )
    return weights


def _reference(path, experts):
    from megatron.lite.model.deepseek_v41.lite import protocol

    torch.manual_seed(351)
    model = protocol.build_model(_prefix_config(experts), impl_cfg=_impl()).chunks[0]
    ids, cu = _inputs()
    out = model(ids, cu_seqlens=cu, return_head_hidden=True)['head_hidden']
    (out * _objective(out)).sum().backward()
    torch.save(
        dict(
            state={k: v.detach().clone() for k, v in model.state_dict().items()},
            hidden=out.detach(),
            grads={
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
        ),
        path,
    )


def _patch_gloo_all_gather_backward():
    """Gloo-only test shim for torch.distributed.nn all_gather backward.

    PyTorch emulates reduce-scatter on Gloo with scatter(src=i), passing a
    group-local index as a global rank, which fails for any subgroup that does
    not contain global rank i (EP x CP CP groups). NCCL uses reduce_scatter and
    is unaffected. Same SUM semantics via all_reduce, then this rank's slot.
    """
    from torch.distributed.nn import functional as dist_nn

    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)
        total = torch.stack(grad_outputs).contiguous()
        dist.all_reduce(total, group=ctx.group)
        return (None, total[rank])

    dist_nn._AllGather.backward = staticmethod(backward)


def _worker(rank, directory, ep, cp, world, experts):
    fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
    import megatron.core.fp8_utils  # noqa: F401
    import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
    import megatron.core.transformer.hyper_connection  # noqa: F401

    fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.set_num_threads(1)
    _patch_gloo_all_gather_backward()
    dist.init_process_group(
        'gloo',
        init_method=f'file://{directory}/store',
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=300),
    )
    reference = torch.load(f'{directory}/reference.pt', weights_only=False)
    bundle = protocol.build_model(
        _prefix_config(experts), impl_cfg=_impl(ParallelConfig(ep=ep, cp=cp))
    )
    model, ps = bundle.chunks[0], bundle.parallel_state
    assert (ps.ep_size, ps.cp_size, ps.dp_cp_size) == (ep, cp, world)
    owned = model.state_dict()
    lookups = {
        name: module.lookup
        for name, module in model.named_modules()
        if getattr(module, 'lookup', None) is not None
    }
    loaded = {}
    for key, value in owned.items():
        source = reference['state'][key]
        if source.shape != value.shape:
            owner = key.rsplit('.', 1)[0]
            lookup = lookups[owner]
            cuts = lookup.boundaries
            source = source[cuts[lookup.rank] : cuts[lookup.rank + 1]]
        loaded[key] = source
    model.load_state_dict(loaded)

    ids, cu = _inputs()
    context = ContiguousCPSequence(ids.shape[1], ps.cp_rank, cp, ps.cp_group)
    out = model(
        context.slice(ids), cu_seqlens=cu, cp_context=context, return_head_hidden=True
    )['head_hidden']
    expected = context.slice(reference['hidden'])
    forward = float((out - expected).abs().max())

    (out * context.slice(_objective(out))).sum().backward()
    mismatches, compared = [], 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        compared += 1
        group = ps.ep_dp_group if '.ffn.experts.' in name else ps.dp_cp_group
        dist.all_reduce(parameter.grad, group=group)
        # world // cp data replicas each see the whole batch; the reference saw
        # it once. A power-of-two division is exact.
        parameter.grad.div_(world // cp)
        wanted = reference['grads'][name]
        if not torch.equal(parameter.grad, wanted):
            error = float((parameter.grad - wanted).abs().max())
            scale = float(wanted.abs().max()) or 1.0
            mismatches.append((name, error / scale, error))
    import json

    Path(f'{directory}/diff-{rank}.json').write_text(
        json.dumps(
            dict(
                forward=forward,
                compared=compared,
                local_reference=sum(
                    1 for n, _ in model.named_parameters() if n in reference['grads']
                ),
                grads=sorted(mismatches, key=lambda m: -m[1]),
            )
        )
    )
    dist.destroy_process_group()


@pytest.mark.parametrize(
    'ep,cp,world,experts',
    # (4, 2, 4): one EP group spans both CP groups, as EP64 x CP4 does.
    [(2, 1, 2, 2), (1, 2, 2, 2), (2, 2, 4, 2), (4, 2, 4, 4)],
)
def test_ep_over_cp_matches_unpartitioned_model(
    v41_core_te, tmp_path, ep, cp, world, experts
):
    import json

    torch.set_num_threads(1)
    _reference(tmp_path / 'reference.pt', experts)
    mp.start_processes(
        _worker,
        args=(str(tmp_path), ep, cp, world, experts),
        nprocs=world,
        join=True,
        start_method='spawn',
    )
    diffs = [
        json.loads((tmp_path / f'diff-{r}.json').read_text()) for r in range(world)
    ]
    print('CP_EP_PARITY', ep, cp, json.dumps(diffs))
    # Every locally owned parameter with a reference gradient was compared.
    assert all(d['compared'] == d['local_reference'] > 0 for d in diffs), diffs
    worst_forward = max(d['forward'] for d in diffs)
    # CPU FP32 BLAS is not batch-invariant and CP/EP reorder reductions, so the
    # criterion is rounding-level: forward <= 2e-6 and each gradient within 5e-5
    # relative or 2e-6 absolute (small, cancelling sums such as attention sinks);
    # a wrong reduction scope shows up at O(1) relative error.
    outliers = [g for d in diffs for g in d['grads'] if g[1] > 5e-5 and g[2] > 2e-6]
    assert worst_forward <= 2e-6 and not outliers, (worst_forward, outliers[:4])


def _recompute_worker(rank, directory, ep, cp, world, experts, recompute):
    """Same partition, same inputs; save this rank's output and local gradients."""
    if world > 1:
        fixtures = runpy.run_path(str(Path(__file__).parents[2] / 'conftest.py'))
        import megatron.core.fp8_utils  # noqa: F401
        import megatron.core.transformer.experimental_attention_variant.csa  # noqa: F401
        import megatron.core.transformer.hyper_connection  # noqa: F401

        fixtures['transformer_engine_import_stub'].__wrapped__(pytest.MonkeyPatch())()
        torch.set_num_threads(1)
        _patch_gloo_all_gather_backward()
        dist.init_process_group(
            'gloo',
            init_method=f'file://{directory}/store-{recompute}',
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=300),
        )
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.model import owns_attention_state
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
    from megatron.lite.runtime.contracts import ParallelConfig

    torch.manual_seed(351)
    bundle = protocol.build_model(
        _prefix_config(experts),
        impl_cfg=_impl(ParallelConfig(ep=ep, cp=cp), recompute=recompute),
    )
    model, ps = bundle.chunks[0], bundle.parallel_state
    assert model.recompute == ('full' if recompute == 'full' else None)
    layers = [layer for layer in model.layers if layer is not None]
    # The 4-layer prefix has both kinds: layer 2 owns KV/index, 0/1/3 consume.
    assert any(owns_attention_state(layer) for layer in layers)
    assert not all(owns_attention_state(layer) for layer in layers)
    ids, cu = _inputs()
    context = (
        None
        if cp == 1
        else ContiguousCPSequence(ids.shape[1], ps.cp_rank, cp, ps.cp_group)
    )
    local = ids if context is None else context.slice(ids)
    out = model(local, cu_seqlens=cu, cp_context=context, return_head_hidden=True)[
        'head_hidden'
    ]
    weights = _objective(out)
    (out * (weights if context is None else context.slice(weights))).sum().backward()
    torch.save(
        dict(
            hidden=out.detach(),
            grads={
                n: p.grad.clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
        ),
        f'{directory}/{recompute}-{rank}.pt',
    )
    if world > 1:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    'ep,cp,world,experts', [(1, 1, 1, 2), (2, 2, 4, 2), (4, 2, 4, 4)]
)
def test_recompute_is_bitwise_at_the_same_partition(
    v41_core_te, tmp_path, ep, cp, world, experts
):
    """Hard gate: recompute replays identical ops, so outputs and grads are equal."""
    torch.set_num_threads(1)
    for recompute in (None, 'full'):
        if world == 1:
            _recompute_worker(0, str(tmp_path), ep, cp, world, experts, recompute)
        else:
            mp.start_processes(
                _recompute_worker,
                args=(str(tmp_path), ep, cp, world, experts, recompute),
                nprocs=world,
                join=True,
                start_method='spawn',
            )
    for rank in range(world):
        plain = torch.load(tmp_path / f'None-{rank}.pt')
        recomputed = torch.load(tmp_path / f'full-{rank}.pt')
        assert torch.equal(recomputed['hidden'], plain['hidden']), rank
        assert recomputed['grads'].keys() == plain['grads'].keys(), rank
        assert plain['grads'], rank
        unequal = [
            name
            for name, value in plain['grads'].items()
            if not torch.equal(recomputed['grads'][name], value)
        ]
        assert not unequal, (rank, unequal[:8])


@pytest.mark.parametrize('ep,cp,world,experts', [(1, 1, 1, 2), (2, 2, 4, 2)])
@pytest.mark.parametrize('selection', ['moe', 'attn', 'moe,attn'])
def test_selective_recompute_keeps_the_forward(
    v41_core_te, tmp_path, ep, cp, world, experts, selection
):
    """Selective recompute replays the same forward; backward only needs to run."""
    torch.set_num_threads(1)
    for recompute in (None, selection):
        if world == 1:
            _recompute_worker(0, str(tmp_path), ep, cp, world, experts, recompute)
        else:
            mp.start_processes(
                _recompute_worker,
                args=(str(tmp_path), ep, cp, world, experts, recompute),
                nprocs=world,
                join=True,
                start_method='spawn',
            )
    for rank in range(world):
        plain = torch.load(tmp_path / f'None-{rank}.pt')
        recomputed = torch.load(tmp_path / f'{selection}-{rank}.pt')
        assert torch.equal(recomputed['hidden'], plain['hidden']), rank
        assert recomputed['grads'].keys() == plain['grads'].keys(), rank
        assert all(torch.isfinite(g).all() for g in recomputed['grads'].values())


def test_selective_recompute_spec_is_validated(v41_core_te):
    from megatron.lite.model.deepseek_v41.lite import protocol

    for bad in ('mlp', 'moe,moe', ['attn', 'core_attn']):
        with pytest.raises(ValueError, match='V4.1 recompute'):
            protocol.build_model(_prefix_config(2), impl_cfg=_impl(recompute=bad))


def test_recompute_carries_container_state_and_hc_carrier(v41_core_te):
    """Deployment state holds containers (IndexCache) and an HC carrier tuple."""
    from collections import namedtuple

    from megatron.lite.model.deepseek_v41.lite.model import recomputed_layer
    from megatron.lite.primitive.modules.attention.csa import AttentionState

    Packed = namedtuple('Packed', 'values scales')

    class Consumer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.linspace(0.5, 1.5, 6, dtype=torch.float64)
            )

        def forward(self, h, p, state, *, attention_kwargs, ffn_kwargs, previous_post):
            mixed = h * self.weight + state.main_kv.sum() * state.index_k.scales.sum()
            out = mixed + previous_post[0] * previous_post[1]
            (
                ffn_kwargs['load_sink'].append(1)
                if ffn_kwargs['load_sink'] is not None
                else None
            )
            return out, p * 2, state, (out * 3, p + 1)

    def run(recompute):
        torch.manual_seed(0)
        layer = Consumer()
        main = torch.randn(4, dtype=torch.float64, requires_grad=True)
        scales = torch.randn(3, dtype=torch.float64, requires_grad=True)
        carrier = (
            torch.randn(6, dtype=torch.float64, requires_grad=True),
            torch.randn(6, dtype=torch.float64, requires_grad=True),
        )
        h = torch.randn(6, dtype=torch.float64, requires_grad=True)
        p = torch.randn(6, dtype=torch.float64, requires_grad=True)
        state = AttentionState(
            kv_owner=2,
            index_owner=2,
            main_kv=main * 1,
            index_k=Packed(torch.zeros(2, dtype=torch.uint8), scales * 1),
        )
        sink = []
        kwargs = dict(load_sink=sink)
        extra = {'previous_post': tuple(t * 1 for t in carrier)}
        if recompute:
            out = recomputed_layer(layer, h, p, state, {}, kwargs, extra)
        else:
            out = layer.forward(
                h, p, state, attention_kwargs={}, ffn_kwargs=kwargs, **extra
            )
        assert out[2] is state
        (out[0].sum() + out[1].sum() + sum(t.sum() for t in out[3])).backward()
        return [t.grad for t in (main, scales, *carrier, h, p, layer.weight)], sink

    plain, plain_sink = run(False)
    recomputed, sink = run(True)
    assert all(torch.equal(a, b) for a, b in zip(plain, recomputed))
    # Router statistics are published once, by the original forward.
    assert sink == plain_sink == [1]
