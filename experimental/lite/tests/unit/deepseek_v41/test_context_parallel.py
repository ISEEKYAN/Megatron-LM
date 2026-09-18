# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Contiguous CP ownership and document-boundary contracts."""

import parallel_test_utils as harness
import pytest
import torch
from parallel_test_utils import assert_exact


def test_cp_nondivisible_contiguous_ownership(moe):
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    ids = torch.arange(7)
    parts = [
        ContiguousCPSequence(7, rank, 2).slice(ids, seq_dim=0) for rank in range(2)
    ]
    assert [p.tolist() for p in parts] == [
        [0, 1, 2, 3],
        [4, 5, 6],
    ], 'CP_NONDIVISIBLE_CONTIGUOUS'


def test_cp_cross_document_intersections(moe):
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    # One global contiguous split, not independently repartitioned documents.
    documents = [(0, 3), (3, 7)]
    actual = []
    for rank in range(2):
        root = ContiguousCPSequence(7, rank, 2)
        actual.append(
            [
                root.document(a, b).slice(torch.arange(a, b), seq_dim=0).tolist()
                for a, b in documents
            ]
        )
    assert actual == [[[0, 1, 2], [3]], [[], [4, 5, 6]]], 'CP_DOCUMENT_GLOBAL_OWNERSHIP'


def test_cp_cross_document_shift_before_slice(moe):
    from megatron.lite.model.deepseek_v41.lite.protocol import _cp_targets
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
    from megatron.lite.runtime.contracts import PackedBatch

    ids = torch.arange(1, 8)
    batch = PackedBatch(ids, ids, torch.tensor([3, 4]))
    actual = [_cp_targets(batch, ContiguousCPSequence(7, r, 2)) for r in range(2)]
    assert [x[0].tolist() for x in actual] == [
        [2, 3, 0, 5],
        [6, 7, 0],
    ], 'CP_GLOBAL_LABEL_SHIFT'
    assert [x[1].tolist() for x in actual] == [
        [1, 1, 0, 1],
        [1, 1, 0],
    ], 'CP_DOCUMENT_TAIL_MASK'
    assert all(x[2].item() == 5 for x in actual), 'CP_GLOBAL_LOSS_DENOMINATOR'


def test_cp_requires_initialized_world(moe, model_config):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(ValueError, match='CP requires an initialized CP-only world'):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', quantized=False, parallel=ParallelConfig(cp=2)
            ),
        )


@pytest.mark.parametrize('key', ['tp', 'vpp'])
def test_cp_preserves_unsupported_parallel_rejection(moe, model_config, key):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.runtime.contracts import ParallelConfig

    with pytest.raises(NotImplementedError):
        protocol.build_model(
            model_config,
            impl_cfg=protocol.ImplConfig(
                device='meta', quantized=False, parallel=ParallelConfig(**{key: 2})
            ),
        )


def _cp_worker(rank, config, trainable, lengths, directory):
    from dataclasses import replace

    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence
    from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

    serial, impl = harness._init_parallel_worker(rank, config, trainable)
    # Both Engram arms consume nonzero row-dependent memory from the first step.
    harness.seed_engram(serial.chunks[0])
    ordered = [protocol.build_model(config, impl_cfg=impl) for _ in range(2)]
    for bundle in ordered:
        bundle.chunks[0].load_state_dict(serial.chunks[0].state_dict())
    with harness.world(rank, directory):
        parallel = protocol.build_model(
            config, impl_cfg=replace(impl, parallel=ParallelConfig(cp=2))
        )
        model, reference = parallel.chunks[0], serial.chunks[0]
        model.load_state_dict(reference.state_dict())
        assert parallel.parallel_state.cp_size == 2, 'CP_REAL_WORLD'
        assert parallel.parallel_state.tp_size == parallel.parallel_state.dp_size == 1
        ownership = ContiguousCPSequence(
            sum(lengths), rank, 2, parallel.parallel_state.cp_group
        )
        errors = dict(
            logits_max_abs=0.0,
            gradient_max_abs=0.0,
            parameter_max_abs=0.0,
            pre_reduction_gradient_max_abs=0.0,
        )
        for step in range(2):
            ids = torch.arange(1 + step, 1 + step + sum(lengths), device=rank)
            batch = PackedBatch(ids, ids, torch.tensor(lengths, device=rank))
            if step == 0:
                _trace_cp_forward(serial, parallel, batch, ownership)
            with torch.no_grad():
                full = (
                    ownership.slice(
                        serial.forward_step(reference, batch)['logits'], seq_dim=0
                    )
                    if step == 0
                    else None
                )
                expected = _ordered_reference([b.chunks[0] for b in ordered], batch)[0][
                    rank
                ]
                actual = parallel.forward_step(model, batch)['logits']
                raw_error = None if full is None else float((actual - full).abs().max())
                errors['logits_max_abs'] = float((actual - expected).abs().max())
                print(
                    f'CP logits rank={rank} step={step} full={raw_error} ordered={errors["logits_max_abs"]}',
                    flush=True,
                )
                assert_exact(actual, expected, msg='CP_ORDERED_LOCAL_LOGITS')
            for b in ordered:
                b.optimizer.zero_grad()
            ref_logits, loads = _ordered_reference(
                [b.chunks[0] for b in ordered], batch
            )
            sum(_reference_losses(ref_logits, batch)).backward()
            parallel.optimizer.zero_grad()
            execution = parallel.forward_step.keywords['execution_model']
            # Isolate every parameter's contribution before DDP averaging.
            with execution.no_sync():
                parallel.forward_step(model, batch)['loss'].backward()
            for name, p, q in harness.parameter_pairs(
                model, ordered[rank].chunks[0], strict=True
            ):
                assert (p.grad is None) == (
                    q.grad is None
                ), f'CP_LOCAL_GRAD_MEMBERSHIP {name}'
                if p.grad is not None:
                    error = float((p.grad - q.grad).abs().max())
                    errors['pre_reduction_gradient_max_abs'] = max(
                        errors['pre_reduction_gradient_max_abs'], error
                    )
                    if error:
                        print(
                            f'CP pre-reduction rank={rank} {name} max_abs={error}',
                            flush=True,
                        )
                    assert_exact(p.grad, q.grad, msg=f'CP_PRE_REDUCTION_GRAD {name}')
            # Clear diagnostic statistics, then exercise production DDP backward.
            parallel.optimizer.zero_grad()
            harness.backward(parallel, [batch])
            for name, p, left, right in harness.parameter_pairs(
                model, *(b.chunks[0] for b in ordered), strict=True
            ):
                if left.grad is None and right.grad is None:
                    assert p.grad is None, f'CP_GRAD_MEMBERSHIP {name}'
                    continue
                average = (
                    (torch.zeros_like(left) if left.grad is None else left.grad)
                    + (torch.zeros_like(right) if right.grad is None else right.grad)
                ) / 2
                assert p.grad is not None, f'CP_GRAD_MEMBERSHIP {name}'
                assert_exact(
                    p.main_grad if p.main_grad is not None else p.grad,
                    p.grad,
                    msg=f'CP_OPTIMIZER_GRAD_VIEW {name}',
                )
                errors['gradient_max_abs'] = max(
                    errors['gradient_max_abs'], float((p.grad - average).abs().max())
                )
                assert_exact(p.grad, average, msg=f'CP_ORDERED_GLOBAL_GRAD {name}')
                left.grad = left.main_grad = average.clone()
                right.grad = right.main_grad = average.clone()
            for b in ordered:
                for item in loads:
                    b.optimizer.accumulate_modality_loads(item)
                assert b.optimizer.step()[0]
            assert parallel.optimizer.step()[0]
            for name, p, q in harness.parameter_pairs(
                model, ordered[0].chunks[0], strict=True
            ):
                harness.record_error(errors, 'parameter_max_abs', p, q)
                assert_exact(p, q, msg=f'CP_STEP_PARAMETER {name}')
            for block, other in zip(
                model.layers, ordered[0].chunks[0].layers, strict=True
            ):
                for name in ('bias', 'bias_vl'):
                    assert_exact(
                        getattr(block.ffn.gate, name),
                        getattr(other.ffn.gate, name),
                        msg='CP_GLOBAL_ROUTER_LOAD',
                    )
                if block.attn.indexer is not None:
                    assert all(
                        not p.requires_grad and p.grad is None
                        for p in block.attn.indexer.parameters()
                    ), 'CP_FROZEN_INDEXER'
        harness.report_rank(directory, rank, errors)


@pytest.mark.gpus(2)
@pytest.mark.parametrize(
    'trainable', [False, True], ids=['frozen_engram', 'trainable_engram']
)
@pytest.mark.parametrize(
    'lengths', [(7,), (5, 7, 9)], ids=['nondivisible', 'cross_document']
)
def test_cp_two_step_training(model_config, trainable, lengths, tmp_path):
    assert torch.cuda.device_count() >= 2, 'CP requires two allocated GPUs'
    harness.run_workers(
        _cp_worker, (model_config, trainable, lengths), tmp_path, report=True
    )


def test_cp_transport_padding_is_not_a_document_token(moe, monkeypatch):
    from megatron.lite.primitive.modules.attention import cp

    root = cp.ContiguousCPSequence(7, 1, 2, group=object())
    document = root.document(3, 7)
    left = torch.tensor([[3.0, 99.0, 99.0, 99.0]], requires_grad=True)
    right = torch.tensor([[4.0, 5.0, 6.0, 99.0]], requires_grad=True)
    monkeypatch.setattr(cp, '_all_gather_cp', lambda value, group: [left, right])
    actual = document.gather(right[:, :3])
    assert actual.tolist() == [[3.0, 4.0, 5.0, 6.0]], 'CP_TRANSPORT_PADDING'
    actual.sum().backward()
    assert left.grad.tolist() == [[1.0, 0.0, 0.0, 0.0]], 'CP_PADDING_GRADIENT'
    assert right.grad.tolist() == [[1.0, 1.0, 1.0, 0.0]], 'CP_PADDING_GRADIENT'


def test_cp_attention_uses_document_global_query_positions(
    moe, model_config, monkeypatch
):
    from megatron.lite.primitive.modules import csa2
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    config = model_config.attention_config(
        linear_fp8=False, main_qat=False, index_qat=False, swa_fp8=False
    )
    # Import the model's configuration adapter, leaving attention transport shared.
    from megatron.lite.model.deepseek_v41.lite.attention import (
        AttentionState,
        CSA2Attention,
    )

    module = CSA2Attention(config, 0).float()
    x = torch.randn(1, 7, config.dim)
    ownership = ContiguousCPSequence(7, 1, 2, group=object())
    monkeypatch.setattr(ContiguousCPSequence, 'gather', lambda self, local, **kwargs: x)
    positions = []
    original = csa2.rotate

    def record(value, where, *args, **kwargs):
        positions.append(where.tolist())
        return original(value, where, *args, **kwargs)

    monkeypatch.setattr(csa2, 'rotate', record)
    module(ownership.slice(x), AttentionState(), cp_context=ownership)
    assert positions == [[4, 5, 6], list(range(7)), [4, 5, 6]], 'CP_GLOBAL_QUERY_ROPE'


def _trace_cp_forward(serial, parallel, batch, ownership):
    """On a parity failure expose the earliest equal-input local computation."""
    from collections import defaultdict

    from megatron.lite.primitive.modules.hyper_connection import HCMixes, RMSNorm
    from megatron.lite.primitive.modules.native_fp32_linear import Linear

    cached, calls = defaultdict(list), defaultdict(int)
    from megatron.lite.model.deepseek_v41.lite.moe import DeepseekV41MoE

    types = (HCMixes, RMSNorm, Linear, torch.nn.Embedding, DeepseekV41MoE)
    eligible = lambda name, module: isinstance(module, types) and ".ffn." not in name
    modules = dict(serial.chunks[0].named_modules())

    def tensors(value):
        return list(value) if isinstance(value, tuple) else [value]

    def capture(name, module, args, output):
        cached[name].append(
            (args[0].detach().clone(), [v.detach().clone() for v in tensors(output)])
        )

    with harness.forward_hooks(serial.chunks[0], eligible, capture), torch.no_grad():
        serial.forward_step(serial.chunks[0], batch)
    mismatches = []

    def compare(name, module, args, output):
        index = calls[name]
        calls[name] += 1
        reference_input, expected = cached[name][index]
        actual = tensors(output)
        # Modules before packed execution (embedding) see the entire packed input.
        context = ownership
        if name.startswith('layers.'):
            begin = int(batch.cu_seqlens[index])
            end = int(batch.cu_seqlens[index + 1])
            context = ownership.document(begin, end)

        def local(value, target):
            if value.shape == target.shape:
                return value
            return context.slice(value)

        expected = [local(v, a) for v, a in zip(expected, actual)]
        if all(torch.equal(v, a) for v, a in zip(expected, actual)):
            return
        if len(mismatches) >= 12:
            return
        same_input = torch.equal(local(reference_input, args[0]), args[0])
        replay = tensors(modules[name](args[0]))
        local_exact = all(torch.equal(v, a) for v, a in zip(replay, actual))
        error = max(
            float((v - a).abs().max()) if a.numel() else 0.0
            for v, a in zip(expected, actual)
        )
        mismatches.append(
            dict(
                module=name,
                call=index,
                equal_input=same_input,
                local_replay_exact=local_exact,
                max_abs=error,
            )
        )

    with harness.forward_hooks(parallel.chunks[0], eligible, compare), torch.no_grad():
        parallel.forward_step(parallel.chunks[0], batch)
    print(f'CP_PRE_REDUCTION rank={ownership.rank}: {mismatches}', flush=True)
    assert all(m['local_replay_exact'] for m in mismatches), 'CP_LOCAL_OPERATOR_REPLAY'


def _ordered_reference(models, batch):
    """Single-process virtual ranks; no process groups or production CP slicing.

    Token-local operators retain each shard's GEMM shape. Each KV replica has a
    separate graph, and the gather adjoint adds replica contributions before
    slicing, matching the two-rank collective's reduction boundary.
    """
    from types import SimpleNamespace

    from megatron.lite.primitive.modules.csa2 import AttentionState
    from megatron.lite.primitive.modules.hyper_connection import (
        contract_hc,
        expand_hc,
        mix_residual,
    )

    class Gather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, left, right):
            ctx.left_length = left.shape[1]
            full = torch.cat((left, right), 1)
            return full.clone(), full.clone()

        @staticmethod
        def backward(ctx, left_grad, right_grad):
            gradient = left_grad + right_grad
            return gradient[:, : ctx.left_length], gradient[:, ctx.left_length :]

    width = (batch.total_tokens + 1) // 2
    ids = [
        batch.input_ids[r * width : min((r + 1) * width, batch.total_tokens)][None]
        for r in range(2)
    ]
    streams = [
        expand_hc(m.embed(tokens), m.layers[0].attn_mixes.copies)
        for m, tokens in zip(models, ids)
    ]
    outputs = [[], []]
    loads = [[[] for _ in m.layers] for m in models]
    consumed = [0, 0]
    for begin, end in zip(
        batch.cu_seqlens[:-1].tolist(), batch.cu_seqlens[1:].tolist()
    ):
        starts = [min(max(r * width - begin, 0), end - begin) for r in range(2)]
        sizes = [
            min(max((r + 1) * width - begin, 0), end - begin) - starts[r]
            for r in range(2)
        ]
        h, pre = [], []
        for r in range(2):
            h.append(streams[r][0][:, consumed[r] : consumed[r] + sizes[r]])
            pre.append(streams[r][1][:, consumed[r] : consumed[r] + sizes[r]])
            consumed[r] += sizes[r]
        hashes = [
            m.engram_hash(batch.input_ids[None, begin:end])[
                :, starts[r] : starts[r] + sizes[r]
            ]
            for r, m in enumerate(models)
        ]
        states = [AttentionState(), AttentionState()]
        for index in range(len(models[0].layers)):
            blocks = [m.layers[index] for m in models]
            for r, block in enumerate(blocks):
                if block.engram is not None:
                    h[r] = block.engram(
                        h[r], hashes[r][:, :, models[r].engram_layer_ids.index(index)]
                    )
            mixes = [b.attn_mixes(hh) for b, hh in zip(blocks, h)]
            local_x = [
                b.attn_norm(contract_hc(hh, pp)) for b, hh, pp in zip(blocks, h, pre)
            ]
            full_x = Gather.apply(*local_x)
            for r, block in enumerate(blocks):
                # Ownership arithmetic above is independent of ContiguousCPSequence.
                context = SimpleNamespace(
                    start=starts[r], gather=lambda x, r=r: full_x[r]
                )
                attn, states[r] = block.attn(local_x[r], states[r], cp_context=context)
                attn_pre, attn_post, attn_comb = mixes[r]
                hh = mix_residual(attn, h[r], attn_post, attn_comb)
                ffn_pre, ffn_post, ffn_comb = block.ffn_mixes(hh)
                x = block.ffn_norm(contract_hc(hh, attn_pre))
                y = block.ffn(x, load_sink=loads[r][index])
                h[r], pre[r] = mix_residual(y, hh, ffn_post, ffn_comb), ffn_pre
        for r in range(2):
            outputs[r].append(contract_hc(h[r], pre[r]))
    logits = [
        torch.nn.functional.linear(
            m.norm(torch.cat(parts, 1)).float(), m.head.weight.float()
        )[0]
        for m, parts in zip(models, outputs)
    ]
    return logits, loads


def _reference_losses(logits, batch):
    # Independent document-local next-token objective, before CP averaging.
    labels = torch.zeros_like(batch.labels)
    mask = torch.zeros_like(batch.labels, dtype=torch.float32)
    source_mask = torch.ones_like(mask) if batch.loss_mask is None else batch.loss_mask
    for begin, end in zip(
        batch.cu_seqlens[:-1].tolist(), batch.cu_seqlens[1:].tolist()
    ):
        labels[begin : end - 1] = batch.labels[begin + 1 : end]
        mask[begin : end - 1] = source_mask[begin + 1 : end]
    width = (len(labels) + 1) // 2
    return [
        (
            torch.nn.functional.cross_entropy(
                value, labels[r * width : r * width + len(value)], reduction='none'
            )
            * mask[r * width : r * width + len(value)]
        ).sum()
        / mask.sum().clamp_min(1)
        * 2
        for r, value in enumerate(logits)
    ]


def test_cp_direct_forward_requires_document_boundaries(moe, model_config):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    bundle = protocol.build_model(
        model_config, impl_cfg=protocol.ImplConfig(device='meta', quantized=False)
    )
    with pytest.raises(
        ValueError, match='CP requires explicit packed document boundaries'
    ):
        bundle.chunks[0](
            torch.ones(1, 4, dtype=torch.int64, device='meta'),
            cp_context=ContiguousCPSequence(7, 0, 2),
        )
