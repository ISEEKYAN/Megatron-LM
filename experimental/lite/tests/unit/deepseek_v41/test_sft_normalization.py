# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SFT normalization against independent, unpacked next-token CE."""

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.runtime.contracts import PackedBatch
from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context
from torch.nn import functional as F


@pytest.mark.parametrize('mask_kind', ['none', 'ones', 'weighted', 'zero'])
def test_packed_sft_matches_unpacked_weighted_mean(mask_kind):
    lengths = [1, 3, 5]
    ids = torch.arange(sum(lengths)) % 4
    weights = {
        'none': None,
        'ones': torch.ones(9, dtype=torch.float64),
        'weighted': torch.tensor(
            [8, 4, 0.5, 1.5, 2, 0, 1, 0.5, 0.5], dtype=torch.float64
        ),
        'zero': torch.zeros(9, dtype=torch.float64),
    }[mask_kind]
    batch = PackedBatch(ids, ids, torch.tensor(lengths), weights)
    logits = (
        torch.arange(36, dtype=torch.float64).reshape(9, 4) % 7 / 8
    ).requires_grad_()
    serial_logits = logits.detach().clone().requires_grad_()
    serial_mask = torch.ones(9, dtype=torch.float64) if weights is None else weights
    # Dense per-sequence baseline: predict labels[1:] from logits[:-1].
    numerators, counts, token_losses = [], [], []
    for scores, labels, mask in zip(
        serial_logits.split(lengths), ids.split(lengths), serial_mask.split(lengths)
    ):
        ce = F.cross_entropy(scores[:-1], labels[1:], reduction='none')
        numerators.append((ce * mask[1:]).sum())
        counts.append(mask[1:].sum())
        token_losses.append(ce)
    total = torch.stack(counts).sum().clamp_min(1)
    reference = torch.stack(numerators).sum() / total
    local = protocol._text_output(logits, batch)
    offset = 0
    for length, ce in zip(lengths, token_losses):
        torch.testing.assert_close(
            -local['log_probs'][offset : offset + length - 1],
            ce,
            rtol=0,
            atol=0,
            msg='packed versus unpacked token CE',
        )
        offset += length
    # Grouped sums may differ by float64 reduction rounding; token CE is exact above.
    torch.testing.assert_close(
        local['loss'], reference, rtol=0, atol=1e-15, msg='packed local weighted mean'
    )
    # Uneven microbatches, each itself packed; preserve the caller's loss policy.
    batches = [
        PackedBatch(
            ids[:4],
            ids[:4],
            torch.tensor([1, 3]),
            None if weights is None else weights[:4],
        ),
        PackedBatch(
            ids[4:],
            ids[4:],
            torch.tensor([5]),
            None if weights is None else weights[4:],
        ),
    ]
    context = LossContext(source_batch='source', return_log_probs=False)
    prepared = protocol.prepare_microbatches(iter((b, context) for b in batches), 2)
    actual = []
    for (microbatch, ctx), scores in zip(prepared, logits.split([4, 5])):
        assert ctx.source_batch == 'source' and not ctx.return_log_probs
        assert ctx.normalization_denominator == float(total) / 2, 'CE token denominator'
        with use_loss_context(ctx):
            actual.append(protocol._text_output(scores, microbatch)['loss'] / 2)
    loss = sum(actual)
    torch.testing.assert_close(
        loss,
        reference,
        rtol=0,
        atol=1e-15,
        msg='prepared packed versus unpacked weighted mean',
    )
    loss.backward()
    reference.backward()
    torch.testing.assert_close(
        logits.grad,
        serial_logits.grad,
        rtol=0,
        atol=0,
        msg='packed versus unpacked per-token gradient',
    )


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('policy', ['native', 'external'])
def test_verl_runtime_sft_uses_one_global_token_denominator(count, policy):
    # Execute the engine's actual adapter without importing optional VERL/CUDA
    # dependencies; only the TensorDict transport is replaced by this CPU seam.
    import ast
    from pathlib import Path
    from types import SimpleNamespace, MethodType
    import weakref
    from megatron.lite.primitive.train_step import run_microbatch_loop

    source = (
        Path(__file__).parents[3] / 'examples/verl/verl_mlite/engine/mlite_engine.py'
    )
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == 'MegatronLiteEngine'
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == '_forward_backward_batch_with_runtime'
    )
    namespace = dict(
        TensorDict=object,
        Any=object,
        torch=torch,
        get_device_id=lambda: 'cpu',
        tu=SimpleNamespace(
            get_non_tensor_data=lambda **kw: 5, assign_non_tensor=lambda *a, **k: None
        ),
        _VerlMetric=None,
        weakref=weakref,
        LossContext=LossContext,
        PackedBatch=PackedBatch,
    )
    hook = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == '_make_runtime_loss_fn'
    )
    exec(
        compile(ast.Module(body=[method, hook], type_ignores=[]), str(source), 'exec'),
        namespace,
    )
    ids = torch.tensor([0, 1, 2, 3, 1, 2, 0])
    # Five predicted tokens, split unevenly across two logical sequences.
    batches = (
        [PackedBatch(ids, ids, torch.tensor([3, 4]))]
        if count == 1
        else [
            PackedBatch(row, row, torch.tensor([len(row)])) for row in ids.split([3, 4])
        ]
    )
    scores = (
        torch.arange(28, dtype=torch.float64).reshape(7, 4) % 7 / 8
    ).requires_grad_()
    expected = scores.detach().clone().requires_grad_()
    reference = (
        F.cross_entropy(expected[:2], ids[1:3], reduction='sum')
        + F.cross_entropy(expected[3:6], ids[4:7], reduction='sum')
    ) / 5

    class Batch:
        def __init__(self, batch):
            self.batch = batch

        def to(self, device):
            return self

    def forward_backward(handle, data_iter, **kwargs):
        pieces = iter(scores.split([7] if count == 1 else [3, 4]))
        output = run_microbatch_loop(
            None,
            data_iter,
            count,
            lambda _, batch: protocol._text_output(next(pieces), batch),
            prepare_microbatches=handle._extras['prepare_microbatches'],
            loss_fn=kwargs['loss_fn'],
        )
        return SimpleNamespace(
            metrics={}, model_output=SimpleNamespace(loss=output['loss'])
        )

    engine = SimpleNamespace(
        handle=SimpleNamespace(
            _extras={'prepare_microbatches': protocol.prepare_microbatches}
        ),
        get_data_parallel_size=lambda: 1,
        _make_runtime_batch=lambda batch: batch.batch,
        _make_runtime_loss_context=lambda batch, **kw: LossContext(
            source_batch=batch, **kw
        ),
        is_mp_src_rank_with_outputs=lambda: False,
        get_data_parallel_group=lambda: None,
        _build_verl_model_output=lambda **kw: kw['raw_output'],
        engine_config=SimpleNamespace(router_replay_mode='disabled'),
        runtime=SimpleNamespace(forward_backward=forward_backward),
    )
    engine._make_runtime_loss_fn = MethodType(
        namespace['_make_runtime_loss_fn'], engine
    )

    def external_loss(model_output, data, **kwargs):
        pieces = model_output['log_probs'].split(data.batch.seq_lens.tolist())
        return -sum(row[:-1].sum() for row in pieces) / 5, {}

    namespace['_forward_backward_batch_with_runtime'](
        engine,
        data=None,
        micro_batches=[Batch(b) for b in batches],
        indices=None,
        loss_function=external_loss if policy == 'external' else None,
        forward_only=False,
    )
    reference.backward()
    assert torch.equal(scores.grad, expected.grad), 'VERL_SFT_GLOBAL_TOKEN_GRADIENT'
