# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Global token-weighted CE under DP x CP gradient averaging (CPU oracle)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.lite.primitive.packed_lm import prepare_microbatches, text_output
from megatron.lite.runtime.contracts.loss import use_loss_context


@pytest.mark.parametrize('equal_tokens', [False, True])
@pytest.mark.parametrize('microbatches', [1, 2])
def test_cp_loss_matches_global_token_weighted_ce(
    monkeypatch, transformer_engine_import_stub, equal_tokens, microbatches
):
    import megatron.core.fp8_utils

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention.cp import ContiguousCPSequence

    # Two DP replicas, each split over two CP ranks; masks also exercise
    # next-token shifting and exclusion of document tails from normalization.
    world = object()
    records = []
    torch.manual_seed(47)
    weight = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    for dp in range(2):
        replica = []
        for mb in range(microbatches):
            n = 8 if equal_tokens else 7 + dp * 2 + mb * 2
            mask = torch.ones(n, dtype=torch.float64)
            if equal_tokens:
                mask[1] = 0  # shifted halves each have three valid targets
            batch = SimpleNamespace(
                input_ids=torch.zeros(n, dtype=torch.long),
                labels=torch.arange(n) % 5,
                loss_mask=mask,
                cu_seqlens=torch.tensor([0, n], dtype=torch.int32),
            )
            features = torch.randn(n, 3, dtype=torch.float64)
            replica.append((batch, features))
        records.append(replica)
    # Independent full-document oracle: no CP helpers or production denominator.
    numerator = sum(
        (
            F.cross_entropy((x @ weight)[:-1], b.labels[1:], reduction='none')
            * b.loss_mask[1:]
        ).sum()
        for replica in records
        for b, x in replica
    )
    total = sum(float(b.loss_mask[1:].sum()) for replica in records for b, _ in replica)
    reference = numerator / total
    local_totals, contributions = [], []
    for dp in range(2):
        for cp in range(2):
            replica = records[dp]
            expected_local = sum(
                float(
                    torch.cat((b.loss_mask[1:], b.loss_mask.new_zeros(1)))[
                        cp
                        * ((len(b.labels) + 1) // 2) : (cp + 1)
                        * ((len(b.labels) + 1) // 2)
                    ].sum()
                )
                for b, _ in replica
            )
            local_totals.append(expected_local)

            def all_reduce(tokens, *, group):
                assert float(tokens) == expected_local
                if group is world:
                    tokens.fill_(total)
                else:
                    # Mutation oracle: sum only DP peers sharing this CP rank.
                    assert group == ('dp-only', cp)
                    dp_total = 0.0
                    for peer in records:
                        for b, _ in peer:
                            shifted = torch.cat(
                                (b.loss_mask[1:], b.loss_mask.new_zeros(1))
                            )
                            width = (len(b.labels) + 1) // 2
                            dp_total += float(
                                shifted[cp * width : (cp + 1) * width].sum()
                            )
                    tokens.fill_(dp_total)

            monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
            monkeypatch.setattr(
                torch.distributed,
                'get_world_size',
                lambda group: 4 if group is world else 2,
            )
            prepared = prepare_microbatches(
                iter(b for b, _ in replica),
                microbatches,
                dp_group=world,
                cp_rank=cp,
                cp_size=2,
            )
            for (batch, context), (_, x) in zip(prepared, replica):
                partition = ContiguousCPSequence(len(batch.labels), cp, 2)
                logits = partition.slice(x @ weight, seq_dim=0)
                with use_loss_context(context):
                    contributions.append(
                        text_output(logits, batch, cp_context=partition)['loss']
                    )
    assert (len(set(local_totals)) == 1) == equal_tokens
    # Runtime averages microbatches; DDP averages the four rank contributions.
    actual = sum(contributions) / (4 * microbatches)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    (expected_grad,) = torch.autograd.grad(reference, weight, retain_graph=True)
    (actual_grad,) = torch.autograd.grad(actual, weight)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('cp_size', [1, 2])
def test_protocol_preparation_uses_the_ddp_reduction_domain(
    monkeypatch, transformer_engine_import_stub, cp_size
):
    import megatron.core.fp8_utils
    import megatron.core.transformer.hyper_connection

    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import model, protocol
    from megatron.lite.primitive.modules import native_fp32_linear
    from megatron.lite.runtime.contracts import ParallelConfig

    dp, dp_cp = object(), object()
    ps = SimpleNamespace(
        dp_group=dp, dp_cp_group=dp_cp, cp_size=cp_size, cp_rank=0, ep_size=1, pp_size=1
    )
    chunk = torch.nn.Module()
    chunk.engram_hash = chunk.engram_group = chunk.vision_schedule = None
    chunk.layers = []
    chunk.parameter_bindings = lambda: []
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(protocol, 'init_parallel', lambda config: ps)
    monkeypatch.setattr(model, 'DeepseekV41Model', lambda *args, **kwargs: chunk)
    monkeypatch.setattr(
        protocol, 'wrap_owned_ddp', lambda model, *args, **kwargs: model
    )
    monkeypatch.setattr(
        native_fp32_linear, 'configure_residual_projections', lambda *args: None
    )
    bundle = protocol.build_model(
        object(),
        impl_cfg=protocol.ImplConfig(
            device='cpu', dtype=torch.float32, parallel=ParallelConfig(cp=cp_size)
        ),
    )
    prepare = bundle.extras['prepare_microbatches']
    assert prepare.keywords['dp_group'] is (dp_cp if cp_size > 1 else dp)
    assert prepare.keywords['cp_size'] == cp_size
    assert prepare.keywords['cp_rank'] == 0
