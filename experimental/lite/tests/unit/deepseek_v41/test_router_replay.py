# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed V4.1 replay against checkpoints with provably disjoint Top-K sets."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import protocol
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
    attach_router_replay,
)
from megatron.lite.runtime.contracts import PackedBatch


@pytest.fixture
def isolated_replay_state():
    instances = RouterReplay.global_router_replay_instances[:]
    stats = RouterReplay.replay_stats()
    RouterReplay.clear_global_router_replay_instances()
    RouterReplay.reset_replay_stats()
    try:
        yield
    finally:
        RouterReplay.clear_global_state()
        RouterReplay.global_router_replay_instances[:] = instances
        RouterReplay.replay_calls = stats['calls']
        RouterReplay.replay_rows_total = stats['rows']
        RouterReplay.replay_rows_changed = stats['changed']


@pytest.mark.parametrize('masked', [False, True])
def test_concatenated_replay_forced_rank_swap_checkpoint(
    moe, model_config, tmp_path, isolated_replay_state, masked
):
    torch.manual_seed(104)
    bundle = protocol.build_model(
        model_config,
        impl_cfg=protocol.ImplConfig(
            device='cpu',
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
        ),
    )
    model = bundle.chunks[0]
    # Zero logits make selection depend ONLY on these strict, known bias ranks.
    # Alternate per layer, so exchanging recorded layer columns is observable.
    with torch.no_grad():
        for layer, block in enumerate(model.layers):
            gate = block.ffn.gate
            gate.router.gate.weight.zero_()
            bias = torch.tensor([4.0, 3.0, 2.0, 1.0])
            if layer % 2:
                bias = bias.flip(0)
            gate.bias.copy_(bias)
            gate.bias_vl.copy_(bias)
    checkpoint_a = deepcopy(model.state_dict())
    checkpoint_b = deepcopy(checkpoint_a)
    for name in checkpoint_b:
        if name.endswith(('.ffn.gate.bias', '.ffn.gate.bias_vl')):
            checkpoint_b[name].neg_()
    path = tmp_path / 'forced_rank_swap.pt'
    torch.save(checkpoint_b, path)

    batch = PackedBatch(
        input_ids=torch.tensor([1, 7, 4, 8, 3, 9, 2, 5]),
        labels=None,
        seq_lens=torch.tensor([3, 5]),
    )
    assert attach_router_replay(model) == 40, 'ALL_LAYERS_USE_SHARED_REPLAY'
    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    with torch.no_grad():
        logits_a = bundle.forward_step(model, batch)['logits']
    recorded_a = [r.detach().clone() for r in RouterReplay.get_recorded_data()]
    expected_a = [
        torch.tensor([2, 3] if layer % 2 else [0, 1]).expand(8, 2)
        for layer in range(40)
    ]
    for actual, expected in zip(recorded_a, expected_a, strict=True):
        assert torch.equal(actual, expected), 'CHECKPOINT_A_STRICT_EXPERT_RANKS'

    # Exercise the existing protocol's full jagged -> concatenated round trip.
    routes = protocol.unpack_recorded_routed_experts(model, batch, recorded_a)
    assert [row.shape for row in routes.unbind()] == [(3, 40, 2), (5, 40, 2)]
    packed_routes = protocol.pack_routed_experts(model, batch, routes)
    for actual, expected in zip(packed_routes, recorded_a, strict=True):
        assert torch.equal(actual, expected), 'PACK_UNPACK_PRESERVES_TOKEN_LAYER_ORDER'

    model.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        native_b = bundle.forward_step(model, batch)['logits']
    recorded_b = [r.detach().clone() for r in RouterReplay.get_recorded_data()]
    for actual, old in zip(recorded_b, expected_a, strict=True):
        assert torch.equal(actual, (old + 2) % 4), 'CHECKPOINT_B_FORCED_RANK_SWAP'
    # Selection-only biases cannot change live scores; any output difference
    # here is caused by the constructed expert-set swap, not a random checkpoint.
    assert not torch.allclose(native_b, logits_a), 'NATIVE_REROUTE_MUST_CHANGE_OUTPUT'

    mask = torch.tensor([True, False, True, False, True, False, True, False])
    if not masked:
        mask.fill_(True)
    replay_batch = replace(batch, routed_experts=routes, r3_replay_mask=mask)
    packed_mask = protocol.pack_r3_replay_mask(model, replay_batch)
    assert torch.equal(packed_mask, mask), 'PACK_MASK_PRESERVES_SAMPLE_BOUNDARIES'
    expected_routes = [
        torch.where(mask[:, None], old, new)
        for old, new in zip(recorded_a, recorded_b, strict=True)
    ]
    seen = [[] for _ in model.layers]
    handles = [
        block.ffn.gate.register_forward_hook(
            lambda module, args, output, layer=layer: seen[layer].append(
                output[1].detach().clone()
            )
        )
        for layer, block in enumerate(model.layers)
    ]
    normalized = []
    handles.append(
        model.norm.register_forward_hook(
            lambda module, args, output: normalized.append(output)
        )
    )

    def snapshot_gradients():
        return {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    try:
        RouterReplay.clear_global_state()
        RouterReplay.reset_replay_stats()
        RouterReplay.set_replay_data(packed_routes, packed_mask)
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        packed_logits = bundle.forward_step(model, replay_batch)['logits']
        packed_hidden = normalized.pop()
        for rows, expected in zip(seen, expected_routes, strict=True):
            assert [r.shape[0] for r in rows] == [3, 5], 'VISIT_BOTH_LOGICAL_SAMPLES'
            assert torch.equal(torch.cat(rows), expected), 'PACKED_REPLAY_EXACT_EXPERTS'
        stats = RouterReplay.replay_stats()
        assert stats == dict(calls=80, rows=640, changed=int(mask.sum()) * 40 * 2)
        if not masked:
            torch.testing.assert_close(packed_logits, logits_a, atol=0, rtol=0)
        packed_logits.square().sum().backward()
        packed_gradients = snapshot_gradients()
        assert packed_gradients, 'NONEMPTY_PACKED_BACKWARD'

        model.zero_grad(set_to_none=True)
        for begin, end in ((0, 3), (3, 8)):
            RouterReplay.clear_global_state()
            RouterReplay.set_replay_data(
                [r[begin:end] for r in recorded_a], mask[begin:end]
            )
            RouterReplay.set_global_router_replay_action(
                RouterReplayAction.REPLAY_FORWARD
            )
            # Independent calls bypass packed_forward/PackedRouterReplay slicing.
            model(batch.input_ids[None, begin:end])
        serial_hidden = torch.cat(normalized, dim=1)
        torch.testing.assert_close(
            packed_hidden, serial_hidden, atol=0, rtol=0, msg='PACKED_VS_SERIAL_HIDDEN'
        )
        # Match the final head GEMM shape. Separate 3/5-row FP32 GEMMs differ
        # by roundoff from an 8-row GEMM even for bitwise-identical hiddens.
        serial_logits = torch.nn.functional.linear(
            serial_hidden[0].float(), model.head.weight.float()
        )
        torch.testing.assert_close(
            packed_logits, serial_logits, atol=0, rtol=0, msg='PACKED_VS_SERIAL_LOGITS'
        )
        serial_logits.square().sum().backward()
        serial_gradients = snapshot_gradients()
        assert packed_gradients.keys() == serial_gradients.keys(), 'GRADIENT_OWNERSHIP'
        for name in packed_gradients:
            torch.testing.assert_close(
                packed_gradients[name],
                serial_gradients[name],
                atol=2e-5,
                rtol=2e-5,
                msg=f'PACKED_VS_SERIAL_GRADIENT {name}',
            )
    finally:
        for handle in handles:
            handle.remove()
