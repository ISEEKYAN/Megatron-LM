# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from collections import defaultdict

import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import image_data, protocol
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.contracts import PackedBatch
from torch.utils.checkpoint import checkpoint


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_modality_bias_follows_committed_step(moe, model_config, device):
    torch.manual_seed(17)
    bundle = protocol.build_model(
        model_config,
        impl_cfg=protocol.ImplConfig(
            device=device,
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(256)),
            optimizer='muon',
            bias_rate=0.03,
            optimizer_config=OptimizerConfig(0.001, 5, 'quintic'),
        ),
    )
    model, optimizer = bundle.chunks[0], bundle.optimizer
    gates = [layer.ffn.gate for layer in model.layers]
    assert all(not gate.router.compute_aux_loss for gate in gates)
    counts = defaultdict(list)

    def observe(gate, args, output):
        indices = output[1]
        mask = args[1]
        if mask is None:
            mask = torch.zeros(len(indices), dtype=torch.bool, device=device)
        # Independent reference: count selected expert IDs, not returned stats.
        counts[gate].append(
            torch.stack(
                [
                    torch.nn.functional.one_hot(
                        indices[mask.reshape(-1) == modality], 4
                    ).sum(dim=(0, 1))
                    for modality in (False, True)
                ]
            )
        )

    handles = [gate.register_forward_hook(observe) for gate in gates]
    ids = torch.tensor([1, 99, 99, 99, 99, 2, 3, 4], device=device)
    image = image_data.ImageInput(
        1,
        torch.randn(4, 3, 14, 14, device=device),
        2,
        2,
        image_data.image_token_types(1, 1).to(device),
    )
    mixed = PackedBatch(
        ids,
        ids,
        torch.tensor([6, 2], device=device),
        torch.ones(8, device=device),
        extras={'images': [[image]]},
    )
    ids = torch.arange(13, 26, device=device)
    text = PackedBatch(
        ids, ids, torch.tensor([5, 8], device=device), torch.ones(13, device=device)
    )
    initial = [(g.bias.clone(), g.bias_vl.clone()) for g in gates]
    try:
        for step in range(4):
            optimizer.zero_grad()
            counts.clear()
            batches = [mixed, text] if step < 2 else [text]
            before = [torch.stack((g.bias, g.bias_vl)).clone() for g in gates]
            run_microbatch_loop(model, iter(batches), len(batches), bundle.forward_step)
            expected = []
            for gate, old in zip(gates, before):
                total = torch.stack(counts[gate]).sum(0).float()
                expected.append(
                    old + 0.03 * torch.sign(total.mean(-1, keepdim=True) - total)
                )
                torch.testing.assert_close(
                    torch.stack((gate.bias, gate.bias_vl)),
                    old,
                    rtol=0,
                    atol=0,
                    msg='forward/backward mutated bias',
                )
            if step == 0:
                # Last packed sample has no images: retaining only its stats must fail.
                assert any(
                    not torch.equal(want[1], old[1])
                    for want, old in zip(expected, before)
                )
            if step == 1:
                parameter = next(p for p in model.parameters() if p.grad is not None)
                parameter.grad.fill_(float('nan'))
            committed = optimizer.step()[0]
            assert committed == (step != 1)
            for gate, old, want in zip(gates, before, expected):
                torch.testing.assert_close(
                    torch.stack((gate.bias, gate.bias_vl)),
                    want if committed else old,
                    rtol=0,
                    atol=1e-8,
                    msg='step-time modality bias differs from summed counts',
                )
            # A second successful step with no new forward must not reuse old stats.
            if step in (0, 1):
                if step == 1:
                    parameter.grad.zero_()
                assert optimizer.step()[0]
                for gate, want in zip(gates, expected if committed else before):
                    torch.testing.assert_close(
                        torch.stack((gate.bias, gate.bias_vl)),
                        want,
                        rtol=0,
                        atol=1e-8,
                        msg='stale load reused',
                    )
        # Abandoning a microbatch or running evaluation must not publish its loads.
        before = [torch.stack((g.bias, g.bias_vl)).clone() for g in gates]
        run_microbatch_loop(model, iter([mixed]), 1, bundle.forward_step)
        optimizer.zero_grad()
        assert optimizer.step()[0]
        model.eval()
        bundle.forward_step(model, mixed)
        model.train()
        with torch.no_grad():
            bundle.forward_step(model, mixed)
        assert optimizer.step()[0]
        for gate, old in zip(gates, before):
            torch.testing.assert_close(
                torch.stack((gate.bias, gate.bias_vl)),
                old,
                rtol=0,
                atol=0,
                msg='discarded/evaluation load published',
            )
        assert any(not torch.equal(g.bias, old[0]) for g, old in zip(gates, initial))
        assert any(not torch.equal(g.bias_vl, old[1]) for g, old in zip(gates, initial))
    finally:
        for handle in handles:
            handle.remove()


def test_moe_recompute_is_bias_pure(moe, model_config):
    from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model

    model = DeepseekV41Model(model_config, token_map=list(range(256)), quantized=False)
    ffn = model.layers[0].ffn.float()
    x = torch.randn(1, 7, 32, requires_grad=True)
    mask = torch.tensor([[False, True, False, True, False, True, False]])
    before = (ffn.gate.bias.clone(), ffn.gate.bias_vl.clone())
    calls = []
    handle = ffn.gate.register_forward_hook(lambda *args: calls.append(1))
    try:
        checkpoint(
            lambda value: ffn(value, image_mask=mask), x, use_reentrant=True
        ).square().sum().backward()
        assert len(calls) == 2, 'fixture must execute forward and recompute'
        for value, old in zip((ffn.gate.bias, ffn.gate.bias_vl), before):
            torch.testing.assert_close(
                value, old, rtol=0, atol=0, msg='forward/recompute mutated bias'
            )
    finally:
        handle.remove()
