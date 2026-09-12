# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from copy import deepcopy
import pytest
import torch
from megatron.lite.model.deepseek_v41.lite import image_data, protocol
from megatron.lite.model.deepseek_v41.lite.optimizer_groups import OptimizerConfig, VisionOptimizerConfig
from megatron.lite.primitive.train_step import run_microbatch_loop
from megatron.lite.runtime.contracts import PackedBatch


@pytest.mark.parametrize('trainable_engram', [False, True])
def test_staged_training_and_restart(moe, model_config, trainable_engram):
    mask = protocol.VisionTrainability(True, True, True, True)
    def build(external):
        return protocol.build_model(model_config, impl_cfg=protocol.ImplConfig(
            device='cpu', dtype=torch.float32, quantized=False, token_map=list(range(256)),
            trainable_engram=trainable_engram, vision_trainability=mask,
            external_vision_device='cpu' if external else None, optimizer='muon',
            optimizer_config=OptimizerConfig(0.001, 5, 'quintic',
                vision_policy=VisionOptimizerConfig(0.5, 1.0, 0.0))))
    actual, serial = build(True), build(False)
    serial.chunks[0].load_state_dict(actual.chunks[0].state_dict())
    ids = torch.tensor([1, 99, 99, 99, 99, 2])
    image = image_data.ImageInput(1, torch.randn(4, 3, 14, 14), 2, 2, image_data.image_token_types(1, 1))
    batch = PackedBatch(ids, ids, torch.tensor([6]), torch.ones(6), extras={'images': [[image]]})
    for bundle in (actual, serial):
        run_microbatch_loop(bundle.chunks[0], iter([batch, batch]), 2, bundle.forward_step)
    for p, q in zip(actual.chunks[0].parameters(), serial.chunks[0].parameters()):
        assert (p.grad is None) == (q.grad is None), 'gradient ownership'
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg='staged gradient')
    assert actual.optimizer.step() == serial.optimizer.step(), 'gradient norm/update'
    state = deepcopy((actual.chunks[0].state_dict(), actual.optimizer.state_dict()))
    resumed = build(True)
    resumed.chunks[0].load_state_dict(state[0])
    resumed.optimizer.load_state_dict(state[1])
    for bundle in (actual, resumed):
        bundle.optimizer.zero_grad()
        bundle.optimizer.reconfigure_vision(protocol.VisionTrainability(False, True, True, False))
        run_microbatch_loop(bundle.chunks[0], iter([batch]), 1, bundle.forward_step)
        assert bundle.optimizer.step()[0]
    for p, q in zip(actual.chunks[0].parameters(), resumed.chunks[0].parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0, msg='stage restart')
