# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Selection/lifecycle identity only; CPU does not execute deployment kernels."""
import inspect

import torch

from megatron.lite.model.deepseek_v41.lite import protocol as native
from megatron.lite.model.deepseek_v41.lite.model import DeepseekV41Model as NativeModel
from megatron.lite.model.deepseek_v41.vllm import protocol
from megatron.lite.model.deepseek_v41.vllm.model import DeepseekV41Model, doc_positions
from megatron.lite.model.registry import get_train_runtime_module, resolve_runtime_model_name


def test_registry_reuses_native_lifecycle_and_dist_opt():
    name = resolve_runtime_model_name('deepseek_v41', 'vllm')
    assert name == 'deepseek_v41_vllm'
    assert get_train_runtime_module(name) is protocol
    for symbol in (
        'ImplConfig',
        'build_model_config',
        'forward_step',
        'load_hf_weights',
        'validate_impl_cfg_keys',
    ):
        assert getattr(protocol, symbol) is getattr(native, symbol)
    assert protocol.ImplConfig().optimizer == 'adam'
    assert inspect.signature(native.build_model).parameters['_model_class'].default is None


def test_deployment_forward_runs_on_the_native_parameter_tree():
    assert DeepseekV41Model._build_attention is NativeModel._build_attention
    assert DeepseekV41Model.set_input_tensor is NativeModel.set_input_tensor
    assert DeepseekV41Model.forward is not NativeModel.forward
    assert 'tracks.md' in DeepseekV41Model.deployment_providers['attention']


def test_packed_positions_restart_per_document():
    positions = doc_positions(torch.tensor([0, 3, 5, 9]), 9)
    assert positions.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
