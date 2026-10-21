# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Selection/lifecycle identity only; CPU does not execute deployment kernels."""
import inspect
from types import SimpleNamespace
from unittest.mock import patch

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
    assert 'FlashMLA' in DeepseekV41Model.deployment_providers['attention']
    assert (
        inspect.signature(DeepseekV41Model.forward).parameters['labels'].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_packed_positions_restart_per_document():
    positions = doc_positions(torch.tensor([0, 3, 5, 9]), 9)
    assert positions.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]


def _two_layers(recompute):
    """Layer 0 owns a latent and writes the indexer top-k; layer 1 reads both, then
    overwrites the top-k, so a replay that read the live state would differ."""
    from megatron.lite.model.deepseek_v41.vllm import forward as F

    w0 = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64, requires_grad=True)
    w1 = torch.tensor([0.5, 3.0, -1.0], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64, requires_grad=True)
    state = F.ForwardState(rope=None)
    replays = []

    def layer0(h, replay=False):
        state.latent[0] = h * w0
        state.indexer.topk = torch.full((3,), 2.0, dtype=torch.float64)
        return (h.tanh(),)

    def layer1(h, replay=False):
        replays.append(replay)
        out = h * w1 * state.indexer.topk + state.latent[0].sin()
        state.indexer.topk = torch.full((3,), 5.0, dtype=torch.float64)
        return (out,)

    run = (lambda f, *a: F.recomputed(f, state, *a)) if recompute else (lambda f, *a: f(*a))
    (h,) = run(layer0, x)
    (h,) = run(layer1, h)
    (h * h).sum().backward()
    assert state.indexer.topk[0] == 5.0 and set(state.latent) == {0}
    return h.detach(), x.grad, w0.grad, w1.grad, replays


def test_recomputed_segments_replay_their_entry_state_bitwise():
    plain, recomputed = _two_layers(False), _two_layers(True)
    for a, b in zip(plain[:4], recomputed[:4]):
        assert torch.equal(a, b)
    assert recomputed[4] == [False, True]


def test_recompute_modes_match_native_names():
    import pytest

    from megatron.lite.model.deepseek_v41.vllm.forward import recompute_modes

    assert recompute_modes(None) == set() and recompute_modes(True) == {"full"}
    assert recompute_modes("attn,moe") == {"attn", "moe"}
    with pytest.raises(ValueError):
        recompute_modes("mlp")


def test_param_gather_finisher_publishes_each_bucket_once():
    """finish_param_gather fires Core's per-bucket readiness callback once per bucket and
    leaves unmarked parameters alone."""
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
    from megatron.lite.model.deepseek_v41.vllm.model import finish_param_gather

    calls = []
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2, bias=False))
    first, second = (lambda: calls.append(0)), (lambda: calls.append(1))
    setattr(model[0].weight, PARAM_READY_CALLBACK_ATTR, first)
    setattr(model[0].bias, PARAM_READY_CALLBACK_ATTR, first)
    setattr(model[1].weight, PARAM_READY_CALLBACK_ATTR, second)
    finish_param_gather(model)
    assert calls == [0, 1]


def test_native_hf_construction_recovers_release_token_map(tmp_path):
    from unittest.mock import Mock

    impl = protocol.ImplConfig(hf_path=str(tmp_path))
    model_cfg = SimpleNamespace(engram_layer_ids=[1])
    native_build = Mock(return_value=object())
    with patch.object(protocol, "release_token_map", return_value=[2, 0, 1]) as recover:
        with patch.object(native, "build_model", native_build):
            protocol.build_model(model_cfg, impl_cfg=impl)
            recovered = native_build.call_args.kwargs["impl_cfg"]
            assert recovered.token_map == [2, 0, 1]
            assert impl.token_map is None
            recover.assert_called_once_with(str(tmp_path))
            recover.reset_mock()
            (tmp_path / "token_map.pt").write_bytes(b"existence-only: native loader owns bytes")
            protocol.build_model(model_cfg, impl_cfg=impl)
            assert native_build.call_args.kwargs["impl_cfg"] is impl
            recover.assert_not_called()


def test_native_hf_construction_keeps_explicit_token_map(tmp_path):
    from unittest.mock import Mock

    impl = protocol.ImplConfig(hf_path=str(tmp_path), token_map=[1, 0])
    with patch.object(protocol, "release_token_map", side_effect=AssertionError("recomputed")):
        with patch.object(native, "build_model", Mock()) as build:
            protocol.build_model(SimpleNamespace(engram_layer_ids=[1]), impl_cfg=impl)
            assert build.call_args.kwargs["impl_cfg"] is impl
