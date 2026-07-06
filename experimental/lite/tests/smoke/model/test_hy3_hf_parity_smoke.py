from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


pytestmark = [
    pytest.mark.mlite,
    pytest.mark.smoke,
    pytest.mark.gpu,
    pytest.mark.distributed,
]


@pytest.fixture(scope="module", autouse=True)
def _single_cuda_rank():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Hy3 HF parity.")
    pytest.importorskip("transformer_engine.pytorch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.6.0" or not hasattr(transformers, "HYV3ForCausalLM"):
        pytest.skip("Hy3 parity requires Transformers 5.6.0 with HYV3ForCausalLM.")
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
    torch.cuda.set_device(0)
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")
        created = True
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _configs():
    from transformers import HYV3Config

    from megatron.lite.model.hy3.config import Hy3Config

    common = dict(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        intermediate_size=48,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=16,
        router_scaling_factor=2.826,
        rms_norm_eps=1e-5,
        max_position_embeddings=32,
        enable_moe_fp32_combine=False,
    )
    hf = HYV3Config(
        **common,
        mlp_layer_types=["dense", "sparse"],
        use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 11_158_840.0},
    )
    native = Hy3Config(
        **common,
        first_k_dense_replace=1,
        num_nextn_predict_layers=0,
    )
    return hf, native


def _build_models():
    from transformers import HYV3ForCausalLM

    from megatron.lite.model.hy3.lite import protocol
    from megatron.lite.runtime.contracts.config import ParallelConfig

    hf_config, native_config = _configs()
    hf_config._attn_implementation = "eager"
    torch.manual_seed(711)
    hf_model = HYV3ForCausalLM(hf_config).cuda().to(torch.bfloat16)
    impl = protocol.ImplConfig(
        parallel=ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1),
        optimizer=None,
        use_deepep=False,
        deterministic=True,
    )
    native_bundle = protocol.build_model(native_config, impl_cfg=impl)
    native_model = native_bundle.chunks[0]
    _copy_hf_to_native(hf_model, native_model, native_config)
    return hf_model, native_model, native_config


def _copy(target: torch.Tensor, source: torch.Tensor) -> None:
    assert target.shape == source.shape, (target.shape, source.shape)
    target.copy_(source.to(device=target.device, dtype=target.dtype))


@torch.no_grad()
def _copy_hf_to_native(hf_model, native_model, config) -> None:
    from megatron.lite.model.hy3.lite.checkpoint import Hy3WeightSpec

    spec = Hy3WeightSpec(config)
    _copy(native_model.embed.embedding.weight, hf_model.model.embed_tokens.weight)
    _copy(native_model.norm.weight, hf_model.model.norm.weight)
    _copy(native_model.head.col.linear.weight, hf_model.lm_head.weight)
    for index, (hf_layer, native_layer) in enumerate(
        zip(hf_model.model.layers, native_model.layers, strict=True)
    ):
        _copy(
            native_layer.attn.qkv.linear.layer_norm_weight,
            hf_layer.input_layernorm.weight,
        )
        packed_qkv = spec.hf_to_native(
            f"layers.{index}.attn.qkv.linear.weight",
            [
                hf_layer.self_attn.q_proj.weight,
                hf_layer.self_attn.k_proj.weight,
                hf_layer.self_attn.v_proj.weight,
            ],
        )
        _copy(native_layer.attn.qkv.linear.weight, packed_qkv)
        _copy(native_layer.attn.q_norm.weight, hf_layer.self_attn.q_norm.weight)
        _copy(native_layer.attn.k_norm.weight, hf_layer.self_attn.k_norm.weight)
        _copy(native_layer.attn.proj.linear.weight, hf_layer.self_attn.o_proj.weight)
        _copy(native_layer.mlp_norm.weight, hf_layer.post_attention_layernorm.weight)
        if native_layer.mlp is not None:
            _copy(
                native_layer.mlp.gate_up.linear.weight,
                torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight]),
            )
            _copy(native_layer.mlp.down.linear.weight, hf_layer.mlp.down_proj.weight)
            continue
        assert native_layer.moe is not None
        native_moe, hf_moe = native_layer.moe, hf_layer.mlp
        _copy(native_moe.router.gate.weight, hf_moe.gate.weight)
        _copy(native_moe.router.expert_bias, hf_moe.e_score_correction_bias)
        _copy(
            native_moe.shared_mlp.gate_up.linear.weight,
            torch.cat(
                [hf_moe.shared_experts.gate_proj.weight, hf_moe.shared_experts.up_proj.weight]
            ),
        )
        _copy(native_moe.shared_mlp.down.linear.weight, hf_moe.shared_experts.down_proj.weight)
        for expert in range(config.num_experts):
            _copy(
                getattr(native_moe.experts.fc1, f"weight{expert}"),
                hf_moe.experts.gate_up_proj[expert],
            )
            _copy(
                getattr(native_moe.experts.fc2, f"weight{expert}"),
                hf_moe.experts.down_proj[expert],
            )


def _batch_first(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    assert tensor.shape[1] == batch
    return tensor.transpose(0, 1).contiguous()


def _router_batch_first(tensor: torch.Tensor, batch: int, sequence: int) -> torch.Tensor:
    return tensor.reshape(sequence, batch, -1).transpose(0, 1).reshape(batch * sequence, -1)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol=3e-2):
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    assert actual_f.shape == expected_f.shape, (name, actual_f.shape, expected_f.shape)
    max_abs = float((actual_f - expected_f).abs().max())
    assert torch.allclose(actual_f, expected_f, atol=atol, rtol=atol), (
        f"{name} mismatch: max_abs={max_abs}"
    )


def test_hy3_transformers_5_6_layer_router_logits_loss_and_gradient_parity():
    hf_model, native_model, config = _build_models()
    hf_model.train()
    native_model.train()
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
        device="cuda",
    )
    labels = torch.tensor(
        [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]],
        device="cuda",
    )
    hf_layers: list[torch.Tensor] = []
    native_layers: list[torch.Tensor] = []
    hf_router: list[tuple[torch.Tensor, torch.Tensor]] = []
    native_router: list[tuple[torch.Tensor, torch.Tensor]] = []
    hooks = []
    for layer in hf_model.model.layers:
        hooks.append(layer.register_forward_hook(lambda _m, _i, out: hf_layers.append(out)))
    for layer in native_model.layers:
        hooks.append(layer.register_forward_hook(lambda _m, _i, out: native_layers.append(out)))
    hooks.append(
        hf_model.model.layers[1].mlp.gate.register_forward_hook(
            lambda _m, _i, out: hf_router.append((out[1], out[2]))
        )
    )
    hooks.append(
        native_model.layers[1].moe.router.register_forward_hook(
            lambda _m, _i, out: native_router.append((out[0], out[1]))
        )
    )

    hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
    native_logits = _batch_first(native_model(input_ids=input_ids)["logits"], input_ids.shape[0])
    for hook in hooks:
        hook.remove()

    assert len(hf_layers) == len(native_layers) == config.num_hidden_layers
    for index, (native_layer, hf_layer) in enumerate(zip(native_layers, hf_layers, strict=True)):
        _assert_close(f"layer_{index}", _batch_first(native_layer, input_ids.shape[0]), hf_layer)
    assert len(hf_router) == len(native_router) == 1
    hf_scores, hf_indices = hf_router[0]
    native_scores, native_indices = native_router[0]
    native_scores = _router_batch_first(native_scores, *input_ids.shape)
    native_indices = _router_batch_first(native_indices, *input_ids.shape)
    assert torch.equal(native_indices, hf_indices)
    _assert_close("router_scores", native_scores, hf_scores, atol=2e-3)
    _assert_close("logits", native_logits, hf_logits)

    hf_loss = F.cross_entropy(hf_logits.flatten(0, 1).float(), labels.flatten())
    native_loss = F.cross_entropy(native_logits.flatten(0, 1).float(), labels.flatten())
    _assert_close("loss", native_loss, hf_loss, atol=2e-3)
    hf_loss.backward()
    native_loss.backward()

    from megatron.lite.model.hy3.lite.checkpoint import Hy3WeightSpec

    spec = Hy3WeightSpec(config)
    native_qkv_grad = native_model.layers[0].attn.qkv.linear.weight.grad
    hf_qkv_grad = spec.hf_to_native(
        "layers.0.attn.qkv.linear.weight",
        [
            hf_model.model.layers[0].self_attn.q_proj.weight.grad,
            hf_model.model.layers[0].self_attn.k_proj.weight.grad,
            hf_model.model.layers[0].self_attn.v_proj.weight.grad,
        ],
    )
    gradient_pairs = {
        "embedding_grad": (
            native_model.embed.embedding.weight.grad,
            hf_model.model.embed_tokens.weight.grad,
        ),
        "qkv_grad": (native_qkv_grad, hf_qkv_grad),
        "router_grad": (
            native_model.layers[1].moe.router.gate.weight.grad,
            hf_model.model.layers[1].mlp.gate.weight.grad,
        ),
        "head_grad": (native_model.head.col.linear.weight.grad, hf_model.lm_head.weight.grad),
    }
    for name, (native_grad, hf_grad) in gradient_pairs.items():
        assert native_grad is not None and hf_grad is not None
        _assert_close(name, native_grad, hf_grad, atol=8e-2)
