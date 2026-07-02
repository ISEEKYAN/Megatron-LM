"""Static tests for dense Qwen2 exact-route readiness."""

from __future__ import annotations

import pytest

from pathlib import Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _exact_qwen2_reference_dir() -> Path:
    return _workspace_root() / "references" / "external" / "hf-deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B"


def test_qwen2_config_reads_exact_fig14_target_hf_config():
    from megatron.lite.model.qwen2.config import Qwen2Config

    cfg = Qwen2Config.from_hf(str(_exact_qwen2_reference_dir()))

    assert cfg.num_hidden_layers == 28
    assert cfg.hidden_size == 1536
    assert cfg.num_attention_heads == 12
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 128
    assert cfg.qkv_size == (12 + 2 * 2) * 128
    assert cfg.intermediate_size == 8960
    assert cfg.vocab_size == 151936
    assert cfg.max_position_embeddings == 131072
    assert cfg.rope_theta == 10000
    assert cfg.attention_bias is True
    assert cfg.tie_word_embeddings is False


def test_qwen2_registry_resolves_exact_target_lite_runtime_name():
    from megatron.lite.model.registry import (
        get_model_package,
        resolve_model_type_from_hf,
        resolve_runtime_model_name,
    )

    assert resolve_model_type_from_hf(str(_exact_qwen2_reference_dir())) == "qwen2"
    assert get_model_package("qwen2").Qwen2Config.__name__ == "Qwen2Config"
    assert resolve_runtime_model_name("qwen2", "lite") == "qwen2"


def test_qwen2_protocol_declares_distopt_training_contract():
    from dataclasses import fields as dc_fields

    from megatron.lite.model.qwen2.lite import protocol as qwen2_protocol

    impl_fields = {field.name for field in dc_fields(qwen2_protocol.ImplConfig)}
    assert "optimizer" in impl_fields
    assert "optimizer_config" in impl_fields
    assert "deterministic" in impl_fields

    protocol_text = Path(qwen2_protocol.__file__).read_text()
    assert "build_dist_opt_training_optimizer" in protocol_text
    assert "attach_model_sharded_state_dict" in protocol_text
    assert "register_training_hooks" in protocol_text
    assert "Qwen2 dense lite optimizer construction is not implemented yet" not in protocol_text


def test_qwen2_protocol_builds_distopt_contract_with_monkeypatched_primitives(monkeypatch):
    pytest.importorskip("megatron.core", reason="dist_opt contract needs Megatron-Core")
    from types import SimpleNamespace

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite import protocol as qwen2_protocol
    from megatron.lite.runtime.contracts import OptimizerConfig

    calls = SimpleNamespace(optimizer=None, attach=None, hooks=None, finalized=False)
    fake_optimizer = SimpleNamespace(name="fake_dist_opt")

    def fake_build_dist_opt_optimizer(chunks, model_cfg, impl_cfg, ps):
        calls.optimizer = {
            "chunks": list(chunks),
            "model_cfg": model_cfg,
            "impl_cfg": impl_cfg,
            "ps": ps,
        }

        def fake_finalize_grads():
            calls.finalized = True

        return fake_optimizer, fake_finalize_grads

    def fake_attach_model_sharded_state_dict(chunks, ps, **kwargs):
        calls.attach = {"chunks": list(chunks), "ps": ps, "kwargs": kwargs}

    def fake_register_training_hooks(chunks, optimizer):
        calls.hooks = {"chunks": list(chunks), "optimizer": optimizer}

    monkeypatch.setattr(
        qwen2_protocol,
        "_build_dist_opt_optimizer",
        fake_build_dist_opt_optimizer,
    )

    import megatron.lite.primitive.ckpt as ckpt_module
    import megatron.lite.runtime.megatron_utils as megatron_utils

    monkeypatch.setattr(
        ckpt_module,
        "attach_model_sharded_state_dict",
        fake_attach_model_sharded_state_dict,
    )
    monkeypatch.setattr(megatron_utils, "register_training_hooks", fake_register_training_hooks)

    cfg = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        intermediate_size=16,
        max_position_embeddings=8,
    )
    opt_cfg = OptimizerConfig(lr=1e-5, weight_decay=0.0, clip_grad=1.0)
    impl_cfg = qwen2_protocol.ImplConfig(
        optimizer="dist_opt",
        optimizer_config=opt_cfg,
        deterministic=False,
        lora={"rank": 2, "alpha": 4, "target_modules": ["all-linear"]},
    )

    bundle = qwen2_protocol.build_model(cfg, impl_cfg=impl_cfg)

    assert calls.optimizer is not None
    assert calls.optimizer["model_cfg"] is cfg
    assert calls.optimizer["impl_cfg"] is impl_cfg
    assert calls.optimizer["chunks"] == bundle.chunks
    assert calls.attach is not None
    assert calls.attach["chunks"] == bundle.chunks
    assert calls.attach["ps"] is bundle.parallel_state
    assert calls.hooks == {"chunks": bundle.chunks, "optimizer": fake_optimizer}
    assert bundle.optimizer is fake_optimizer
    assert bundle.extras["optimizer_backend"] == "dist_opt"
    assert bundle.finalize_grads is not None
    bundle.finalize_grads()
    assert calls.finalized is True


def test_qwen2_protocol_builds_tiny_lora_forward_backward():
    import torch

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite.protocol import (
        ImplConfig,
        build_model,
        unpack_forward_output,
    )
    from megatron.lite.runtime.contracts.loss import LossContext, use_loss_context
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.manual_seed(1234)
    cfg = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    bundle = build_model(
        cfg,
        impl_cfg=ImplConfig(
            optimizer=None,
            lora={
                "rank": 2,
                "alpha": 4,
                "target_modules": ["all-linear"],
                "use_rslora": True,
            },
        ),
    )

    assert bundle.extras["adapter_lifecycle_supported"] is True
    assert bundle.extras["olora_tail_supported"] is True
    assert bundle.extras["lora_stats"]["chunks"][0]["lora_tensors"] > 0
    assert bundle.extras["lora_stats"]["chunks"][0]["trainable_tensors"] > 0

    model = bundle.chunks[0]
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        labels=torch.tensor([2, 3, 4, 5, 6], dtype=torch.long),
        seq_lens=torch.tensor([3, 2], dtype=torch.long),
        loss_mask=torch.ones(5),
    )
    assert bundle.forward_step is not None
    with use_loss_context(LossContext(temperature=0.7, calculate_entropy=True)):
        output = bundle.forward_step(model, batch)
    assert output["log_probs"].shape == (2, 3)
    assert output["entropy"].shape == (2, 3)
    assert unpack_forward_output(model, batch, output["log_probs"]).shape == (5,)
    loss = output["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    lora_grads = [
        param.grad
        for name, param in model.named_parameters()
        if "lora" in name.lower() and param.requires_grad
    ]
    base_grads = [
        param.grad
        for name, param in model.named_parameters()
        if "lora" not in name.lower() and "adapter" not in name.lower()
    ]
    assert lora_grads
    assert any(grad is not None and torch.isfinite(grad).all() for grad in lora_grads)
    assert all(grad is None for grad in base_grads)


def test_qwen2_protocol_lora_init_runs_as_post_load_hook():
    import torch

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite.protocol import ImplConfig, build_model

    torch.manual_seed(7)
    cfg = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    bundle = build_model(
        cfg,
        impl_cfg=ImplConfig(
            lora={
                "rank": 2,
                "alpha": 4,
                "target_modules": ["all-linear"],
                "init_lora_weights": "olora_tail",
            },
        ),
    )
    assert bundle.extras["lora_init"] == "olora_tail"
    assert bundle.extras["olora_tail_supported"] is True
    post_load_hook = bundle.extras["post_model_load_hook"]
    assert post_load_hook is not None

    model = bundle.chunks[0]
    layer = model.model.layers[0]
    pairs = [
        (layer.self_attn.qkv_lora, layer.self_attn.qkv),
        (layer.self_attn.proj_lora, layer.self_attn.proj),
        (layer.mlp.gate_up_lora, layer.mlp.gate_up),
        (layer.mlp.down_lora, layer.mlp.down),
    ]
    assert all(lora is not None for lora, _ in pairs)
    base_outs = []
    with torch.no_grad():
        for lora, base in pairs:
            x = torch.randn(3, base.weight.shape[1])
            base_outs.append((x, base(x)))
            assert lora.lora_b.abs().sum() == 0  # standard zero-delta before the hook

    result = post_load_hook()
    assert result["extras"]["lora_init_result"]["olora_initialized"] == len(pairs)

    with torch.no_grad():
        for (lora, base), (x, out0) in zip(pairs, base_outs, strict=True):
            assert lora.lora_b.abs().sum() > 0  # tail factors installed (non-zero delta)
            # PiSSA-style residual: base was shifted so base(x) + lora(x) == original base(x)
            torch.testing.assert_close(base(x) + lora(x), out0, atol=1e-4, rtol=1e-4)

def test_qwen2_protocol_lora_init_requires_enabled_lora():
    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite.protocol import ImplConfig, build_model

    cfg = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        intermediate_size=16,
        max_position_embeddings=8,
    )

    try:
        build_model(cfg, impl_cfg=ImplConfig(lora={"rank": 0}, lora_init="olora_tail"))
    except ValueError as exc:
        assert "requires enabled LoRA" in str(exc)
    else:
        raise AssertionError("dense Qwen2 OLoRA-tail accepted disabled LoRA")


def test_qwen2_hf_checkpoint_state_dict_round_trip(tmp_path):
    import torch

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite import protocol as qwen2_protocol
    from megatron.lite.primitive.parallel import ParallelState

    cfg = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        intermediate_size=12,
        max_position_embeddings=16,
    )
    ps = ParallelState()

    def make_tensor(shape, offset):
        return (torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32) + offset).view(
            shape
        )

    def make_hf_state():
        state = {
            "model.embed_tokens.weight": make_tensor((cfg.vocab_size, cfg.hidden_size), 1_000),
            "model.norm.weight": make_tensor((cfg.hidden_size,), 2_000),
            "lm_head.weight": make_tensor((cfg.vocab_size, cfg.hidden_size), 3_000),
        }
        for layer_idx in range(cfg.num_hidden_layers):
            base = 10_000 * (layer_idx + 1)
            prefix = f"model.layers.{layer_idx}"
            state[f"{prefix}.input_layernorm.weight"] = make_tensor((cfg.hidden_size,), base + 1)
            state[f"{prefix}.self_attn.q_proj.weight"] = make_tensor(
                (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size), base + 10
            )
            state[f"{prefix}.self_attn.k_proj.weight"] = make_tensor(
                (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size), base + 20
            )
            state[f"{prefix}.self_attn.v_proj.weight"] = make_tensor(
                (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size), base + 30
            )
            state[f"{prefix}.self_attn.q_proj.bias"] = make_tensor(
                (cfg.num_attention_heads * cfg.head_dim,), base + 35
            )
            state[f"{prefix}.self_attn.k_proj.bias"] = make_tensor(
                (cfg.num_key_value_heads * cfg.head_dim,), base + 36
            )
            state[f"{prefix}.self_attn.v_proj.bias"] = make_tensor(
                (cfg.num_key_value_heads * cfg.head_dim,), base + 37
            )
            state[f"{prefix}.self_attn.o_proj.weight"] = make_tensor(
                (cfg.hidden_size, cfg.hidden_size), base + 40
            )
            state[f"{prefix}.post_attention_layernorm.weight"] = make_tensor(
                (cfg.hidden_size,), base + 50
            )
            state[f"{prefix}.mlp.gate_proj.weight"] = make_tensor(
                (cfg.intermediate_size, cfg.hidden_size), base + 60
            )
            state[f"{prefix}.mlp.up_proj.weight"] = make_tensor(
                (cfg.intermediate_size, cfg.hidden_size), base + 70
            )
            state[f"{prefix}.mlp.down_proj.weight"] = make_tensor(
                (cfg.hidden_size, cfg.intermediate_size), base + 80
            )
        return state

    bundle = qwen2_protocol.build_model(cfg, impl_cfg=qwen2_protocol.ImplConfig())
    model = bundle.chunks[0]
    hf_state = make_hf_state()
    result = qwen2_protocol.load_hf_state_dict([model], hf_state, cfg, ps)
    assert result["loaded_native_tensors"] == 3 + cfg.num_hidden_layers * 7
    assert result["loaded_hf_tensors"] == len(hf_state)

    layer0 = model.model.layers[0]
    torch.testing.assert_close(model.model.embed_tokens.weight, hf_state["model.embed_tokens.weight"])
    torch.testing.assert_close(model.model.norm.weight, hf_state["model.norm.weight"])
    torch.testing.assert_close(model.lm_head.weight, hf_state["lm_head.weight"])
    torch.testing.assert_close(
        layer0.self_attn.qkv.weight,
        torch.cat(
            [
                hf_state["model.layers.0.self_attn.q_proj.weight"],
                hf_state["model.layers.0.self_attn.k_proj.weight"],
                hf_state["model.layers.0.self_attn.v_proj.weight"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        layer0.self_attn.qkv.bias,
        torch.cat(
            [
                hf_state["model.layers.0.self_attn.q_proj.bias"],
                hf_state["model.layers.0.self_attn.k_proj.bias"],
                hf_state["model.layers.0.self_attn.v_proj.bias"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        layer0.mlp.gate_up.weight,
        torch.cat(
            [
                hf_state["model.layers.0.mlp.gate_proj.weight"],
                hf_state["model.layers.0.mlp.up_proj.weight"],
            ],
            dim=0,
        ),
    )

    exported = qwen2_protocol.export_hf_state_dict([model], cfg, ps)
    assert sorted(exported) == sorted(hf_state)
    for key, tensor in hf_state.items():
        torch.testing.assert_close(exported[key], tensor)

    save_dir = tmp_path / "qwen2_hf"
    qwen2_protocol.save_hf_weights([model], save_dir, cfg, ps)
    target_bundle = qwen2_protocol.build_model(cfg, impl_cfg=qwen2_protocol.ImplConfig())
    qwen2_protocol.load_hf_weights(target_bundle.chunks[0], save_dir, cfg, ps)
    reexported = qwen2_protocol.export_hf_state_dict(target_bundle.chunks, cfg, ps)
    for key, tensor in hf_state.items():
        torch.testing.assert_close(reexported[key], tensor)

    missing = dict(hf_state)
    missing.pop("model.layers.0.self_attn.k_proj.weight")
    try:
        qwen2_protocol.load_hf_state_dict([model], missing, cfg, ps)
    except KeyError as exc:
        assert "self_attn.k_proj.weight" in str(exc)
    else:
        raise AssertionError("dense Qwen2 checkpoint loader accepted a missing k_proj tensor")

    missing_bias = dict(hf_state)
    missing_bias.pop("model.layers.0.self_attn.k_proj.bias")
    try:
        qwen2_protocol.load_hf_state_dict([model], missing_bias, cfg, ps)
    except KeyError as exc:
        assert "self_attn.k_proj.bias" in str(exc)
    else:
        raise AssertionError("dense Qwen2 checkpoint loader accepted a missing k_proj bias")

    bad_shape = dict(hf_state)
    bad_shape["model.layers.0.mlp.up_proj.weight"] = torch.zeros(
        cfg.intermediate_size + 1, cfg.hidden_size
    )
    try:
        qwen2_protocol.load_hf_state_dict([model], bad_shape, cfg, ps)
    except ValueError as exc:
        assert "up_proj.weight" in str(exc)
    else:
        raise AssertionError("dense Qwen2 checkpoint loader accepted a bad up_proj shape")

def test_qwen2_forward_step_loss_is_next_token_shifted():
    # Regression: the protocol must roll labels/mask per sequence (THD roll_labels
    # convention). Unshifted CE scores P(x_t|x_<=t) against x_t itself — on real
    # weights that read CE ~15 vs the HF reference 1.7 while every hidden state
    # matched (runs/20260702-qwen2-15b-realweight-smoke).
    import torch
    import torch.nn.functional as F

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite.protocol import ImplConfig, build_model, _forward_step
    from megatron.lite.runtime.contracts.data import PackedBatch

    torch.manual_seed(3)
    cfg = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    bundle = build_model(cfg, impl_cfg=ImplConfig())
    model = bundle.chunks[0]
    ids = torch.tensor([5, 9, 2, 7, 11, 3], dtype=torch.long)
    batch = PackedBatch(
        input_ids=ids,
        labels=ids.clone(),
        seq_lens=torch.tensor([4, 2], dtype=torch.long),
        loss_mask=torch.ones(6),
    )
    with torch.no_grad():
        out = _forward_step(model, batch)
        # reference: per-sequence explicit next-token CE from the raw logits
        logits = model(
            input_ids=torch.stack([ids[:4], torch.cat([ids[4:], ids.new_zeros(2)])])
        )["logits"].float()
    # masked mean over shifted positions: (sum of per-target CE) / (#targets = 3+1)
    per_target = []
    for row, (row_ids, n) in zip(logits, ((ids[:4], 4), (ids[4:], 2)), strict=True):
        per_target.append(F.cross_entropy(row[: n - 1], row_ids[1:n], reduction="sum"))
    expected = torch.stack(per_target).sum() / 4.0
    torch.testing.assert_close(out["loss"].float(), expected, atol=1e-4, rtol=1e-4)


def _tiny_lora_bundle(seed: int):
    import torch

    from megatron.lite.model.qwen2.config import Qwen2Config
    from megatron.lite.model.qwen2.lite.protocol import ImplConfig, build_model

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    bundle = build_model(
        cfg,
        impl_cfg=ImplConfig(
            lora={"rank": 2, "alpha": 4, "target_modules": ["all-linear"], "use_rslora": True}
        ),
    )
    return cfg, bundle


def test_qwen2_lora_adapter_save_load_round_trip(tmp_path):
    import torch

    from megatron.lite.model.qwen2.lite.protocol import load_lora_adapter, save_lora_adapter

    cfg, src = _tiny_lora_bundle(11)
    _, dst = _tiny_lora_bundle(22)
    with torch.no_grad():
        for name, p in src.chunks[0].named_parameters():
            if "lora" in name:
                p.copy_(torch.randn_like(p))

    out = save_lora_adapter(
        src.chunks, cfg, src.parallel_state, tmp_path / "adapter",
        lora_config=src.extras["lora_config"],
    )
    assert (out / "adapter_model.safetensors").is_file()
    assert (out / "adapter_config.json").is_file()
    assert (out / "megatron.lite_adapter_meta.json").is_file()

    result = load_lora_adapter(
        dst.chunks, out, cfg, dst.parallel_state, lora_config=dst.extras["lora_config"]
    )
    assert result["loaded_lora_modules"] == 2 * 4  # 2 layers x (qkv, proj, gate_up, down)

    src_params = dict(src.chunks[0].named_parameters())
    for name, p in dst.chunks[0].named_parameters():
        if "lora" in name:
            assert torch.equal(p, src_params[name]), name


def test_qwen2_lora_adapter_peft_key_syntax_and_strictness(tmp_path):
    import torch

    from megatron.lite.model.qwen2.lite.protocol import (
        export_lora_adapter_state,
        load_lora_adapter_state,
    )

    cfg, bundle = _tiny_lora_bundle(33)
    state = export_lora_adapter_state(bundle.chunks, cfg, bundle.parallel_state)
    # 2 layers x 7 PEFT projections x (A, B)
    assert len(state) == 2 * 7 * 2
    sample = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert sample in state
    # fused surfaces share lora_A across their PEFT projections
    for proj in ("k_proj", "v_proj"):
        assert torch.equal(
            state[sample], state[sample.replace("q_proj", proj)]
        )

    with pytest.raises(ValueError, match="unexpected keys"):
        load_lora_adapter_state(
            bundle.chunks, {**state, "base_model.model.model.bogus": state[sample]},
            cfg, bundle.parallel_state,
        )

    broken = dict(state)
    k_key = sample.replace("q_proj", "k_proj")
    broken[k_key] = broken[k_key] + 1.0
    with pytest.raises(ValueError, match="identical\\s+lora_A"):
        load_lora_adapter_state(bundle.chunks, broken, cfg, bundle.parallel_state)
