# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HF-save entry-point contract for every registered Lite model protocol."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.lite.model.registry import TRAIN_RUNTIME_MODULES

LITE_ROOT = Path(__file__).resolve().parents[3]
_REGISTERED_PROTOCOLS = sorted(TRAIN_RUNTIME_MODULES.items())


@pytest.mark.parametrize(
    ("runtime_name", "module_name"),
    _REGISTERED_PROTOCOLS,
    ids=[runtime_name for runtime_name, _ in _REGISTERED_PROTOCOLS],
)
def test_registered_protocol_exposes_hf_save(
    runtime_name: str, module_name: str
) -> None:
    protocol_path = LITE_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    tree = ast.parse(protocol_path.read_text())
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert (
        "save_hf_weights" in functions
    ), f"{runtime_name} ({module_name}) cannot honor save_contents=['hf_model']"


@pytest.mark.parametrize("model_name", ["kimi_k2", "qwen3_moe"])
def test_new_hf_save_protocols_delegate_all_arguments(
    model_name: str, monkeypatch: pytest.MonkeyPatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    protocol = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.protocol"
    )
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    calls = []
    monkeypatch.setattr(
        checkpoint,
        "save_hf_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    chunks, model_cfg, parallel_state = object(), object(), object()

    protocol.save_hf_weights(chunks, "/tmp/hf-save-contract", model_cfg, parallel_state)

    assert calls == [((chunks, "/tmp/hf-save-contract", model_cfg, parallel_state), {})]


# Engine-side ``save_contents=['hf_model']`` unconditionally forwards
# ``export_dtype`` / ``target`` / ``resync_config`` — protocols that don't
# consume them must still accept them without raising ``TypeError``.
_ENGINE_EXPORT_KWARGS = {
    "export_dtype": "bfloat16",
    "target": "mxfp4",
    "resync_config": {"expert_dtype": "fp4"},
}


@pytest.mark.parametrize(
    "model_name", ["kimi_k2", "qwen3_moe", "qwen3_5", "deepseek_v4", "deepseek_v41"]
)
def test_hf_save_protocols_accept_engine_export_kwargs(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
    transformer_engine_import_stub,
    tmp_path,
) -> None:
    transformer_engine_import_stub()
    if model_name == "deepseek_v41":
        _check_v41_save(tmp_path)
        return
    if model_name == "deepseek_v4":
        # DS4 protocol drags in megatron.core via the CSA module at import time.
        pytest.importorskip(
            "megatron.core",
            reason="deepseek_v4 protocol needs megatron.core in the test env",
        )
    protocol = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.protocol"
    )
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    calls = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))

    # kimi_k2 / qwen3_moe use function-local imports of the checkpoint
    # writer; qwen3_5 aliases it at module load as ``_save_hf_weights_impl``.
    # Patch both surfaces so the test targets whichever the protocol uses.
    monkeypatch.setattr(checkpoint, "save_hf_weights", _record)
    if hasattr(protocol, "_save_hf_weights_impl"):
        monkeypatch.setattr(protocol, "_save_hf_weights_impl", _record)

    chunks, model_cfg, parallel_state = object(), object(), object()

    protocol.save_hf_weights(
        chunks,
        "/tmp/hf-save-kwargs",
        model_cfg,
        parallel_state,
        **_ENGINE_EXPORT_KWARGS,
    )

    assert calls == [
        (
            (chunks, "/tmp/hf-save-kwargs", model_cfg, parallel_state),
            _ENGINE_EXPORT_KWARGS,
        )
    ]


@pytest.mark.parametrize("model_name", ["qwen3_moe", "qwen3_5"])
def test_hf_save_checkpoint_warns_on_unused_export_kwargs(
    model_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    transformer_engine_import_stub,
) -> None:
    """Protocols without an MXFP4/block-FP8 save path warn (with kwargs printed)
    when engine-level export kwargs arrive, and never forward them to the
    shared writer.  Stays fully in-memory: the writer is monkeypatched."""
    transformer_engine_import_stub()
    checkpoint = importlib.import_module(
        f"megatron.lite.model.{model_name}.lite.checkpoint"
    )
    calls = []
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.save_hf_weights",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class _Config:
        vocab_size = 32

    model, parallel_state = object(), object()
    with caplog.at_level("WARNING", logger=checkpoint.__name__):
        checkpoint.save_hf_weights(
            model,
            "/tmp/hf-save-drop",
            _Config(),
            parallel_state,
            **_ENGINE_EXPORT_KWARGS,
        )

    assert len(calls) == 1
    forwarded_kwargs = calls[0][1]
    for key in _ENGINE_EXPORT_KWARGS:
        assert key not in forwarded_kwargs
    assert forwarded_kwargs.get("vocab_size") == 32

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "WARNING"
    ]
    assert any("ignoring unsupported kwargs" in msg for msg in warnings)
    joined = "\n".join(warnings)
    for key in _ENGINE_EXPORT_KWARGS:
        assert key in joined


def _v41_export_model(tmp_path):
    from megatron.lite.model.deepseek_v41.lite.checkpoint import CheckpointTensorStore
    from safetensors.torch import save_file

    archive_path = tmp_path / "archive.safetensors"
    archive = {"mtp.weight": torch.tensor([1.1234567, -2.7654321])}
    save_file(archive, archive_path)
    store = CheckpointTensorStore.load([archive_path], expected_keys=archive)
    values = {
        "model.weight": torch.arange(24, dtype=torch.float32).reshape(3, 8) / 7,
        "model.norm": torch.arange(8, dtype=torch.float32) / 3,
    }
    model = SimpleNamespace(
        local_layer_range=(0, 1),
        layers=[None],
        tensor_bindings={
            name: SimpleNamespace(role="weight", tensor=value, owner=None)
            for name, value in values.items()
        },
        archival_bindings=archive,
        archival_store=store,
        validate_parameter_bindings=lambda: None,
        config=SimpleNamespace(to_hf_dict=lambda: {"model_type": "deepseek_v41"}),
    )
    return model, values, archive


def _check_v41_save(tmp_path):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from safetensors.torch import load_file

    model, values, archive = _v41_export_model(tmp_path)
    destination = tmp_path / "saved"
    protocol.save_hf_weights(
        [model],
        destination,
        model.config,
        None,
        export_dtype="bfloat16",
        cpu=True,
        buffer_max_size_bytes=48,
    )
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in set(index["weight_map"].values()):
        part = load_file(destination / shard)
        assert sum(t.numel() * t.element_size() for t in part.values()) <= 48
        tensors.update(part)
    assert set(tensors) == set(values) | set(archive)
    for name, value in values.items():
        assert tensors[name].dtype == torch.bfloat16
        assert torch.equal(tensors[name], value.bfloat16())
    assert torch.equal(
        tensors["mtp.weight"].view(torch.uint8), archive["mtp.weight"].view(torch.uint8)
    )
    assert (
        json.loads((destination / "config.json").read_text())
        == model.config.to_hf_dict()
    )


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("export_dtype", ["bfloat16", torch.float16, None])
def test_v41_online_export_engine_kwargs(
    tmp_path, transformer_engine_import_stub, cpu, export_dtype
):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import protocol

    model, values, archive = _v41_export_model(tmp_path)
    tensors = dict(
        protocol.export_hf_weights(
            [model],
            model.config,
            None,
            export_dtype=export_dtype,
            cpu=cpu,
            buffer_max_size_bytes=16,
        )
    )
    dtype = (
        torch.bfloat16 if export_dtype == "bfloat16" else export_dtype or torch.float32
    )
    assert set(tensors) == set(values) | set(archive)
    for name, value in values.items():
        assert tensors[name].dtype == dtype
        assert tensors[name].device == (torch.device("cpu") if cpu else value.device)
        assert torch.equal(tensors[name], value.to(dtype))
        assert value.dtype == torch.float32
    assert torch.equal(
        tensors["mtp.weight"].view(torch.uint8), archive["mtp.weight"].view(torch.uint8)
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("target", "mxfp4"),
        ("resync_config", {}),
        ("buffer_max_size_bytes", 0),
        ("export_dtype", "int8"),
    ],
)
def test_v41_export_rejects_named_unsupported_options(
    tmp_path, transformer_engine_import_stub, option, value
):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import protocol

    model, _, _ = _v41_export_model(tmp_path)
    with pytest.raises((ValueError, TypeError), match=option):
        list(protocol.export_hf_weights([model], model.config, None, **{option: value}))
    with pytest.raises((ValueError, TypeError), match=option):
        protocol.save_hf_weights(
            [model], tmp_path / "invalid", model.config, None, **{option: value}
        )
    assert not (tmp_path / "invalid").exists()


def test_v41_conversion_copy_obeys_buffer_budget(
    tmp_path, transformer_engine_import_stub, monkeypatch
):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import protocol

    model, _, _ = _v41_export_model(tmp_path)
    original = torch.Tensor.copy_
    copies = []

    def record_copy(destination, source, *args, **kwargs):
        copies.append(
            source.numel() * max(source.element_size(), destination.element_size())
        )
        return original(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    list(
        protocol.export_hf_weights(
            [model],
            model.config,
            None,
            export_dtype="bfloat16",
            cpu=True,
            buffer_max_size_bytes=16,
        )
    )
    assert copies and max(copies) <= 16
    assert sum(copies) == 128


def test_v41_export_preserves_encoded_engram(tmp_path, transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.lite.engram import EngramTable

    model, _, _ = _v41_export_model(tmp_path)
    weight = (
        torch.arange(64, dtype=torch.float32).reshape(2, 32).to(torch.float8_e4m3fn)
    )
    scale = torch.ones(2, 1).to(torch.float8_e8m0fnu)
    table = EngramTable(weight, scale, trainable=False)
    model.tensor_bindings['model.engram.embed.weight'] = SimpleNamespace(
        role='engram_table', tensor=table.weight, owner=table
    )
    tensors = dict(
        protocol.export_hf_weights(
            [model],
            model.config,
            None,
            export_dtype="bfloat16",
            cpu=True,
            buffer_max_size_bytes=16,
        )
    )
    for key, expected in [('weight', weight), ('scale', scale)]:
        actual = tensors['model.engram.embed.' + key]
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_v41_default_engine_save_into_precreated_directory(
    tmp_path, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.model.deepseek_v41.lite import protocol
    from safetensors.torch import load_file

    model, values, _ = _v41_export_model(tmp_path)
    destination = tmp_path / "huggingface"
    destination.mkdir()  # The engine creates this directory before protocol dispatch.
    protocol.save_hf_weights(
        [model], destination, model.config, None, export_dtype="bfloat16"
    )
    tensors = load_file(destination / 'model.safetensors')
    assert all(tensors[name].dtype == torch.bfloat16 for name in values)


@pytest.mark.gpus(1)
@pytest.mark.parametrize("cpu", [False, True])
def test_v41_export_cpu_controls_gpu_tensor_destination(tmp_path, cpu):
    from megatron.lite.model.deepseek_v41.lite import protocol

    model, values, _ = _v41_export_model(tmp_path)
    for binding in model.tensor_bindings.values():
        binding.tensor = binding.tensor.cuda()
    tensors = dict(
        protocol.export_hf_weights(
            [model],
            model.config,
            None,
            export_dtype="bfloat16",
            cpu=cpu,
            buffer_max_size_bytes=16,
        )
    )
    assert all(t.device.type == ('cpu' if cpu else 'cuda') for t in tensors.values())
    for name, expected in values.items():
        assert tensors[name].dtype == torch.bfloat16
        assert torch.equal(tensors[name].cpu(), expected.bfloat16())
