import json

import pytest
import torch


def test_math_prompts_are_fixed_and_cover_at_least_32_cases() -> None:
    from examples.verl.ds4_resync_tp4 import math_prompts

    prompts = math_prompts()
    assert len(prompts) >= 32
    assert len(set(prompts)) == len(prompts)
    assert all(prompt.startswith("Solve briefly:") for prompt in prompts)


def test_distribution_comparison_reports_kl_and_selected_token_delta() -> None:
    from examples.verl.ds4_resync_tp4 import compare_distributions

    reference = [
        {
            "logprobs": torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1),
            "token_ids": torch.tensor([2]),
        }
    ]
    candidate = [
        {
            "logprobs": torch.log_softmax(torch.tensor([[1.1, 1.9, 3.0]]), -1),
            "token_ids": torch.tensor([2]),
        }
    ]
    report = compare_distributions(reference, candidate)
    assert report["token_count"] == 1
    assert report["fp32"]["max_abs"] > 0
    assert report["fp32"]["max_kl"] > 0
    assert report["fp32"]["max_selected_token_logprob_delta"] >= 0
    assert report["bf16_rounded"]["max_abs"] > 0
    assert report["bf16_rounded"]["max_ratio_deviation"] >= 0


def test_payload_row_preserves_fp32_artifact_precision() -> None:
    from examples.verl.ds4_resync_tp4 import payload_row

    row = payload_row(
        [3, 7],
        torch.log_softmax(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dim=-1),
    )

    assert row["token_ids"].dtype == torch.int32
    assert row["logprobs"].dtype == torch.float32


def test_distribution_comparison_rejects_old_fp16_artifact() -> None:
    from examples.verl.ds4_resync_tp4 import compare_distributions

    fp32 = torch.log_softmax(torch.tensor([[1.0, 2.0]]), dim=-1)
    reference = [{"token_ids": torch.tensor([1]), "logprobs": fp32}]
    candidate = [{"token_ids": torch.tensor([1]), "logprobs": fp32.half()}]

    with pytest.raises(ValueError, match="must be FP32"):
        compare_distributions(reference, candidate)


def test_mlite_teacher_forcing_uses_previous_position_for_each_prompt_token() -> None:
    from examples.verl.ds4_resync_tp4 import mlite_payload_row

    logits = torch.tensor(
        [
            [
                [3.0, 1.0, 0.0, -1.0],
                [0.0, 4.0, 1.0, -2.0],
                [1.0, 0.0, 5.0, -3.0],
            ]
        ]
    )
    row = mlite_payload_row([0, 1, 2], logits)

    assert row["token_ids"].tolist() == [1, 2]
    torch.testing.assert_close(
        row["logprobs"], torch.log_softmax(logits[0, :-1].float(), dim=-1)
    )


def test_pure_fp8_config_overrides_mixed_checkpoint_experts() -> None:
    from examples.verl.ds4_resync_tp4 import pure_block_fp8_config

    config = pure_block_fp8_config(
        {
            "expert_dtype": "fp4",
            "quantization_config": {
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            },
        }
    )

    assert config["expert_dtype"] == "fp8"
    assert config["quantization_config"]["expert_dtype"] == "fp8"
    assert config["quantization_config"]["scale_fmt"] == "float32"


def test_online_resync_uses_native_checkpoint_reload_lifecycle(tmp_path) -> None:
    from examples.verl.ds4_resync_tp4 import reload_resync_checkpoint

    calls = []

    class LLM:
        def collective_rpc(self, method, *, args, timeout):
            calls.append((method, args, timeout))

    reload_resync_checkpoint(LLM(), tmp_path)

    assert calls == [("reload_checkpoint_from_path", (str(tmp_path),), None)]


def test_three_arm_comparison_reports_fp32_bf16_and_dapo_gate() -> None:
    from examples.verl.ds4_resync_tp4 import compare_three_arms

    base = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    cold = [{"token_ids": torch.tensor([2]), "logprobs": base}]
    online = [
        {
            "token_ids": torch.tensor([2]),
            "logprobs": torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0001]]), dim=-1),
        }
    ]
    mlite = [
        {
            "token_ids": torch.tensor([2]),
            "logprobs": torch.log_softmax(torch.tensor([[1.0, 2.0, 2.9999]]), dim=-1),
        }
    ]

    report = compare_three_arms(cold, online, mlite, minimum_prompts=1)

    assert set(report["pairs"]) == {
        "vllm_fp8_cold__vllm_fp8_online",
        "mlite_bf16__vllm_fp8_cold",
        "mlite_bf16__vllm_fp8_online",
    }
    pair = report["pairs"]["vllm_fp8_cold__vllm_fp8_online"]
    assert pair["fp32"]["max_ratio_deviation"] > 0
    assert pair["bf16_rounded"]["max_ratio_deviation"] >= 0
    assert report["gate"]["acceptable"] is True


def test_three_arm_comparison_rejects_token_misalignment() -> None:
    from examples.verl.ds4_resync_tp4 import compare_three_arms

    logprobs = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), dim=-1)
    direct = [{"token_ids": torch.tensor([2]), "logprobs": logprobs}]
    resync = [{"token_ids": torch.tensor([1]), "logprobs": logprobs}]

    with pytest.raises(ValueError, match="tokenized prompts differ"):
        compare_three_arms(direct, resync, direct, minimum_prompts=1)


def test_percentile_uses_exact_order_statistic() -> None:
    from examples.verl.ds4_resync_tp4 import percentile

    values = torch.arange(1, 101, dtype=torch.float32)
    assert percentile(values, 0.99) == 99.0


def test_copy_checkpoint_metadata_recurses_but_excludes_weights(tmp_path) -> None:
    from examples.verl.ds4_resync_tp4 import copy_checkpoint_metadata

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text("{}")
    (source / "inference").mkdir()
    (source / "inference" / "model.py").write_text("class Model: pass\n")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "model.safetensors.index.json").write_text('{"stale": true}')
    copy_checkpoint_metadata(source, output)
    assert (output / "config.json").read_text() == "{}"
    assert (output / "inference" / "model.py").read_text() == "class Model: pass\n"
    assert not (output / "model.safetensors").exists()
    assert not (output / "model.safetensors.index.json").exists()


def test_runtime_export_writes_pure_fp8_shards_and_index(tmp_path) -> None:
    from safetensors.torch import load_file

    from examples.verl.ds4_resync_tp4 import write_exported_checkpoint
    from megatron.lite.primitive.quantization.block_fp8 import quantize_block_fp8

    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    config = {
        "expert_dtype": "fp8",
        "quantization_config": {
            "quant_method": "fp8",
            "scale_fmt": "float32",
            "weight_block_size": [128, 128],
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors.index.json").write_text('{"stale": true}')
    dense = torch.linspace(-2.0, 2.0, 128 * 128).reshape(128, 128)
    expert = torch.linspace(-4.0, 4.0, 128 * 128).reshape(128, 128)
    dense_weight, dense_scale = quantize_block_fp8(dense, scale_format="float32")
    expert_weight, expert_scale = quantize_block_fp8(expert, scale_format="float32")
    weights = iter(
        [
            ("layers.2.attn.wo.weight", dense_weight),
            ("layers.2.attn.wo.scale", dense_scale),
            ("layers.2.ffn.experts.0.w1.weight", expert_weight),
            ("layers.2.ffn.experts.0.w1.scale", expert_scale),
        ]
    )

    write_exported_checkpoint(
        weights,
        source,
        output,
        max_shard_bytes=dense_weight.numel() * dense_weight.element_size() + 1,
    )

    converted_config = json.loads((output / "config.json").read_text())
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert converted_config["expert_dtype"] == "fp8"
    assert converted_config["quantization_config"]["scale_fmt"] == "float32"
    assert len(set(index["weight_map"].values())) >= 2
    tensors = {}
    for shard in set(index["weight_map"].values()):
        tensors.update(load_file(output / shard))
    assert tensors["layers.2.attn.wo.weight"].dtype == torch.float8_e4m3fn
    assert tensors["layers.2.attn.wo.scale"].dtype == torch.float32
    assert tensors["layers.2.ffn.experts.0.w1.weight"].dtype == torch.float8_e4m3fn
    assert tensors["layers.2.ffn.experts.0.w1.scale"].dtype == torch.float32


def test_pure_fp8_export_hook_changes_export_contract_only() -> None:
    from types import SimpleNamespace

    from examples.verl.ds4_resync_tp4 import configure_pure_block_fp8_export

    model_config = SimpleNamespace(
        expert_dtype="fp4",
        quantization_config={
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )

    configured = configure_pure_block_fp8_export(model_config)

    assert configured is model_config
    assert configured.expert_dtype == "fp8"
    assert configured.quantization_config == {
        "expert_dtype": "fp8",
        "quant_method": "fp8",
        "scale_fmt": "float32",
        "weight_block_size": [128, 128],
    }


def test_formal_sbatch_uses_mixed_source_for_mlite_and_fp8_artifact_for_vllm() -> None:
    script = (
        __import__("pathlib").Path(__file__).parents[3]
        / "examples/verl/slurm/run_ds4_resync_tp4.sbatch"
    ).read_text()

    assert "collect-mlite" in script
    assert "--model '${CHECKPOINT_DIR}'" in script
    assert "--fp8-output '${RESYNC_DIR}'" in script
    assert "collect --model '${RESYNC_DIR}'" in script
    assert "--resync-model '${RESYNC_DIR}'" in script
    assert 'export PYTHONPATH="${MLITE_SRC}/experimental/lite/examples/verl:' in script
    assert "convert --source" not in script


def test_model_vocab_size_includes_non_tokenizer_slots() -> None:
    from examples.verl.ds4_resync_tp4 import model_vocab_size

    class Config:
        vocab_size = 129280

    class Tokenizer:
        vocab_size = 128000

    assert model_vocab_size(Config(), Tokenizer()) == 129280
