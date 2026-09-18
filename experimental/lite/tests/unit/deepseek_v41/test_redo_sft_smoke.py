# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slurm GPU smoke on nv/dev c82b9495a (native hyper_connection required).

Select this file with MLITE_TEST_SELECTION in redo-v4parity-gpu4.sbatch.
DS41_REFERENCE_DIR must contain the pinned release config.json. This uses
random weights, FP32 diagnostic execution, and synthetic inactive archive
bytes: it proves pipeline execution, not checkpoint or native FP4 parity.
"""
import copy
import hashlib
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import pytest
import torch

RELEASE_SHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"
RELEASE_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def smoke_config(reference_dir):
    from megatron.lite.model.deepseek_v41.config import DeepseekV41Config
    from megatron.lite.primitive.modules.engram_lookup import prime_buckets

    raw = (Path(reference_dir) / "config.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RELEASE_SHA256
    original = json.loads(raw)
    release = copy.deepcopy(original)
    text = release["text_config"]
    overrides = dict(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=1,
        head_dim=32,
        qk_rope_head_dim=4,
        q_lora_rank=32,
        o_lora_rank=8,
        o_groups=1,
        index_n_heads=1,
        index_head_dim=32,
        index_topk=2,
        sliding_window=4,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        engram_n_heads=1,
        engram_vocab_size=7,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
        dspark_n_routed_experts=2,
        dspark_num_experts_per_tok=1,
        dspark_markov_rank=32,
    )
    text.update(overrides)
    text["engram_num_embeddings"] = (
        prime_buckets(
            text["engram_layer_ids"],
            text["engram_max_ngram_size"],
            text["engram_n_heads"],
            text["engram_vocab_size"],
        )
        .sum(dim=(1, 2))
        .tolist()
    )
    overrides["engram_num_embeddings"] = text["engram_num_embeddings"]
    config = DeepseekV41Config(release)
    assert len(config.topology) == text["num_hidden_layers"] == 40
    for key in (
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "candidate_source_layer_id",
        "engram_layer_ids",
        "num_nextn_predict_layers",
    ):
        assert text[key] == original["text_config"][key], key
    return config, overrides


@pytest.mark.gpus(1)
def test_release_sft_save_reload(tmp_path, capsys):
    from megatron.lite.model.deepseek_v41.lite import protocol
    from megatron.lite.model.deepseek_v41.vision_config import OptimizerConfig
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
    from megatron.lite.primitive.train_step import run_microbatch_loop
    from megatron.lite.runtime.contracts.data import PackedBatch
    from safetensors.torch import save_file

    started = time.perf_counter()
    report = dict(
        status="failed",
        job_id=os.environ.get("SLURM_JOB_ID"),
        release_revision=RELEASE_REVISION,
        release_sha256=RELEASE_SHA256,
        weights="dummy random initialization; synthetic inactive archival bytes",
        execution="FP32, quantized=False; DP=1, TP=PP=CP=EP=1",
        loss_curve=[],
        completed_steps=0,
    )
    report_path = Path(
        os.environ.get("DS41_SMOKE_REPORT", "ds41-sft-smoke-report.json")
    )

    def phase(name):
        report["phase"] = name
        with capsys.disabled():
            print(f"DS41_SMOKE_PHASE={name}", flush=True)

    try:
        assert report["job_id"], "GPU smoke must run inside Slurm"
        assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
        assert int(os.environ.get("WORLD_SIZE", "1")) == 1
        report["gpu"] = torch.cuda.get_device_name()
        torch.manual_seed(351)
        torch.cuda.manual_seed_all(351)
        phase("build")
        config, report["numeric_overrides"] = smoke_config(
            os.environ["DS41_REFERENCE_DIR"]
        )
        impl = protocol.ImplConfig(
            device="cuda",
            dtype=torch.float32,
            quantized=False,
            token_map=list(range(64)),
            trainable_engram=True,
            optimizer="muon",
            optimizer_config=OptimizerConfig(
                lr=1e-3, ns_steps=2, coefficient_type="quintic"
            ),
        )
        bundle = protocol.build_model(config, impl_cfg=impl)
        model, optimizer = bundle.chunks[0], bundle.optimizer
        ids = torch.tensor([3, 7, 11, 5, 3, 7, 11, 5], device="cuda")
        batch = PackedBatch(
            input_ids=ids,
            labels=ids.clone(),
            seq_lens=torch.tensor([4, 4], device="cuda", dtype=torch.int32),
        )

        def backward():
            optimizer.zero_grad()
            return run_microbatch_loop(
                model,
                iter([batch]),
                1,
                bundle.forward_step,
                optimizer=optimizer,
                prepare_microbatches=bundle.extras["prepare_microbatches"],
            )

        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        for step in range(8):
            phase(f"sft_{step + 1}")
            output = backward()
            loss = float(output["loss"].detach())
            report["loss_curve"].append(loss)
            assert torch.isfinite(output["loss"]).all(), report
            assert optimizer.step()[0], "finite SFT step was rejected"
            report["completed_steps"] += 1
        curve = report["loss_curve"]
        assert len(set(curve)) > 1, curve
        assert sum(curve[-3:]) < sum(curve[:3]), curve
        report["changed_parameters"] = sum(
            not torch.equal(p, initial[name]) for name, p in model.named_parameters()
        )
        assert report["changed_parameters"] > 0
        del initial
        phase("nonfinite_skip")
        backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        gradients = [
            (
                getattr(p, "main_grad", None)
                if getattr(p, "main_grad", None) is not None
                else p.grad
            )
            for p in model.parameters()
        ]
        grad = next(g for g in gradients if g is not None)
        grad.flatten()[0] = float("nan")
        assert not optimizer.step()[0], "nonfinite step was accepted"
        assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
        report["nonfinite_step_skipped"] = True
        del before
        phase("save_reload")
        archive_dir = tmp_path / "dummy-archive"
        archive_dir.mkdir()
        archive = {
            name: torch.arange(17, dtype=torch.uint8)
            for name in model.archival_bindings
        }
        save_file(archive, str(archive_dir / "model.safetensors"))
        model.archival_store = SafeTensorReader(str(archive_dir))
        model.archival_keys = sorted(archive)
        model.eval()
        inference_batch = replace(batch, labels=None)
        with torch.no_grad():
            expected = bundle.forward_step(model, inference_batch)["logits"].clone()
        saved = tmp_path / "saved"
        protocol.save_hf_weights(
            [model],
            saved,
            config,
            model.ps,
            export_dtype="float32",
            buffer_max_size_bytes=65536,
        )
        restored = protocol.build_model(config, impl_cfg=impl)
        loaded = restored.chunks[0]
        protocol.load_hf_weights(loaded, str(saved), config, loaded.ps)
        loaded.eval()
        with torch.no_grad():
            actual = restored.forward_step(loaded, inference_batch)["logits"]
        assert torch.equal(actual, expected), (actual - expected).abs().max().item()
        report["reload_logits_bitwise_equal"] = True
        torch.cuda.synchronize()
        report["status"] = "passed"
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        with capsys.disabled():
            print("DS41_SFT_SMOKE_RESULT=" + json.dumps(report), flush=True)
