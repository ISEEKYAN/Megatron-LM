# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Debridge invariants for the DS4 GRPO hero path (TASK-1.1.12).

The hero path was moved OFF the self-built fp8-checkpoint bridge
(VllmCheckpointWorkerExtension + LayerClusterBuffer IPC, which shipped
SENDER-side pre-quantized block-fp8 and crashed the receiver on non-8-aligned
bucket offsets) ONTO the verl-native path: mlite exports bf16, and verl's
vLLMColocateWorkerExtension quantizes bf16->fp8 receiver-side via quant_weights
(gated by VERL_VLLM_FP8_QUANT_ENABLED=1 + is_fp8_model). These text-level
tripwires lock that wiring so a silent regression to the sender-side fp8 bridge
(or a stray FP8_QUANT=0 that disables receiver quant) fails on CPU, before GPU.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3] / "examples/verl"
_HERO_SH = _ROOT / "scripts/run_deepseek_v4_gsm8k_grpo.sh"
_HERO_SBATCH = _ROOT / "slurm/run_ds4_gsm8k_grpo.sbatch"


def _sh() -> str:
    return _HERO_SH.read_text(encoding="utf-8")


def _sbatch() -> str:
    return _HERO_SBATCH.read_text(encoding="utf-8")


def test_hero_selects_verl_native_worker_not_self_built_bridge() -> None:
    sh = _sh()
    # verl-native receiver drives update_weights + receiver-side quant.
    assert (
        "worker_extension_cls=verl.workers.rollout.vllm_rollout.utils"
        ".vLLMColocateWorkerExtension" in sh
    )
    # The retired self-built bridge must not be wired in as the extension class
    # (check the wiring value, not prose: the .sh comment names it on purpose).
    assert "worker_extension_cls=verl_mlite" not in sh
    assert "=verl_mlite.rollout.verl_worker.VllmCheckpointWorkerExtension" not in sh
    assert "VllmCheckpointWorkerExtension" not in _sbatch()


def test_hero_exports_bf16_and_not_sender_side_fp8() -> None:
    sh = _sh()
    assert "actor_rollout_ref.actor.engine.resync_format=bf16" in sh
    # Sender-side fp8 (expert_dtype=fp8 in resync_config) is what misaligned the
    # bucket; it must be gone so the bucket stays bf16 (always 8-aligned).
    assert "resync_config.expert_dtype=fp8" not in sh


def test_hero_enables_receiver_side_fp8_quant() -> None:
    sh = _sh()
    # The .sh must NOT hard-clobber FP8_QUANT to 0 (that was only correct for the
    # retired sender-side bridge). It must inherit so the sbatch's =1 flows to
    # verl, arming quant_weights on the bf16 stream.
    assert "export VERL_VLLM_FP8_QUANT_ENABLED=0" not in sh
    assert 'VERL_VLLM_FP8_QUANT_ENABLED="${VERL_VLLM_FP8_QUANT_ENABLED:-1}"' in sh
    # vLLM stays an fp8 model so is_fp8_model=True and quant_weights actually
    # fires on the incoming bf16 weights.
    assert "hf_overrides.quantization_config.quant_method=fp8" in sh


def test_sbatch_arms_fp8_quant_and_gates_bf16_resync() -> None:
    sbatch = _sbatch()
    assert "export VERL_VLLM_FP8_QUANT_ENABLED=1" in sbatch
    # Zero-GPU CONFIG_ONLY gate asserts the resolved config carries bf16 resync.
    assert '"resync_format: bf16"' in sbatch
