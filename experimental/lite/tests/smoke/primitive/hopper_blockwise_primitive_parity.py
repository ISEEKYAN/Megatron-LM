# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hopper blockwise BF16-weight primitive parity under torchrun.

The reference path uses the frozen Megatron-Core Transformer Engine wrappers.
It does not import MLite modules or precision helpers. Three complete reference
repeats are sealed before the target path is constructed. The target path then
exercises the production MLite primitives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


SEED = 1234
PROFILES = ("bf16", "hopper_blockwise_bf16_weight")
REPEATS = 3


class _OptionalNvrxBlocker(importlib.abc.MetaPathFinder):
    """Make the unused checkpoint-only NVRx integration deterministically absent."""

    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname == "nvidia_resiliency_ext" or fullname.startswith(
            "nvidia_resiliency_ext."
        ):
            raise ModuleNotFoundError(
                "NVRx is disabled for primitive parity", name=fullname
            )
        return None


def _block_optional_nvrx() -> None:
    if any(isinstance(finder, _OptionalNvrxBlocker) for finder in sys.meta_path):
        return
    if any(name.startswith("nvidia_resiliency_ext") for name in sys.modules):
        raise RuntimeError("NVRx was imported before the primitive parity isolation")
    sys.meta_path.insert(0, _OptionalNvrxBlocker())


@dataclass
class _Bundle:
    linear: torch.nn.Module
    attention: Any
    experts: Any
    parameters: dict[str, torch.nn.Parameter]
    coverage_manifest: Any = None
    reference_config: Any = None


def _stable_seed(label: str, rank: int) -> int:
    digest = hashlib.sha256(f"{SEED}:{rank}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _tensor(shape: tuple[int, ...], label: str, rank: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(label, rank))
    return torch.randn(shape, generator=generator, dtype=torch.bfloat16) * 0.02


class _ReferenceAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        from megatron.core.extensions.transformer_engine import (
            TEDotProductAttention,
            TELayerNormColumnParallelLinear,
            TENorm,
            TERowParallelLinear,
        )
        from megatron.core.transformer.enums import AttnMaskType

        self.qkv = TELayerNormColumnParallelLinear(
            input_size=1024,
            output_size=2048,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
            name="reference.attention.qkv",
        )
        self.proj = TERowParallelLinear(
            input_size=1024,
            output_size=1024,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="proj",
            name="reference.attention.proj",
        )
        self.q_norm = TENorm(config, 64, eps=1e-6)
        self.k_norm = TENorm(config, 64, eps=1e-6)
        self._attn_mask_type = AttnMaskType.causal
        self.core = TEDotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=self._attn_mask_type,
            attention_type="self",
            attention_dropout=0.0,
        )

    def forward(self, x: torch.Tensor):
        qkv, _ = self.qkv(x)
        qkv = qkv.view(*qkv.shape[:-1], 16, 64)
        q = self.q_norm(qkv[..., :8, :])
        k = self.k_norm(qkv[..., 8:12, :])
        v = qkv[..., 12:, :]
        core = self.core(q, k, v, None, self._attn_mask_type)
        if core.dim() > x.dim():
            core = core.reshape(*core.shape[:-2], 512)
        output, _ = self.proj(core)
        return output, core


def _named_te_parameters(prefix: str, module: torch.nn.Module):
    return {
        f"{prefix}.{name}": parameter for name, parameter in module.named_parameters()
    }


def _reference_config(profile: str):
    from megatron.core.transformer.transformer_config import TransformerConfig

    fp8 = "e4m3" if profile != "bf16" else None
    return TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
        ffn_hidden_size=4096,
        moe_ffn_hidden_size=4096,
        num_moe_experts=4,
        moe_router_topk=2,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_activation_fusion=True,
        deterministic_mode=True,
        fp8=fp8,
        fp8_recipe=("blockwise" if fp8 else None),
        fp8_param=False,
    )


def _build_reference(profile: str) -> _Bundle:
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoESubmodules
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.spec_utils import get_submodules

    config = _reference_config(profile)
    linear = TEColumnParallelLinear(
        input_size=1024,
        output_size=4096,
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=False,
        is_expert=False,
        name="reference.linear",
    )
    attention = _ReferenceAttention(config)
    layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=4, moe_grouped_gemm=True
    )
    mlp_submodules = get_submodules(layer_submodules.mlp)
    if not isinstance(mlp_submodules, MoESubmodules):
        raise TypeError("Megatron reference did not construct MoE submodules")
    expert_submodules = get_submodules(mlp_submodules.experts)
    if not isinstance(expert_submodules, GroupedMLPSubmodules):
        raise TypeError(
            "Megatron reference did not construct grouped expert submodules"
        )
    experts = TEGroupedMLP(
        2,
        config,
        expert_submodules,
        get_default_pg_collection(),
        name="reference.experts",
    )
    modules = {
        "linear": linear,
        "attention": attention,
        "experts.fc1": experts.linear_fc1,
        "experts.fc2": experts.linear_fc2,
    }
    parameters: dict[str, torch.nn.Parameter] = {}
    for prefix, module in modules.items():
        module.cuda().to(torch.bfloat16)
        parameters.update(_named_te_parameters(prefix, module))
    return _Bundle(linear, attention, experts, parameters, reference_config=config)


def _build_target(ps, profile: str) -> _Bundle:
    from megatron.lite.primitive.modules.experts import Experts
    from megatron.lite.primitive.modules.gqa import GQAttention
    from megatron.lite.primitive.parallel.linear import ColumnParallelLinear
    from megatron.lite.primitive.precision import (
        PrecisionCoverage,
        PrimitiveCapability,
        SemanticSite,
        precision_model_init_context,
        resolve_precision,
    )

    implementation = resolve_precision(profile)
    coverage = PrecisionCoverage(implementation) if implementation is not None else None
    init_context = (
        precision_model_init_context(implementation)
        if implementation is not None
        else nullcontext()
    )
    with init_context:
        linear = ColumnParallelLinear(
            1024,
            4096,
            ps,
            bias=False,
            sequence_parallel=True,
            precision_coverage=coverage,
            precision_site=(SemanticSite.ATTENTION_PROJECTION if coverage else None),
        )
        attention = GQAttention(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=64,
            ps=ps,
            rotary_percent=0.0,
            qkv_layout="flat",
            precision_coverage=coverage,
        )
        experts = Experts(
            SimpleNamespace(
                num_experts=4,
                hidden_size=1024,
                moe_intermediate_size=4096,
                swiglu_limit=0.0,
            ),
            ps,
            precision_coverage=coverage,
        )
        manifest = None
        if coverage is not None:
            coverage.require(
                linear,
                SemanticSite.ATTENTION_PROJECTION,
                frozenset({PrimitiveCapability.TE_LINEAR}),
                diagnostic="tp column linear",
            )
            coverage.require(
                attention.qkv,
                SemanticSite.ATTENTION_PROJECTION,
                frozenset({PrimitiveCapability.TE_LAYERNORM_LINEAR}),
                diagnostic="gqa qkv projection",
            )
            coverage.require(
                attention.proj,
                SemanticSite.ATTENTION_PROJECTION,
                frozenset({PrimitiveCapability.TE_LINEAR}),
                diagnostic="gqa output projection",
            )
            coverage.require(
                experts.fc1,
                SemanticSite.MOE_EXPERT,
                frozenset({PrimitiveCapability.TE_GROUPED_LINEAR}),
                diagnostic="expert fc1",
            )
            coverage.require(
                experts.fc2,
                SemanticSite.MOE_EXPERT,
                frozenset({PrimitiveCapability.TE_GROUPED_LINEAR}),
                diagnostic="expert fc2",
            )
            for owner, site, diagnostic in (
                (attention.core_attn, SemanticSite.ATTENTION_CORE, "TE DPA"),
                (attention.q_norm, SemanticSite.NORM, "query norm"),
                (attention.k_norm, SemanticSite.NORM, "key norm"),
                (object(), SemanticSite.ROUTER, "router boundary"),
                (object(), SemanticSite.EMBEDDING, "embedding boundary"),
                (object(), SemanticSite.LM_HEAD, "LM head boundary"),
            ):
                coverage.require(owner, site, diagnostic=diagnostic)
            manifest = coverage.seal()

    modules = {
        "linear": linear.linear,
        "attention.qkv": attention.qkv.linear,
        "attention.proj": attention.proj.linear,
        "attention.q_norm": attention.q_norm,
        "attention.k_norm": attention.k_norm,
        "experts.fc1": experts.fc1,
        "experts.fc2": experts.fc2,
    }
    parameters: dict[str, torch.nn.Parameter] = {}
    for module in (linear, attention, experts):
        module.cuda().to(torch.bfloat16)
    for prefix, module in modules.items():
        parameters.update(_named_te_parameters(prefix, module))
    return _Bundle(linear, attention, experts, parameters, manifest)


def _canonical_key(key: str) -> str:
    return key.replace("attention.qkv.linear.", "attention.qkv.").replace(
        "attention.proj.linear.", "attention.proj."
    )


def _canonical_parameters(bundle: _Bundle) -> dict[str, torch.nn.Parameter]:
    canonical = {_canonical_key(key): value for key, value in bundle.parameters.items()}
    if len(canonical) != len(bundle.parameters):
        raise RuntimeError("duplicate canonical parameter name")
    return canonical


def _make_artifact(reference: _Bundle, rank: int) -> dict[str, Any]:
    parameters = _canonical_parameters(reference)
    state = {
        name: _tensor(tuple(parameter.shape), f"parameter:{name}", rank)
        for name, parameter in parameters.items()
    }
    return {
        "parameters": state,
        "linear_input": _tensor((128, 2, 1024), "linear_input", rank),
        "attention_input": _tensor((128, 2, 1024), "attention_input", rank),
        "expert_input": _tensor((16, 1024), "expert_input", rank),
        "expert_probs": torch.sigmoid(_tensor((16,), "expert_probs", rank).float()).to(
            torch.bfloat16
        ),
        "tokens_per_expert": torch.tensor([7, 9], dtype=torch.int64),
    }


def _load_artifact(bundle: _Bundle, artifact: dict[str, Any]) -> None:
    parameters = _canonical_parameters(bundle)
    if parameters.keys() != artifact["parameters"].keys():
        raise RuntimeError(
            f"parameter mapping mismatch: target={sorted(parameters)} "
            f"artifact={sorted(artifact['parameters'])}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = artifact["parameters"][name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"shape mismatch for {name}: {value.shape} != {parameter.shape}"
                )
            parameter.copy_(value.to(device="cuda", dtype=parameter.dtype))


def _adamw_step(parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    master = parameter.detach().float().clone()
    grad = gradient.detach().float()
    beta1, beta2 = 0.9, 0.95
    first = (1.0 - beta1) * grad
    second = (1.0 - beta2) * grad.square()
    master.mul_(1.0 - 1.0e-4 * 0.1)
    master.addcdiv_(
        first / (1.0 - beta1),
        (second / (1.0 - beta2)).sqrt().add_(1.0e-8),
        value=-1.0e-4,
    )
    return master.cpu()


def _run(bundle: _Bundle, artifact: dict[str, Any], profile: str, reference: bool):
    _load_artifact(bundle, artifact)
    for parameter in bundle.parameters.values():
        parameter.grad = None
    fp8 = profile != "bf16"
    implementation = None
    forward_context = nullcontext()
    if not reference and fp8:
        from megatron.lite.primitive.precision import (
            precision_forward_context,
            resolve_precision,
        )

        implementation = resolve_precision(profile)
        forward_context = precision_forward_context(implementation)

    linear_input = artifact["linear_input"].cuda().requires_grad_(True)
    attention_input = artifact["attention_input"].cuda().requires_grad_(True)
    expert_input = artifact["expert_input"].cuda().requires_grad_(True)
    expert_probs = artifact["expert_probs"].cuda()
    tokens_per_expert = artifact["tokens_per_expert"].cuda()
    captured: dict[str, Any] = {}

    with forward_context:
        if reference:
            if fp8:
                from megatron.core.fp8_utils import get_fp8_context

                reference_context = get_fp8_context(bundle.reference_config)
            else:
                reference_context = nullcontext()
            with reference_context:
                linear_output, _ = bundle.linear(linear_input)
                attention_output, core_output = bundle.attention(attention_input)
                expert_output, _ = bundle.experts(
                    expert_input, tokens_per_expert, expert_probs
                )
        else:
            linear_output = bundle.linear(linear_input)

            def _capture_input(_module, args):
                captured["core_input_dtypes"] = tuple(value.dtype for value in args[:3])

            def _capture_output(_module, _args, output):
                captured["core_output"] = output.detach()

            pre_handle = bundle.attention.core_attn.register_forward_pre_hook(
                _capture_input
            )
            post_handle = bundle.attention.core_attn.register_forward_hook(
                _capture_output
            )
            try:
                attention_output = bundle.attention(attention_input)
            finally:
                pre_handle.remove()
                post_handle.remove()
            core_output = captured["core_output"]
            if core_output.dim() > attention_input.dim():
                core_output = core_output.reshape(*core_output.shape[:-2], 512)
            expert_output = bundle.experts(
                expert_input, tokens_per_expert, permuted_probs=expert_probs
            )

    loss = sum(
        output.float().square().mean()
        for output in (linear_output, attention_output, expert_output)
    )
    loss.backward()
    parameters = _canonical_parameters(bundle)
    metrics: dict[str, torch.Tensor] = {
        "linear.forward": linear_output.detach().cpu(),
        "linear.dx": linear_input.grad.detach().cpu(),
        "attention.forward": attention_output.detach().cpu(),
        "attention.core": core_output.detach().cpu(),
        "attention.dx": attention_input.grad.detach().cpu(),
        "experts.forward": expert_output.detach().cpu(),
        "experts.dx": expert_input.grad.detach().cpu(),
        "loss": loss.detach().cpu(),
    }
    for name, parameter in parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        metrics[f"gradient.{name}"] = parameter.grad.detach().cpu()
        metrics[f"update.{name}"] = _adamw_step(parameter, parameter.grad)
    if reference:
        metrics["boundary.dtypes"] = torch.tensor(
            [int(core_output.dtype is torch.bfloat16), 1, 1, 1], dtype=torch.int8
        )
    else:
        core_dtypes = captured["core_input_dtypes"]
        metrics["boundary.dtypes"] = torch.tensor(
            [
                int(all(dtype is torch.bfloat16 for dtype in core_dtypes)),
                int(core_output.dtype is torch.bfloat16),
                int(expert_probs.dtype is torch.bfloat16),
                int(
                    all(
                        parameter.dtype is torch.bfloat16
                        for name, parameter in parameters.items()
                        if "q_norm" in name or "k_norm" in name
                    )
                ),
            ],
            dtype=torch.int8,
        )
    if not all(
        torch.isfinite(value).all()
        for value in metrics.values()
        if value.is_floating_point()
    ):
        raise RuntimeError("non-finite primitive metric")
    return metrics


def _noise(repeats: list[dict[str, torch.Tensor]]) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for name in repeats[0]:
        values = [repeat[name] for repeat in repeats]
        max_abs = 0.0
        max_l2 = 0.0
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left = values[left_index].float()
                right = values[right_index].float()
                diff = left - right
                max_abs = max(max_abs, float(diff.abs().max()))
                denominator = max(
                    float(torch.linalg.vector_norm(left)),
                    float(torch.linalg.vector_norm(right)),
                    torch.finfo(torch.float32).tiny,
                )
                max_l2 = max(
                    max_l2, float(torch.linalg.vector_norm(diff)) / denominator
                )
        thresholds[name] = {
            "atol": 4.0 * max_abs,
            "rtol_l2": 4.0 * max_l2,
            "shape": list(values[0].shape),
            "dtype": str(values[0].dtype),
        }
    return thresholds


def _compare(
    reference: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    target_noise = _noise(targets)
    for name, expected in reference.items():
        gate = thresholds[name]
        for repeat_index, target in enumerate(targets):
            actual = target[name]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise AssertionError(
                    f"{name} repeat {repeat_index} shape/dtype mismatch"
                )
            if gate["atol"] == 0.0 and gate["rtol_l2"] == 0.0:
                if not torch.equal(actual, expected):
                    raise AssertionError(
                        f"{name} repeat {repeat_index} is not bitwise equal"
                    )
                max_abs = 0.0
                rel_l2 = 0.0
            else:
                diff = actual.float() - expected.float()
                max_abs = float(diff.abs().max())
                denominator = max(
                    float(torch.linalg.vector_norm(expected.float())),
                    torch.finfo(torch.float32).tiny,
                )
                rel_l2 = float(torch.linalg.vector_norm(diff)) / denominator
                if max_abs > gate["atol"] or rel_l2 > gate["rtol_l2"]:
                    raise AssertionError(
                        f"{name} repeat {repeat_index} exceeds frozen gate"
                    )
        if (
            target_noise[name]["atol"] > gate["atol"]
            or target_noise[name]["rtol_l2"] > gate["rtol_l2"]
        ):
            raise AssertionError(f"{name} target repeat noise exceeds reference gate")
        report[name] = {
            "atol": gate["atol"],
            "rtol_l2": gate["rtol_l2"],
            "target_noise_atol": target_noise[name]["atol"],
            "target_noise_rtol_l2": target_noise[name]["rtol_l2"],
        }
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 4:
        raise RuntimeError("primitive parity requires exactly four ranks")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False

    _block_optional_nvrx()
    from megatron.core import parallel_state as mcore_parallel_state
    from megatron.lite.primitive.parallel import init_parallel

    mcore_parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        order="tp-ep-dp-pp",
        create_gloo_process_groups=False,
    )
    ps = init_parallel(SimpleNamespace(tp=2, ep=2, etp=1, cp=1, pp=1, pp_layout=None))
    reference_bundle = _build_reference("bf16")
    artifact = _make_artifact(reference_bundle, rank)
    artifact_path = args.output / f"artifact-rank{rank}.pt"
    torch.save(artifact, artifact_path)
    artifact_hash = _sha256(artifact_path)
    del reference_bundle
    torch.cuda.empty_cache()

    all_reports: dict[str, Any] = {}
    for profile in PROFILES:
        reference_repeats = []
        for repeat_index in range(REPEATS):
            bundle = _build_reference(profile)
            repeat = _run(bundle, artifact, profile, reference=True)
            torch.save(
                repeat,
                args.output / f"reference-{profile}-repeat{repeat_index}-rank{rank}.pt",
            )
            reference_repeats.append(repeat)
            del bundle
            torch.cuda.empty_cache()

        thresholds = _noise(reference_repeats)
        threshold_path = args.output / f"thresholds-{profile}-rank{rank}.json"
        threshold_path.write_text(
            json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
        )
        if any(
            gate["atol"] != 0.0 or gate["rtol_l2"] != 0.0
            for gate in thresholds.values()
        ):
            raise RuntimeError(
                f"{profile} reference noise is non-zero; review the sealed threshold before target"
            )

        target_repeats = []
        for repeat_index in range(REPEATS):
            bundle = _build_target(ps, profile)
            repeat = _run(bundle, artifact, profile, reference=False)
            torch.save(
                repeat,
                args.output / f"target-{profile}-repeat{repeat_index}-rank{rank}.pt",
            )
            target_repeats.append(repeat)
            if profile != "bf16":
                assert bundle.coverage_manifest is not None
                assert len(bundle.coverage_manifest.entries) == 11
            del bundle
            torch.cuda.empty_cache()

        report = _compare(reference_repeats[0], target_repeats, thresholds)
        report_path = args.output / f"comparison-{profile}-rank{rank}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        all_reports[profile] = {
            "threshold_sha256": _sha256(threshold_path),
            "comparison_sha256": _sha256(report_path),
        }

    hashes = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(
        hashes, {"rank": rank, "artifact_sha256": artifact_hash, **all_reports}
    )
    if rank == 0:
        print("OPTIONAL_NVRX_DISABLED_FOR_PRIMITIVE_PARITY")
        manifest = {
            "seed": SEED,
            "repeats": REPEATS,
            "world_size": 4,
            "tp": 2,
            "ep": 2,
            "global_sequence_length": 256,
            "micro_batch_size": 2,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_experts": 4,
            "top_k": 2,
            "expert_padding": 16,
            "optimizer": {
                "name": "AdamW",
                "lr": 1.0e-4,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.1,
                "master": "fp32",
            },
            "ranks": hashes,
        }
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"manifest_sha256={_sha256(manifest_path)}")
        print("HOPPER_BLOCKWISE_BF16_BASELINE_PARITY_OK")
        print("HOPPER_BLOCKWISE_PRIMITIVE_PARITY_OK")
    dist.barrier()
    mcore_parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
