"""BF16 master-state load and shared online-FP8 export contracts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.ckpt.hf_weights import to_global_layer_name
from megatron.lite.primitive.quantization.checkpoint_block_fp8 import (
    BLOCK_SHAPE,
    BlockFP8CheckpointDequantAdapter,
)
from megatron.lite.primitive.quantization.deployment_block_fp8 import (
    quantize_block_fp8_weight,
    requantize_block_fp8_weight,
)
from megatron.lite.primitive.quantization.mxfp4 import (
    dequantize_mxfp4,
)

_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_TOP_LEVEL = {
    "embed_tokens.embedding.weight": "embed.weight",
    "norm.weight": "norm.weight",
    "hc_head.hc_fn": "hc_head_fn",
    "hc_head.hc_base": "hc_head_base",
    "hc_head.hc_scale": "hc_head_scale",
    "lm_head.col.linear.weight": "head.weight",
}


def EXPERT_CLASSIFIER(name: str) -> bool:
    return ".experts." in name and ".shared_experts." not in name


def PLACEMENT_FN(param_name: str) -> list:
    from torch.distributed.tensor import Replicate

    del param_name
    return [Replicate(), Replicate(), Replicate(), Replicate()]


def _hf_names(native_name: str) -> list[str]:
    if native_name in _TOP_LEVEL:
        return [_TOP_LEVEL[native_name]]
    match = _LAYER_RE.match(native_name)
    if match is None:
        return []
    layer_idx, attr = match.groups()
    prefix = f"layers.{layer_idx}"
    direct = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "attn_hc.hc_fn": "hc_attn_fn",
        "attn_hc.hc_base": "hc_attn_base",
        "attn_hc.hc_scale": "hc_attn_scale",
        "ffn_hc.hc_fn": "hc_ffn_fn",
        "ffn_hc.hc_base": "hc_ffn_base",
        "ffn_hc.hc_scale": "hc_ffn_scale",
        "self_attn.q_norm": "attn.q_norm.weight",
        "self_attn.kv_norm": "attn.kv_norm.weight",
        "self_attn.wq_b": "attn.wq_b.weight",
        "self_attn.wo_a": "attn.wo_a.weight",
        "self_attn.wo_b": "attn.wo_b.weight",
        "self_attn.attn_sink": "attn.attn_sink",
        "self_attn.compressor.ape": "attn.compressor.ape",
        "self_attn.compressor.norm.weight": "attn.compressor.norm.weight",
        "self_attn.indexer.wq_b": "attn.indexer.wq_b.weight",
        "self_attn.indexer.weights_proj": "attn.indexer.weights_proj.weight",
        "self_attn.indexer.compressor.ape": "attn.indexer.compressor.ape",
        "self_attn.indexer.compressor.norm.weight": (
            "attn.indexer.compressor.norm.weight"
        ),
    }
    if attr in direct:
        return [f"{prefix}.{direct[attr]}"]
    if attr == "self_attn.fused_wqa_wkv":
        return [f"{prefix}.attn.wq_a.weight", f"{prefix}.attn.wkv.weight"]
    compressor_fused = {
        "self_attn.compressor.fused_wkv_wgate": "attn.compressor",
        "self_attn.indexer.compressor.fused_wkv_wgate": (
            "attn.indexer.compressor"
        ),
    }
    if attr in compressor_fused:
        base = f"{prefix}.{compressor_fused[attr]}"
        return [f"{base}.wkv.weight", f"{base}.wgate.weight"]
    if attr == "mlp.gate.gate.weight":
        return [f"{prefix}.ffn.gate.weight"]
    if attr == "mlp.gate.tid2eid":
        return [f"{prefix}.ffn.gate.tid2eid"]
    if attr == "mlp.gate.expert_bias":
        return [f"{prefix}.ffn.gate.bias"]
    if attr == "mlp.shared_experts.gate_up.weight":
        base = f"{prefix}.ffn.shared_experts"
        return [f"{base}.w1.weight", f"{base}.w3.weight"]
    if attr == "mlp.shared_experts.down.weight":
        return [f"{prefix}.ffn.shared_experts.w2.weight"]
    return []


def _is_block_fp8_weight(native_name: str) -> bool:
    return (
        ".self_attn.fused_wqa_wkv" in native_name
        or ".self_attn.wq_b" in native_name
        or ".self_attn.indexer.wq_b" in native_name
        or ".self_attn.wo_a" in native_name
        or ".self_attn.wo_b" in native_name
        or ".mlp.shared_experts.gate_up.weight" in native_name
        or ".mlp.shared_experts.down.weight" in native_name
        or re.match(
            r"^layers\.\d+\.mlp\.experts\.(?:w13|w2)\.\d+$",
            native_name,
        )
        is not None
    )


def _is_fp32_hf_tensor(native_name: str) -> bool:
    return native_name.endswith(
        (
            ".hc_fn",
            ".hc_base",
            ".hc_scale",
            ".attn_sink",
            ".compressor.ape",
            ".mlp.gate.expert_bias",
        )
    )


def _scale_to_float32(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.float32:
        return scale
    if scale.dtype == torch.uint8:
        return (scale.to(torch.int32) << 23).view(torch.float32)
    if scale.dtype == torch.float8_e8m0fnu:
        return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    raise TypeError(f"unsupported checkpoint block scale dtype {scale.dtype}")


class DeepseekV4WeightSpec:
    """Small shared-loader adapter for skeleton-owned state."""

    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        source_block_fp8: bool = True,
    ):
        self.config = config
        self.source_block_fp8 = source_block_fp8
        self.ps = ParallelState()
        self.source_block_scales: dict[str, torch.Tensor] = {}
        self.export_source_block_scales: dict[str, torch.Tensor] = {}

    @property
    def num_experts(self) -> int:
        return self.config.n_routed_experts

    def weight_map(self) -> dict[str, list[str]]:
        return {}

    def validate_load(self, ps: ParallelState) -> None:
        if (ps.tp_size, ps.etp_size) != (1, 1):
            raise NotImplementedError(
                "vLLM checkpoint load requires TP/ETP=1; PP/CP are supported."
            )
        if ps.ep_size <= 0 or self.config.n_routed_experts % ps.ep_size:
            raise ValueError(
                f"EP={ps.ep_size} must divide {self.config.n_routed_experts} routed experts."
            )
        self.ps = ps

    def _expert_hf_names(self, native_name: str) -> list[str]:
        match = re.match(r"^layers\.(\d+)\.mlp\.experts\.(w13|w2)\.(\d+)$", native_name)
        if match is None:
            return []
        layer, kind, local_id = match.groups()
        local_count = self.config.n_routed_experts // self.ps.ep_size
        expert_id = self.ps.ep_rank * local_count + int(local_id)
        base = f"layers.{layer}.ffn.experts.{expert_id}"
        if kind == "w13":
            return [f"{base}.w1.weight", f"{base}.w3.weight"]
        return [f"{base}.w2.weight"]

    def _names(self, native_name: str) -> list[str]:
        return self._expert_hf_names(native_name) or _hf_names(native_name)

    def _load_names(self, native_name: str) -> list[str]:
        names = self._names(native_name)
        if not self.source_block_fp8 or not _is_block_fp8_weight(native_name):
            return names
        paired: list[str] = []
        for name in names:
            if not name.endswith(".weight"):
                raise ValueError(
                    f"block-FP8 checkpoint source must end in .weight: {name}"
                )
            paired.extend((name, name.removesuffix(".weight") + ".scale"))
        return paired

    def load_weight_map(
        self,
        base_model: nn.Module,
        ps: ParallelState,
        logical_state_keys: tuple[str, ...],
    ) -> dict[str, list[str]]:
        self.validate_load(ps)
        layer_map = (
            {
                local_idx: base_model.layer_indices[local_idx]
                for local_idx in range(len(base_model.layer_indices))
            }
            if hasattr(base_model, "layer_indices")
            else {}
        )
        return {
            global_name: names
            for name in logical_state_keys
            if (global_name := to_global_layer_name(name, layer_map))
            if (names := self._load_names(global_name))
        }

    def hf_to_native(
        self, native_name: str, hf_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        if self.source_block_fp8 and _is_block_fp8_weight(native_name):
            source_names = self._names(native_name)
            if len(hf_tensors) != 2 * len(source_names):
                raise ValueError(
                    f"{native_name} requires weight/scale pairs for "
                    f"{len(source_names)} source weights"
                )
            dequant = BlockFP8CheckpointDequantAdapter()
            masters: list[torch.Tensor] = []
            fp8_scales: list[torch.Tensor] = []
            for source_name, qweight, source_scale in zip(
                source_names,
                hf_tensors[::2],
                hf_tensors[1::2],
                strict=True,
            ):
                if qweight.dtype == torch.float8_e4m3fn:
                    scale = _scale_to_float32(source_scale)
                    master = dequant(qweight, source_scale)
                    reconstructed = requantize_block_fp8_weight(master, scale)
                    if not torch.equal(reconstructed.qweight, qweight):
                        mismatch = int((reconstructed.qweight != qweight).sum().item())
                        raise RuntimeError(
                            f"{source_name} is not reversible through BF16 master: "
                            f"{mismatch}/{qweight.numel()} FP8 values changed"
                        )
                    fp8_scales.append(scale)
                elif qweight.dtype == torch.int8:
                    # The release checkpoint is mixed: dense projections use
                    # block FP8 while routed experts use packed MXFP4 (two
                    # E2M1 values per int8 byte with one E8M0 scale per 32).
                    master = dequantize_mxfp4(qweight, source_scale).to(torch.bfloat16)
                else:
                    raise TypeError(
                        f"unsupported quantized checkpoint dtype for {source_name}: "
                        f"{qweight.dtype}"
                    )
                masters.append(master)
            if fp8_scales:
                if len(fp8_scales) != len(source_names):
                    raise RuntimeError(
                        f"{native_name} mixes FP8 and MXFP4 within one fused parameter"
                    )
                self.source_block_scales[native_name] = (
                    torch.cat(fp8_scales, dim=0)
                    if len(fp8_scales) > 1
                    else fp8_scales[0]
                )
            return torch.cat(masters, dim=0) if len(masters) > 1 else masters[0]
        output = (
            torch.cat(hf_tensors, dim=0)
            if len(hf_tensors) == 2
            else hf_tensors[0]
        )
        if native_name.endswith("tid2eid"):
            return output.to(dtype=torch.int32)
        if output.is_floating_point():
            return output.to(
                dtype=(
                    torch.float32
                    if _is_fp32_hf_tensor(native_name)
                    else torch.bfloat16
                )
            )
        return output

    def hf_target_shape(
        self, native_name: str, source_index: int, target_shape: torch.Size
    ) -> torch.Size:
        names = self._names(native_name)
        paired = self.source_block_fp8 and _is_block_fp8_weight(native_name)
        pair_index = source_index // 2 if paired else source_index
        if native_name.endswith("self_attn.fused_wqa_wkv"):
            rows = (self.config.q_lora_rank, self.config.head_dim)[pair_index]
            weight_shape = torch.Size((rows, target_shape[1]))
        elif len(names) == 2:
            weight_shape = torch.Size(
                (target_shape[0] // 2, *target_shape[1:])
            )
        else:
            weight_shape = target_shape
        if paired and source_index % 2:
            return torch.Size(
                tuple(
                    (size + block - 1) // block
                    for size, block in zip(
                        weight_shape, BLOCK_SHAPE, strict=True
                    )
                )
            )
        return weight_shape

    def read_hf_source_raw(
        self, native_name: str, source_index: int, source_name: str
    ) -> bool:
        """Keep explicit FP8 weight/scale pairs serialized until fused dequant."""
        del source_index, source_name
        return self.source_block_fp8 and _is_block_fp8_weight(native_name)

    def bind_source_scales(self, base_model: nn.Module) -> None:
        parameters = dict(base_model.named_parameters())
        missing = sorted(set(self.source_block_scales) - set(parameters))
        if missing:
            raise RuntimeError(
                "cannot bind reversible FP8 scales to parameters: "
                + ", ".join(missing)
            )
        registry: dict[str, torch.Tensor] = {}
        for native_name, scales in self.source_block_scales.items():
            parameter = parameters[native_name]
            parameter._fp8_source_scales = scales.to(
                device=parameter.device, dtype=torch.float32
            ).contiguous()
            parameter._fp8_source_scale_version = parameter._version
            # FSDP2 CPU offload/reload may replace its DTensor Parameter and
            # thereby drop arbitrary Parameter attributes.  Keep one small
            # CPU-side source-scale registry on the model as the durable
            # initial-sync contract.  Routed MXFP4 experts do not enter this
            # map, so its footprint is limited to checkpoint block-FP8 state.
            registry[native_name] = scales.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous()
        base_model._fp8_source_scales_by_name = registry
        base_model._fp8_source_scales_valid = True

    @staticmethod
    def replica_group_for_load(native_name: str, ps: ParallelState):
        if _is_block_fp8_weight(native_name):
            # The shared loader only broadcasts native BF16 tensors, not the
            # source scales required by the reversible deployment contract.
            return None
        if EXPERT_CLASSIFIER(native_name):
            # EP ranks own disjoint expert IDs.  Only expert-DP replicas may
            # share a load; broadcasting over dense DP would copy rank 0's
            # local experts onto every EP rank.
            return ps.ep_dp_group
        return ps.dp_cp_group

    @staticmethod
    def replica_source_rank_for_load(native_name: str, ps: ParallelState) -> int:
        del native_name, ps
        return 0

    def native_to_hf(
        self, native_name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        names = self._names(native_name)
        source_scales = getattr(tensor, "_fp8_source_scales", None)
        source_scale_version = getattr(tensor, "_fp8_source_scale_version", None)
        source_scales_from_export_registry = False
        if source_scales is None and native_name in self.export_source_block_scales:
            source_scales = self.export_source_block_scales[native_name]
            source_scales_from_export_registry = True
        source_scales_are_current = (
            source_scales is not None
            and (
                source_scales_from_export_registry
                or source_scale_version == tensor._version
            )
        )
        if len(names) == 2:
            if native_name.endswith("self_attn.fused_wqa_wkv"):
                row_sizes = [self.config.q_lora_rank, self.config.head_dim]
                tensors = tensor.split(row_sizes, dim=0)
            else:
                row_sizes = [tensor.shape[0] // 2] * 2
                tensors = tensor.chunk(2, dim=0)
        else:
            row_sizes = [tensor.shape[0]] if names else []
            tensors = (tensor,) if names else ()
        if _is_block_fp8_weight(native_name):
            scale_tensors = (
                source_scales.split(
                    [rows // BLOCK_SHAPE[0] for rows in row_sizes], dim=0
                )
                if source_scales_are_current
                else (None,) * len(tensors)
            )
            outputs: list[tuple[str, torch.Tensor]] = []
            for name, source, scale in zip(
                names, tensors, scale_tensors, strict=True
            ):
                canonical = (
                    requantize_block_fp8_weight(source, scale)
                    if scale is not None
                    else quantize_block_fp8_weight(source)
                )
                outputs.extend(
                    (
                        (name, canonical.qweight),
                        (
                            name.removesuffix(".weight") + ".scale",
                            canonical.scales,
                        ),
                    )
                )
            return outputs
        return list(zip(names, tensors, strict=True))

    def qkv_spec(self, native_name: str):
        del native_name
        return None

    def tp_spec(self, native_name: str):
        del native_name
        return None

    def is_expert(self, native_name: str) -> bool:
        return EXPERT_CLASSIFIER(native_name)

    def expert_global_id(self, native_name: str):
        match = re.match(r"^layers\.\d+\.mlp\.experts\.(?:w13|w2)\.(\d+)$", native_name)
        if match is None:
            return None
        local_count = self.config.n_routed_experts // self.ps.ep_size
        return self.ps.ep_rank * local_count + int(match.group(1))

    @staticmethod
    def expert_local_name(native_name: str, local_idx: int) -> str:
        if re.match(r"^layers\.\d+\.mlp\.experts\.(?:w13|w2)\.\d+$", native_name) is None:
            raise ValueError(f"not a routed-expert parameter: {native_name}")
        return native_name.rsplit(".", 1)[0] + f".{local_idx}"


def _models(chunks: nn.Module | Iterable[nn.Module]) -> list[nn.Module]:
    return [chunks] if isinstance(chunks, nn.Module) else list(chunks)


def invalidate_bound_source_scales(model: nn.Module) -> None:
    """Invalidate checkpoint scales after the first successful optimizer step."""

    model._fp8_source_scales_valid = False
    model._fp8_source_scales_by_name = {}


def _pipeline_export_source_scales(
    chunks: nn.Module | Iterable[nn.Module], ps: ParallelState
) -> dict[str, torch.Tensor]:
    """Replicate the small reversible-scale registry across PP stages.

    The common HF exporter already streams full parameters over TP/EP/PP.  The
    registry is model-specific metadata rather than state_dict data, so gather
    only these CPU scale tensors and let the common exporter own weight traffic.
    """

    local: dict[str, torch.Tensor] = {}
    for model in _models(chunks):
        if not bool(getattr(model, "_fp8_source_scales_valid", False)):
            continue
        layer_map = (
            {
                local_idx: model.layer_indices[local_idx]
                for local_idx in range(len(model.layer_indices))
            }
            if hasattr(model, "layer_indices")
            else {}
        )
        for native_name, scale in getattr(
            model, "_fp8_source_scales_by_name", {}
        ).items():
            global_name = to_global_layer_name(native_name, layer_map)
            value = scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
            previous = local.get(global_name)
            if previous is not None and not torch.equal(previous, value):
                raise RuntimeError(
                    f"conflicting reversible FP8 scales for {global_name}"
                )
            local[global_name] = value

    if ps.pp_size <= 1:
        return local
    if not dist.is_initialized() or ps.pp_group is None:
        raise RuntimeError("PP resync requires an initialized pipeline group")
    stage_registries: list[dict[str, torch.Tensor] | None] = [None] * ps.pp_size
    dist.all_gather_object(stage_registries, local, group=ps.pp_group)
    combined: dict[str, torch.Tensor] = {}
    for registry in stage_registries:
        if registry is None:
            raise RuntimeError("PP source-scale gather returned an empty stage")
        overlap = set(combined).intersection(registry)
        conflicts = sorted(
            name for name in overlap if not torch.equal(combined[name], registry[name])
        )
        if conflicts:
            raise RuntimeError(
                "conflicting reversible FP8 scales across PP stages: "
                + ", ".join(conflicts)
            )
        combined.update(registry)
    return combined


def export_hf_weights(
    chunks: nn.Module | Iterable[nn.Module],
    model_cfg: DeepseekV4Config,
    ps: ParallelState,
    **kwargs,
) -> Iterator[tuple[str, torch.Tensor]]:
    if (ps.tp_size, ps.etp_size) != (1, 1):
        raise NotImplementedError(
            "vLLM BF16-master/online-FP8 export requires TP/ETP=1."
        )
    spec = DeepseekV4WeightSpec(model_cfg)
    spec.validate_load(ps)
    if ps.pp_size > 1:
        # Reuse the same bounded TP/EP/PP streaming exporter as mlite.lite.
        # Only the checkpoint's reversible block scales are DS4-vLLM-specific;
        # replicate that small registry so native_to_hf can preserve the
        # original deployment bytes after each PP-stage tensor is broadcast.
        from megatron.lite.primitive.ckpt.hf_weights import (
            export_hf_weights as _export_common,
        )

        spec.export_source_block_scales = _pipeline_export_source_scales(chunks, ps)
        yield from _export_common(
            chunks,
            spec,
            ps,
            vocab_size=model_cfg.vocab_size,
            **kwargs,
        )
        return
    fp8_export_count = 0
    source_scale_count = 0
    current_source_scale_count = 0
    for model in _models(chunks):
        layer_map = (
            {
                local_idx: model.layer_indices[local_idx]
                for local_idx in range(len(model.layer_indices))
            }
            if hasattr(model, "layer_indices")
            else {}
        )
        model_scale_registry = getattr(model, "_fp8_source_scales_by_name", {})
        model_scale_registry_valid = bool(
            getattr(model, "_fp8_source_scales_valid", False)
        )
        # FSDP2's state-dict hook may synthesize DTensor values even with
        # ``keep_vars=True``.  Custom Parameter metadata is restored on the
        # actual named parameters by the FSDP wrapper, not copied to those
        # synthesized state values.  Preserve state_dict ordering/buffers but
        # prefer the live Parameter whenever the key names one.
        parameters = dict(model.named_parameters())
        for native_name, state_tensor in model.state_dict(keep_vars=True).items():
            # Pipeline stages store their layers under stage-local ModuleDict
            # indices.  The rollout model is PP1 and expects global layer IDs,
            # so globalize only the exported name; live parameters and the
            # durable source-scale registry remain keyed by the local name.
            global_native_name = to_global_layer_name(native_name, layer_map)
            tensor = parameters.get(native_name, state_tensor)
            if _is_block_fp8_weight(native_name):
                fp8_export_count += 1
            source_scales = getattr(tensor, "_fp8_source_scales", None)
            source_scale_version = getattr(
                tensor, "_fp8_source_scale_version", None
            )
            source_scales_from_registry = False
            if (
                source_scales is None
                and model_scale_registry_valid
                and native_name in model_scale_registry
            ):
                source_scales = model_scale_registry[native_name]
                source_scales_from_registry = True
            if source_scales is not None:
                source_scale_count += 1
            source_scales_are_current = (
                source_scales is not None
                and (
                    source_scales_from_registry
                    or source_scale_version == getattr(tensor, "_version", None)
                )
            )
            if source_scales_are_current:
                current_source_scale_count += 1
            full_tensor = getattr(tensor, "full_tensor", None)
            if callable(full_tensor):
                # FSDP2 state_dict values are DTensors. Export is an online
                # inference boundary and must materialize each BF16 master
                # parameter before dtype conversion and bucket packing.
                tensor = full_tensor()
            if source_scales_are_current:
                tensor._fp8_source_scales = source_scales.to(
                    device=tensor.device, dtype=torch.float32
                ).contiguous()
                tensor._fp8_source_scale_version = tensor._version
            if not tensor.is_floating_point():
                output = tensor.detach()
            else:
                output = tensor.detach().to(
                    dtype=(
                        torch.float32
                        if _is_fp32_hf_tensor(native_name)
                        else torch.bfloat16
                    )
                )
            # ``detach`` and ``to`` return a fresh Tensor and therefore drop
            # the reversible checkpoint-scale metadata carried by the BF16
            # master Parameter/DTensor.  Reattach it at the final tensor
            # boundary consumed by ``native_to_hf``; otherwise initial RL
            # synchronization silently recomputes scales and changes the FP8
            # bytes loaded by rollout before the first optimizer step.
            if source_scales_are_current:
                output._fp8_source_scales = source_scales.to(
                    device=output.device, dtype=torch.float32
                ).contiguous()
                output._fp8_source_scale_version = output._version
            yield from spec.native_to_hf(global_native_name, output)
    if os.getenv("MLITE_WEIGHT_SYNC_FINGERPRINT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        print(
            "MLITE_FP8_EXPORT_SCALE_COVERAGE "
            + json.dumps(
                {
                    "rank": int(getattr(ps, "rank", 0)),
                    "fp8_parameter_count": fp8_export_count,
                    "source_scale_count": source_scale_count,
                    "current_source_scale_count": current_source_scale_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def load_hf_weights(
    chunk: nn.Module,
    hf_path: str,
    model_cfg: DeepseekV4Config,
    ps: ParallelState,
) -> None:
    if not hf_path:
        return
    from megatron.lite.primitive.ckpt.hf_weights import load_hf_weights as _load

    index_path = Path(hf_path) / "model.safetensors.index.json"
    source_block_fp8 = False
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} has no weight_map")
        source_block_fp8 = any(str(name).endswith(".scale") for name in weight_map)
    spec = DeepseekV4WeightSpec(
        model_cfg,
        source_block_fp8=source_block_fp8,
    )
    _load(
        chunk,
        hf_path,
        spec,
        ps,
        vocab_size=model_cfg.vocab_size,
    )
    spec.bind_source_scales(chunk)


def save_hf_weights(
    chunks: nn.Module | Iterable[nn.Module],
    path: str,
    model_cfg: DeepseekV4Config,
    ps: ParallelState,
    **kwargs,
) -> None:
    from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards

    shard_size_bytes = int(kwargs.pop("shard_size_bytes", 5 * 1024**3))
    stream_export_to_shards(
        export_hf_weights(chunks, model_cfg, ps, **kwargs),
        path,
        shard_size_bytes=shard_size_bytes,
    )


__all__ = [
    "EXPERT_CLASSIFIER",
    "DeepseekV4WeightSpec",
    "PLACEMENT_FN",
    "export_hf_weights",
    "invalidate_bound_source_scales",
    "load_hf_weights",
    "save_hf_weights",
]
