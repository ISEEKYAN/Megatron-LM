"""Normal-DeepEP prepare/finalize for vLLM's batched expert layout.

The communication and route reconstruction follow the Slime design implemented
by :class:`TokenDispatcher`.  Expert quantization and compute remain vLLM's
official batched DeepGEMM path.
"""

from __future__ import annotations

import os

import torch


def _quantize_batched_input(
    batched,
    quant_dtype,
    block_shape,
    tokens_per_expert,
):
    """Use vLLM's official FP8 quantizer with the LL scale layout."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    if block_shape != [128, 128]:
        raise ValueError("vLLM LL-compatible quantization requires 128-wide groups")
    if quant_dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise TypeError(f"unsupported vLLM LL activation dtype: {quant_dtype}")
    if batched.ndim != 3:
        raise ValueError("vLLM batched expert input must be rank-3")
    if tokens_per_expert.ndim != 1 or tokens_per_expert.shape[0] != batched.shape[0]:
        raise ValueError("tokens_per_expert must describe every local expert")

    # The libtorch-stable CUDA op currently requires output_s.dim() == 2,
    # although the Python API advertises ndim >= 2 and allocates rank-3 scales
    # for batched experts. Quantize each expert as an official 2D call, then
    # copy its column-major scales into the expected batched TMA layout.
    experts, capacity, hidden = batched.shape
    groups = hidden // 128
    quantized = torch.empty_like(batched, dtype=quant_dtype)
    # BatchedDeepGemmExperts consumes the compact batched scale contract from
    # ``scales_shape_stride_dtype``: (E, T, G) with strides (T*G, 1, T).
    # The official 2-D quantizer may use a larger TMA-aligned leading dimension
    # for its per-expert temporary when T is small.  That padding is private to
    # the temporary and must not become the expert stride of the batched tensor;
    # DeepGEMM's masked grouped kernel requires adjacent expert scale blocks.
    scales = torch.empty_strided(
        (experts, capacity, groups),
        (capacity * groups, 1, capacity),
        device=batched.device,
        dtype=torch.float32,
    )
    for expert in range(experts):
        _, expert_scales = per_token_group_quant_fp8(
            batched[expert],
            128,
            eps=1e-10,
            dtype=quant_dtype,
            column_major_scales=True,
            tma_aligned_scales=True,
            out_q=quantized[expert],
            use_ue8m0=False,
        )
        scales[expert].copy_(expert_scales)
    return quantized, scales


def _compact_fused_expert_output(
    fused_expert_output: torch.Tensor,
    counts: tuple[int, ...],
) -> torch.Tensor:
    expert_axis = 0 if fused_expert_output.shape[0] == len(counts) else 1
    if (
        fused_expert_output.ndim != 3
        or fused_expert_output.shape[expert_axis] != len(counts)
    ):
        raise RuntimeError(
            "normal DeepEP cannot locate expert axis in output "
            f"{tuple(fused_expert_output.shape)} for {len(counts)} experts"
        )
    rows = [
        (
            fused_expert_output[expert, :count]
            if expert_axis == 0
            else fused_expert_output[:count, expert]
        )
        for expert, count in enumerate(counts)
        if count
    ]
    if rows:
        return torch.cat(rows, dim=0)
    return fused_expert_output.new_empty((0, fused_expert_output.shape[-1]))


class NormalDeepEPAlignedPrepareAndFinalize:
    """Build a vLLM modular prepare/finalize around normal DeepEP.

    This factory returns an actual ``FusedMoEPrepareAndFinalizeModular``
    subclass without importing vLLM when this module is imported.
    """

    @staticmethod
    def build(dispatcher, *, max_tokens_per_rank: int, num_dispatchers: int):
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceDelegate,
        )

        class _NormalDeepEPAlignedPrepareAndFinalize(
            mk.FusedMoEPrepareAndFinalizeModular
        ):
            def __init__(self):
                super().__init__()
                self.dispatcher = dispatcher
                self.max_tokens_per_rank_ = int(max_tokens_per_rank)
                self.num_dispatchers_ = int(num_dispatchers)
                self.capacity = self.max_tokens_per_rank_ * self.num_dispatchers_
                self._counts: tuple[int, ...] | None = None

            def num_dispatchers(self) -> int:
                return self.num_dispatchers_

            def output_is_reduced(self) -> bool:
                return True

            @property
            def activation_format(self):
                return mk.FusedMoEActivationFormat.BatchedExperts

            def max_num_tokens_per_rank(self) -> int:
                return self.max_tokens_per_rank_

            def topk_indices_dtype(self) -> torch.dtype:
                return torch.int64

            def prepare(
                self,
                a1,
                topk_weights,
                topk_ids,
                num_experts,
                expert_map,
                apply_router_weight_on_input,
                quant_config,
                defer_input_quant=False,
            ):
                if defer_input_quant:
                    raise NotImplementedError(
                        "normal DeepEP batched prepare requires quantized inputs"
                    )
                if apply_router_weight_on_input:
                    raise NotImplementedError(
                        "normal DeepEP alignment applies router weights in finalize"
                    )
                if quant_config.block_shape != [128, 128]:
                    raise ValueError(
                        "normal DeepEP alignment requires vLLM 128x128 block FP8"
                    )

                compact, tokens_per_expert, _ = self.dispatcher.dispatch(
                    a1,
                    topk_weights,
                    topk_ids,
                )
                self.dispatcher.wait_dispatch_event()
                counts = tuple(
                    int(value)
                    for value in tokens_per_expert.detach().cpu().tolist()
                )
                if len(counts) != self.dispatcher.num_local_experts:
                    raise RuntimeError("normal DeepEP returned the wrong expert count")
                if max(counts, default=0) > self.capacity:
                    raise RuntimeError(
                        "normal DeepEP expert rows exceed the vLLM LL capacity: "
                        f"max={max(counts)} capacity={self.capacity}"
                    )
                batched = compact.new_zeros(
                    (self.dispatcher.num_local_experts, self.capacity, a1.shape[1])
                )
                source = 0
                for expert, count in enumerate(counts):
                    if count:
                        batched[expert, :count].copy_(compact[source : source + count])
                    source += count
                if source != compact.shape[0]:
                    raise RuntimeError("normal DeepEP expert counts do not cover all rows")

                quantized, scales = _quantize_batched_input(
                    batched,
                    quant_config.quant_dtype,
                    quant_config.block_shape,
                    tokens_per_expert,
                )
                self._counts = counts
                metadata = mk.ExpertTokensMetadata(
                    expert_num_tokens=tokens_per_expert.to(dtype=torch.int32),
                    expert_num_tokens_cpu=None,
                )
                return quantized, scales, metadata, None, None

            def finalize(
                self,
                output,
                fused_expert_output,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                weight_and_reduce_impl,
            ) -> None:
                if not isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                    raise TypeError("normal DeepEP finalize requires delegated gather")
                if self._counts is None:
                    raise RuntimeError("normal DeepEP finalize called without prepare")
                compact = _compact_fused_expert_output(
                    fused_expert_output,
                    self._counts,
                )
                if (
                    os.environ.get("MLITE_VALIDATE_FINITE") == "1"
                    and not bool(torch.isfinite(compact).all())
                ):
                    raise FloatingPointError(
                        "MLITE_NONFINITE "
                        "stage=normal_deepep.finalize_compact "
                        f"fused_shape={tuple(fused_expert_output.shape)} "
                        f"nonfinite={int((~torch.isfinite(compact)).sum().item())}"
                    )
                combined = self.dispatcher.combine(compact)
                output.copy_(combined)
                self._counts = None

        return _NormalDeepEPAlignedPrepareAndFinalize()


__all__ = ["NormalDeepEPAlignedPrepareAndFinalize"]
