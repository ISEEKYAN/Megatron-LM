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
    quant_config,
    tokens_per_expert,
):
    """Use vLLM's official FP8 quantizer with the LL scale layout."""
    from vllm.model_executor.layers.fused_moe.utils import (
        moe_kernel_quantize_input,
        normalize_batched_scales_shape,
    )

    if quant_config.block_shape != [128, 128]:
        raise ValueError("vLLM LL-compatible quantization requires 128-wide groups")
    if quant_config.quant_dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise TypeError(
            "unsupported vLLM LL activation dtype: "
            f"{quant_config.quant_dtype}"
        )
    if batched.ndim != 3:
        raise ValueError("vLLM batched expert input must be rank-3")
    if tokens_per_expert.ndim != 1 or tokens_per_expert.shape[0] != batched.shape[0]:
        raise ValueError("tokens_per_expert must describe every local expert")

    # Keep this call identical to DeepEPLLPrepareAndFinalize._do_quant. In
    # particular, the active DeepGEMM oracle decides FLOAT32 versus packed
    # UE8M0 scales; a transport adapter must not choose a second scale format.
    experts, _, hidden = batched.shape
    flat = batched.view(-1, hidden)
    quantized, scales = moe_kernel_quantize_input(
        flat,
        quant_config.a1_scale,
        quant_config.quant_dtype,
        quant_config.per_act_token_quant,
        quant_config.block_shape,
    )
    quantized = quantized.view_as(batched)
    scales = normalize_batched_scales_shape(scales, experts)
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

                # Slime derives the exact per-expert counts from received
                # top-k metadata on CUDA while scattering routes.  Reuse that
                # tensor for vLLM's masked DeepGEMM ABI; normal DeepEP's public
                # CPU count metadata remains the ownership/source-of-shape
                # contract and is never uploaded again.
                device_tokens_per_expert = getattr(
                    self.dispatcher, "_aligned_device_tokens_per_expert", None
                )
                if device_tokens_per_expert is None:
                    if tokens_per_expert.device.type != "cuda":
                        raise RuntimeError(
                            "aligned DeepEP did not expose Slime-derived CUDA expert counts"
                        )
                    device_tokens_per_expert = tokens_per_expert
                quantized, scales = _quantize_batched_input(
                    batched,
                    quant_config,
                    device_tokens_per_expert,
                )
                self._counts = counts
                metadata = mk.ExpertTokensMetadata(
                    expert_num_tokens=device_tokens_per_expert.to(dtype=torch.int32),
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
