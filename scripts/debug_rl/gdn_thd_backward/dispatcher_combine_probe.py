#!/usr/bin/env python3
"""Run the correctness CLI while fingerprinting mLite routed-combine tensors."""

from __future__ import annotations

import hashlib
import os
import runpy

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

try:
    from megatron.core.transformer.moe import token_dispatcher as mcore_token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
except ImportError:
    mcore_token_dispatcher = None
    MoEAlltoAllTokenDispatcher = None


def _fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    square_sum = value.float().square().sum().item()
    return f"shape={tuple(value.shape)} dtype={value.dtype} sha256={digest} sq={square_sum:.17g}"


_original_combine = TokenDispatcher.combine


def _combine_with_probe(self: TokenDispatcher, expert_output: torch.Tensor) -> torch.Tensor:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"MLITE_DISPATCHER_COMBINE_INPUT rank={rank} {_fingerprint(expert_output)}", flush=True)
    result = _original_combine(self, expert_output)
    print(f"MLITE_DISPATCHER_COMBINE_OUTPUT rank={rank} {_fingerprint(result)}", flush=True)
    return result


TokenDispatcher.combine = _combine_with_probe

if MoEAlltoAllTokenDispatcher is not None:
    _original_combine_postprocess = MoEAlltoAllTokenDispatcher.combine_postprocess

    def _combine_postprocess_with_probe(self, *args, **kwargs):
        permuted = args[0] if args else kwargs["permutated_local_input_tokens"]
        routed = mcore_token_dispatcher.unpermute(
            permuted,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        ).view(self.hidden_shape)
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(
            f"MCORE_ROUTED_UNPERMUTE_OUTPUT rank={rank} {_fingerprint(routed)}",
            flush=True,
        )
        result = _original_combine_postprocess(self, *args, **kwargs)
        print(
            f"MCORE_COMBINE_POSTPROCESS_OUTPUT rank={rank} {_fingerprint(result)}",
            flush=True,
        )
        return result

    MoEAlltoAllTokenDispatcher.combine_postprocess = _combine_postprocess_with_probe

runpy.run_path(os.environ["MLITE_CORRECTNESS_ENTRY"], run_name="__main__")
