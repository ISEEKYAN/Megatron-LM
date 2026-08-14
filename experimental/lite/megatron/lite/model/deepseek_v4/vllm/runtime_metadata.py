"""Runtime-owned DeepSeek-V4 layer-0 FlashMLA metadata.

This module deliberately builds the small runtime contract directly.  It does
not construct a vLLM model or read vLLM's process-global forward context.
"""

from __future__ import annotations

import importlib
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.primitive.parallel import ParallelState


DS4_SWA_BLOCK_SIZE = 256
DS4_FP8_MLA_TOKEN_BYTES = 584
DS4_FLASHMLA_INDEX_ALIGNMENT = 128


def _symbol(module: str, name: str) -> Any:
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise NotImplementedError(
            f"vLLM runtime primitive {module}.{name} is unavailable"
        ) from exc


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _local_num_tokens(batch: Any) -> int:
    total_tokens = getattr(batch, "total_tokens", None)
    if total_tokens is not None:
        value = int(total_tokens)
    else:
        input_ids = getattr(batch, "input_ids", None)
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("DeepSeek V4 vLLM batches require tensor input_ids")
        value = int(input_ids.numel())
    if value <= 0:
        raise ValueError("DeepSeek V4 vLLM batches must contain at least one token")
    return value


def initialize_ds4_vllm_batch_invariance() -> None:
    """Mirror vLLM's optional process-local batch-invariant mode."""

    vllm_envs = importlib.import_module("vllm.envs")
    if not vllm_envs.VLLM_BATCH_INVARIANT:
        return
    _symbol(
        "vllm.model_executor.layers.batch_invariant",
        "init_batch_invariance",
    )()

    deep_gemm = importlib.import_module("deep_gemm")
    setter = getattr(deep_gemm, "set_batch_invariant", None)
    getter = getattr(deep_gemm, "get_batch_invariant", None)
    if setter is None or getter is None:
        raise RuntimeError(
            "DeepSeek V4 vLLM requires DeepGEMM "
            "set_batch_invariant/get_batch_invariant"
        )
    setter(True)
    if not getter():
        raise RuntimeError("DeepGEMM batch-invariant mode failed to enable")
    supports = _symbol(
        "vllm.utils.deep_gemm",
        "supports_deep_gemm_batch_invariance",
    )
    if not supports():
        raise RuntimeError(
            "vLLM BatchedDeepGemmExperts does not expose native "
            "batch-invariant capability"
        )
    os.environ["DS4_BATCH_INVARIANT_CAPABILITY_MODE"] = "native"


@contextmanager
def ds4_vllm_forward_context(
    batch: Any,
    parallel_state: ParallelState,
    *,
    vllm_config: Any,
) -> Iterator[None]:
    """Install the vLLM per-microbatch token contract used by MoE kernels."""

    input_ids = getattr(batch, "input_ids", None)
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("DeepSeek V4 vLLM batches require tensor input_ids")
    if parallel_state.ep_group is None:
        raise RuntimeError("DeepSeek V4 vLLM requires an initialized EP group")

    local_tokens = torch.tensor(
        [_local_num_tokens(batch)],
        dtype=torch.int32,
        device=input_ids.device,
    )
    gathered_tokens = [
        torch.empty_like(local_tokens) for _ in range(parallel_state.ep_size)
    ]
    dist.all_gather(
        gathered_tokens,
        local_tokens,
        group=parallel_state.ep_group,
    )

    dp_metadata = _symbol("vllm.forward_context", "DPMetadata")(
        torch.cat(gathered_tokens).cpu()
    )
    if not torch.equal(
        dp_metadata.num_tokens_across_dp_cpu,
        torch.cat(gathered_tokens).cpu(),
    ):
        raise RuntimeError("DPMetadata token counts do not match the runtime batch")
    os.environ["DS4_DP_METADATA_MODE"] = "dynamic"
    forward_context = _symbol("vllm.forward_context", "create_forward_context")(
        None,
        vllm_config,
        dp_metadata=dp_metadata,
    )
    override_forward_context = _symbol(
        "vllm.forward_context", "override_forward_context"
    )
    with override_forward_context(forward_context):
        yield


class _RuntimeGateLinear(nn.Module):
    """TP-independent composition of vLLM's DS4 router GEMM primitives."""

    def __init__(self, input_size: int, output_size: int, *, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.bfloat16, device=device)
        )

    def forward(self, hidden_states: torch.Tensor):
        if hidden_states.shape[0] <= 16:
            is_available = _symbol(
                "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16",
                "is_available",
            )
            if is_available():
                kernel = _symbol(
                    "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16",
                    "ll_bf16_gemm",
                )
                return kernel(hidden_states, self.weight), None
        return torch.mm(hidden_states, self.weight.T, out_dtype=torch.float32), None


def _build_rope(
    hf_config: Any,
    config: DeepseekV4Config,
    *,
    compress_ratio: int,
    device: torch.device | str,
) -> Any:
    """Instantiate vLLM's CustomOp-backed RoPE in its required scoped context."""
    build_rope = _symbol(
        "vllm.models.deepseek_v4.common.rope", "build_deepseek_v4_rope"
    )
    vllm_config_type = _symbol("vllm.config", "VllmConfig")
    set_current_vllm_config = _symbol("vllm.config", "set_current_vllm_config")
    target_device = torch.device(device)
    with set_current_vllm_config(vllm_config_type()), torch.device(target_device):
        rotary = build_rope(
            hf_config,
            head_dim=config.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=compress_ratio,
        )
    return rotary.to(device=target_device)


@dataclass(frozen=True)
class DS4RuntimeLayout:
    """Observable scheduler/cache layout retained for contract tests."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor


@dataclass
class DS4CompressorRuntimeMetadata:
    """Caller-owned tensors consumed by one official DS4 compressor launch."""

    state_cache: torch.Tensor
    state_slot_mapping: torch.Tensor
    state_block_table: torch.Tensor
    state_block_size: int
    token_to_req_indices: torch.Tensor
    k_cache: torch.Tensor
    k_slot_mapping: torch.Tensor
    cos_sin_cache: torch.Tensor
    rms_norm_eps: float
    rope_head_dim: int


@dataclass
class DS4IndexerRuntimeMetadata:
    """Official indexer metadata plus its caller-owned cache and output."""

    compressor: DS4CompressorRuntimeMetadata
    attention_metadata: Any
    k_cache_prefix: str
    topk_indices: torch.Tensor
    max_model_len: int
    max_total_seq_len: int


class DS4SparseAttentionMetadataBuilderAdapter:
    """Build production metadata for DS4's layer-0 SWA-only attention.

    Layer 0 has ``compress_ratio=max(1, config.compress_ratios[0]) == 1`` in
    the official model.  Its sparse set is therefore the causal sliding window;
    no learned indexer or compressor output exists at this layer.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        device: torch.device | str,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        self.config = config
        requested_device = torch.device(device)
        self.device = (
            cos_sin_cache.device
            if requested_device.index is None
            and isinstance(cos_sin_cache, torch.Tensor)
            and cos_sin_cache.device.type == requested_device.type
            else requested_device
        )
        self.cos_sin_cache = cos_sin_cache
        ratio = max(1, config.compress_ratios[0]) if config.compress_ratios else 1
        if ratio != 1:
            raise ValueError(
                "layer-0 metadata builder only supports the official SWA-only "
                f"contract; got compress_ratio={ratio}"
            )
        if config.head_dim != 512:
            raise ValueError(
                f"DS4 FlashMLA requires head_dim=512, got {config.head_dim}"
            )
        if config.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if (
            not isinstance(cos_sin_cache, torch.Tensor)
            or cos_sin_cache.ndim != 2
            or cos_sin_cache.device != self.device
            or cos_sin_cache.dtype != torch.float32
        ):
            raise ValueError(
                "cos_sin_cache must be a 2D float32 tensor on the runtime device"
            )

    @classmethod
    def from_hf(
        cls,
        model_path: str | Path,
        config: DeepseekV4Config,
        *,
        device: torch.device | str,
    ) -> "DS4SparseAttentionMetadataBuilderAdapter":
        """Create the exact RoPE cache through vLLM's public DS4 rope builder."""

        get_config = _symbol("vllm.transformers_utils.config", "get_config")
        hf_config = get_config(
            str(model_path),
            trust_remote_code=True,
        )
        rotary = _build_rope(
            hf_config,
            config,
            compress_ratio=1,
            device=device,
        )
        return cls(
            config,
            device=device,
            cos_sin_cache=rotary.cos_sin_cache.to(
                device=device,
                dtype=torch.float32,
            ),
        )

    def _cache_and_layout(self, seq_len: int) -> tuple[torch.Tensor, DS4RuntimeLayout]:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        num_blocks = (seq_len + DS4_SWA_BLOCK_SIZE - 1) // DS4_SWA_BLOCK_SIZE
        # Retain vLLM's owning page layout. The fused insert call temporarily
        # views it as [num_blocks, block_size * 584 bytes].
        cache = torch.empty(
            (num_blocks, DS4_SWA_BLOCK_SIZE, DS4_FP8_MLA_TOKEN_BYTES),
            dtype=torch.uint8,
            device=self.device,
        )
        layout = DS4RuntimeLayout(
            block_table=torch.arange(
                num_blocks, dtype=torch.int32, device=self.device
            ).unsqueeze(0),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=self.device),
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=self.device
            ),
        )
        return cache, layout

    def _prefill_indices(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = _round_up(self.config.sliding_window, DS4_FLASHMLA_INDEX_ALIGNMENT)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=self.device)
        offsets = torch.arange(width, dtype=torch.int32, device=self.device)
        starts = torch.clamp(positions - self.config.sliding_window + 1, min=0)
        lengths = torch.minimum(
            positions + 1,
            torch.tensor(
                self.config.sliding_window, dtype=torch.int32, device=self.device
            ),
        )
        indices = starts[:, None] + offsets[None, :]
        indices.masked_fill_(offsets[None, :] >= lengths[:, None], -1)
        return indices.unsqueeze(1).contiguous(), lengths.contiguous()

    def build_prefill(self, num_tokens: int):
        """Build one-request prompt metadata and its post-insert gather closure."""

        from megatron.lite.model.deepseek_v4.vllm.model import AttentionKernelMetadata

        cache, layout = self._cache_and_layout(num_tokens)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        slot_mapping = positions.clone()
        indices, topk_length = self._prefill_indices(num_tokens)
        workspace_3d = torch.empty(
            (1, num_tokens, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.empty(
            (
                num_tokens,
                self.config.num_attention_heads,
                self.config.head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        def prepare_flash() -> None:
            gather = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "dequantize_and_gather_k_cache",
            )
            gather(
                workspace_3d,
                cache,
                seq_lens=layout.seq_lens,
                gather_lens=layout.seq_lens,
                block_table=layout.block_table,
                block_size=DS4_SWA_BLOCK_SIZE,
                offset=0,
            )

        metadata = AttentionKernelMetadata(
            positions=positions,
            slot_mapping=slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=cache,
            block_size=DS4_SWA_BLOCK_SIZE,
            flash_kind="prefill",
            indices=indices,
            topk_length=topk_length,
            output=output,
            kv_workspace=workspace_3d.view(-1, 1, self.config.head_dim),
            padded_heads=self.config.num_attention_heads,
            prepare_flash=prepare_flash,
        )
        metadata.runtime_layout = layout
        return metadata

    def build_prefill_batch(self, token_counts: list[int]):
        """Build packed prefill metadata for multiple independent requests."""

        from megatron.lite.model.deepseek_v4.vllm.model import AttentionKernelMetadata

        if not token_counts or any(count <= 0 for count in token_counts):
            raise ValueError("token_counts must contain positive sequence lengths")
        block_counts = [
            (count + DS4_SWA_BLOCK_SIZE - 1) // DS4_SWA_BLOCK_SIZE
            for count in token_counts
        ]
        block_offsets = [0]
        for count in block_counts:
            block_offsets.append(block_offsets[-1] + count)
        cache = torch.empty(
            (
                block_offsets[-1],
                DS4_SWA_BLOCK_SIZE,
                DS4_FP8_MLA_TOKEN_BYTES,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        block_table = torch.full(
            (len(token_counts), max(block_counts)),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        positions = []
        slot_mapping = []
        indices = []
        topk_lengths = []
        query_start = [0]
        max_tokens = max(token_counts)
        for batch_index, count in enumerate(token_counts):
            block_table[batch_index, : block_counts[batch_index]] = torch.arange(
                block_offsets[batch_index],
                block_offsets[batch_index + 1],
                dtype=torch.int32,
                device=self.device,
            )
            sequence_positions = torch.arange(
                count, dtype=torch.int64, device=self.device
            )
            positions.append(sequence_positions)
            slot_mapping.append(
                sequence_positions + block_offsets[batch_index] * DS4_SWA_BLOCK_SIZE
            )
            sequence_indices, sequence_lengths = self._prefill_indices(count)
            valid = sequence_indices >= 0
            sequence_indices = sequence_indices.clone()
            sequence_indices[valid] += batch_index * max_tokens
            indices.append(sequence_indices)
            topk_lengths.append(sequence_lengths)
            query_start.append(query_start[-1] + count)

        layout = DS4RuntimeLayout(
            block_table=block_table,
            seq_lens=torch.tensor(
                token_counts, dtype=torch.int32, device=self.device
            ),
            query_start_loc=torch.tensor(
                query_start, dtype=torch.int32, device=self.device
            ),
        )
        workspace_3d = torch.empty(
            (len(token_counts), max_tokens, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        output = torch.empty(
            (
                sum(token_counts),
                self.config.num_attention_heads,
                self.config.head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        def prepare_flash() -> None:
            gather = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "dequantize_and_gather_k_cache",
            )
            gather(
                workspace_3d,
                cache,
                seq_lens=layout.seq_lens,
                gather_lens=layout.seq_lens,
                block_table=layout.block_table,
                block_size=DS4_SWA_BLOCK_SIZE,
                offset=0,
            )

        metadata = AttentionKernelMetadata(
            positions=torch.cat(positions).contiguous(),
            slot_mapping=torch.cat(slot_mapping).contiguous(),
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=cache,
            block_size=DS4_SWA_BLOCK_SIZE,
            flash_kind="prefill",
            indices=torch.cat(indices).contiguous(),
            topk_length=torch.cat(topk_lengths).contiguous(),
            output=output,
            kv_workspace=workspace_3d.view(-1, 1, self.config.head_dim),
            padded_heads=self.config.num_attention_heads,
            prepare_flash=prepare_flash,
        )
        metadata.runtime_layout = layout
        return metadata

    def build_decode(
        self,
        seq_len: int,
        *,
        swa_cache: torch.Tensor | None = None,
    ):
        """Build single-token decode metadata for a sequential one-request cache."""

        from megatron.lite.model.deepseek_v4.vllm.model import AttentionKernelMetadata

        allocated_cache, layout = self._cache_and_layout(seq_len)
        cache = allocated_cache if swa_cache is None else swa_cache
        expected_blocks = layout.block_table.shape[1]
        expected_shape = (
            expected_blocks,
            DS4_SWA_BLOCK_SIZE,
            DS4_FP8_MLA_TOKEN_BYTES,
        )
        if cache.shape != expected_shape or cache.dtype != torch.uint8:
            raise ValueError(
                f"swa_cache must have shape {expected_shape} and dtype uint8"
            )
        position = seq_len - 1
        start = max(0, seq_len - self.config.sliding_window)
        width = _round_up(self.config.sliding_window, DS4_FLASHMLA_INDEX_ALIGNMENT)
        indices = torch.full((1, 1, width), -1, dtype=torch.int32, device=self.device)
        length = seq_len - start
        indices[0, 0, :length] = torch.arange(
            start, seq_len, dtype=torch.int32, device=self.device
        )
        scheduler, _ = _symbol("vllm.v1.attention.ops.flashmla", "get_mla_metadata")()
        metadata = AttentionKernelMetadata(
            positions=torch.tensor([position], dtype=torch.int64, device=self.device),
            slot_mapping=torch.tensor(
                [position], dtype=torch.int64, device=self.device
            ),
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=cache,
            block_size=DS4_SWA_BLOCK_SIZE,
            flash_kind="decode",
            indices=indices,
            topk_length=torch.tensor([length], dtype=torch.int32, device=self.device),
            output=torch.empty(
                (
                    1,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                ),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            tile_scheduler_metadata=scheduler,
            padded_heads=self.config.num_attention_heads,
        )
        metadata.runtime_layout = layout
        return metadata


class DS4SparseIndexerCompressorMetadataAdapter:
    """Runtime metadata for DS4 layers 1--3 without a vLLM model instance.

    Parameters remain owned by the mLite model.  This adapter owns only
    ephemeral cache/state tensors and calls vLLM's operation-level compressor,
    indexer, gather, and metadata helpers.
    """

    _MAIN_TOKEN_BYTES = 584
    _INDEXER_TOKEN_BYTES = 132
    _SWA_BLOCK_SIZE = DS4_SWA_BLOCK_SIZE
    _MLA_BLOCK_SIZE = 256

    def _allocate_k_cache(
        self, num_blocks: int, block_size: int, token_bytes: int
    ) -> torch.Tensor:
        """Allocate logical cache rows with the 32-byte block stride CuTe expects."""
        block_stride = _round_up(block_size * token_bytes, 32)
        storage = torch.zeros(
            num_blocks * block_stride, dtype=torch.uint8, device=self.device
        )
        return torch.as_strided(
            storage,
            size=(num_blocks, block_size, token_bytes),
            stride=(block_stride, token_bytes, 1),
        )

    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        layer_idx: int,
        device: torch.device | str,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        if layer_idx < 1 or layer_idx > 3:
            raise ValueError("extended metadata adapter supports layers 1 through 3")
        self.config = config
        self.layer_idx = layer_idx
        self.device = torch.device(device)
        self.cos_sin_cache = cos_sin_cache
        self.compress_ratio = max(1, config.compress_ratios[layer_idx])
        if self.compress_ratio not in (1, 4, 128):
            raise ValueError(
                f"unsupported layer {layer_idx} compress_ratio={self.compress_ratio}"
            )
        if config.head_dim != 512 or config.index_head_dim != 128:
            raise ValueError(
                "official DS4 metadata requires head_dim=512/index_dim=128"
            )
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.device != self.device
            or cos_sin_cache.dtype != torch.float32
        ):
            raise ValueError(
                "cos_sin_cache must be 2D float32 and on the runtime device"
            )

    @classmethod
    def from_hf(
        cls,
        model_path: str | Path,
        config: DeepseekV4Config,
        *,
        layer_idx: int,
        device: torch.device | str,
    ) -> "DS4SparseIndexerCompressorMetadataAdapter":
        get_config = _symbol("vllm.transformers_utils.config", "get_config")
        hf_config = get_config(
            str(model_path),
            trust_remote_code=True,
        )
        ratio = max(1, config.compress_ratios[layer_idx])
        rotary = _build_rope(
            hf_config,
            config,
            compress_ratio=ratio,
            device=device,
        )
        return cls(
            config,
            layer_idx=layer_idx,
            device=device,
            cos_sin_cache=rotary.cos_sin_cache.to(
                device=device,
                dtype=torch.float32,
            ),
        )

    def _base_prefill(self, num_tokens: int):
        return DS4SparseAttentionMetadataBuilderAdapter(
            self.config,
            device=self.device,
            cos_sin_cache=self.cos_sin_cache,
        ).build_prefill(num_tokens)

    def _compressed_mapping(self, num_tokens: int, ratio: int) -> torch.Tensor:
        mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=self.device)
        positions = torch.arange(num_tokens, device=self.device)
        complete = (positions + 1).remainder(ratio).eq(0)
        mapping[complete] = positions[complete].div(ratio, rounding_mode="floor")
        return mapping

    def _compressor_metadata(
        self,
        num_tokens: int,
        *,
        head_dim: int,
        token_bytes: int,
    ) -> DS4CompressorRuntimeMetadata:
        ratio = self.compress_ratio
        coff = 2 if ratio == 4 else 1
        state_block_size = 4 if ratio == 4 else 8
        num_state_blocks = max(
            1, _round_up(num_tokens, state_block_size) // state_block_size
        )
        compressed_block_size = self._MLA_BLOCK_SIZE // ratio
        num_compressed = num_tokens // ratio
        num_k_blocks = max(
            1,
            _round_up(max(num_compressed, 1), compressed_block_size)
            // compressed_block_size,
        )
        return DS4CompressorRuntimeMetadata(
            state_cache=torch.zeros(
                num_state_blocks,
                state_block_size,
                2 * coff * head_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            state_slot_mapping=torch.arange(
                num_tokens, dtype=torch.int64, device=self.device
            ),
            state_block_table=torch.arange(
                num_state_blocks, dtype=torch.int32, device=self.device
            ).unsqueeze(0),
            state_block_size=state_block_size,
            token_to_req_indices=torch.zeros(
                num_tokens, dtype=torch.int32, device=self.device
            ),
            k_cache=self._allocate_k_cache(
                num_k_blocks, compressed_block_size, token_bytes
            ),
            k_slot_mapping=self._compressed_mapping(num_tokens, ratio),
            cos_sin_cache=self.cos_sin_cache,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_head_dim=self.config.qk_rope_head_dim,
        )

    def _compressor_metadata_batch(
        self,
        token_counts: list[int],
        *,
        head_dim: int,
        token_bytes: int,
    ) -> tuple[DS4CompressorRuntimeMetadata, torch.Tensor]:
        """Allocate sequence-isolated compressor state and cache tables."""

        ratio = self.compress_ratio
        coff = 2 if ratio == 4 else 1
        state_block_size = 4 if ratio == 4 else 8
        compressed_block_size = self._MLA_BLOCK_SIZE // ratio
        state_block_counts = [
            max(1, _round_up(count, state_block_size) // state_block_size)
            for count in token_counts
        ]
        k_block_counts = [
            max(
                1,
                _round_up(max(count // ratio, 1), compressed_block_size)
                // compressed_block_size,
            )
            for count in token_counts
        ]
        state_offsets = [0]
        k_offsets = [0]
        for state_count, k_count in zip(state_block_counts, k_block_counts):
            state_offsets.append(state_offsets[-1] + state_count)
            k_offsets.append(k_offsets[-1] + k_count)

        state_block_table = torch.full(
            (len(token_counts), max(state_block_counts)),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        k_block_table = torch.full(
            (len(token_counts), max(k_block_counts)),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        state_slots = []
        k_slots = []
        request_indices = []
        for request, count in enumerate(token_counts):
            state_block_table[request, : state_block_counts[request]] = torch.arange(
                state_offsets[request],
                state_offsets[request + 1],
                dtype=torch.int32,
                device=self.device,
            )
            k_block_table[request, : k_block_counts[request]] = torch.arange(
                k_offsets[request],
                k_offsets[request + 1],
                dtype=torch.int32,
                device=self.device,
            )
            local_positions = torch.arange(count, device=self.device)
            state_slots.append(
                local_positions.to(torch.int64)
                + state_offsets[request] * state_block_size
            )
            local_k_slots = self._compressed_mapping(count, ratio)
            valid = local_k_slots >= 0
            local_k_slots[valid] += k_offsets[request] * compressed_block_size
            k_slots.append(local_k_slots)
            request_indices.append(
                torch.full(
                    (count,), request, dtype=torch.int32, device=self.device
                )
            )

        metadata = DS4CompressorRuntimeMetadata(
            state_cache=torch.zeros(
                state_offsets[-1],
                state_block_size,
                2 * coff * head_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            state_slot_mapping=torch.cat(state_slots).contiguous(),
            state_block_table=state_block_table,
            state_block_size=state_block_size,
            token_to_req_indices=torch.cat(request_indices).contiguous(),
            k_cache=self._allocate_k_cache(
                k_offsets[-1], compressed_block_size, token_bytes
            ),
            k_slot_mapping=torch.cat(k_slots).contiguous(),
            cos_sin_cache=self.cos_sin_cache,
            rms_norm_eps=self.config.rms_norm_eps,
            rope_head_dim=self.config.qk_rope_head_dim,
        )
        return metadata, k_block_table

    @staticmethod
    def compressor_operation(
        *,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        compress_ratio: int,
        head_dim: int,
        metadata: DS4CompressorRuntimeMetadata | DS4IndexerRuntimeMetadata,
    ) -> None:
        """Call the two official operation-level compressor launches."""

        if isinstance(metadata, DS4IndexerRuntimeMetadata):
            metadata = metadata.compressor
        coff = 2 if compress_ratio == 4 else 1
        width = coff * head_dim
        kv, score = kv_score.split([width, width], dim=-1)
        save = _symbol(
            "vllm.models.deepseek_v4.common.ops.save_partial_states",
            "save_partial_states",
        )
        save(
            kv=kv,
            score=score,
            ape=ape,
            positions=positions,
            state_cache=metadata.state_cache,
            slot_mapping=metadata.state_slot_mapping,
            block_size=metadata.state_block_size,
            state_width=width,
            compress_ratio=compress_ratio,
            pdl_kwargs={"launch_pdl": False},
        )
        use_cutedsl = head_dim == 512 and kv_score.device.type == "cuda"
        compress = _symbol(
            (
                "vllm.models.deepseek_v4.nvidia.ops."
                "sparse_attn_compress_cutedsl"
                if use_cutedsl
                else "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
            ),
            (
                "compress_norm_rope_store_cutedsl"
                if use_cutedsl
                else "compress_norm_rope_store_triton"
            ),
        )
        quant_block = 64 if head_dim == 512 else 128
        token_stride = (
            head_dim - metadata.rope_head_dim + 2 * metadata.rope_head_dim
            if head_dim == 512
            else head_dim
        )
        compress(
            state_cache=metadata.state_cache,
            num_actual=positions.numel(),
            token_to_req_indices=metadata.token_to_req_indices,
            positions=positions,
            slot_mapping=metadata.state_slot_mapping,
            block_table=metadata.state_block_table,
            block_size=metadata.state_block_size,
            state_width=width,
            cos_sin_cache=metadata.cos_sin_cache,
            kv_cache=metadata.k_cache,
            k_cache_metadata=SimpleNamespace(slot_mapping=metadata.k_slot_mapping),
            pdl_kwargs={"launch_pdl": False},
            head_dim=head_dim,
            rope_head_dim=metadata.rope_head_dim,
            compress_ratio=compress_ratio,
            overlap=compress_ratio == 4,
            use_fp4_cache=False,
            rms_norm_weight=norm_weight,
            rms_norm_eps=metadata.rms_norm_eps,
            quant_block=quant_block,
            token_stride=token_stride,
            scale_dim=8 if head_dim == 512 else 4,
            **(
                {
                    "store_full_kv": False,
                    "store_full_fp8": False,
                    "fp8_scale": None,
                }
                if use_cutedsl
                else {}
            ),
        )

    def _indexer_attention_metadata(
        self, num_tokens: int, block_table: torch.Tensor
    ) -> Any:
        module = "vllm.v1.attention.backends.mla.indexer"
        metadata_cls = _symbol(module, "DeepseekV32IndexerMetadata")
        prefill_cls = _symbol(module, "DeepseekV32IndexerPrefillMetadata")
        build_chunk = _symbol(module, "build_prefill_chunk_metadata")
        query_start = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=self.device
        )
        seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)
        compressed = seq_lens // self.compress_ratio
        chunk = build_chunk(
            0,
            1,
            query_start,
            query_start.cpu(),
            seq_lens,
            compressed,
            compressed.cpu(),
            block_table,
            self.compress_ratio,
        )
        return metadata_cls(
            seq_lens=seq_lens,
            max_seq_len=num_tokens,
            slot_mapping=self._compressed_mapping(num_tokens, self.compress_ratio),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=num_tokens,
            prefill=prefill_cls([] if chunk is None else [chunk]),
            decode=None,
        )

    def _indexer_attention_metadata_batch(
        self,
        token_counts: list[int],
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> Any:
        module = "vllm.v1.attention.backends.mla.indexer"
        metadata_cls = _symbol(module, "DeepseekV32IndexerMetadata")
        prefill_cls = _symbol(module, "DeepseekV32IndexerPrefillMetadata")
        build_chunk = _symbol(module, "build_prefill_chunk_metadata")
        query_start = torch.tensor(
            [0, *torch.tensor(token_counts).cumsum(0).tolist()],
            dtype=torch.int32,
            device=self.device,
        )
        seq_lens = torch.tensor(token_counts, dtype=torch.int32, device=self.device)
        compressed = seq_lens // self.compress_ratio
        chunk = build_chunk(
            0,
            len(token_counts),
            query_start,
            query_start.cpu(),
            seq_lens,
            compressed,
            compressed.cpu(),
            block_table,
            self.compress_ratio,
        )
        return metadata_cls(
            seq_lens=seq_lens,
            max_seq_len=max(token_counts),
            slot_mapping=slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=len(token_counts),
            num_prefill_tokens=sum(token_counts),
            prefill=prefill_cls([] if chunk is None else [chunk]),
            decode=None,
        )

    @staticmethod
    def indexer_operation(
        *,
        qr: torch.Tensor,
        index_q: torch.Tensor,
        index_weights: torch.Tensor,
        positions: torch.Tensor,
        compress_ratio: int,
        topk: int,
        metadata: DS4IndexerRuntimeMetadata,
    ) -> torch.Tensor:
        """Call official indexer quantization/metadata and sparse selection ops."""

        if metadata.attention_metadata.max_seq_len // compress_ratio <= topk:
            fill = _symbol(
                "vllm.models.deepseek_v4.attention",
                "_fill_short_context_topk_indices",
            )
            padded_topk = 1 << (topk - 1).bit_length()
            fill[(positions.numel(),)](
                metadata.topk_indices,
                positions,
                TOP_K=topk,
                COMPRESS_RATIO=compress_ratio,
                PADDED_TOP_K=padded_topk,
                num_warps=8,
            )
            return metadata.topk_indices

        quantize = _symbol(
            "vllm.models.deepseek_v4.common.ops.fused_indexer_q",
            "fused_indexer_q_rope_quant",
        )
        q_quant, weights = quantize(
            positions,
            index_q,
            metadata.compressor.cos_sin_cache,
            index_weights,
            index_q.shape[-1] ** -0.5,
            index_q.shape[1] ** -0.5,
            use_fp4=False,
        )
        sparse = _symbol(
            "vllm.model_executor.layers.sparse_attn_indexer",
            "sparse_attn_indexer",
        )
        forward_module = importlib.import_module("vllm.forward_context")
        context = forward_module.ForwardContext(
            no_compile_layers={},
            attn_metadata={metadata.k_cache_prefix: metadata.attention_metadata},
            slot_mapping={},
        )
        with forward_module.override_forward_context(context):
            return sparse(
                qr,
                metadata.k_cache_prefix,
                metadata.compressor.k_cache,
                q_quant,
                None,
                None,
                weights,
                128,
                "ue8m0",
                topk,
                index_q.shape[-1],
                metadata.max_model_len,
                metadata.max_total_seq_len,
                metadata.topk_indices,
                True,
                False,
                "",
                False,
            )

    def _c128_prefill_indices(self, num_tokens: int) -> torch.Tensor:
        width = max(128, _round_up(max(num_tokens // 128, 1), 128))
        if self.device.type == "cuda":
            build = _symbol(
                "vllm.models.deepseek_v4.sparse_mla",
                "build_c128a_topk_metadata",
            )
            positions = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
            global_buffer = torch.empty(
                (1, width), dtype=torch.int32, device=self.device
            )
            decode_lens = torch.empty(1, dtype=torch.int32, device=self.device)
            prefill_buffer = torch.empty(
                (num_tokens, width), dtype=torch.int32, device=self.device
            )
            _, _, prefill = build(
                positions,
                128,
                0,
                torch.zeros(num_tokens, dtype=torch.int32, device=self.device),
                torch.zeros((1, 1), dtype=torch.int32, device=self.device),
                2,
                torch.arange(num_tokens, dtype=torch.int64, device=self.device),
                global_buffer,
                decode_lens,
                prefill_buffer,
                max_compressed_tokens=width,
            )
            return prefill
        output = torch.full(
            (num_tokens, width), -1, dtype=torch.int32, device=self.device
        )
        counts = (torch.arange(num_tokens, device=self.device) + 1) // 128
        columns = torch.arange(width, device=self.device)
        output.masked_fill_(columns.unsqueeze(0) >= counts.unsqueeze(1), -1)
        output.copy_(
            torch.where(
                columns.unsqueeze(0) < counts.unsqueeze(1),
                columns.unsqueeze(0).expand(num_tokens, -1),
                output,
            ).to(torch.int32)
        )
        return output

    def build_prefill(self, num_tokens: int):
        if self.compress_ratio == 1:
            return self._base_prefill(num_tokens)
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")

        from megatron.lite.model.deepseek_v4.vllm.model import AttentionKernelMetadata

        base = self._base_prefill(num_tokens)
        main = self._compressor_metadata(
            num_tokens,
            head_dim=self.config.head_dim,
            token_bytes=self._MAIN_TOKEN_BYTES,
        )
        topk = (
            torch.full(
                (num_tokens, self.config.index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            if self.compress_ratio == 4
            else self._c128_prefill_indices(num_tokens)
        )
        indexer_runtime = None
        if self.compress_ratio == 4:
            indexer_compressor = self._compressor_metadata(
                num_tokens,
                head_dim=self.config.index_head_dim,
                token_bytes=self._INDEXER_TOKEN_BYTES,
            )
            block_table = torch.arange(
                indexer_compressor.k_cache.shape[0],
                dtype=torch.int32,
                device=self.device,
            ).unsqueeze(0)
            indexer_runtime = DS4IndexerRuntimeMetadata(
                compressor=indexer_compressor,
                attention_metadata=self._indexer_attention_metadata(
                    num_tokens, block_table
                ),
                k_cache_prefix=f"mlite.layers.{self.layer_idx}.indexer.k_cache",
                topk_indices=topk,
                max_model_len=self.config.max_position_embeddings // 4,
                max_total_seq_len=max(1, num_tokens // 4),
            )

        compressed_len = num_tokens // self.compress_ratio
        # A single prefill invocation covers every query row in the sequence.
        # Keep the complete SWA source in the workspace: combine_topk_swa_indices
        # computes each row relative to ``seq_len - gather_len``.  Gathering only
        # the final sliding-window tail would therefore make all earlier query
        # rows point at negative workspace indices.  The combine kernel still
        # limits every row to ``sliding_window`` selected keys.
        gathered_len = num_tokens
        workspace = torch.empty(
            (1, compressed_len + gathered_len, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        combined_width = _round_up(
            topk.shape[-1] + self.config.sliding_window,
            DS4_FLASHMLA_INDEX_ALIGNMENT,
        )
        combined = torch.empty(
            (num_tokens, combined_width), dtype=torch.int32, device=self.device
        )
        combined_lens = torch.empty(num_tokens, dtype=torch.int32, device=self.device)
        prepared = False

        metadata = AttentionKernelMetadata(
            positions=base.positions,
            slot_mapping=base.slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=base.swa_cache,
            block_size=base.block_size,
            flash_kind="prefill",
            indices=topk.unsqueeze(1),
            topk_length=combined_lens,
            output=base.output,
            kv_workspace=workspace.view(-1, 1, self.config.head_dim),
            padded_heads=self.config.num_attention_heads,
            compressor_operation=self.compressor_operation,
            compressor_metadata=main,
            indexer_operation=(
                self.indexer_operation if indexer_runtime is not None else None
            ),
            indexer_metadata=indexer_runtime,
        )

        def prepare_flash() -> None:
            nonlocal prepared
            if prepared:
                return
            gather = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "dequantize_and_gather_k_cache",
            )
            block_table = torch.arange(
                main.k_cache.shape[0], dtype=torch.int32, device=self.device
            ).unsqueeze(0)
            if compressed_len:
                gather(
                    workspace,
                    main.k_cache,
                    seq_lens=torch.tensor(
                        [compressed_len], dtype=torch.int32, device=self.device
                    ),
                    gather_lens=None,
                    block_table=block_table,
                    block_size=self._MLA_BLOCK_SIZE // self.compress_ratio,
                    offset=0,
                )
            gather(
                workspace,
                base.swa_cache,
                seq_lens=torch.tensor(
                    [num_tokens], dtype=torch.int32, device=self.device
                ),
                gather_lens=torch.tensor(
                    [gathered_len], dtype=torch.int32, device=self.device
                ),
                block_table=base.runtime_layout.block_table,
                block_size=self._SWA_BLOCK_SIZE,
                offset=compressed_len,
            )
            combine = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "combine_topk_swa_indices",
            )
            indices, lengths = combine(
                metadata.indices.squeeze(1),
                base.runtime_layout.query_start_loc,
                base.runtime_layout.seq_lens,
                torch.tensor([gathered_len], dtype=torch.int32, device=self.device),
                self.config.sliding_window,
                self.compress_ratio,
                topk.shape[-1],
                workspace.shape[1],
                compressed_len,
                out=(combined, combined_lens),
            )
            metadata.indices = indices.unsqueeze(1)
            metadata.topk_length = lengths
            prepared = True

        metadata.prepare_flash = prepare_flash
        metadata.runtime_layout = SimpleNamespace(
            swa=base.runtime_layout,
            compressor=main,
            indexer=indexer_runtime,
        )
        return metadata

    def build_prefill_batch(self, token_counts: list[int]):
        """Build packed, sequence-isolated metadata for extended DS4 layers."""

        if not token_counts or any(count <= 0 for count in token_counts):
            raise ValueError("token_counts must contain positive sequence lengths")
        if self.compress_ratio == 1:
            return DS4SparseAttentionMetadataBuilderAdapter(
                self.config,
                device=self.device,
                cos_sin_cache=self.cos_sin_cache,
            ).build_prefill_batch(token_counts)

        from megatron.lite.model.deepseek_v4.vllm.model import AttentionKernelMetadata

        base = DS4SparseAttentionMetadataBuilderAdapter(
            self.config,
            device=self.device,
            cos_sin_cache=self.cos_sin_cache,
        ).build_prefill_batch(token_counts)
        main, main_block_table = self._compressor_metadata_batch(
            token_counts,
            head_dim=self.config.head_dim,
            token_bytes=self._MAIN_TOKEN_BYTES,
        )
        topk_parts = [
            (
                torch.full(
                    (count, self.config.index_topk),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.compress_ratio == 4
                else self._c128_prefill_indices(count)
            )
            for count in token_counts
        ]
        topk = torch.cat(topk_parts).contiguous()

        indexer_runtime = None
        indexer_block_table = None
        if self.compress_ratio == 4:
            indexer_compressor, indexer_block_table = (
                self._compressor_metadata_batch(
                    token_counts,
                    head_dim=self.config.index_head_dim,
                    token_bytes=self._INDEXER_TOKEN_BYTES,
                )
            )
            indexer_runtime = DS4IndexerRuntimeMetadata(
                compressor=indexer_compressor,
                attention_metadata=self._indexer_attention_metadata_batch(
                    token_counts,
                    indexer_block_table,
                    indexer_compressor.k_slot_mapping,
                ),
                k_cache_prefix=f"mlite.layers.{self.layer_idx}.indexer.k_cache",
                topk_indices=topk,
                max_model_len=self.config.max_position_embeddings // 4,
                max_total_seq_len=max(count // 4 for count in token_counts),
            )

        compressed_lens = [count // self.compress_ratio for count in token_counts]
        # See the single-sequence path above.  Each packed sequence needs its
        # full SWA source because this metadata describes the whole prefill, not
        # only a final decode-style tail.
        gathered_lens = list(token_counts)
        max_compressed = max(compressed_lens)
        max_gathered = max(gathered_lens)
        workspace_width = max_compressed + max_gathered
        workspace = torch.empty(
            (len(token_counts), workspace_width, self.config.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        combined_width = _round_up(
            topk.shape[-1] + self.config.sliding_window,
            DS4_FLASHMLA_INDEX_ALIGNMENT,
        )
        combined = torch.empty(
            (sum(token_counts), combined_width),
            dtype=torch.int32,
            device=self.device,
        )
        combined_lens = torch.empty(
            sum(token_counts), dtype=torch.int32, device=self.device
        )
        metadata = AttentionKernelMetadata(
            positions=base.positions,
            slot_mapping=base.slot_mapping,
            cos_sin_cache=self.cos_sin_cache,
            swa_cache=base.swa_cache,
            block_size=base.block_size,
            flash_kind="prefill",
            indices=topk.unsqueeze(1),
            topk_length=combined_lens,
            output=base.output,
            kv_workspace=workspace.view(-1, 1, self.config.head_dim),
            padded_heads=self.config.num_attention_heads,
            compressor_operation=self.compressor_operation,
            compressor_metadata=main,
            indexer_operation=(
                self.indexer_operation if indexer_runtime is not None else None
            ),
            indexer_metadata=indexer_runtime,
        )
        prepared = False

        def prepare_flash() -> None:
            nonlocal prepared
            if prepared:
                return
            gather = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "dequantize_and_gather_k_cache",
            )
            if max_compressed:
                gather(
                    workspace,
                    main.k_cache,
                    seq_lens=torch.tensor(
                        compressed_lens, dtype=torch.int32, device=self.device
                    ),
                    gather_lens=None,
                    block_table=main_block_table,
                    block_size=self._MLA_BLOCK_SIZE // self.compress_ratio,
                    offset=0,
                )
            gather(
                workspace,
                base.swa_cache,
                seq_lens=base.runtime_layout.seq_lens,
                gather_lens=torch.tensor(
                    gathered_lens, dtype=torch.int32, device=self.device
                ),
                block_table=base.runtime_layout.block_table,
                block_size=self._SWA_BLOCK_SIZE,
                offset=max_compressed,
            )
            combine = _symbol(
                "vllm.models.deepseek_v4.common.ops",
                "combine_topk_swa_indices",
            )
            indices, lengths = combine(
                metadata.indices.squeeze(1),
                base.runtime_layout.query_start_loc,
                base.runtime_layout.seq_lens,
                torch.tensor(gathered_lens, dtype=torch.int32, device=self.device),
                self.config.sliding_window,
                self.compress_ratio,
                topk.shape[-1],
                workspace_width,
                max_compressed,
                out=(combined, combined_lens),
            )
            metadata.indices = indices.unsqueeze(1)
            metadata.topk_length = lengths
            prepared = True

        metadata.prepare_flash = prepare_flash
        metadata.runtime_layout = SimpleNamespace(
            swa=base.runtime_layout,
            compressor=main,
            compressor_block_table=main_block_table,
            indexer=indexer_runtime,
            indexer_block_table=indexer_block_table,
        )
        return metadata


class DS4MoEKernelMetadataBuilderAdapter:
    """Build vLLM router metadata, optionally with the LL oracle kernel."""

    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        device: torch.device | str,
        deepep_buffer: Any | None = None,
        ep_size: int,
        max_tokens_per_rank: int,
        layer_idx: int = 0,
    ) -> None:
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(f"layer_idx is outside the model: {layer_idx}")
        if ep_size <= 0 or config.n_routed_experts % ep_size:
            raise ValueError("routed experts must divide evenly across EP ranks")
        self.config = config
        self.device = torch.device(device)
        self.ep_size = ep_size
        self.layer_idx = layer_idx
        max_tokens_per_rank = self.align_max_tokens_per_rank(
            max_tokens_per_rank, ep_size
        )
        self.gate_linear = _RuntimeGateLinear(
            config.hidden_size,
            config.n_routed_experts,
            device=self.device,
        )
        from megatron.lite.primitive.kernels.vllm_ds4 import (
            GroupedMoEKernelBuilderAdapter,
        )

        self.grouped_kernel = GroupedMoEKernelBuilderAdapter(
            deepep_buffer,
            device=self.device,
            num_experts=config.n_routed_experts,
            num_local_experts=config.n_routed_experts // ep_size,
            experts_per_token=config.num_experts_per_tok,
            hidden_dim=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            max_tokens_per_rank=max_tokens_per_rank,
            num_dispatchers=ep_size,
            use_fp8_dispatch=True,
        )

    @staticmethod
    def align_max_tokens_per_rank(max_tokens_per_rank: int, ep_size: int) -> int:
        if max_tokens_per_rank <= 0 or ep_size <= 0:
            raise ValueError("DeepEP token capacity and EP size must be positive")
        alignment = 4 // math.gcd(ep_size, 4)
        return _round_up(max_tokens_per_rank, alignment)

    @staticmethod
    def create_deepep_buffer(
        config: DeepseekV4Config,
        *,
        group: dist.ProcessGroup,
        ep_size: int,
        max_tokens_per_rank: int,
    ) -> Any:
        """Create the caller-owned low-latency buffer used by vLLM's oracle."""

        if config.n_routed_experts % ep_size:
            raise ValueError("routed experts must be divisible by EP size")
        local_experts = config.n_routed_experts // ep_size
        max_tokens_per_rank = (
            DS4MoEKernelMetadataBuilderAdapter.align_max_tokens_per_rank(
                max_tokens_per_rank, ep_size
            )
        )
        buffer_cls = _symbol("deep_ep", "Buffer")
        rdma_bytes = buffer_cls.get_low_latency_rdma_size_hint(
            max_tokens_per_rank,
            config.hidden_size,
            ep_size,
            config.n_routed_experts,
        )
        return buffer_cls(
            group,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=local_experts,
            allow_nvlink_for_low_latency_mode=True,
            explicitly_destroy=True,
        )

    def build(self):
        from megatron.lite.model.deepseek_v4.vllm.moe import MoEKernelMetadata
        return MoEKernelMetadata(
            gate_linear=self.gate_linear,
            build_grouped_moe=self.grouped_kernel,
        )


# Compatibility for existing layer-0 callers.
DS4HashMoEKernelMetadataBuilderAdapter = DS4MoEKernelMetadataBuilderAdapter


__all__ = [
    "DS4_FLASHMLA_INDEX_ALIGNMENT",
    "DS4_FP8_MLA_TOKEN_BYTES",
    "DS4_SWA_BLOCK_SIZE",
    "DS4RuntimeLayout",
    "DS4HashMoEKernelMetadataBuilderAdapter",
    "DS4MoEKernelMetadataBuilderAdapter",
    "DS4SparseIndexerCompressorMetadataAdapter",
    "DS4SparseAttentionMetadataBuilderAdapter",
    "ds4_vllm_forward_context",
    "initialize_ds4_vllm_batch_invariance",
]
