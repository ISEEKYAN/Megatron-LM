# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native R1 operator composition, not a serving-engine or C8 claim."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import pad

from megatron.lite.model.deepseek_v41.vllm import lora

pytestmark = pytest.mark.gpus(1)


def _bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def test_native_punica_zero_nonzero_and_grouped_wo_a():
    from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    torch.manual_seed(42)
    rows, groups, width, out, rank = 7, 8, 256, 128, 32
    x = torch.randn(rows, groups, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    base = torch.randn(rows, groups, out, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    a = torch.randn(groups, rank, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b = torch.zeros(groups, out, rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    module = SimpleNamespace(lora_a=a, lora_b=b, scale=0.5)
    _bits(lora.grouped_add(x, base, module), base)
    with torch.no_grad():
        b.normal_(std=0.03)
    actual = lora.grouped_add(x, base, module)
    meta = LoRAKernelMeta.make(1, rows, x.device)
    meta.prepare_tensors(torch.zeros(rows, device=x.device, dtype=torch.int32))
    args = meta.meta_args(rows, specialize_active_lora=True)
    hidden = torch.zeros(groups, rows, rank, device=x.device, dtype=torch.float32)
    for group in range(groups):
        lora_shrink(
            x[:, group].contiguous(),
            [a[group][None, None].contiguous()],
            hidden[group : group + 1],
            *args,
            1.0,
        )
    expected = base.detach().flatten(1).clone()
    weights = [(value.detach() * 0.5)[None, None].contiguous() for value in b]
    lora_expand(hidden, weights, expected, *args, add_inputs=True)
    _bits(actual, expected.reshape_as(actual))
    # The ordinary one-slice path must have the same rounding chain too.
    single = SimpleNamespace(lora_a=a[0], lora_b=b[0], scale=0.5)
    _bits(lora.add(x[:, 0], base[:, 0], single), expected.reshape_as(actual)[:, 0])
    actual.float().square().mean().backward()
    for parameter in (x, base, a, b):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero()


def test_routed_native_grouped_deltas_with_empty_owner():
    from vllm.lora.layers.moe_slot import MoELoRASlotWithLoRA
    from vllm.utils.deep_gemm import _import_deep_gemm

    torch.manual_seed(43)
    dg = _import_deep_gemm()
    counts, rank, width, out = (3, 0, 4), 32, 256, 128
    rows = sum(counts)
    hidden = torch.randn(rows, rank, device='cuda', dtype=torch.float32, requires_grad=True)
    x = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b = [
        torch.randn(out, rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        for _ in counts
    ]
    a = [
        torch.randn(rank, width, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        for _ in counts
    ]
    shared_b = torch.randn(out, rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    module = SimpleNamespace(
        rank=rank,
        scale=0.5,
        fc1_lora_b=b,
        fc2_lora_a=a,
        fc2_lora_b=shared_b,
        per_expert=lambda values: values,
    )
    alignment = dg.get_theoretical_mk_alignment_for_contiguous_layout()
    positions, owners, offset = [], [], 0
    for expert, count in enumerate(counts):
        positions.extend(range(offset, offset + count))
        owners.extend([expert] * count)
        offset += (count + alignment - 1) // alignment * alignment
    pos = torch.tensor(positions, device=x.device)
    layout = torch.full((offset,), -1, device=x.device, dtype=torch.int32)
    layout[pos] = torch.tensor(owners, device=x.device, dtype=torch.int32)
    padded_hidden = hidden.new_zeros(offset, rank).index_copy_(0, pos, hidden.detach())
    gate_slot = SimpleNamespace(
        _b_per_expert=True, expert_w=pad(torch.stack(b).detach() * 0.5, (0, 64 - rank))
    )
    expected_gate = MoELoRASlotWithLoRA.expert_delta(gate_slot, padded_hidden, layout)[pos]
    actual_gate = lora.expert_gate_up(hidden, module, counts)
    _bits(actual_gate, expected_gate)
    padded_x = x.new_zeros(offset, width).index_copy_(0, pos, x.detach())
    down_slot = SimpleNamespace(
        _b_per_expert=False,
        expert_w=pad(torch.stack(a).detach(), (0, 0, 0, 64 - rank)),
        shared_w=pad(shared_b.detach() * 0.5, (0, 64 - rank)),
    )
    expected_down = MoELoRASlotWithLoRA.expert_delta(down_slot, padded_x, layout)[pos]
    actual_down = lora.expert_down(x, module, counts)
    _bits(actual_down, expected_down)
    (actual_gate.float().square().mean() + actual_down.float().square().mean()).backward()
    for parameter in (x, hidden, shared_b, *a, *b):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert a[1].grad.count_nonzero() == b[1].grad.count_nonzero() == 0
    assert a[0].grad.count_nonzero() and b[2].grad.count_nonzero()


def test_fp8_wo_a_native_rotated_input_and_finite_adjoint():
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.layers.grouped_linear import GroupedColumnParallelLinearWithLoRA
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

    from megatron.lite.model.deepseek_v41.lite.export import to_release
    from megatron.lite.model.deepseek_v41.vllm.attention import (
        _FP8OutputProjection,
        inverse_rope_bf16,
    )

    torch.manual_seed(44)
    rows, groups, heads, dim, out, rank = 7, 8, 64, 512, 128, 32
    width = heads * dim // groups
    x = torch.randn(rows, heads, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(groups * out, width, device='cuda', dtype=torch.bfloat16) * 0.01
    positions = torch.arange(rows, device='cuda')
    angles = torch.randn(64, 32, device='cuda')
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    a = (
        torch.randn(groups, rank, width, device='cuda', dtype=torch.bfloat16) * 0.01
    ).requires_grad_()
    b = torch.zeros(groups, out, rank, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    adapter = SimpleNamespace(lora_a=a, lora_b=b, scale=0.5)
    config = LoRAConfig(max_loras=1, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperGPU(256, 8, 'cuda', lora_config=config)
    punica.update_metadata(LoRAMapping((1,) * rows, (1,) * rows), [1], max_loras=1, vocab_size=0)
    native = SimpleNamespace(
        groups=groups, lora_config=config, punica_wrapper=punica, output_slices=(out,) * groups
    )
    tensors = dict(to_release('layers.0.attn.wo_a.weight', weight))
    codes, scales = deepgemm_post_process_fp8_weight_block(
        tensors['layers.0.attn.wo_a.weight'],
        tensors['layers.0.attn.wo_a.scale'],
        (32, 32),
        True,
        is_bmm=True,
        bmm_batch_size=groups,
    )
    seen = {}

    def apply_native(rotated, z):
        seen['rotated'] = rotated.clone()
        GroupedColumnParallelLinearWithLoRA.apply_grouped_lora(native, rotated, z)

    projection = SimpleNamespace(weight=codes, weight_scale=scales, apply_grouped_lora=apply_native)
    rotated = inverse_rope_bf16(x, positions, cache, groups)
    base = _FP8OutputProjection.apply(x, weight, positions, cache, groups)
    for nonzero in (False, True):
        if nonzero:
            with torch.no_grad():
                b.normal_(std=0.03)
        native.lora_a_stacked = tuple(v[None, None].detach().contiguous() for v in a)
        native.lora_b_stacked = tuple((v.detach() * 0.5)[None, None].contiguous() for v in b)
        expected = deep_gemm_fp8_o_proj(
            x.detach(),
            positions,
            cache,
            projection,
            lambda z: z,
            n_groups=groups,
            heads_per_group=heads // groups,
            nope_dim=448,
            rope_dim=64,
            o_lora_rank=out,
            einsum_recipe=(1, 1, 32),
            tma_aligned_scales=True,
        ).reshape_as(base)
        actual = lora.grouped_add(rotated, base, adapter)
        _bits(rotated, seen['rotated'])
        _bits(actual, expected)
        if not nonzero:
            _bits(actual, base)
    actual.float().square().mean().backward()
    for parameter in (x, a, b):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero()
