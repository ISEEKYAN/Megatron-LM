"""Slurm-only CP transport isolation; the reference is one unsharded process.

This is a GDN component probe, not whole-model CP acceptance. Save raw tensors
before exact assertions so a reduction mismatch never becomes a tolerance pass.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.lite.model.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from megatron.lite.model.qwen3_8_flash_next.model import Qwen38GatedDeltaNet
from megatron.lite.primitive.modules import gated_delta_net as shared
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.utils.packed_seq import PackedSeqParams
from megatron.lite.runtime.contracts import ParallelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    assert world in (1, 2), 'CP_GDN_WORLD'
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(3852)
    ps = init_parallel(ParallelConfig(cp=world))
    assert shared._HAS_FLA, 'CP_GDN_PACKED_FLA_REQUIRED'
    cfg = Qwen3_8_FlashNextTextConfig.from_hf_dict(
        dict(
            model_type='qwen4_exp_text',
            hidden_size=128,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )
    )
    model = Qwen38GatedDeltaNet(cfg, ps).to(dtype=torch.bfloat16, device='cuda')
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reference = torch.load(args.reference, weights_only=True) if world > 1 else None
    run_id = os.environ['QWEN_CP_RUN_ID']
    if reference is not None:
        assert (
            reference['world'] == 1 and reference['run_id'] == run_id
        ), 'CP_GDN_SINGLE_PROCESS_REFERENCE'
        model.load_state_dict(reference['initial'])
        initial = reference['initial']
    # Global physical length 16, real length 13; document 5..13 crosses CP boundary 8.
    cu = torch.tensor([0, 5, 13, 16], device='cuda', dtype=torch.int32)
    full_x = torch.randn(16, 1, 128, device='cuda', dtype=torch.bfloat16)
    full_dy = torch.randn_like(full_x)
    full_x[13:] = 0
    full_dy[13:] = 0
    if reference is not None:
        full_x, full_dy = reference['x'].cuda(), reference['dy'].cuda()
    x = full_x.chunk(world, 0)[rank].clone().requires_grad_()
    dy = full_dy.chunk(world, 0)[rank].clone()
    traces = {}

    def capture(name):
        def hook(module, inputs, output):
            traces[name] = output.detach().cpu().clone()

        return hook

    model.in_proj.register_forward_hook(capture('projected'))
    original_rule = model._gated_delta_rule

    def rule(q, k, v, *a, **kw):
        traces['recurrence_q'] = q.detach().cpu().clone()
        out = original_rule(q, k, v, *a, **kw)
        traces['recurrence_output'] = out[0].detach().cpu().clone()
        return out

    model._gated_delta_rule = rule
    calls = []
    original_exchange = dist.all_to_all_single

    def exchange(*a, **kw):
        if kw.get('group') is ps.cp_group and world > 1:
            calls.append([list(t.shape) for t in a if isinstance(t, torch.Tensor)])
        return original_exchange(*a, **kw)

    dist.all_to_all_single = exchange
    y = model(x, packed_seq_params=PackedSeqParams.from_cu_seqlens(cu, 8))
    y.backward(dy)
    # Capture local contributions before any parameter-gradient all_reduce.
    local_grads = {
        n: p.grad.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    data = dict(
        world=world,
        rank=rank,
        run_id=run_id,
        initial=initial,
        x=full_x.cpu(),
        dy=full_dy.cpu(),
        y=y.detach().cpu(),
        dx=x.grad.cpu(),
        local_grads=local_grads,
        traces=traces,
        calls=calls,
        real_length=13,
        padded_length=16,
        cu=cu.cpu(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output / f'rank{rank}.pt')
    if world == 1:
        torch.save(data, args.reference)
        print('CP_GDN_SERIAL_SAVED', flush=True)
    else:
        assert x.shape[0] < full_x.shape[0], 'CP_GDN_LOCAL_TOKEN_STORAGE'
        assert (
            traces['recurrence_q'].shape[2]
            < reference['traces']['recurrence_q'].shape[2]
        ), 'CP_GDN_LOCAL_HEAD_STORAGE'
        assert len(calls) == 4, ('CP_GDN_REAL_ALL_TO_ALL', len(calls))
        assert torch.equal(
            traces['projected'], reference['traces']['projected'].chunk(world, 0)[rank]
        ), 'CP_GDN_PRE_TRANSPORT_PROJECTION'
        print('CP_GDN_PRE_TRANSPORT_PROJECTION_EXACT', rank, flush=True)
        for key in ('recurrence_q', 'recurrence_output'):
            assert torch.equal(
                traces[key], reference['traces'][key].chunk(world, 2)[rank]
            ), ('CP_GDN_HEAD_COMPUTE_EXACT', key, rank)
        assert torch.equal(data['y'], reference['y'].chunk(world, 0)[rank]), (
            'CP_GDN_FORWARD_BITWISE',
            rank,
        )
        print('CP_GDN_FORWARD_BITWISE', rank, flush=True)
        assert torch.equal(data['dx'], reference['dx'].chunk(world, 0)[rank]), (
            'CP_GDN_DX_BITWISE',
            rank,
        )
        print('CP_GDN_DX_BITWISE', rank, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
