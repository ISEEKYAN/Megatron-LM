"""Independent serial versus real CP2 QSA/PLE, before parameter reductions.

Only real-token outputs enter the objective; padded rows retain zero input
adjoints. Full raw tensors are retained, including unscored padding outputs.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from megatron.lite.model.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from megatron.lite.model.qwen3_8_flash_next.model import Qwen38Layer
from megatron.lite.model.qwen3_8_flash_next.protocol import _forward_step
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig
from test_qwen38_training import tiny_training_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
    assert world in (1, 2)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.manual_seed(3853)
    ps = init_parallel(ParallelConfig(cp=world))
    cfg = Qwen3_8_FlashNextTextConfig.from_hf_dict(tiny_training_config())
    layer = Qwen38Layer(
        cfg,
        ps,
        1,
        ngram_primes=(17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79),
    ).to(device='cuda', dtype=torch.bfloat16)
    modules = {'PLE': layer.ple, 'QSA': layer.self_attn}
    reference = torch.load(args.reference, weights_only=True) if world > 1 else None
    if reference is not None:
        assert (
            reference['world'] == 1
            and reference['run_id'] == os.environ['QWEN_CP_RUN_ID']
        ), 'CP_COMPONENT_REFERENCE_IDENTITY'
    ids = torch.arange(13, device='cuda')
    loss_mask = torch.ones_like(ids, dtype=torch.bool)
    loss_mask[10] = False
    batch = PackedBatch(
        ids, ids.clone(), torch.tensor([5, 8], device='cuda'), loss_mask=loss_mask
    )

    class Inputs:
        def __init__(self):
            self.ps = ps

        def __call__(self, **kwargs):
            return kwargs

    prepared = _forward_step(Inputs(), batch)
    context = prepared.get('cp_context')
    start, end = (
        (0, 16)
        if world == 1
        else (context.local_sequence_start, context.local_sequence_end)
    )
    serial_ids = torch.nn.functional.pad(
        ids.reshape(1, -1), (0, 3), value=cfg.eos_token_id
    )
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]], device='cuda'
    )
    cu = torch.tensor([0, 5, 13, 16], device='cuda', dtype=torch.int32)
    rotary_dim = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv = cfg.rope_theta ** (
        -torch.arange(0, rotary_dim, 2, device='cuda').float() / rotary_dim
    )
    pos = positions if world == 1 else prepared['position_ids']
    angles = pos.float().unsqueeze(-1) * inv
    angles = torch.cat((angles, angles), -1).unsqueeze(-2)
    result = dict(
        world=world, rank=rank, run_id=os.environ['QWEN_CP_RUN_ID'], modules={}
    )
    checks = []
    for name, module in modules.items():
        projection_lengths = []
        projection = module.key_proj if name == 'PLE' else module.q_proj
        hook = projection.register_forward_pre_hook(
            lambda module, inputs: projection_lengths.append(inputs[0].shape[1])
        )
        if reference is not None:
            module.load_state_dict(reference['modules'][name]['initial'])
        width = cfg.hidden_size * (cfg.hc_count if name == 'PLE' else 1)
        full_x = torch.randn(1, 16, width, device='cuda', dtype=torch.bfloat16)
        full_dy = torch.randn_like(full_x)
        full_x[:, 13:] = 0
        full_dy[:, 13:] = 0
        if reference is not None:
            full_x = reference['modules'][name]['x'].cuda()
            full_dy = reference['modules'][name]['dy'].cuda()
        initial = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

        def run():
            module.zero_grad(set_to_none=True)
            x = full_x[:, start:end].clone().requires_grad_()
            if name == 'PLE':
                y = module(
                    x,
                    serial_ids if world == 1 else prepared['input_ids'],
                    cu_seqlens=cu if world == 1 else None,
                    cp_context=context,
                )
            else:
                y = module(
                    x, angles, cu_seqlens=cu if world == 1 else None, cp_context=context
                )
            y.backward(full_dy[:, start:end])
            return dict(
                y=y.detach().cpu(),
                dx=x.grad.cpu(),
                local_grads={
                    n: p.grad.detach().cpu().clone()
                    for n, p in module.named_parameters()
                    if p.grad is not None
                },
            )

        actual = run()
        if world == 1:
            repeated = run()
            for field in ('y', 'dx'):
                assert torch.equal(actual[field], repeated[field]), (
                    'CP_COMPONENT_SERIAL_REPEAT_BITWISE',
                    name,
                    field,
                )
        else:
            target = reference['modules'][name]
            valid = torch.arange(start, end) < 13
            checks.extend(
                [
                    (
                        f'CP_{name}_VALID_FORWARD_BITWISE',
                        torch.equal(
                            actual['y'][:, valid], target['y'][:, start:end][:, valid]
                        ),
                    ),
                    (
                        f'CP_{name}_DX_BITWISE',
                        torch.equal(actual['dx'], target['dx'][:, start:end]),
                    ),
                    (
                        f'CP_{name}_PAD_DX_ZERO',
                        not bool(actual['dx'][:, ~valid].count_nonzero()),
                    ),
                ]
            )
        if world > 1:
            checks.append((f'CP_{name}_LOCAL_PROJECTION', projection_lengths == [8]))
        hook.remove()
        actual['projection_lengths'] = projection_lengths
        result['modules'][name] = dict(
            initial=initial, x=full_x.cpu(), dy=full_dy.cpu(), **actual
        )
    result['checks'] = checks
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output / f'rank{rank}.pt')
    if world == 1:
        torch.save(result, args.reference)
        print('CP_COMPONENT_INDEPENDENT_SERIAL_SAVED', flush=True)
    else:
        for tag, passed in checks:
            print(tag, rank, passed, flush=True)
        for tag, passed in checks:
            assert passed, (tag, rank)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
