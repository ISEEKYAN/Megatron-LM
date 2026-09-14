"""Diagnostic only: intervene on query grouping using independent serial operands."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'unit' / 'model'))
from megatron.lite.model.qwen3_8_flash_next import qsa
from megatron.lite.model.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from test_qwen38_training import tiny_training_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = torch.load(args.source / 'serial.pt', weights_only=True)
    serial = reference['modules']['QSA']
    cp = [
        torch.load(args.source / f'cp2/rank{rank}.pt', weights_only=True)['modules'][
            'QSA'
        ]
        for rank in range(2)
    ]
    config = Qwen3_8_FlashNextTextConfig.from_hf_dict(tiny_training_config())
    model = qsa.Qwen3_8_FlashNextQSAAttention(config).cuda().bfloat16()
    model.load_state_dict(serial['initial'])
    positions = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]], device='cuda'
    )
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    inv = config.rope_theta ** (
        -torch.arange(0, rotary_dim, 2, device='cuda').float() / rotary_dim
    )
    angles = positions.float().unsqueeze(-1) * inv
    angles = torch.cat((angles, angles), -1).unsqueeze(-2)
    cu = torch.tensor([0, 5, 13, 16], device='cuda', dtype=torch.int32)
    native = qsa.sparse_attention
    records = []

    def split_query(q, k, v, routes):
        if q.shape[1] != 8:
            return native(q, k, v, routes)
        return torch.cat(
            [
                native(q[:, :3], k, v, routes[:, :3]),
                native(q[:, 3:], k, v, routes[:, 3:]),
            ],
            1,
        )

    for name, operation in [
        ('full', native),
        ('split', split_query),
        ('restored', native),
    ]:
        for repeat in range(2):
            model.zero_grad(set_to_none=True)
            x = serial['x'].cuda().requires_grad_()
            with patch.object(qsa, 'sparse_attention', operation):
                y = model(x, angles, cu_seqlens=cu)
                y.backward(serial['dy'].cuda())
            records.append(
                dict(name=name, repeat=repeat, y=y.detach().cpu(), dx=x.grad.cpu())
            )
    reports = []

    def compare(name, a, b):
        reports.append(
            dict(
                name=name,
                different=int((a != b).sum()),
                byte_equal=torch.equal(
                    a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
                ),
            )
        )

    for i in (0, 2, 4):
        for key in ('y', 'dx'):
            compare(
                f'{records[i]["name"]}.repeat.{key}',
                records[i][key],
                records[i + 1][key],
            )
            compare(f'{records[i]["name"]}.serial.{key}', records[i][key], serial[key])
            for rank in range(2):
                compare(
                    f'{records[i]["name"]}.cp{rank}.{key}',
                    records[i][key][:, rank * 8 : rank * 8 + 8],
                    cp[rank][key],
                )
    # Also isolate the attention VJP, with every numerical operand from serial.
    attention = [r for r in serial['boundaries'] if r['kind'] == 'attention'][1]
    partials = []
    for left, right in ((0, 3), (3, 8)):
        operands = [
            attention[f'input{i}'].cuda().clone().requires_grad_() for i in range(3)
        ]
        output = native(
            operands[0][:, left:right],
            operands[1],
            operands[2],
            attention['input3'][:, left:right].cuda(),
        )
        output.backward(attention['output_grad'][:, left:right].cuda())
        partials.append(
            {f'input{i}_grad': x.grad.cpu() for i, x in enumerate(operands)}
        )
    for rank, partial in enumerate(partials):
        target = [r for r in cp[rank]['boundaries'] if r['kind'] == 'attention'][
            1 if rank == 0 else 0
        ]
        for key in ('input1_grad', 'input2_grad'):
            compare(f'independent.partial{rank}.{key}', partial[key], target[key])
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(records=records, partials=partials, reports=reports),
        args.output / 'raw.pt',
    )
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2), flush=True)
    for item in reports:
        if '.repeat.' in item['name'] or item['name'].startswith(
            ('full.serial.', 'restored.serial.')
        ):
            assert item['different'] == 0, (
                'CP_QSA_QUERY_SPLIT_DIAGNOSTIC_REPEAT',
                item,
            )


if __name__ == '__main__':
    main()
