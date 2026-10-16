# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""python examples/checkpoint/row_stream.py --output /tmp/row-checkpoint"""
import argparse

import torch
from megatron.lite.primitive.ckpt.hf_weights import stream_export_to_shards
from megatron.lite.primitive.ckpt.row_stream import RowReceiver, stream_rows


def main():
    parser = argparse.ArgumentParser(
        description='Export row-wise FP8 with bounded scratch'
    )
    parser.add_argument('--output', required=True)
    parser.add_argument('--buffer-bytes', type=int, default=65536)
    args = parser.parse_args()
    master = torch.full((4096, 32), 1.3)
    weight = torch.empty(103, 32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(103, 1, dtype=torch.uint8)
    receiver = RowReceiver('table.weight', 4096, 991, weight, scales)
    # A rollout shard owns global rows [991,1094), not the sender's partition.
    for chunk in stream_rows(
        'table.weight',
        master,
        quantize=True,
        buffer_max_size_bytes=args.buffer_bytes // 2,
    ):
        receiver.copy(chunk)
    receiver.finish()
    # Checkpoint files are standard safetensors despite the in-memory row wire.
    stream_export_to_shards(
        stream_rows(
            'table.weight',
            master,
            quantize=True,
            buffer_max_size_bytes=args.buffer_bytes // 2,
        ),
        args.output,
        shard_size_bytes=args.buffer_bytes,
    )
    print(f'Saved checkpoint and filled local rows [991,1094): {args.output}')


if __name__ == '__main__':
    main()
