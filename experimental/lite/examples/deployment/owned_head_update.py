# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Small owned Muon head update/state-residency example, without a rollout."""
import argparse
from types import SimpleNamespace

import torch
from megatron.lite.primitive.optimizers.headwise_muon import MixedOptimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    torch.manual_seed(19)
    masters = torch.nn.ParameterDict(
        {'head': torch.nn.Parameter(torch.randn(17, 8, device=args.device))}
    )
    config = SimpleNamespace(
        lr=1e-3, ns_steps=2, coefficient_type='quintic', clip_grad=1.0
    )

    def groups():
        return [
            dict(
                params=[masters['head']],
                owner_key='head',
                algorithm='muon',
                matrix_shape=(17, 8),
                weight_decay=0.1,
            )
        ]

    optimizer = MixedOptimizer(
        masters,
        config,
        group_builder=groups,
        owners=lambda: ([], [], [], None),
        stats_factory=SimpleNamespace,
    )
    initial = masters['head'].detach().clone()
    hidden = torch.randn(7, 8, device=args.device, dtype=torch.bfloat16)
    labels = torch.arange(7, device=args.device)
    for _ in range(2):
        optimizer.zero_grad()
        logits = hidden.float() @ masters['head'].T
        loss = -torch.log_softmax(logits, -1).gather(-1, labels[:, None]).mean()
        loss.backward()
        assert masters['head'].grad.dtype == torch.float32
        assert optimizer.step()[0]
        optimizer.offload_state_to_cpu()
        optimizer.load_state_to_device()
    assert not torch.equal(initial, masters['head'])
    assert masters['head'].dtype == torch.float32
    print('Two owned Muon head updates and state roundtrips: PASS')


if __name__ == '__main__':
    main()
