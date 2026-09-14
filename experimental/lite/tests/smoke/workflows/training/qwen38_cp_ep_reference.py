"""Independent virtual EP owners for the contiguous CP2 reference.

Only independently built reference states, raw batches and fixed CP/EP layout
enter here. No tested routing maps, counts, gradients or collective output.
"""

import torch
from qwen38_cp_reference import forward
from torch.nn import functional as F


class Exchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, splits, left, right):
        ctx.splits = splits
        rows = [torch.split(x, n, 0) for x, n in zip((left, right), splits)]
        return tuple(torch.cat([rows[s][d] for s in range(2)], 0) for d in range(2))

    @staticmethod
    def backward(ctx, left, right):
        reverse = list(zip(*ctx.splits))
        rows = [torch.split(x, n, 0) for x, n in zip((left, right), reverse)]
        return (None, *(torch.cat([rows[d][s] for d in range(2)], 0) for s in range(2)))


def lookup(modules, hashes):
    weights = [m.ple_embedding.ngram_embedding.weight for m in modules]
    local_rows = weights[0].shape[0]
    ids = [hashes[:, r * 8 : (r + 1) * 8].reshape(-1) for r in range(2)]
    locations = [
        [torch.where(x // local_rows == owner)[0] for owner in range(2)] for x in ids
    ]
    counts = [[v.numel() for v in row] for row in locations]
    values = []
    for owner, weight in enumerate(weights):
        requested = torch.cat([ids[src][locations[src][owner]] for src in range(2)])
        values.append(F.embedding(requested - owner * local_rows, weight))
    returned = Exchange.apply(list(zip(*counts)), *values)
    outputs = []
    for src in range(2):
        order = torch.cat(locations[src])
        inverse = torch.argsort(order)
        outputs.append(returned[src][inverse].reshape(1, 8, 16, -1))
    return outputs


def experts(modules, inputs, routing):
    from megatron.lite.primitive.utils.moe import permute, unpermute

    num_experts = modules[0].router.num_experts
    local = num_experts // 2
    values, probabilities, indices, counts = [], [], [], []
    for x, (scores, topk) in zip(inputs, routing):
        mapping = torch.zeros(
            x.shape[0], num_experts, dtype=torch.bool, device=x.device
        )
        mapping.scatter_(1, topk, True)
        dense = scores.new_zeros(mapping.shape).scatter_add_(1, topk, scores)
        value, prob, index, *_ = permute(
            x, mapping, probs=dense, num_out_tokens=int(mapping.sum()), fused=False
        )
        values.append(value)
        probabilities.append(prob.unsqueeze(-1))
        indices.append(index)
        counts.append(mapping.sum(0).tolist())
    splits = [
        [sum(row[o * local : (o + 1) * local]) for o in range(2)] for row in counts
    ]
    received = Exchange.apply(splits, *values)
    received_probs = Exchange.apply(splits, *probabilities)
    output = []
    for owner, m in enumerate(modules):
        sizes = [
            counts[s][e]
            for s in range(2)
            for e in range(owner * local, (owner + 1) * local)
        ]
        parts = torch.split(received[owner], sizes)
        probs = torch.split(received_probs[owner], sizes)
        order = [s * local + e for e in range(local) for s in range(2)]
        dispatched = torch.cat([parts[i] for i in order])
        weights = torch.cat([probs[i] for i in order]).squeeze(-1)
        tpe = torch.tensor(
            [
                counts[0][e] + counts[1][e]
                for e in range(owner * local, (owner + 1) * local)
            ],
            device=dispatched.device,
        )
        result = m.experts(dispatched, tpe, weights, tokens_per_expert_list=None)
        parts = torch.split(result, [sizes[i] for i in order])
        inverse = [e * 2 + s for s in range(2) for e in range(local)]
        output.append(torch.cat([parts[i] for i in inverse]))
    returned = Exchange.apply(list(zip(*splits)), *output)
    return [
        unpermute(y, i, restore_shape=x.shape, fused=False)
        for y, i, x in zip(returned, indices, inputs)
    ]


def ordered_forward(models, batch, trace=None):
    return forward(models, batch, trace, lookup=lookup, expert_forward=experts)
