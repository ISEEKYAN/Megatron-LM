"""Strict C5 comparison at the actual differentiable PPO loss boundary."""

import torch


def compare(actual, rollout, old, mask, *, grad_enabled):
    if not grad_enabled or not actual.requires_grad:
        raise AssertionError('C5 requires the grad-enabled actor forward')
    if (
        actual.shape != rollout.shape
        or actual.shape != old.shape
        or actual.shape != mask.shape
    ):
        raise AssertionError('C5 same-batch probability/mask shapes differ')
    mask = mask.bool()
    selected = tuple(value[mask] for value in (actual, rollout, old))
    if not selected[0].numel() or not all(
        torch.isfinite(value).all() for value in selected
    ):
        raise AssertionError('C5 requires finite valid response probabilities')
    if not torch.equal(selected[0], selected[1]) or not torch.equal(
        selected[0], selected[2]
    ):
        raise AssertionError(
            f'C5 grad-enabled actor logprob mismatch: '
            f'n={int((selected[0] != selected[1]).sum())}, '
            f'max_abs={float((selected[0].detach() - selected[1]).abs().max())}'
        )
    return {
        'actor_log_probs': selected[0].detach().cpu().contiguous(),
        'rollout_log_probs': selected[1].detach().cpu().contiguous(),
        'old_log_probs': selected[2].detach().cpu().contiguous(),
        'grad_enabled': True,
        'actor_requires_grad': True,
    }


def install(module, save, event, counters):
    original = module.ppo_loss

    def loss(config, model_output, data, dp_group=None):
        actual = module.no_padding_2_padding(model_output['log_probs'], data)
        paired = data.select(
            'rollout_log_probs', 'old_log_probs', 'response_mask'
        ).to_padded_tensor()
        receipt = compare(
            actual,
            paired['rollout_log_probs'],
            paired['old_log_probs'],
            paired['response_mask'],
            grad_enabled=torch.is_grad_enabled(),
        )
        receipt['update'] = counters['optimizer'] + 1
        receipt['rank'] = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        receipt['input_ids'] = [
            row.detach().cpu() for row in data['input_ids'].unbind(0)
        ]
        target = save('grad', receipt)
        event(
            'grad_forward',
            {
                'update': receipt['update'],
                'rank': receipt['rank'],
                'tokens': receipt['actor_log_probs'].numel(),
                'tensor_file': target,
            },
        )
        return original(config, model_output, data, dp_group)

    module.ppo_loss = loss


def expected_pp_updates(optimizer_rows, *, ranks, pp_size, steps):
    """Validate every stage owner, then require C5 only from the head stage."""
    assert pp_size > 1 and ranks > 0 and ranks % pp_size == 0 and steps > 0
    assert len(optimizer_rows) == ranks, 'PP optimizer census incomplete'
    owners = {}
    for rows in optimizer_rows.values():
        identities = {
            (r['global_rank'], r['pp_rank'], r['pp_size'], r['world_size'])
            for r in rows
        }
        assert len(identities) == 1, 'PP optimizer identity changed'
        rank, stage, size, world = identities.pop()
        assert type(rank) is int and type(stage) is int
        assert 0 <= rank < ranks and size == pp_size and world == ranks
        assert stage == rank // (ranks // pp_size), 'PP stage/rank mapping differs'
        assert rank not in owners, 'Duplicate PP optimizer rank'
        owners[rank] = stage
    assert set(owners) == set(range(ranks)), 'PP optimizer rank missing'
    return {
        (rank, update)
        for rank, stage in owners.items()
        if stage == pp_size - 1
        for update in range(1, steps + 1)
    }
