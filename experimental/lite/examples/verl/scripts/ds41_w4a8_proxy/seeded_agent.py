"""Evidence recipe: explicit independent reproducible request RNG through VERL API.

Model, rollout distribution, reward and original SingleTurnAgentLoop outputs stay
owned by the original implementations. Only SamplingParams.seed is supplied.
"""

import json
import operator
import os
from pathlib import Path

import verl.experimental.agent_loop.single_turn_agent_loop as single_turn

SingleTurnAgentLoop = single_turn.SingleTurnAgentLoop


def request_sampling_params(original, base_seed, priority):
    base = operator.index(base_seed)
    index = operator.index(priority)
    if base < 0 or index < 0 or base + index >= 2**63:
        raise ValueError('seed and priority must fit nonnegative signed int64')
    params = dict(original)
    params.setdefault('seed', base + index)
    return params


class SeededSingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params, priority=0, **kwargs):
        params = request_sampling_params(
            sampling_params, self.rollout_config.seed, priority
        )
        root = Path(os.environ['W4_GRPO_OUT']) / 'sampling-audit'
        root.mkdir(parents=True, exist_ok=True)
        record = {
            'protocol': 'base-seed-plus-global-priority-v1',
            'priority': int(priority),
            'base_seed': int(self.rollout_config.seed),
            'request_seed': params['seed'],
            'caller_explicit_seed': 'seed' in sampling_params,
            'original_sampling_params': sampling_params,
        }
        with (root / f'seeds-{os.getpid()}.jsonl').open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')
        return await super().run(params, priority=priority, **kwargs)
