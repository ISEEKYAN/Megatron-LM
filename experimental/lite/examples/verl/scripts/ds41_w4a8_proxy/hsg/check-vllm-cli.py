"""Check actual VERL CLI parsing, replay eligibility and per-rank host storage."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from verl_mlite.compat import _patch_verl_vllm_prefix_caching_cli

# Match container-node.sh: enable the DS41 BooleanOptionalAction adapter.
os.environ.setdefault('VERL_MLITE_HF_CONFIG_MODEL_TYPE', 'deepseek_v41')
_patch_verl_vllm_prefix_caching_cli()
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config

resolved = Path(sys.argv[1])
raw = yaml.load(resolved.read_text(), Loader=yaml.CSafeLoader)
rollout = raw['actor_rollout_ref']['rollout']
engine_kwargs = dict(rollout['engine_kwargs']['vllm'] or {})
# These server fields precede engine_kwargs in VERL's production argument map.
server_args = {
    key: rollout[key]
    for key in ('max_model_len', 'max_num_batched_tokens', 'enable_prefix_caching')
}
server_args.update(engine_kwargs)
cli = build_cli_args_from_config(server_args)

# A CPU node cannot infer vLLM's platform; pin the CUDA platform the GPU job
# runs on before the parser builds its configuration classes.
import vllm.platforms as platforms  # noqa: E402
from vllm.platforms.cuda import CudaPlatform  # noqa: E402

platforms._current_platform = CudaPlatform()
from vllm.config import EngramConfig, ParallelConfig  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
# parse_args is the path `vllm serve` takes (it normalizes --a_b to --a-b).
parsed = parser.parse_args(['--model', 'ds41', *cli])
args = AsyncEngineArgs.from_cli_args(parsed)
engram = args.engram_config or EngramConfig()
parallel = ParallelConfig(
    data_parallel_size=int(raw['actor_rollout_ref']['rollout']['data_parallel_size']),
    tensor_parallel_size=int(
        raw['actor_rollout_ref']['rollout']['tensor_model_parallel_size']
    ),
)
engram.resolve_dp_shared_memory(parallel)
engram.verify_parallel_config(parallel)
from vllm.models.deepseek_v41.attention import resolve_swa_bounded_replay

effective_replay = resolve_swa_bounded_replay(
    SimpleNamespace(
        cache_config=SimpleNamespace(
            swa_bounded_replay=parsed.swa_bounded_replay,
            enable_prefix_caching=parsed.enable_prefix_caching,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=parsed.max_num_batched_tokens
        ),
        model_config=SimpleNamespace(max_model_len=parsed.max_model_len),
    )
)
record = {
    'layers': os.environ.get('DS41_LAYERS'),
    'resolved': str(resolved),
    'cli_replay_args': [a for a in cli if 'bounded_replay' in a],
    'swa_bounded_replay_requested': parsed.swa_bounded_replay,
    'swa_bounded_replay': effective_replay,
    'enable_prefix_caching': parsed.enable_prefix_caching,
    'tensor_parallel_size': parallel.tensor_parallel_size,
    'engram_config': {
        'cpu_offload': engram.cpu_offload,
        'dp_shared_memory': engram.dp_shared_memory,
        'embedding_across_dp': engram.embedding_across_dp,
        'host_table_layout': engram.host_table_layout,
    },
}
(resolved.parent / 'vllm-cli.json').write_text(json.dumps(record, indent=2) + '\n')
print('DS41_VLLM_CLI ' + json.dumps(record), flush=True)
if parsed.swa_bounded_replay is not None or effective_replay is not True:
    raise SystemExit('The default replay policy did not resolve enabled')
if parsed.enable_prefix_caching is not False:
    raise SystemExit('Prefix caching was not disabled through the VERL CLI')
if (
    not engram.cpu_offload
    or engram.dp_shared_memory is not False
    or engram.host_table_layout != 'per_rank'
    or parallel.tensor_parallel_size != 1
):
    raise SystemExit(f'Invalid per-rank Engram profile: {engram}')
