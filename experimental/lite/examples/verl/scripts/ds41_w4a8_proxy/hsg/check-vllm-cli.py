"""CPU gate: the rollout's vLLM arguments, as VERL passes them, turn replay off.

VERL serves the rollout through ``vllm serve`` and converts every
``rollout.engine_kwargs.vllm`` entry with ``build_cli_args_from_config``, which
drops False booleans. Convert the resolved engine kwargs the same way, parse
them with vLLM's own engine-argument parser and require decoder SWA bounded
replay off. The receipt goes next to the resolved config as vllm-cli.json.
"""

import json
import os
import sys
from pathlib import Path

import yaml
from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config

resolved = Path(sys.argv[1])
raw = yaml.load(resolved.read_text(), Loader=yaml.CSafeLoader)
engine_kwargs = dict(raw['actor_rollout_ref']['rollout']['engine_kwargs']['vllm'] or {})
cli = build_cli_args_from_config(engine_kwargs)

# A CPU node cannot infer vLLM's platform; pin the CUDA platform the GPU job
# runs on before the parser builds its configuration classes.
import vllm.platforms as platforms  # noqa: E402
from vllm.platforms.cuda import CudaPlatform  # noqa: E402

platforms._current_platform = CudaPlatform()
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())
# parse_args is the path `vllm serve` takes (it normalizes --a_b to --a-b).
parsed = parser.parse_args(['--model', 'ds41', *cli])
record = {
    'layers': os.environ.get('DS41_LAYERS'),
    'resolved': str(resolved),
    'cli_replay_args': [a for a in cli if 'bounded_replay' in a],
    'swa_bounded_replay': parsed.swa_bounded_replay,
}
(resolved.parent / 'vllm-cli.json').write_text(json.dumps(record, indent=2) + '\n')
print('DS41_VLLM_CLI ' + json.dumps(record), flush=True)
if '--no_swa_bounded_replay' not in cli:
    raise SystemExit('The negated replay flag did not reach the vLLM CLI')
if parsed.swa_bounded_replay is not False:
    raise SystemExit(f'vLLM would run swa_bounded_replay={parsed.swa_bounded_replay}')
