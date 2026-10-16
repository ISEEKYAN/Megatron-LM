"""Run the production C8 suite or collect the same test without GPUs."""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group()
for layers in (2, 4, 6, 8, 9, 40):
    mode.add_argument(
        f'--c8-{layers}l', dest='layers', action='store_const', const=layers
    )
parser.add_argument('--collect-only', action='store_true')
parser.add_argument('--runtime-preflight', action='store_true')
args = parser.parse_args()
layers = args.layers or 2
gpus = int(os.environ.get('DS41_C8_GPUS', '8'))
output = Path('/test-output')
import torch
import wandb

assert torch.cuda.device_count() == (0 if args.collect_only else 4)
run = wandb.init(
    project='ds41-fp4-c8',
    job_type='cpu-preflight' if args.collect_only else 'c8',
    config={'layers': layers, 'bi': 1, 'gpu_count': 0 if args.collect_only else gpus},
)
(output / 'wandb.json').write_text(json.dumps({'url': run.url}) + '\n')
print('DS41_WANDB_URL', run.url, flush=True)
lite = '/opt/src/mlite/experimental/lite'
case = (
    lite
    + f'/tests/smoke/workflows/test_ds41_grpo_recipe.py::test_two_updates_and_checkpoint_resume[{layers}-1]'
)
command = [
    sys.executable,
    '-u',
    '-m',
    'pytest',
    '--import-mode=importlib',
    '--confcutdir=' + lite,
    '--rootdir=' + lite,
    '-q',
    '-rA',
    '--junitxml=/test-output/c8.xml',
    case,
]
if args.collect_only:
    command.append('--collect-only')
with (output / 'pytest.log').open('w') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
passed = result.returncode == 0
if args.collect_only:
    passed = passed and '1 test collected' in (output / 'pytest.log').read_text()
else:
    counts = {
        key: sum(
            int(n.attrib.get(key, 0))
            for n in ET.parse(output / 'c8.xml').getroot().iter('testsuite')
        )
        for key in ('tests', 'failures', 'errors', 'skipped')
    }
    passed = passed and counts == dict(tests=1, failures=0, errors=0, skipped=0)
receipt = dict(
    passed=passed,
    collect_only=args.collect_only,
    pytest_exit=result.returncode,
    wandb_url=run.url,
)
(output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
run.log({'passed': int(passed)})
run.finish(exit_code=0 if passed else 1)
print('DS41_C8_RESULT', json.dumps(receipt), flush=True)
raise SystemExit(0 if passed else 1)
