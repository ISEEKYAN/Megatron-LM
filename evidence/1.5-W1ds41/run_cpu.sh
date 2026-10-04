#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
repo_root="$PWD"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${W1DS41_DEPS:-/tmp/w1ds41-deps}:${W1DS41_CORE:-/tmp/mrg}:${W1DS41_EMERGING:-/tmp/ds41-f12-emerging-audit}:$repo_root/experimental/lite"
evidence="$repo_root/evidence/1.5-W1ds41"
scratch=$(mktemp -d /tmp/w1ds41-parity.XXXXXX)
export DS41_DEFAULT_SNAPSHOT="$scratch/current.pt"
python -m pytest -c /dev/null --confcutdir=experimental/lite/tests \
  experimental/lite/tests/unit/deepseek_v41/test_w4a8_fp32.py \
  -q --disable-warnings > "$evidence/ds41-cpu.txt"
python -m pytest -c /dev/null --confcutdir=experimental/lite/tests \
  experimental/lite/tests/unit/primitive/quantization/test_w4a8_experts_unit.py \
  experimental/lite/tests/unit/model/test_qwen3_moe_w4a8_unit.py \
  experimental/lite/tests/unit/deepseek_v41/test_redo_bindings.py \
  experimental/lite/tests/unit/deepseek_v41/test_redo_parallel_guards.py \
  -q --disable-warnings > "$evidence/regression-cpu.txt"
mkdir "$scratch/baseline"
git archive 01b252f37 experimental/lite | tar -x -C "$scratch/baseline"
cp experimental/lite/tests/unit/deepseek_v41/{test_w4a8_fp32.py,w4a8_reference.py} \
  "$scratch/baseline/experimental/lite/tests/unit/deepseek_v41/"
export PYTHONPATH="${W1DS41_DEPS:-/tmp/w1ds41-deps}:${W1DS41_CORE:-/tmp/mrg}:${W1DS41_EMERGING:-/tmp/ds41-f12-emerging-audit}:$scratch/baseline/experimental/lite"
cd "$scratch/baseline"
DS41_DEFAULT_SNAPSHOT="$scratch/baseline.pt" python -m pytest -c /dev/null \
  --confcutdir=experimental/lite/tests \
  experimental/lite/tests/unit/deepseek_v41/test_w4a8_fp32.py::test_default_snapshot \
  -q --disable-warnings > "$evidence/baseline-cpu.txt"
python - "$scratch" "$evidence/default-parity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
a = torch.load(root / 'baseline.pt', weights_only=True)
b = torch.load(root / 'current.pt', weights_only=True)
assert a.keys() == b.keys()
assert all(torch.equal(a[k], b[k]) for k in a)
h = hashlib.sha256()
for k in sorted(a):
    h.update(k.encode())
    h.update(a[k].numpy().tobytes())
result = dict(baseline='01b252f37', tensor_count=len(a),
              byte_count=sum(v.numel() for v in a.values()),
              all_tensor_bytes_equal=True, sha256=h.hexdigest(),
              scope='FP32 residual full tiny logits/loss/weights/gradients and BF16 default MoE; no mHC adapter')
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
PY
