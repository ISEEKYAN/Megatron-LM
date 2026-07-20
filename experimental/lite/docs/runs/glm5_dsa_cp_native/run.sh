#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/bin:/usr/bin:/bin"
unset PYTHONHOME CC CXX FC CPP CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache_$$}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_$$}
export PYTHONPATH="$DSA_SITE:$REPO_ROOT/experimental/lite:$CORE_TREE"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 - <<'PY'
import torch
import transformer_engine.pytorch as te
assert torch.cuda.is_available() and hasattr(te, "Linear")
print("GLM5_DSA_CP_PREFLIGHT_OK", torch.__version__, torch.cuda.device_count())
PY

OUT_JSON="$OUTPUT_DIR/glm5_dsa_cp_${SLURM_JOB_ID}.json"
torchrun --nproc_per_node=2 "$WORK_DIR/validate_dsa_cp.py" \
  --seq-lens "${SEQ_LENS:-512,1024,2048}" \
  --warmup "${WARMUP:-2}" \
  --steps "${STEPS:-5}" \
  --output "$OUT_JSON"
echo "GLM5_DSA_CP_ALL_GREEN output=$OUT_JSON"
