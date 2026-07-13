#!/usr/bin/env bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=00:40:00
#SBATCH --job-name=ds4-pp-export-ab
#
# 8-GPU pp>1 HF-export sensitization (three arms), all on the current commit's
# streaming exporter plus a legacy A/B leg:
#   arm 1  DS4 (deepseek_v4) pp2 export streaming peak MEMLOG
#   arm 2  qwen3.5 pp2 export OLD(all_gather_object) vs NEW(streaming) timing +
#          per-tensor fingerprint equivalence  (speed regression >5% = BLOCKER)
#   arm 3  CPU streaming-equivalence unit regression (no GPU)
#
# Required env (all cw lustre paths):
#   BASE_IMAGE   qualified DS4 container (TE + fla + cudnn/DSA stack)
#   MLITE_NEW    checkout at the streaming HEAD (this branch)
#   MLITE_OLD    checkout at the legacy all_gather_object commit 0bbe4fc37
#   OUTPUT_DIR   fresh result dir
set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:?set BASE_IMAGE to the qualified DS4 container}
MLITE_NEW=${MLITE_NEW:?set MLITE_NEW to the streaming HEAD checkout}
MLITE_OLD=${MLITE_OLD:?set MLITE_OLD to the legacy all_gather_object checkout}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to a fresh result directory}

for p in "${BASE_IMAGE}" "${MLITE_NEW}" "${MLITE_OLD}"; do
  [[ -e "${p}" ]] || { echo "missing required path: ${p}" >&2; exit 2; }
done
mkdir -p "${OUTPUT_DIR}"

# The OLD leg reuses the NEW harness file (it only calls export_hf_weights and
# never depends on streaming-specific symbols), so the A/B differs solely in the
# exporter implementation each checkout carries.
cp "${MLITE_NEW}/experimental/lite/examples/verl/pp_export_sensitize.py" \
   "${MLITE_OLD}/experimental/lite/examples/verl/pp_export_sensitize.py"

run() {  # <mlite_src> -- <cmd...>
  local src=$1; shift
  [[ "${1:-}" == "--" ]] && shift
  srun --mpi=pmix --export=ALL --no-container-mount-home \
    --container-image="${BASE_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${src}/experimental/lite" \
    bash -lc "
      set -euo pipefail
      export PATH=/usr/local/bin:/usr/bin:/bin; hash -r
      unset CONDA_DEFAULT_ENV CONDA_PREFIX PYTHONHOME VIRTUAL_ENV ROCR_VISIBLE_DEVICES || true
      export PYTHONNOUSERSITE=1
      export PYTHONPATH='${src}/experimental/lite':\${PYTHONPATH:-}
      export TMPDIR=/tmp/ppab-\${SLURM_JOB_ID}
      export TORCHINDUCTOR_CACHE_DIR=\${TMPDIR}/ti TRITON_CACHE_DIR=\${TMPDIR}/triton
      export CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
      mkdir -p \${TORCHINDUCTOR_CACHE_DIR} \${TRITON_CACHE_DIR}
      $*
    "
}

# GPU arms fire first (炮响优先); each is non-fatal so one failure never starves
# the others — the final verdict reads whatever JSON landed.
echo "==== arm 1: DS4 pp2 export streaming peak (NEW) ===="
run "${MLITE_NEW}" -- \
  "torchrun --standalone --nproc_per_node=8 -m examples.verl.pp_export_sensitize \
     --model deepseek_v4 --tag new --out '${OUTPUT_DIR}/ds4_new.json'" || echo "ARM1 rc=$?"

echo "==== arm 2a: qwen3.5 pp2 export NEW (streaming) ===="
run "${MLITE_NEW}" -- \
  "torchrun --standalone --nproc_per_node=8 -m examples.verl.pp_export_sensitize \
     --model qwen3_5 --tag new --out '${OUTPUT_DIR}/qwen_new.json'" || echo "ARM2a rc=$?"

echo "==== arm 2b: qwen3.5 pp2 export OLD (all_gather_object) ===="
run "${MLITE_OLD}" -- \
  "torchrun --standalone --nproc_per_node=8 -m examples.verl.pp_export_sensitize \
     --model qwen3_5 --tag old --out '${OUTPUT_DIR}/qwen_old.json'" || echo "ARM2b rc=$?"

# arm 3 last: the pp_export unit tests are CPU-designed (they compare CPU params
# against device-resident export output), so hide CUDA to match the login-host
# CPU run; non-fatal.
echo "==== arm 3: CPU streaming-equivalence unit regression (CUDA hidden) ===="
run "${MLITE_NEW}" -- \
  "CUDA_VISIBLE_DEVICES= python -m pytest -q \
     tests/unit/primitive/test_hf_weights_streaming.py -k pp_export 2>&1 | tail -30" \
  || echo "ARM3 rc=$?"

echo "==== A/B verdict ===="
python3 - "${OUTPUT_DIR}" <<'PY'
import json, sys, os
d = sys.argv[1]
def load(n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else {"status": "missing"}
ds4, new, old = load("ds4_new.json"), load("qwen_new.json"), load("qwen_old.json")
print("DS4  :", json.dumps(ds4))
print("qNEW :", json.dumps(new))
print("qOLD :", json.dumps(old))
if new.get("status") == "ok" and old.get("status") == "ok":
    fp_ok = new["fingerprint"] == old["fingerprint"]
    dpct = 100.0 * (new["wall_s"] - old["wall_s"]) / max(old["wall_s"], 1e-9)
    print(f"FINGERPRINT_MATCH={fp_ok}  WALL_DELTA_PCT={dpct:.1f}  (BLOCKER if >5)")
    print("AB_VERDICT=" + ("PASS" if (fp_ok and dpct <= 5.0) else "FAIL"))
else:
    print("AB_VERDICT=INCONCLUSIVE (a leg skipped/failed — see status/reason)")
PY
echo "DONE ds4-pp-export-ab job=${SLURM_JOB_ID}"
