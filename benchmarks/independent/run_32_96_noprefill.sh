#!/usr/bin/env bash
# Sequential: GMaster on GPU 1 (no XLA prefill), then NaMaster 32c, then NaMaster 96c.
set -euo pipefail
ROOT=/home/lxu/scratch/agent_dev/auto_research_agent/GMaster
cd "$ROOT"
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
OUT="$ROOT/benchmarks/independent"
PY=benchmarks/run_paper_v2_benchmarks.py
COMMON=(--mode all --nsides 64,128,256,512,1024,2048,4096 --sht-nsides 64,128,256,512,1024,2048,4096 --spins 0,2 --repeats 3)

echo "CORES_BENCH_START $(date -Is) nproc=$(nproc)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

echo "=== GMaster GPU1 PREALLOCATE=false $(date -Is) ==="
export CUDA_VISIBLE_DEVICES=1
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GMASTER_MARCH_V2=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
python "$PY" "${COMMON[@]}" --skip-ref --output "$OUT/results_gm_noprefill.json"
echo "GM_DONE $(date -Is)"

pin_nm() {
  local n="$1" out="$2"
  echo "=== NaMaster ${n} cores $(date -Is) ==="
  unset CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=""
  export JAX_PLATFORMS=cpu
  export OMP_NUM_THREADS="$n"
  export MKL_NUM_THREADS="$n"
  export OPENBLAS_NUM_THREADS="$n"
  export NUMEXPR_NUM_THREADS="$n"
  taskset -c "0-$((n - 1))" python "$PY" "${COMMON[@]}" --skip-gm --output "$out"
  echo "NM${n}_DONE $(date -Is)"
}

pin_nm 32 "$OUT/results_nm32.json"
pin_nm 96 "$OUT/results_nm96.json"

python "$OUT/merge_cores.py"
echo "CORES_BENCH_DONE $(date -Is)"
