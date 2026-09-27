#!/usr/bin/env bash
# Sequential: GMaster 4096 spin-2 pipeline with XLA prefill, then NaMaster 32c/96c with i64 bin patch.
set -euo pipefail
ROOT=/home/lxu/scratch/agent_dev/auto_research_agent/GMaster
cd "$ROOT"
source /scratch/scratch-lxu/venv/cmbagent_env/bin/activate
OUT="$ROOT/benchmarks/independent"
PY=benchmarks/run_paper_v2_benchmarks.py
CELL=(--mode pipeline --nsides 4096 --spins 2 --repeats 1)

echo "S2_START $(date -Is)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

echo "=== GMaster 4096 spin2 PREALLOCATE=true MEM_FRACTION=0.93 GPU1 skip-ref $(date -Is) ==="
export CUDA_VISIBLE_DEVICES=1
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
export GMASTER_MARCH_V2=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
unset JAX_PLATFORMS
python "$PY" "${CELL[@]}" --skip-ref --output "$OUT/results_gm_noprefill.json"

echo "GM_4096S2_DONE $(date -Is)"

pin_nm() {
  local n="$1" out="$2"
  echo "=== NaMaster ${n}c 4096 spin2 $(date -Is) ==="
  unset JAX_PLATFORMS
  export CUDA_VISIBLE_DEVICES=""
  export JAX_PLATFORMS=cpu
  export OMP_NUM_THREADS="$n"
  export MKL_NUM_THREADS="$n"
  export OPENBLAS_NUM_THREADS="$n"
  export NUMEXPR_NUM_THREADS="$n"
  unset XLA_PYTHON_CLIENT_PREALLOCATE
  taskset -c "0-$((n - 1))" python "$PY" "${CELL[@]}" --skip-gm --output "$out"
  echo "NM${n}_4096S2_DONE $(date -Is)"
}

pin_nm 32 "$OUT/results_nm32.json"
pin_nm 96 "$OUT/results_nm96.json"

python "$OUT/merge_cores.py"
echo "S2_DONE $(date -Is)"
