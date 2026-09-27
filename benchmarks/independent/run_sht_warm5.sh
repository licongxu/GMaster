#!/bin/bash
# Transform-only warm times. One process per cell so the memory figure is that cell's.
# GMaster is the v2 march (GMASTER_DC=0). SHTns spin 2 is its spin-1 vector transform.
set -u
PY=/scratch/scratch-lxu/venv/cmbagent_env/bin/python
SP=/scratch/scratch-lxu/venv/cmbagent_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cuda_nvrtc/lib:$SP/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
NS=64,128,256,512,1024,2048,4096,8192
export REPS=5 GMASTER_DC=0
for n in ${NS//,/ }; do
  for sp in 0 2; do
    echo "BEGIN shtns $n $sp"
    "$PY" benchmarks/independent/bench_three_way.py shtns "$n" "$sp"
  done
done
echo done
