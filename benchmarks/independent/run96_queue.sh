#!/bin/bash
# 96-core NaMaster warm ladder. Each nside prints five times.
PY=/scratch/scratch-lxu/venv/cmbagent_env/bin/python
NS=64,128,256,512,1024,2048,4096,8192
"$PY" benchmarks/independent/bench_act_warm.py "$NS" 0 namaster 96
"$PY" benchmarks/independent/bench_act_warm.py "$NS" 2 namaster 96
