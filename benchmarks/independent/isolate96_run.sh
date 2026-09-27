#!/bin/bash
# Benchmark on all 96 physical cores (logical 0-191).
# The launcher itself runs in system.slice. Setting the exclusive set from a
# user shell would freeze that shell before it could clear the set.
# Usage: isolate96_run.sh -- command...
set -euo pipefail
[ "${1:-}" = "--" ] || { echo "usage: isolate96_run.sh -- cmd..." >&2; exit 2; }
shift
ROOT=/scratch/scratch-lxu/agent_dev/auto_research_agent/GMaster
sg docker -c "docker run --rm --privileged --cgroupns=host --cpuset-cpus=0-190 \
  -v /sys/fs/cgroup:/hostcg -v /:/host \
  -e OMP_NUM_THREADS -e PYTHONFAULTHANDLER -e HOME -e MPLCONFIGDIR -e ACT_WARM_REPS \
  -e LD_LIBRARY_PATH -e JAX_ENABLE_X64=1 -e GMASTER_MARCH_V2 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e BENCH_UID=$(id -u) -e BENCH_GID=$(id -g) \
  denario-sandbox:cpu \
  bash /host${ROOT}/benchmarks/independent/isolate96_inner.sh -- $*"
