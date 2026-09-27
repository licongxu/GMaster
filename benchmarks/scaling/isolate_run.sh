#!/bin/bash
# Run one command inside a Docker container on an exclusive cpuset.
# Both hyperthreads of each reserved physical core must be in CPUS.
# Partition "root" keeps load balancing inside the set. "isolated" disables it,
# which leaves every worker on the first CPU.
# Refuses to exec unless the partition reads exactly "root" and a host thread
# is rejected from the first reserved CPU.
# Usage: isolate_run.sh CPUS [--gpus device=0] -- command...
set -euo pipefail
CPUS="$1"
shift
GPU=()
if [ "${1:-}" = "--gpus" ]; then
    GPU=(--gpus "$2")
    shift 2
fi
[ "${1:-}" = "--" ] || { echo "usage: isolate_run.sh CPUS [--gpus spec] -- cmd..." >&2; exit 2; }
shift
NAME="gm-iso-$$"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root, seen at /host$REPO in the container
FIRST="${CPUS%%,*}"
FIRST="${FIRST%%-*}"
cleanup() {
    sg docker -c "docker rm -f $NAME >/dev/null 2>&1" || true
    sg docker -c "docker run --rm --privileged --cgroupns=host -v /sys/fs/cgroup:/hostcg ${BENCH_IMAGE:-ubuntu:24.04} bash -c 'echo > /hostcg/system.slice/cpuset.cpus.exclusive'" || true
}
trap cleanup EXIT
sg docker -c "docker run --rm --privileged --cgroupns=host -v /sys/fs/cgroup:/hostcg ${BENCH_IMAGE:-ubuntu:24.04} bash -c 'echo $CPUS > /hostcg/system.slice/cpuset.cpus.exclusive'"
PRIV=()
if [ ${#GPU[@]} -gt 0 ]; then
    PRIV=(--privileged)
fi
cid=$(sg docker -c "docker run -d --name $NAME ${GPU[*]} ${PRIV[*]} --cpuset-cpus=$CPUS \
    -v /:/host \
    -e OMP_NUM_THREADS -e OMP_PROC_BIND -e OMP_PLACES \
    -e LD_LIBRARY_PATH -e HOME -e MPLCONFIGDIR \
    -e CUDA_VISIBLE_DEVICES -e JAX_ENABLE_X64=1 -e GMASTER_DC -e GMASTER_MARCH_V2 \
    -e PYTHONFAULTHANDLER -e REPS \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e XLA_PYTHON_CLIENT_ALLOCATOR \
    -e ACT_WARM_REPS -e ACT_CACHE -e ACT_OUT -e BENCH_IMAGE \
    -w /host \
    ${BENCH_IMAGE:-ubuntu:24.04} sleep infinity")
sg docker -c "docker run --rm --privileged --cgroupns=host -v /sys/fs/cgroup:/hostcg ${BENCH_IMAGE:-ubuntu:24.04} bash -c 'echo root > /hostcg/system.slice/docker-${cid}.scope/cpuset.cpus.partition'"
part=$(sg docker -c "docker run --rm --privileged --cgroupns=host -v /sys/fs/cgroup:/hostcg ${BENCH_IMAGE:-ubuntu:24.04} bash -c 'cat /hostcg/system.slice/docker-${cid}.scope/cpuset.cpus.partition'")
echo "PARTITION=$part"
[ "$part" = "root" ] || { echo "refusing to run: partition is '$part'" >&2; exit 1; }
if python3 -c "import os,sys; os.sched_setaffinity(0,{int(sys.argv[1])})" "$FIRST" 2>/dev/null; then
    echo "refusing to run: host was allowed onto CPU $FIRST" >&2
    exit 1
fi
echo "HOST_DENIED cpu $FIRST"
if [ ${#GPU[@]} -gt 0 ]; then
    sg docker -c "docker exec $NAME bash -c 'chmod a+rw /dev/nvidia-caps/nvidia-cap1 /dev/nvidia-caps/nvidia-cap2 || true; mount --bind /dev /host/dev; mount --bind /proc /host/proc'"
fi
sg docker -c "docker exec -w /host $NAME chroot --userspec=$(id -u):$(id -g) /host /bin/bash -c 'cd $REPO && exec \"\$@\"' bash $*"
