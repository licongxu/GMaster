#!/bin/bash
# Privileged system.slice container. Exclusive set is logical 0-190:
# every physical core (0-95), leaving CPU 191 for the kernel. An exclusive
# set of 0-191 is rejected while the root cgroup has tasks.
# Usage: isolate96_inner.sh -- cmd...
set -euo pipefail
[ "${1:-}" = "--" ] || { echo "usage: isolate96_inner.sh -- cmd..." >&2; exit 2; }
shift
cg=$(cut -d: -f3 /proc/1/cgroup)
cg=${cg#/}
partf="/hostcg/${cg}/cpuset.cpus.partition"
cleanup() {
    echo member > "$partf" || true
    echo > /hostcg/system.slice/cpuset.cpus.exclusive || true
}
trap cleanup EXIT
echo 0-190 > /hostcg/system.slice/cpuset.cpus.exclusive
echo root > "$partf"
part=$(cat "$partf")
echo "PARTITION=$part"
test "$part" = root
eff=$(tr -d '[:space:]' < /hostcg/user.slice/cpuset.cpus.effective)
echo "HOST_CPUS=$eff"
test "$eff" = 191
chroot --userspec="${BENCH_UID}:${BENCH_GID}" /host /bin/bash -c 'cd $REPO && exec "$@"' bash "$@"
