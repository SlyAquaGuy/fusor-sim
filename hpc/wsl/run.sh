#!/usr/bin/env bash
# Launch one case on this machine (WSL) in the warpx image, detached so it keeps running after the
# command returns. Watch it with fusor-monitor (http://localhost:8050) or `tail -f runs/<name>.log`.
#   hpc/wsl/run.sh R1_bound_5mT                        # default: 1 rank x half the logical CPUs
#   RANKS=2 THREADS=4 hpc/wsl/run.sh R1_bound_5mT --nr 400   # extra args go to fusor-warpx (CLI wins over case)
#   FOREGROUND=1 RUN_NAME=bench_4x2 hpc/wsl/run.sh coarse_test   # block until done (benchmarks, scripts)
#   LIVE_SRC=1 hpc/wsl/run.sh coarse_test                  # use ./src instead of the image's copy (dev only:
#                                                          # provenance then is the checkout, rebuild for real runs)
# Output: runs/<RUN_NAME>/ plus runs/<RUN_NAME>.log and runs/<RUN_NAME>.pid (default RUN_NAME = <case>_<YYYYmmdd_HHMMSS>).
# (Not NAME: WSL exports NAME=<hostname>, which silently named every run after the PC.)
# Stop:   kill -TERM -"$(cat runs/<RUN_NAME>.pid)"           # the whole process group (mpirun + ranks)
# Ranks x threads: MPI ranks split the mesh AND the Python physics callback (serial within a rank), so
# more ranks usually beat more threads; the mesh has ~28 boxes at --max-grid-size 64 (docs/mpi.md).
# The best split is machine-specific: time it with a cases/bench_<machine>.yaml case and set RANKS/THREADS.
set -euo pipefail
cd "$(dirname "$0")/../.."
CASE="${1:?usage: hpc/wsl/run.sh <case in cases/> [extra fusor-warpx flags]}"; shift
[ -f "cases/$CASE.yaml" ] || { echo "no cases/$CASE.yaml" >&2; exit 1; }
RANKS="${RANKS:-1}"
THREADS="${THREADS:-$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 ))}"
SIF=sif/warpx.sif
[ -e "$SIF" ] || { echo "no $SIF: run containers/build.sh warpx" >&2; exit 1; }
RUN_NAME="${RUN_NAME:-${CASE}_$(date +%Y%m%d_%H%M%S)}"
OUT="runs/$RUN_NAME" LOG="runs/$RUN_NAME.log"
[ -e "$OUT/timeseries.csv" ] && { echo "$OUT already holds a run: pick another RUN_NAME" >&2; exit 1; }
mkdir -p runs

export APPTAINERENV_FUSORSIM_SIF_SHA256="$(cut -d' ' -f1 "$(readlink -f "$SIF").sha256")"
export APPTAINERENV_OMP_NUM_THREADS="$THREADS"
bind=()
if [ -n "${LIVE_SRC:-}" ]; then   # the image installs fusorsim into site-packages, so prepend ./src
  bind=(--bind "$PWD/src:/opt/live-src") APPTAINERENV_PYTHONPATH=/opt/live-src
  export APPTAINERENV_PYTHONPATH APPTAINERENV_FUSORSIM_GIT="$(git describe --always --dirty --tags)-live"
fi
if [ "$RANKS" -gt 1 ]; then
  # Everything inside one container (WSL has no host MPI). Each rank gets its OWN THREADS cores:
  # the image sets OMP_PROC_BIND=spread, and unbound ranks each "spread" over the same CPUs, which
  # stacks their OpenMP threads and made the Poisson solve ~75x slower with 2 ranks.
  cmd=(apptainer exec "${bind[@]}" "$SIF" mpirun -n "$RANKS" --map-by "slot:PE=$THREADS" --bind-to core
       --report-bindings -x OMP_NUM_THREADS fusor-warpx)
else
  cmd=(apptainer exec "${bind[@]}" "$SIF" fusor-warpx)
fi
cmd+=(--case "cases/$CASE.yaml" --outdir "$OUT" "$@")
echo "$RUN_NAME: $RANKS rank(s) x $THREADS thread(s) -> $OUT  (log $LOG)"
if [ -n "${FOREGROUND:-}" ]; then
  "${cmd[@]}" > "$LOG" 2>&1
else
  setsid nohup "${cmd[@]}" > "$LOG" 2>&1 < /dev/null &
  echo $! > "runs/$RUN_NAME.pid"
  echo "started, pid $(cat "runs/$RUN_NAME.pid").  Monitor: fusor-monitor --runs runs  ->  http://localhost:8050"
fi
