#!/bin/bash
#SBATCH --job-name=fusor-warpx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out
##SBATCH --account=...   --partition=...   (fill in for your M3 allocation)
# Hybrid MPI: srun launches one container per rank, host PMIx wires them up.
#   sbatch --export=ALL,CASE=R1_bound_5mT hpc/m3/warpx.slurm
#   sbatch --nodes=2 ... for larger meshes (--nr 400): see docs/mpi.md on how many ranks the mesh supports
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
CASE="${CASE:?CASE=<name in cases/>}"
OUT="$FUSOR_RUNS/${CASE}_$(date +%Y%m%d)"
export APPTAINERENV_FUSORSIM_SIF_SHA256="$(sif_sha warpx)"
mkdir -p "$FUSOR_RUNS"
srun --mpi=pmix apptainer exec "$(sif warpx)" \
    fusor-warpx --case "$FUSOR_REPO/cases/$CASE.yaml" --outdir "$OUT" ${EXTRA_ARGS:-}
echo "$OUT" > "$FUSOR_RUNS/${CASE}.latest"
