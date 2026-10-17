#!/bin/bash
#SBATCH --job-name=fusor-matrix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-0           # set to 0-(N-1) for N files in MATRIX_DIR
#SBATCH --output=%x-%A_%a.out
# Run matrix: one WarpX job per case file. Throughput beats multi-node for the default mesh.
#   sbatch --array=0-5 --export=ALL,MATRIX_DIR=cases hpc/m3/array.slurm
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
mapfile -t CASES < <(ls "$FUSOR_REPO/${MATRIX_DIR:-cases}"/*.yaml | sort)
F="${CASES[$SLURM_ARRAY_TASK_ID]}"
export CASE="$(basename "$F" .yaml)"
export APPTAINERENV_FUSORSIM_SIF_SHA256="$(sif_sha warpx)"
srun --mpi=pmix apptainer exec "$(sif warpx)" fusor-warpx --case "$F" --outdir "$FUSOR_RUNS/${CASE}_$(date +%Y%m%d)"
