#!/bin/bash
#SBATCH --job-name=fusor-post
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
#   sbatch --export=ALL,RUN=/scratch/<id>/fusor-sim/runs/<run> hpc/m3/analysis.slurm
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
apptainer exec "$(sif analysis)" fusor-post "${RUN:?RUN=<run dir>}"
