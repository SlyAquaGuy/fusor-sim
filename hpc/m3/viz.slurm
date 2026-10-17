#!/bin/bash
#SBATCH --job-name=fusor-viz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
apptainer exec "$(sif viz)" fusor-viz "${RUN:?RUN=<run dir>}"
