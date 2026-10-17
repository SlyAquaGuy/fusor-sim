#!/bin/bash
#SBATCH --job-name=fusor-openmc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=%x-%j.out
# OpenMC dose model. Needs <run>/post from analysis.slurm.
#   sbatch --export=ALL,RUN=<run>,SHIELD="0 10 20 30" hpc/m3/openmc.slurm
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
export APPTAINERENV_OPENMC_CROSS_SECTIONS=/nucdata/endfb-viii.0-hdf5/cross_sections.xml
srun --mpi=pmix apptainer exec --bind "$FUSOR_NUCDATA:/nucdata:ro" "$(sif openmc)" \
    fusor-openmc "${RUN:?RUN=<run dir>}/post" --shield ${SHIELD:-20} ${EXTRA_ARGS:-}
