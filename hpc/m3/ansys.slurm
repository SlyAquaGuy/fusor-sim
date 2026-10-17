#!/bin/bash
#SBATCH --job-name=fusor-ansys
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
# Host ANSYS (licensed; request access via mcc-help@monash.edu) + container pre/post. Single node:
# M3 docs say Fluent/MAPDL MPI does not understand Slurm across nodes.
set -euo pipefail
source "$FUSOR_REPO/hpc/m3/env.sh"
RUN="${RUN:?RUN=<run dir>}"
apptainer exec "$(sif ansys)" python3 -m fusorsim.ansys prepare "$RUN"       # -> $RUN/ansys/*.inp, *.csv
module load ansys/25r1
( cd "$RUN/ansys" && mapdl -b -np "$SLURM_CPUS_PER_TASK" -i grid_thermal.inp -o grid_thermal.out )
apptainer exec "$(sif ansys)" python3 -m fusorsim.ansys collect "$RUN"
