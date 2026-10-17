#!/bin/bash
# Chain the stages with dependencies:  warpx -> analysis -> (openmc | viz | ansys)
#   FUSOR_PROJECT=<id> hpc/m3/pipeline.sh R1_bound_5mT
# Run from a login node. The run dir is fixed up front so every stage sees the same path.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CASE="${1:?usage: pipeline.sh <case>}"
RUN="$FUSOR_RUNS/${CASE}_$(date +%Y%m%d_%H%M%S)"
H="$FUSOR_REPO/hpc/m3"
j1=$(sbatch --parsable --export=ALL,CASE="$CASE",EXTRA_ARGS="--outdir $RUN" "$H/warpx.slurm")
j2=$(sbatch --parsable --dependency=afterok:$j1 --export=ALL,RUN="$RUN" "$H/analysis.slurm")
j3=$(sbatch --parsable --dependency=afterok:$j2 --export=ALL,RUN="$RUN" "$H/openmc.slurm")
j4=$(sbatch --parsable --dependency=afterok:$j2 --export=ALL,RUN="$RUN" "$H/viz.slurm")
echo "run dir: $RUN"; echo "warpx $j1 -> analysis $j2 -> openmc $j3, viz $j4   (ansys: submit hpc/m3/ansys.slurm once the model is defined)"
