# Source from every job script:  source "$(dirname "$0")/env.sh"
# Set these once in ~/.bashrc on M3 (or export before sbatch): FUSOR_PROJECT is your /projects/<id> name.
: "${FUSOR_PROJECT:?set FUSOR_PROJECT to your M3 project id (the <id> in /projects/<id>)}"
export FUSOR_REPO="${FUSOR_REPO:-/projects/$FUSOR_PROJECT/fusor-sim}"
export FUSOR_SIF="${FUSOR_SIF:-/projects/$FUSOR_PROJECT/sif}"
export FUSOR_RUNS="${FUSOR_RUNS:-/scratch/$FUSOR_PROJECT/fusor-sim/runs}"
export FUSOR_NUCDATA="${FUSOR_NUCDATA:-/projects/$FUSOR_PROJECT/nucdata}"
export FUSOR_VERSION="${FUSOR_VERSION:-0.1.0}"

# apptainer is installed system-wide on M3 (no module needed)
export APPTAINER_BINDPATH="/projects,/scratch${APPTAINER_BINDPATH:+,$APPTAINER_BINDPATH}"
export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=spread OMP_PLACES=threads

sif() { echo "$FUSOR_SIF/$1-$FUSOR_VERSION.sif"; }
sif_sha() { sha256sum "$(sif "$1")" | cut -d' ' -f1; }
git_desc() { git -C "$FUSOR_REPO" describe --always --dirty --tags 2>/dev/null || echo unknown; }
# provenance recorded in run_params.json by the driver
export APPTAINERENV_FUSORSIM_GIT="$(git_desc)"
