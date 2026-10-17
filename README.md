# fusor-sim

Source terms and radiation dose for the Monash IEC fusor:

```
cases/*.yaml ──► WarpX (2D-RZ PIC + MCC + Python physics) ──► analysis ──┬─► OpenMC  (neutron / X-ray dose)
                 containers/warpx.def   [MPI]              analysis.def  ├─► ANSYS   (grid wire thermal, stub)
                                                                          └─► viz     (PNG / MP4)
```

One Apptainer image per component, the same images in WSL (development) and on Monash M3 (production,
Slurm + MPI). Layout, conventions and the references behind them: [docs/](docs/).

| path | what |
|---|---|
| `src/fusorsim/` | the package: `physics/` (models, shared by all components), `warpx/`, `analysis/`, `openmc/`, `viz/`, `ansys/` |
| `cases/` | one YAML per scenario (the run matrix); keys are the driver's CLI flags |
| `data/` | read-only cross-section tables, with provenance and checksums |
| `containers/` | `.def` files, pinned `lock/` requirements, `build.sh` |
| `hpc/m3/` | Slurm scripts + `pipeline.sh` |
| `tests/` | `unit/`, `verification/` (analytic, serial-vs-MPI), `validation/` (vs experiment), `regression/` |
| `runs/` | results; never committed (on M3 this is on `/scratch`) |

## Develop (WSL)

```bash
pip install -e '.[test]'                       # or: PYTHONPATH=src
pytest tests/unit
fusor-warpx --case cases/coarse_test.yaml --preflight --outdir runs/pf    # no WarpX needed
```

Build and use the images (needs `apptainer`; `sudo apt install apptainer` / see apptainer.org):

```bash
containers/build.sh warpx                      # -> sif/warpx-<ver>.sif (+ mpi-base)
apptainer exec --bind $PWD/src:/opt/fusor-sim/src sif/warpx.sif \
    python3 -c "import fusorsim"               # live-edit code against frozen dependencies
mpirun -n 4 apptainer exec sif/warpx.sif fusor-warpx --case cases/coarse_test.yaml --outdir runs/t4
```

## Deploy (M3)

M3 cannot build images (no root); build in WSL/CI and copy:

```bash
rsync -av sif/*-0.1.0.sif m3:/projects/$FUSOR_PROJECT/sif/        # ssh host alias `m3`
ssh m3 'cd /projects/$FUSOR_PROJECT/fusor-sim && git pull'
ssh m3 'FUSOR_PROJECT=<id> /projects/<id>/fusor-sim/hpc/m3/pipeline.sh R1_bound_5mT'
```

MPI uses the *hybrid* model (OpenMPI in the image, launched by the host `srun --mpi=pmix`); the image's
OpenMPI must match M3's — see [docs/mpi.md](docs/mpi.md) before the first build.

## Provenance

Every run's `run_params.json` records the git description, the image SHA-256, the rank/thread counts, the
seed and all derived inputs. Results are reproducible for a fixed (seed, number of ranks).

## Status

Verified: package imports, physics self-test, preflight via `--case`, unit tests.
**Not yet verified:** container builds, MPI runs (the driver's MPI path has only been exercised with one rank),
M3 launch. See [docs/mpi.md](docs/mpi.md#status).
