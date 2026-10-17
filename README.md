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

Clone into the WSL filesystem (`~/fusor-sim`), not `/mnt/<drive>`: builds there are slow and symlinks/permissions misbehave.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[test]'                       # or: PYTHONPATH=src
pytest tests/unit
fusor-warpx --case cases/coarse_test.yaml --preflight --outdir runs/pf    # no WarpX needed
```

Build and use the images (Ubuntu: `sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt install -y apptainer`):

```bash
containers/build.sh warpx                      # -> sif/warpx-<ver>.sif (+ mpi-base)
containers/build.sh analysis
RANKS=2 THREADS=4 hpc/wsl/run.sh R1_bound_5mT  # detached; -> runs/R1_bound_5mT_<time>/ + .log + .pid
fusor-monitor --runs runs                      # live progress page: http://localhost:8050 (stdlib only)
apptainer run sif/analysis.sif runs/<run>     # -> runs/<run>/post/report.md
apptainer run sif/viz.sif runs/<run> --movie  # -> runs/<run>/viz/*.png + ions_movie.mp4
LIVE_SRC=1 hpc/wsl/run.sh coarse_test          # dev: run ./src against the image's frozen dependencies
```

`hpc/wsl/run.sh` runs MPI inside the image (WSL has no host MPI; on M3 the host `srun` launches the image) and
gives each rank its own cores. The best ranks x threads split is machine-specific: time it with a
`cases/bench_<machine>.yaml` case (see `cases/bench_5800X3D.yaml`).

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

Verified: package imports, physics self-test, preflight via `--case`, unit tests; `mpi-base`, `warpx`,
`analysis` and `viz` images built in WSL (2026-10-17); serial `coarse_test` run in the warpx image (1 rank x 16
threads, 263 s), post-processed in the analysis image, rendered (incl. `--movie`) in the viz image.
MPI in WSL: `coarse_test` with 2, 4 and 8 ranks agrees with 1 rank within Monte Carlo noise.
3D grid kicks (`--grid-kicks`, branch `grid-kicks`): field model and tracer verified (wire potentials within
~3 %, no torque from an axisymmetric grid); **not yet calibrated** (`data/grid_kicks_3d.dat` to be generated)
or run in WarpX, so the default stays `off`.
**Not yet verified:** `openmc`, `ansys` image builds, multi-node MPI, M3 launch.
See [docs/mpi.md](docs/mpi.md#status).
