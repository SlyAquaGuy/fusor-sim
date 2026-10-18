# CLAUDE.md

## Purpose

`fusor-sim` produces the **neutron and X-ray source terms** and the **grid-wire heating** for the Monash
IEC fusor. These results feed its radiation-dose certification, so treat them as safety evidence: they
must be correct, conservative where uncertain, and reproducible.

```
cases/*.yaml ─► WarpX 2D-RZ PIC + MCC + Python physics ─► analysis ─┬─► OpenMC (neutron / X-ray dose)
                                                                     ├─► ANSYS  (wire thermal, stub)
                                                                     └─► viz    (PNG / MP4)
```

There is one Apptainer image per component, and the **same images** run in WSL (development) and on
Monash M3 (production, Slurm + MPI). The layout is in [README.md](README.md); the design notes are in [docs/](docs/).

## Rules: physics and code

- Physical models (cross sections, stopping, thermal, kinematics) live in `src/fusorsim/physics/core.py`
  and are shared by every component. Don't re-implement physics inside `warpx/`, `analysis/` or `openmc/`.
- Use SI units everywhere, except where a flag says otherwise (e.g. `--pressure` in mTorr). Every
  flag's help text gives its unit.
- The driver defaults are the certification **bounding case**: −60 kV, 50 mA, 5 mTorr D2, γ = 1,
  reflection on, `--makeup anode`. Don't change a default silently. Record the reason in the commit and in docs.
- When a modelling choice is uncertain, pick the conservative option (more neutrons, more heating) and say so in a comment.
- Every model or constant from the literature cites its source in [docs/references.md](docs/references.md).
- 2D-RZ has no torque about z, so without help every ion crosses the axis (the RZ axis line). `--grid-kicks on`
  adds the l_z change the real 3D wire grid gives at each cathode crossing, sampled from
  `data/grid_kicks_3d.dat` (made by `python -m fusorsim.physics.grid3d --case <case>`). The table is
  geometry-specific: regenerate it after changing radii, wires, `--great-circles`, voltage or chamber size
  (the driver refuses a mismatch). Kicks lower the bounding outputs slightly (A/B 2026-10-18: peak neutrons
  −0.3 %, hottest cathode latitude 935 → 778 K), so **R1 (`R1_bound_5mT`, kicks off) is the certification
  bound** and **R2 (`R2_best_5mT`, kicks on) is the best estimate** (source shape, pictures).
- Keep the verified conventions:
  - WarpX implicit function: >0 inside solid, <0 in vacuum. The inverted sign fills the cathode.
  - `--nr` must be a multiple of 8 (AMReX blocking factor).
- Match the surrounding code: its naming, its comment density, and comments that explain the *physics
  reasoning*, not the syntax. Python ≥ 3.11 (the images use 3.11).
- Component modules are still run-as-script (`cli._run` executes them as `__main__`). Keep module-level
  code import-safe up to the `--preflight` exit, so preflight works without pywarpx.

## Rules: cases and data

- `cases/*.yaml` are flat `key: value` files, and each key is a driver CLI flag (`t_end` → `--t-end`).
  `cli._expand_case` expands them; explicit CLI flags override the case. `name` and `description` are metadata only.
- `data/` is read-only. A new table needs its provenance and SHA-256 in `data/README.md`.

## Rules: MPI (details in [docs/mpi.md](docs/mpi.md))

- `physics_callback` runs on **every** rank and sees only that rank's particles.
- Monte Carlo is rank-local, with one RNG stream per rank: `SeedSequence(seed).spawn(size)[rank]`.
- Global quantities (interval sums, macro counts, max velocity) are `Allreduce`d before the controller,
  so `alpha` comes out identical on all ranks.
- Only rank 0 writes shared files.
- Collective pywarpx calls must happen the **same number of times on every rank**, never behind a rank-local
  condition: `add_particles` (ends in `Redistribute`; use `inject` + `flush_injections`), field access
  (`PhiFPWrapper(0)[...]`), `get_particle_count()` / `get_particle_boundary_buffer_size()` (default `local=False`).
  Breaking this deadlocks only with ≥2 ranks, so test MPI changes with `RANKS=2 hpc/wsl/run.sh coarse_test`.
- When ranks share a machine, bind each to its own cores (the image sets `OMP_PROC_BIND=spread`).
- Restart needs the same rank count. Results are reproducible for a fixed (seed, ranks) pair, not across rank counts.

## Rules: runs and provenance

- Never commit `runs/`, `diags/`, `*.sif`, `*.npz`. Only the small reference figures in `examples/` are kept.
- Never reuse an `--outdir`. Old diags must never mix with new ones; the driver refuses, so use `--restart`.
- Every run writes `run_params.json` (git describe, image SHA-256, ranks/threads, seed, derived inputs).
  Anything new that affects results must be recorded there too.
- Run outputs besides the tallies: `tracks.npz` (`--track-n` ions every callback over the last `--track-len`)
  and `diags/movie/frame_*.npz` (every ion's packed side-view position + energy, `--movie-frames` evenly over
  the run, ~2.6 GB for R1; 0 = off). Viz turns them into `ion_tracks.png`, `ion_tracks_movie.mp4` and
  `ions_movie.mp4`. Runs made before these existed fall back to the snapshot movie.
- Rebuild the images after committing code changes and before any run you keep, so provenance is clean.

## Rules: containers and HPC

- Python dependencies are pinned in `containers/lock/<component>.txt`. Freeze a loose pin (e.g. `mpi4py`)
  to the exact version once a build has succeeded.
- Build with `containers/build.sh` in WSL/CI (`--fakeroot`). M3 cannot build images, so copy the `.sif` there.
- The OpenMPI in `mpi-base` must be ABI-compatible with the M3 module. Run the checklist in `docs/mpi.md` before changing it.
- On M3, launch with host `srun --mpi=pmix` (hybrid model). In WSL there is no host MPI, so use
  `apptainer exec sif/warpx.sif mpirun ...`.

## Rules: tests and status

- Test tiers: `tests/unit/` (fast, no WarpX), `verification/` (analytic, serial-vs-MPI), `validation/`
  (vs experiment), `regression/`. Mark tests that need the image or mpirun with `@pytest.mark.warpx` / `@pytest.mark.mpi`.
- Keep the "Status" sections in README.md and docs/mpi.md accurate. Anything that hasn't been built or run is
  **not verified**; update them when something is verified for the first time.

## Commands

```bash
# host (WSL), no WarpX: work in a clone on the WSL filesystem (~/fusor-sim), not /mnt/<drive>
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[test]'
pytest tests/unit
fusor-warpx --case cases/coarse_test.yaml --preflight --outdir runs/pf

# images (-> sif/<component>.sif)
containers/build.sh warpx            # builds mpi-base first if missing
containers/build.sh analysis

# runs (WSL): detached, ranks bound to their own cores; layout per machine from cases/bench_<machine>.yaml
RANKS=2 THREADS=4 hpc/wsl/run.sh R1_bound_5mT                      # -> runs/<case>_<time>/ (+ .log, .pid)
RUN_NAME=R2_x FOREGROUND=1 RANKS=2 THREADS=4 hpc/wsl/run.sh R2_best_5mT   # named, blocking (not NAME: WSL sets it)
fusor-monitor --runs runs                                          # live progress page, http://localhost:8050
apptainer run sif/analysis.sif runs/<run>                          # -> runs/<run>/post/report.md
apptainer run sif/viz.sif runs/<run> --movie                       # -> runs/<run>/viz/ (PNGs, both movies)
LIVE_SRC=1 RANKS=2 THREADS=4 hpc/wsl/run.sh coarse_test            # test ./src changes without a rebuild
python -m fusorsim.physics.grid3d --case cases/R1_bound_5mT.yaml   # regenerate data/grid_kicks_3d.dat (~15 min)
```

## Working from Windows (VS Code) on this machine

- The Windows checkout (`D:\Github\MORF\fusor-sim`, a submodule of MORF) is where code is edited and committed;
  runs happen in the WSL clone `~/fusor-sim` (ext4, fast). Bring the clone up to date with
  `git pull --ff-only /mnt/d/Github/MORF/fusor-sim <branch>` (no GitHub credentials needed in WSL).
- `wsl.exe -- bash -c '...'` mangles `$` (variables, `$?`): put multi-step WSL commands in a script file and run
  that. Strip CRLF when copying scripts in (`tr -d '\r'`).
- Long jobs: the PC must stay awake with a terminal attached (WSL stops otherwise), and heavy Windows load
  (games) halves the speed. Layout here: 2 ranks x 4 threads (`cases/bench_5800X3D.yaml`).
