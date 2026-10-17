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
- Only rank 0 writes shared files. pywarpx field access is collective, so all ranks must call it.
- Restart needs the same rank count. Results are reproducible for a fixed (seed, ranks) pair, not across rank counts.

## Rules: runs and provenance

- Never commit `runs/`, `diags/`, `*.sif`, `*.npz`. Only the small reference figures in `examples/` are kept.
- Never reuse an `--outdir`. Old diags must never mix with new ones; the driver refuses, so use `--restart`.
- Every run writes `run_params.json` (git describe, image SHA-256, ranks/threads, seed, derived inputs).
  Anything new that affects results must be recorded there too.

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

# runs
OMP_NUM_THREADS=8 apptainer run sif/warpx.sif --case cases/coarse_test.yaml --outdir runs/coarse_test
apptainer run sif/analysis.sif runs/coarse_test                   # -> runs/coarse_test/post/report.md
apptainer exec sif/warpx.sif mpirun -n 2 fusor-warpx --case cases/coarse_test.yaml --outdir runs/ct_np2
```
