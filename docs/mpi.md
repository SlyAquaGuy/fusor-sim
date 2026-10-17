# MPI design and M3 checklist

## How the WarpX driver is parallelised

WarpX splits the mesh into boxes owned by ranks. The Python callback (`physics_callback` in
`src/fusorsim/warpx/driver.py`) runs on **every** rank and sees only that rank's ions and boundary buffer.

| quantity | treatment |
|---|---|
| Monte Carlo (CX, ionisation, neutral tracing, reflections) | rank-local; independent RNG stream per rank (`SeedSequence(seed).spawn(size)[rank]`) |
| births from local ions / local scraped particles, fast-ion re-injection | rank-local, injected with `add_particles(unique_particles=True)` |
| bootstrap source (a *global* rate) | each rank draws `Poisson(rate/size)` |
| interval sums (`acc`), `n_macro`, `injected_real`, `phys_real`, max velocity | `Allreduce` before the controller (step 4b) |
| controller (`alpha`), time-series rows | computed from the reduced sums, so identical on all ranks; only rank 0 writes `timeseries.csv` |
| histograms `H` | rank-local; summed to rank 0 when `tallies.npz` is written |
| restart state | one file per rank (`python_state_<step>_r<rank>.npz`) — **restart needs the same rank count** |
| vacuum potential map | all ranks call the (collective) pywarpx field access; rank 0 saves, result broadcast |

Reproducibility holds for a fixed (seed, number of ranks), not across rank counts. Serial vs MPI agree
statistically, not bitwise (`tests/verification/test_mpi_equivalence.py`).

## How many ranks are useful

Default mesh 200x400 with `--max-grid-size 64` gives ~28 boxes, so more than ~28 ranks cannot all hold work;
use `--max-grid-size 32` (~100 boxes) for more. At this size a single node is usually fastest; multi-node pays
off for `--nr 400`. For the run matrix prefer `hpc/m3/array.slurm` (one job per case).

## Before the first build — check M3 (login node)

```bash
module avail openmpi
ompi_info | grep -E "Open MPI:|pmix|ucx"
srun --mpi=list              # is pmix / pmix_v4 listed? if not, use --mpi=pmi2 and build OpenMPI --with-pmi
apptainer --version
```

Then set `OMPI_VERSION` (and PMIx flavour) to match: `OMPI_VERSION=x.y.z containers/build.sh mpi-base`.
The image uses OpenMPI's *internal* PMIx; if the host `srun` speaks a different PMIx major version, switch
`mpi-base.def` to `--with-pmix=external` against a matching PMIx or use PMI2.

## Status

* Done and verified: serial path through the package; `--case`; unit tests; `mpi-base` (OpenMPI 4.1.6),
  `warpx` and `analysis` images built in WSL; serial `coarse_test` run in the warpx image.
* Written, **not built or run**: `openmc`, `viz`, `ansys` `.def` files; Slurm scripts.
* MPI in WSL (2026-10-17, `hpc/wsl/run.sh`, 2/4/8 ranks on `coarse_test`): one `timeseries.csv` header,
  monotonic steps, end state (n_macro, current, neutron rates) equal to the 1-rank run within Monte Carlo noise.
  Two bugs found on the way, both fixed:
  1. **deadlock**: `add_particles` ends in a collective `Redistribute`, but the driver skipped it on ranks with
     no births and called it once per (rank-local) fast-ion batch. Now one call per callback on every rank
     (`flush_injections`).
  2. **75x slower Poisson solve**: unbound ranks each applied the image's `OMP_PROC_BIND=spread` over the same
     CPUs. Ranks are now bound to disjoint cores (`--map-by slot:PE=<threads> --bind-to core`).
* Still to check on M3 / multi-node (items from the first MPI plan, 1 and 4 not yet tested directly):
  1. pywarpx field access (`PhiFPWrapper(0)[...]`) returns the **global** array on every rank (else the vacuum map is wrong);
  2. `add_particles(..., unique_particles=True)` really keeps each rank's particles (particle count conserved);
  3. exactly one `timeseries.csv` header and monotonic `step` column;
  4. `alpha` identical across ranks (print it per rank once).
* `ansys` is a stub (`docs/ansys.md`).
