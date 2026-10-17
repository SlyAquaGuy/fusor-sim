# Coding rules and practices

These are the rules the fusor-sim code follows, with the published practice each one comes from. Keys in
brackets point to [sim-import-library.bib](sim-import-library.bib). The reasons are expanded in
[references.md](references.md).

## 1. Repository layout

| rule | where | source |
|---|---|---|
| Code, data, configuration and results live in separate directories: `src/`, `data/`, `cases/`, `runs/` | repo root | [noble2009quick], [wilson2017good] |
| Raw input data is read-only. It ships with its provenance and SHA-256 sums, and nothing edits it in place | `data/README.md` | [wilson2017good], [sandve2013ten] |
| Results are never committed. `runs/`, `diags/`, `*.npz`, `*.log` and `*.sif` are git-ignored; on M3, runs go to `/scratch` | `.gitignore` | [noble2009quick] |
| One scenario is one YAML file in `cases/`. Its keys are the driver's CLI flags, and a flag given on the command line overrides the YAML value | `cases/`, `cli.py` | [wilson2017good] |

## 2. Code structure

| rule | where | source |
|---|---|---|
| There is a single installable package, `fusorsim` (src layout, PEP 621), and no `sys.path` hacks | `pyproject.toml` | [wilson2014best] |
| The physics models are written once, in `fusorsim.physics.core`. WarpX, analysis, viz and OpenMC import them and never copy them | `src/fusorsim/physics/` | [wilson2014best] ("don't repeat yourself") |
| Data files are found through `FUSORSIM_DATA`, then `<repo>/data`, then `./data`, so the same code runs in a checkout and in a container | `physics/core.py:_find_data_dir` | [taschuk2017ten] |
| Every component has a console entry point: `fusor-warpx`, `fusor-post`, `fusor-openmc`, `fusor-viz` | `cli.py` | [taschuk2017ten] |
| Optional dependencies (mpi4py, matplotlib, pyyaml) degrade gracefully: a serial MPI stub, skipped plots, a flat YAML parser | `warpx/mpi.py`, `cli.py` | [taschuk2017ten] |

## 3. Physics-code conventions (from `physics/core.py`)

- **Every literature number is cited where it is used.** The citation gives authors, journal, volume, page
  and year, and also the reaction number or fit parameters taken from the source.
- **Validity and accuracy are stated.** Each fit records its energy range and quoted uncertainty, for
  example Bosch–Hale at ~2 %, Rudd at ±20 % and Kramers at ±30 %.
- **Approximations lean conservative, and are labelled.** Where a model has to approximate, it chooses the
  side that over-predicts the hazard and says so ("conservative"). Examples:
  - nuclear stopping is omitted, which lengthens the range and so raises the thick-target yield;
  - the bremsstrahlung spectrum is unfiltered;
  - the emissivity floor is low.
- **Uncertain surface parameters are scenario inputs, not constants.** These are the secondary-electron
  yield γ, reflection coefficients and D saturation fractions. Each bound case (neutron, X-ray,
  wire-heating) sets them in the direction that makes its own output an upper bound.
- **Isotope scaling is explicit.** Electronic cross sections are scaled at equal velocity. Elastic
  cross sections are mapped at equal centre-of-mass energy (Langevin regime) [barnett1990atomic].
- **Units are SI internally.** Conversions happen only at the interfaces (OpenMC uses cm and eV) and are
  written in the function name or docstring.
- **Constants come from CODATA 2018** [tiesinga2021codata].
- **Implementations are cross-checked against independent ones.** The Bosch–Hale coefficients were
  compared with WarpX's own implementation, and the ICRU 49 coefficients were taken from Geant4.

## 4. Testing: verification and validation hierarchy

| level | content | source |
|---|---|---|
| unit | the physics self-test checks values against published numbers (e.g. D(d,n) at 50 keV ~16–17 mb, BEB H₂ peak, ICRU 49 stopping); gas density; data-table integrity | [wilson2014best], [taschuk2017ten] |
| verification | analytic or manufactured solutions (vacuum Laplace φ against concentric spheres); serial vs MPI agree statistically | [roache2002code], [oberkampf2010verification] |
| benchmark | PIC-MCC checked against community benchmarks | [turner2013simulation] |
| validation | comparison against measured fusor data, kept separate from verification | [oberkampf2010verification] |
| regression | the coarse reference run, with tolerances | [taschuk2017ten] |

## 5. Reproducibility and provenance

- **Each run writes `run_params.json`.** It records the CLI and case inputs, all derived quantities, the
  seed, `git describe --dirty`, the image SHA-256, and the rank and thread counts [sandve2013ten].
- **Results reproduce for a fixed (seed, number of ranks).** Each rank's RNG stream comes from
  `SeedSequence(seed).spawn(size)[rank]`. Results across different rank counts agree statistically, not
  bitwise.
- **Restart files are per rank,** so a restart must use the same rank count.

## 6. Containers

| rule | source |
|---|---|
| There is one Apptainer image per component: mpi-base, warpx, openmc, analysis, viz, ansys | [kurtzer2017singularity] |
| `.def` recipes live in git with exact version pins (`containers/lock/`). Images are named `<component>-<version>.sif` and never `latest`. Each image's SHA-256 is recorded | [nust2020ten] |
| Large data is bind-mounted, not baked into images; for example nuclear data at `/nucdata` [brown2018endf] | [nust2020ten] |
| Images are built in WSL or CI with `--fakeroot` and copied to M3, which cannot build them | [monash_m3_containers] |
| For development, `--bind $PWD/src` runs live code against frozen dependencies; a release bakes the package in | [nust2020ten] |
| Licensed solvers (ANSYS) stay on the host module; the image holds only the Python pre/post tooling | [monash_m3_ansys] |

## 7. MPI

- **Hybrid model.** OpenMPI inside the image must be ABI-matched to the host, which launches the job with
  `srun --mpi=pmix` [apptainer_mpi].
- **Monte Carlo is rank-local,** using independent RNG streams.
- **Global quantities are reduced with `Allreduce` before the controller runs,** so the controller's
  `alpha` is identical on every rank.
- **Histograms are summed to rank 0 at write time.**
- **Only rank 0 writes shared files.**
- **Global source rates are split per rank** as `Poisson(rate/size)`.
- **Throughput.** For the run matrix, use a job array of single-node jobs. Multi-node runs are only worth
  it for refined meshes and OpenMC.

## 8. Release and citation

- `CITATION.cff` cites this repo and the codes and data it drives: WarpX [fedeli2022pushing], OpenMC
  [romano2015openmc] and Tabata & Shirai [tabata2000analytic].
- Releases are tagged and archived (Zenodo or Monash Bridges) for a DOI [barker2022introducing].

## 9. Citation errors found while building the .bib (to fix in the code)

| location | as written | correct |
|---|---|---|
| `physics/core.py` (W emissivity) | Matsumoto et al., Int. J. Thermophys. **20 (1999) 1539** | 20(3) (1999) **943** [matsumoto1999hemispherical]; 18 (1997) 1539 is their tantalum paper |
| `physics/core.py` (thoriated W) | Dushman & Ewald **1923** | Phys. Rev. 29 (**1927**) 857 [dushman1927electron] |
| `analysis/postprocess.py` (`--f-wall`) | Nanoscale Res. Lett. 11 (2016) **1251** | article **44**, Morozov et al. [morozov2016structural] |
| `physics/core.py` (γ) | "Thomas 1985" | not identified; there is no .bib entry, so add the full reference |
| `physics/core.py` (Straub 1996) | Straub et al. (1996), incomplete | Phys. Rev. A 54, 2146 [straub1996absolute] |

Entries whose `note` says "not checked online" (Barnett ORNL-6086, Eckstein IPP 9/132, Compton & Allison,
the CRC edition, the NIST composition tables) still need checking before they are cited in a paper.
