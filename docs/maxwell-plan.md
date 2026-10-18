# CAD → ANSYS Maxwell → grid kicks → WarpX: fixing the RZ axis line with real 3D geometry

> **Status: planned, not started.** Stage 0 starts once ANSYS Maxwell (Electronics Desktop) is installed on the
> project's Windows PC. Nothing below is implemented or verified yet. The current grid kicks come from the analytic
> solver in `fusorsim.physics.grid3d`.

## Context
2D-RZ gives the grids no torque about z, so every ion crosses the axis. The result is the bright vertical line in the
emission maps. A fix already exists (commits 61ce9f9, 6c8d6e8). [grid3d.py](../src/fusorsim/physics/grid3d.py) solves
the 3D vacuum field of *idealised* great-circle wires with an analytic thin-wire charge-simulation model. It traces
800 test D+ ions, tabulates Δl_z per cathode crossing into `data/grid_kicks_3d.dat`, and the driver applies these
kicks with `--grid-kicks on`. A/B result: axis emission fraction 8.7 → 5.5 %, peak neutrons −0.3 %.

The analytic model can't represent the real assembly (stalk, feedthrough, insulator, wire joints, supports). The stalk
is the largest asymmetry in a real fusor. So the **field stage** becomes CAD-driven:
STEP assembly → Maxwell 3D electrostatic (on the project's Windows PC, via PyAEDT) → exported field → the existing tracer,
plus the run's axisymmetric space charge → kick table → WarpX → normal post-processing.
The analytic `GridField` stays as an independent cross-check of the Maxwell solve.

Only the electric field matters. The self-B of a 50 mA beam is ~10⁻⁷ T, which has no effect on 60 keV ions. So the
Maxwell solve is electrostatic only.

Defaults are not changed. R1 (the certification bound) keeps `--grid-kicks off`, and the kicks only feed the
best-estimate case R2.

## Where things run
ANSYS is a **local pre-compute step only**, on the project's Windows PC. M3 never runs ANSYS. It receives finished data and
runs WarpX and the stages after it.

```
LOCAL (Windows PC, one-off per CAD revision)                        │  M3 (production)
cad/<asm>.step + .map.yaml ─► fusor-maxwell ─► fields/<name>/       │
                               (E grid .npz, STL, field.json, PNGs) │
fields/<name>/ ─► fusor-ee-report ─► ee/report.md (for EEs)         │
fields/<name>/ ─► grid3d --field ─► data/grid_kicks_3d.dat ─────────┼─► fusor-warpx --grid-kicks on (R2)
                                     (small, committed, SHA-256)    │   ─► fusor-post / openmc / viz
```
- The only file that crosses to M3 is the kick table, about 30 rows of text. It is committed in `data/` with its
  SHA-256 and reaches the run through the image (rebuild) or `--grid-kicks-table`. The driver's existing geometry
  check still refuses a mismatched table.
- The large field package (`fields/<name>/`) stays local and is archived for provenance, not committed. The table
  header records its SHA-256 and the CAD's.
- The tracer (Stage 2) is plain numpy and also runs locally, so all pre-data is computed before any M3 job.
- The existing `hpc/m3/ansys.slurm` thermal stub assumes ANSYS on M3. It is out of scope here; it gets a one-line
  "superseded" note in `docs/ansys.md`, to settle when the thermal model is designed.

## Stage 0: spike, before writing much code (½ day, on the Windows PC)
- Check the AEDT version and that `ansys-aedt-core` connects. Confirm that the Monash teaching licence enables
  **Maxwell 3D electrostatic** in batch/non-graphical mode, and find out its mesh or problem-size limits.
- Time a field export at about 1 M points, to size the grid in Stage 1.
- **Risk to confirm with Monash (licence admin):** Ansys *Academic Teaching* licences usually exclude research
  use. Confirm the rolling teaching licence covers project and certification work, or get the right licence for
  that PC. This is recorded as an open item in docs/ansys.md.

## Stage 1: Maxwell field package. New file `src/fusorsim/ansys/maxwell.py`, entry point `fusor-maxwell`
- Runs on the Windows host, not in the container (images carry no ANSYS binaries). `ansys.aedt.core` is imported
  lazily, so the module stays import-safe.
- New pyproject extra: `maxwell = ["ansys-aedt-core==<exact>"]`. Freeze the pin in
  `containers/lock/maxwell.txt` once the first solve works.
- **Geometry sources:**
  - `--cad cad/<asm>.step --map cad/<asm>.map.yaml`: the evolving assembly.
  - `--geometry ideal`: PyAEDT builds the great-circle grids from the case parameters, using the same
    construction as `grid3d.great_circles`. This is used for the cross-solver check and before real CAD exists.
- **Body mapping file** (flat YAML, glob on the body name → role):
  - `conductor` + material + potential (`V_c` / `0` / a number in V), `dielectric` + material/ε_r (e.g. the
    alumina insulator), or `ignore`.
  - Any unmapped body is a **hard error**: no silent defaults for safety evidence.
- **Conventions, checked on import:**
  - origin = grid centre; +z = stalk axis (the RZ symmetry axis).
  - Cathode and anode radii and wire diameters are measured from the bodies and must match the case
    (R_c, d_c, R_a, d_a) within 1 %, otherwise the run is refused. This keeps the RZ driver and the 3D table
    describing the same device.
- **Setup:**
  - Vacuum region equal to the chamber interior (the chamber is a 0 V conductor body, or the region boundary at
    0 V when the chamber is not in the CAD).
  - Voltage excitations from the map; electrostatic solver.
  - Adaptive mesh to an energy-error target, plus a length-based mesh operation on the wires (wire Ø ≥ ~4 elements).
- **Export:**
  - E vector sampled at our own point list (PyAEDT sample-points export).
  - Coarse ~2 mm grid over the chamber, plus fine ~0.5 mm points in shells |r−R_c| < 12 mm and |r−R_a| < 6 mm,
    plus a fine box around every other conductor (stalk). Resolutions are flags.
  - Stored as float32 `.npz` (not committed, like other `*.npz`).
  - Conductor STL via a model export, used for wire-hit clearance.
  - `field.json` metadata: AEDT and PyAEDT versions, CAD and map SHA-256, mesh stats, adaptive convergence,
    measured geometry, centre potential.
  - `--keep-project` keeps the `.aedt` for inspection in the GUI.
- **Visuals:**
  - AEDT field plots exported as PNG: φ and |E| on the equator and meridian planes, and |E| on the cathode wire
    surfaces.
  - The surface field also gives the field-emission / breakdown margin at no extra cost.

## Stage 1b: electrical-engineering diagnostics (arcing / hold-off). New file `src/fusorsim/ansys/ee_report.py`
For the project's electrical engineers, rerun on every CAD revision.

**Extra Maxwell exports** in Stage 1, so the report runs anywhere (numpy/matplotlib) from `fields/<name>/`:
- surface |E| and its normal and tangential parts, sampled on every conductor and dielectric face;
- E inside the dielectrics;
- the capacitance matrix, from a Maxwell matrix setup;
- field lines between conductors at different potentials.

**Output:** `fields/<name>/ee/report.md` plus PNGs. Each criterion gets a value, its limit, the margin (limit /
value), and the voltage at which the margin reaches 1 (fields ∝ V).

| Check | What it flags |
|---|---|
| Peak surface field per conductor, top-N hotspots (body, xyz, \|E\|), 3D surface maps | vacuum breakdown / field-emission sites (cathode wires, joints, stalk tip, screws, weld beads) |
| Fowler–Nordheim emission current at the hotspots, for an assumed enhancement β (sweep) | dark-current and electron-loading risk |
| Field at **triple junctions** (insulator–metal–vacuum), cathode side especially | the usual starting point of surface flashover on the stalk insulator |
| Tangential field along the insulator surfaces | surface flashover |
| Field inside the dielectrics vs dielectric strength | bulk insulator breakdown |
| Minimum gap and mean field V/d for each conductor pair at different potentials | clearance hold-off |
| Paschen check: p·(field-line length) vs the H2/D2 Paschen curve, at 5 mTorr and over a pressure sweep (e.g. gas-puff transients) | long-path gas breakdown (e.g. behind the cathode inside the feedthrough shield), or in air if the external feedthrough is modelled |
| Capacitance matrix and stored energy ½CV² at 60 kV | arc energy, for sizing the supply's series current-limit resistor |

**Mesh honesty:**
- Peak surface E is mesh-sensitive, and it is singular at sharp CAD edges. The report shows its convergence over
  the adaptive passes plus a local-refinement rerun.
- Hotspots that keep growing with refinement are labelled **"geometric singularity: add the real edge radius /
  fillet in CAD"** rather than reported as a number.

**Limits and caveats:**
- All limits live in one cited table (`docs/references.md`). They are conservative: use the lower end of
  published ranges for unconditioned stainless steel and alumina.
- The report states that these are screening indicators, not breakdown predictions.
- A diff against the previous CAD revision's report shows whether a change made things better or worse.

## Stage 2: field map + space charge in the tracer (`src/fusorsim/physics/grid3d.py`)
- **`FieldMap(field_dir)`**
  - Has the same interface as `GridField`: `.field(P) → (E, clearance)`, `.potential(P)`, `.geom`. This lets
    `trace_kicks`, `birth_positions`, `kick_table` and `write_table` be reused **unchanged**.
  - Field: trilinear interpolation of E, fine level first, falling back to coarse.
  - Clearance: `scipy.spatial.cKDTree` distance to densely sampled STL surface points, signed by an inside-solid
    mask. No new dependencies.
- **`SpaceChargeRZ(run_dir)`**
  - φ_sc(r,z) = time-averaged φ from `diags/` − `phi_vacuum.npy`. The driver already writes both; see
    `load_vacuum_phi` in [driver.py](../src/fusorsim/warpx/driver.py).
  - E_sc = −∇φ_sc, mapped to 3D and added to the vacuum field (superposition; for a given ρ, Laplace and Poisson
    are linear).
  - Approximation, stated in a comment: the RZ conductors are rings, not wires. Space charge sits mostly in the
    core, where that difference is small.
  - Cost is negligible. This is the "most realistic without the cost" option. A small 3D PIC is only worth it if
    this term turns out to move the table much.
- **CLI:**
  - `--field fields/<name>` (default stays the analytic solver, for back-compatibility and the cross-check) and
    `--space-charge-run runs/<run>`.
  - Workflow:
    1. Ship the vacuum table first.
    2. Optional refinement, which needs no ANSYS: copy one finished R2 run's `phi_vacuum.npy` and its averaged φ
       (a few MB) back from M3.
    3. Retrace locally, ship the space-charge table, and report how much it changed. If the change is within
       Monte Carlo noise, keep the vacuum table.
- **Table v2:**
  - When the geometry is not z-mirror symmetric (stalk), the cos bins are signed (−1…1, 8 bins).
  - New header fields: `field_source`, `field_sha256`, `cad_sha256`, `space_charge_run`. `GEOM_KEYS` are unchanged,
    so the driver's mismatch check still works.

## Stage 3: driver and core (small changes)
- `core.load_grid_kicks` sets `signed = cos_lo.min() < 0`. `core.sample_grid_kick` takes signed cos when the table
  is signed and |cos| otherwise, so old tables still load.
- The driver kick block passes `z/rs` instead of `|z|/rs` for signed tables. Record `grid_kicks_field_source` /
  `cad_sha256` in `run_params.json`.
- Collective-call rules: the kicks are rank-local numpy only, with no new pywarpx calls.

## Docs, data, provenance
- `docs/ansys.md`:
  - New "Maxwell field stage" section: Windows setup, map-file format, conventions, the licence open item.
  - The thermal stub section is kept.
- `cad/README.md`: each STEP and map file with its SHA-256 and revision.
- `data/README.md`: the new table's provenance and SHA-256.
- `docs/references.md`: Ansys Electronics Desktop/Maxwell (version), PyAEDT, and the superposition argument.
- CLAUDE.md: update the grid-kick rule (regenerate after a CAD change too).
- README "Status": each piece is **not verified** until it has run.

## Verification
1. **Unit tests** (`tests/unit/`, no ANSYS):
   - `FieldMap` built from points sampled from a coarse `GridField` reproduces its E within 1 % away from wires.
   - Clearance on a synthetic sphere STL.
   - The map file refuses unmapped bodies and a geometry mismatch.
   - Signed and unsigned tables both load and sample.
   - `SpaceChargeRZ` on an analytic uniform-sphere ρ.
2. **Cross-solver gate** (Maxwell `--geometry ideal` vs `GridField`), required before trusting CAD runs:
   - centre potential (CSM: −53.05 kV);
   - E along lines through the wire gaps within a few %;
   - kick-table rms Δl_z equal within Monte Carlo noise + 5 %;
   - results written into README status.
3. **Convergence:** Maxwell adaptive passes are converged, and the kick table at 0.5 vs 0.75 mm fine-grid spacing
   agrees.
4. **EE report checks:**
   - On the ideal geometry, compare against analytic results:
     - peak wire-surface field ≈ the thin-wire estimate λ/(2πε₀a) from `GridField`;
     - capacitance of the concentric-sphere limit;
   - unit tests for the margin, Paschen and FN functions on hand-checked numbers.
5. **As-built CAD:**
   - make the table, then run R2 A/B (`RANKS=2 hpc/wsl/run.sh`) and compare with the earlier numbers: axis emission
     fraction, peak neutron rate, hottest cathode latitude;
   - repeat the A/B with the space-charge table;
   - `RANKS=2 hpc/wsl/run.sh coarse_test` with `--grid-kicks on` as an MPI sanity check after the driver change;
   - `pytest tests/unit` + `tests/verification`.
