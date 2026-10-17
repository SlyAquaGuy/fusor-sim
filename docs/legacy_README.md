# fusor_v6 — source terms for radiation certification

WarpX 26.01 (2D-RZ electrostatic PIC) + reduced physics callbacks that produce
the **neutron and X-ray source terms for OpenMC** and the **grid wire
temperatures over a 15 s shot** for the Monash IEC fusor.

| file | what it is |
|---|---|
| `fusor_v6.py` | the WarpX run. `--preflight` checks everything in seconds without WarpX. |
| `viz_v6.py` | pictures and animations of a run: potential, ion density, neutron emission, side-view "camera" images, phase space, spectra, ion movie and tracks |
| `postprocess_v6.py` | steady-state checks, 15 s totals, wire temperatures, OpenMC source files |
| `fusor_physics.py` | cross sections, stopping, thermal and kinematics models (`python fusor_physics.py` self-tests) |
| `data/` | Tabata & Shirai (2000) H+/H0 on H2 tables (charge exchange, stripping, momentum transfer) |

Quick start (inside `conda activate warpx`):

```bash
python fusor_v6.py --preflight --chamber-half-length 0.144      # set the REAL half-length
python fusor_v6.py --t-end 0.2e-6 --t-avg 0.1e-6 --outdir timing_test   # how fast is this PC?
nohup python fusor_v6.py --outdir run_R1 > run_R1.log 2>&1 &     # production (bounding case)
python postprocess_v6.py run_R1                                  # -> run_R1/post/report.md
python viz_v6.py run_R1                                          # -> run_R1/viz/*.png
```

Defaults are the certification bounding case: 60 kV, 50 mA, 5 mTorr, SEE yield
gamma = 1 (most of the current carried by ions), reflection on, and any ionisation the
model does not capture placed at the anode (where it gives the most fusion). See the
methods document for the reasoning behind every model and the run matrix.

Data provenance: `data/TS2000_R6_*` and `data/TS2000_R31_*` are copies of
BLAST-WarpX/warpx-data `MCC_cross_sections/H/Hion_on_H2_charge_exchange.dat` and
`H_on_H2_ionization.dat` (Tabata & Shirai, At. Data Nucl. Data Tables 76 (2000) 1,
reactions 6 and 31, energies converted to the H+H2 centre-of-mass frame).
`data/TS2000_R10_*` is reaction 10 (momentum transfer) digitised in v4.
