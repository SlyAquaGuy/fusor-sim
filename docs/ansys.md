# ANSYS component (not yet implemented)

Assumed role: transient thermal FEA of the cathode/anode grid wires, driven by the per-wire heat
loads from `fusor-post`, as a cross-check of the reduced wire-temperature model in
`src/fusorsim/analysis/postprocess.py`. Solver = host module `ansys/25r1` (licence: request access via
mcc-help@monash.edu); the `ansys` container only prepares inputs and reads results.
Open questions: geometry source (`cathode_anode_grids.stl` from the old WSL tree?), radiation BCs, 15 s shot profile.

Planned (not started): a local ANSYS Maxwell field stage (CAD → 3D electrostatic field → grid kicks, plus arcing /
hold-off diagnostics for the electrical engineers). ANSYS runs only on the project's Windows PC as a pre-compute
step; M3 never runs ANSYS. See [maxwell-plan.md](maxwell-plan.md).
