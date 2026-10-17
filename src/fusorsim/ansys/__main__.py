"""ANSYS pre/post (stub). The solver is the host `ansys/25r1` module; this package will
  prepare <run>   : per-wire heat loads from <run>/post (wire power, impact maps) -> <run>/ansys/{grid_thermal.inp,*.csv}
  collect <run>   : read MAPDL results back into <run>/ansys/summary.json
The thermal model itself (geometry, materials, radiation BCs) is not defined yet."""
import sys

sys.exit("fusorsim.ansys is a stub: define the grid thermal model first (see docs/ansys.md).")
