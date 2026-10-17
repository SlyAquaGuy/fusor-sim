"""Verification of the grid-kick calibration (grid3d): an axisymmetric grid exerts no torque about z,
so test ions keep l_z = 0; the real 7-great-circle grid does twist them. A few minutes of numpy."""
import numpy as np

from fusorsim.physics import grid3d

G = dict(R_c=0.069, d_c=0.002, R_a=0.129, d_a=0.001, V_c=-60e3, R_ch=0.144, H_ch=0.144)


def _rms_dlz(n_circles):
    gf = grid3d.GridField(n_circles=n_circles, seg_c=4e-3, seg_a=6e-3, panel=16e-3, **G)
    X0 = grid3d.birth_positions(40, G["R_c"], G["d_c"], G["R_a"], G["d_a"], np.random.default_rng(5))
    ev, st = grid3d.trace_kicks(gf, X0, max_cross=4)
    assert st["n_events"] > 20
    return np.sqrt(np.mean(ev[:, 3] ** 2))


def test_axisymmetric_grid_conserves_lz_and_real_grid_does_not():
    sym = _rms_dlz(1)            # equator rings only: axisymmetric (up to the wall panels' discreteness)
    real = _rms_dlz(7)           # equator + 6 meridians
    assert real > 20 * sym, (sym, real)
