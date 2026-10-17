import numpy as np
import pytest

from fusorsim.physics import core as fp
from fusorsim.physics import grid3d


GEOM = dict(R_c=0.069, d_c=0.002, R_a=0.129, d_a=0.001, great_circles=7, V_c=-60e3, R_ch=0.144, H_ch=0.144)


def _coarse_field():
    g = dict(GEOM); nc = g.pop("great_circles")
    return grid3d.GridField(n_circles=nc, seg_c=6e-3, seg_a=8e-3, panel=24e-3, **g)


def test_wire_potentials_reproduced_off_collocation():
    """Thin-wire CSM: potential on the wire surface between collocation points (1/4 along each segment,
    sideways) within the thin-wire accuracy (2 mm wire, coarse 6 mm segments)."""
    gf = _coarse_field()
    P = gf.A + 0.25 * gf.L[:, None] * gf.U
    P = P + gf.a[:, None] * np.cross(gf.U, P / np.linalg.norm(P, axis=1)[:, None])
    err = np.abs(gf.potential(P) - np.where(gf.grid == 0, GEOM["V_c"], 0.0)) / abs(GEOM["V_c"])
    assert np.median(err) < 0.03 and err.max() < 0.06
    assert -57e3 < gf.potential(np.zeros((1, 3)))[0] < -50e3          # centre: below |V_c| (finite grid)


def _write(tmp_path, q_row, geom=GEOM):
    """A small table in the real file format: 2 directions x 2 |cos| bins x 1 energy bin, same quantiles."""
    rows = np.array([[inw, c0, c1, 0.0, 72e3, 100, 0, *q_row]
                     for inw in (1, 0) for c0, c1 in ((0.0, 0.5), (0.5, 1.0))])
    p = tmp_path / "k.dat"
    grid3d.write_table(str(p), rows, geom, {"n_events": 400}, "test")
    return fp.load_grid_kicks(str(p))


def test_zero_table_gives_zero_kicks(tmp_path):
    t = _write(tmp_path, np.zeros(len(grid3d.QUANTILES)))
    rng = np.random.default_rng(0)
    dl = fp.sample_grid_kick(t, rng.random(1000) < 0.5, rng.random(1000), rng.uniform(0, 9e4, 1000), rng)
    assert np.all(dl == 0)


def test_symmetric_table_sign_symmetric_and_in_range(tmp_path):
    q = np.linspace(-1e3, 1e3, len(grid3d.QUANTILES))                 # uniform on [-1e3, 1e3]
    t = _write(tmp_path, q)
    rng = np.random.default_rng(1)
    dl = fp.sample_grid_kick(t, np.ones(20000, bool), np.full(20000, 0.3), np.full(20000, 5e4), rng)
    assert abs(dl.mean()) < 20 and dl.min() >= -1e3 and dl.max() <= 1e3 and dl.std() > 500


def test_kick_turns_velocity_without_changing_energy():
    rng = np.random.default_rng(2)
    n = 500
    th = rng.uniform(0, 2 * np.pi, n); r = rng.uniform(1e-3, 0.1, n)
    # meridional velocities (no azimuthal part), as for every RZ ion without kicks
    vr, vz = rng.normal(0, 2e6, n), rng.normal(0, 2e6, n)
    ux, uy, uz = vr * np.cos(th), vr * np.sin(th), vz
    dl = rng.normal(0, 3e3, n)
    nx, ny, nz = fp.apply_azimuthal_kick(ux, uy, uz, th, r, dl)
    assert np.allclose(nx**2 + ny**2 + nz**2, ux**2 + uy**2 + uz**2, rtol=1e-12)   # energy unchanged
    v = np.sqrt(ux**2 + uy**2 + uz**2)
    ok = np.abs(dl / r) < 0.05 * v                                     # cap not active; renormalisation ~ 0.1 %
    lz = r * (-nx * np.sin(th) + ny * np.cos(th))
    assert np.allclose(lz[ok], dl[ok], rtol=0.01)
    assert np.all(np.abs(lz) <= 0.5 * v * r + 1e-9)                    # capped near the axis


def test_geometry_mismatch_refused(tmp_path):
    t = _write(tmp_path, np.zeros(len(grid3d.QUANTILES)))
    fp.check_grid_kicks_geometry(t, **GEOM)
    bad = dict(GEOM, d_c=0.003)
    with pytest.raises(ValueError, match="d_c"):
        fp.check_grid_kicks_geometry(t, **bad)
