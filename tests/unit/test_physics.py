"""The built-in physics self-test (formerly `python fusor_physics.py`) as pytest, plus a few
independent invariants so a change to a table can't silently pass."""
import numpy as np
import pytest

from fusorsim.physics import core as fp


def test_builtin_selftest(capsys):
    assert bool(fp._selftest()), capsys.readouterr().out


def test_gas_density_ideal_gas():
    p, T = 5.0, 300.0                       # mTorr, K
    n = fp.gas_density(p, T)
    assert n == pytest.approx(p * fp.MTORR_TO_PA / (fp.kB * T), rel=1e-12)


def test_data_tables_present():
    import os
    for f in ("TS2000_R6_Hp_H2_charge_exchange_Ecm.dat", "TS2000_R10_Hp_H2_momentum_transfer_Ecm.dat",
              "TS2000_R31_H_H2_stripping_Ecm.dat"):
        assert os.path.exists(os.path.join(fp.DATA_DIR, f)), f


def test_elastic_table_is_uniform_grid(tmp_path):
    out = tmp_path / "xs.dat"
    fp.write_elastic_table_for_warpx(str(out))
    x = np.loadtxt(out)
    assert np.allclose(np.diff(x[:, 0]), x[1, 0] - x[0, 0], rtol=1e-6)   # WarpX MCC requirement
