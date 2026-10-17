"""Serial vs MPI statistical equivalence of the WarpX driver (needs the warpx image + mpirun).
Not run by default:  pytest -m "warpx and mpi" --sif sif/warpx.sif

Method: same case (cases/coarse_test.yaml), serial and 4 ranks. Rank-local RNG streams mean the runs
are NOT bitwise equal; compare steady-state means of timeseries.csv columns (I_ema, alpha,
fus_ion_n_rate, fus_neu_n_rate) within combined Monte-Carlo error (block-averaged), and check that
exactly one timeseries.csv header exists. TODO: implement once the image builds."""
import pytest

pytestmark = [pytest.mark.warpx, pytest.mark.mpi]


@pytest.mark.skip(reason="needs the built warpx.sif; see docstring")
def test_serial_vs_mpi_means_agree():
    ...
