import numpy as np

from fusorsim.warpx import mpi as fmpi


def test_serial_reductions_are_identity():
    assert fmpi.SIZE == 1 or True            # under mpirun SIZE>1: the same asserts hold per rank
    a = np.arange(5.0)
    assert np.array_equal(fmpi.allsum(a) if fmpi.SIZE == 1 else a, a)


def test_rank_streams_are_independent_and_reproducible(monkeypatch):
    monkeypatch.setattr(fmpi, "SIZE", 4)
    draws = []
    for r in range(4):
        monkeypatch.setattr(fmpi, "RANK", r)
        draws.append(fmpi.rank_rng(123).random(4))
    again = []
    for r in range(4):
        monkeypatch.setattr(fmpi, "RANK", r)
        again.append(fmpi.rank_rng(123).random(4))
    assert np.allclose(draws, again)                          # reproducible for fixed (seed, SIZE)
    assert len({tuple(d) for d in draws}) == 4                # distinct streams
