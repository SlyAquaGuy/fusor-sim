"""MPI helpers for the WarpX driver. Falls back to a one-rank stub when mpi4py is absent,
so `--preflight`, unit tests and plain `python` runs need no MPI.

Design (see docs/mpi.md): WarpX decomposes the grid into boxes owned by ranks. The Python
callback runs on every rank and only ever sees that rank's particles and boundary buffer, so
 * Monte Carlo (CX, ionisation, neutral tracing) is rank-local, with an independent RNG stream;
 * every quantity the controller or the timeseries needs is Allreduce'd, so all ranks hold
   identical `alpha` and identical rows (only rank 0 writes);
 * histograms stay rank-local and are summed to rank 0 when written.
"""
import numpy as np

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
except ImportError:  # serial
    MPI = None
    COMM = None
    RANK, SIZE = 0, 1


def barrier():
    if SIZE > 1:
        COMM.Barrier()


def bcast(obj, root=0):
    return COMM.bcast(obj, root=root) if SIZE > 1 else obj


def allsum(a):
    a = np.asarray(a, dtype=float)
    return COMM.allreduce(a, op=MPI.SUM) if SIZE > 1 else a


def allmax(x):
    return COMM.allreduce(x, op=MPI.MAX) if SIZE > 1 else x


def rank_rng(seed):
    """Independent, reproducible stream per rank: SeedSequence(seed).spawn(SIZE)[RANK].
    Results are reproducible for a fixed (seed, SIZE), not across different SIZE."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(SIZE)[RANK])


def reduce_tree(tree, skip=(), root=0):
    """Sum a dict of ndarrays / dicts of ndarrays onto `root` (other ranks get None).
    Keys in `skip` are taken from the local copy (e.g. a time that is already identical on all ranks).
    Collective: every rank must call it."""
    if SIZE == 1:
        return tree
    out = {}
    for k, v in tree.items():
        if k in skip:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = reduce_tree(v, root=root)
        else:
            out[k] = COMM.reduce(np.asarray(v), op=MPI.SUM, root=root)
    return out
