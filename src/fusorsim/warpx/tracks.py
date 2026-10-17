"""Ion tracks: a few ions followed by particle id at every callback through a short window, written
to tracks.npz. Snapshot diagnostics are ~0.2 us apart in a production run -- longer than an ion lives
and longer than it takes to cross the chamber -- so paths can only come from the callback cadence
(--ncb steps, ~1.5 ns). Pure numpy so it is testable without WarpX; the driver makes the collective calls.

Ids are AMReX `idcpu` words: unique across ranks and unchanged when a particle moves to another rank.
"""
import numpy as np


class Recorder:
    def __init__(self):
        self.ids = np.zeros(0, np.uint64)
        self._id, self._x = [], []

    def set_ids(self, ids):
        self.ids = np.unique(np.asarray(ids, dtype=np.uint64))

    def record(self, t, idcpu, x, y, z, E):
        """Keep the tracked ions among these (one tile's arrays). Returns how many were kept."""
        if not self.ids.size or not len(idcpu):
            return 0
        m = np.isin(np.asarray(idcpu, dtype=np.uint64), self.ids)
        n = int(m.sum())
        if n:
            self._id.append(np.asarray(idcpu, dtype=np.uint64)[m])
            self._x.append(np.column_stack([np.full(n, t), x[m], y[m], z[m], E[m]]))
        return n

    def rows(self):
        """(ids, [t, x, y, z, E] rows) recorded on this rank, for gathering."""
        if not self._id:
            return np.zeros(0, np.uint64), np.zeros((0, 5))
        return np.concatenate(self._id), np.vstack(self._x)


def merge(parts):
    """Gathered per-rank rows -> one table sorted by (id, t): dict of id, t, x, y, z, E (m, s, eV)."""
    ids = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, np.uint64)
    X = np.vstack([p[1] for p in parts]) if parts else np.zeros((0, 5))
    o = np.lexsort((X[:, 0], ids))
    ids, X = ids[o], X[o]
    return dict(id=ids, t=X[:, 0], x=X[:, 1], y=X[:, 2], z=X[:, 3], E=X[:, 4])


def split(d):
    """tracks.npz content -> {id: dict(t, x, y, z, E)} per ion, each in time order."""
    ids = np.asarray(d["id"])
    if not ids.size:
        return {}
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], ids.size]
    return {int(ids[s]): {k: np.asarray(d[k])[s:e] for k in ("t", "x", "y", "z", "E")}
            for s, e in zip(starts, ends)}
