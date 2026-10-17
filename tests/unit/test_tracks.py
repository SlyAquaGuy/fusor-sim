import numpy as np

from fusorsim.warpx import tracks


def test_tracks_survive_rank_changes_and_keep_full_ids():
    big = np.uint64(0x8000_0123_4567_0001)          # valid-flag bit set: > 2**53, must not pass through float
    ids = np.array([big, big + np.uint64(1), np.uint64(7)], dtype=np.uint64)
    r0, r1 = tracks.Recorder(), tracks.Recorder()
    for r in (r0, r1):
        r.set_ids(ids[:2])                           # every rank knows the whole tracked set
    one = np.ones(1)
    # ion `big` is on rank 0 at t=1, moves to rank 1 at t=2 (and back at t=3); id 7 is never tracked
    assert r0.record(1.0, np.array([big, np.uint64(7)]), 1 + 0 * np.ones(2), np.zeros(2), np.zeros(2), np.full(2, 5e4)) == 1
    r1.record(2.0, np.array([big]), 2 * one, 0 * one, 0 * one, 6e4 * one)
    r0.record(3.0, np.array([big]), 3 * one, 0 * one, 0 * one, 4e4 * one)
    r1.record(1.0, np.array([big + np.uint64(1)]), 9 * one, 0 * one, 0 * one, 1e3 * one)
    d = tracks.merge([r0.rows(), r1.rows()])
    per = tracks.split(d)
    assert set(per) == {int(big), int(big) + 1}
    assert list(per[int(big)]["t"]) == [1.0, 2.0, 3.0] and list(per[int(big)]["x"]) == [1.0, 2.0, 3.0]
    assert d["id"].dtype == np.uint64 and int(d["id"].max()) == int(big) + 1


def test_empty_recorder_merges_to_empty():
    d = tracks.merge([tracks.Recorder().rows(), tracks.Recorder().rows()])
    assert d["t"].size == 0 and tracks.split(d) == {}
