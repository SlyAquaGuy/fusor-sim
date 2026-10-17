import json
import os
import time

from fusorsim import monitor


def _run(tmp_path, rows, partial=None, log=None):
    run = tmp_path / "r1"
    run.mkdir()
    (run / "run_params.json").write_text(json.dumps(dict(t_end=1e-6, t_avg=5e-7, max_steps=100, I_target=0.05)))
    text = "t,step,I_ema,alpha\n" + "".join(f"{t:.8e},{s},{0.05:.8e},{1.0:.8e}\n" for t, s in rows)
    (run / "timeseries.csv").write_text(text + (partial or ""))
    if log is not None:
        (tmp_path / "r1.log").write_text(log)
    return run


def test_half_written_row_skipped_and_downsampled(tmp_path):
    rows = [(i * 1e-9, i) for i in range(1000)]
    run = _run(tmp_path, rows, partial="1.0e-06,10")          # rank 0 mid-append
    cols, n = monitor.read_timeseries(str(run / "timeseries.csv"), max_points=100)
    assert n == 1000
    assert len(cols["t"]) <= 101 and cols["step"][-1] == 999   # last row always kept


def test_status_from_files(tmp_path):
    _run(tmp_path, [(5e-7, 50)], log="[v6] t= 500 ns step 50/100\n")
    s = monitor.summarise(str(tmp_path), "r1")
    assert s["status"] == "running" and abs(s["frac"] - 0.5) < 1e-9 and s["log"].endswith("r1.log")
    old = time.time() - 10 * monitor.ACTIVE_S
    for f in ("r1/run_params.json", "r1/timeseries.csv", "r1.log"):
        os.utime(tmp_path / f, (old, old))
    assert monitor.summarise(str(tmp_path), "r1")["status"] == "stopped"
    # AMReX prints its finalize/timer table after the driver's done line
    (tmp_path / "r1.log").write_text("[v6] done in 0.10 h.\n" + "AMReX timer line\n" * 200)
    assert monitor.summarise(str(tmp_path), "r1")["status"] == "done"
    assert [r["name"] for r in monitor.list_runs(str(tmp_path))] == ["r1"]
