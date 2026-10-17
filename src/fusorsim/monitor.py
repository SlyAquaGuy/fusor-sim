"""Thin live monitor for WarpX runs: a stdlib-only HTTP server over a runs/ directory.

    fusor-monitor --runs runs --port 8050        # then open http://localhost:8050

It only READS what the driver already writes (run_params.json, timeseries.csv and the run log),
so it is safe next to a running sim, and it needs nothing beyond the Python standard library:
run it from the host venv, or from any image (`apptainer exec sif/analysis.sif fusor-monitor ...`).
On M3, run it on the login node and tunnel it: `ssh -L 8050:localhost:8050 m3`.

Log lookup (first hit wins): <runs>/<name>.log, <run>/*.log, <run>/*.out.
"""
import argparse
import csv
import glob
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ACTIVE_S = 120          # a run whose files changed this recently counts as running
MAX_POINTS = 600        # timeseries rows sent to the page (strided, last row always kept)
LOG_TAIL = 40
COLS = ("t", "step", "I_ema", "I_meas", "alpha", "n_macro", "fus_ion_n_rate", "fus_neu_n_rate",
        "vmax_cfl", "boot_frac", "E_ion_mean_eV")
PARAMS = ("t_end", "t_avg", "t_boot", "max_steps", "dt", "dt_cb", "I_target", "n_macro_target", "V_c",
          "pressure_mTorr", "nr", "nz", "nranks", "omp_threads", "seed", "fusorsim_git", "sif_digest")


def read_timeseries(path, max_points=MAX_POINTS):
    """Columns of timeseries.csv as lists of floats. Tolerates a half-written last line
    (rank 0 appends while we read) and strides long runs down to ~max_points rows."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        return {}, 0
    if not rows:
        return {}, 0
    header, good = rows[0], []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        try:
            good.append([float(x) for x in r])
        except ValueError:
            continue
    n = len(good)
    if n > max_points:
        stride = -(-n // max_points)
        good = good[::stride] + ([good[-1]] if (n - 1) % stride else [])
    keep = [i for i, k in enumerate(header) if k in COLS]
    return {header[i]: [r[i] for r in good] for i in keep}, n


def find_log(root, name):
    run = os.path.join(root, name)
    for cand in [os.path.join(root, name + ".log")] + sorted(glob.glob(os.path.join(run, "*.log"))) \
            + sorted(glob.glob(os.path.join(run, "*.out"))):
        if os.path.isfile(cand):
            return cand
    return None


def tail(path, nbytes=64 * 1024):
    """Lines of the last nbytes of the log: enough to find the '[v6] done' marker behind
    AMReX's finalize/timer output, which follows it."""
    if not path:
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - nbytes))
        return f.read().decode("utf-8", "replace").splitlines()


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except (FileNotFoundError, TypeError):
        return None


def run_status(params, last_t, log_lines, last_change, now):
    """done / failed / running / stopped / preflight, from the files alone."""
    text = "\n".join(log_lines)
    if "[v6] done" in text:
        return "done"
    if last_t is None:
        if "Preflight complete" in text:
            return "preflight"
        if "Traceback" in text or "Error" in text:
            return "failed"
        return "running" if last_change and now - last_change < ACTIVE_S else "preflight"
    if "Traceback" in text:
        return "failed"
    # rows come once per callback interval, so the last one can fall up to dt_cb short of t_end
    if params.get("t_end") and last_t >= params["t_end"] - 1.01 * (params.get("dt_cb") or 0) \
            and now - last_change >= ACTIVE_S:
        return "done"
    return "running" if now - last_change < ACTIVE_S else "stopped"


def summarise(root, name, series=True):
    run = os.path.join(root, name)
    with open(os.path.join(run, "run_params.json")) as f:
        allp = json.load(f)
    params = {k: allp.get(k) for k in PARAMS}
    ts_path = os.path.join(run, "timeseries.csv")
    cols, n_rows = read_timeseries(ts_path, MAX_POINTS if series else 1)
    log = find_log(root, name)
    lines = tail(log)
    t0 = _mtime(os.path.join(run, "run_params.json"))      # written just before WarpX starts
    last_change = max(x for x in (t0, _mtime(ts_path), _mtime(log)) if x is not None)
    now = time.time()
    last_t = cols["t"][-1] if cols.get("t") else None
    status = run_status(params, last_t, lines, last_change, now)
    wall = (now if status == "running" else last_change) - t0
    frac = (last_t / params["t_end"]) if (last_t is not None and params.get("t_end")) else 0.0
    eta = wall * (1 - frac) / frac if status == "running" and frac > 0.02 else None
    out = dict(name=name, status=status, frac=min(frac, 1.0), wall_s=wall, eta_s=eta,
               last_change=last_change, n_rows=n_rows, params=params, log=log)
    if series:
        out.update(series=cols, log_tail=lines[-LOG_TAIL:],
                   last={k: v[-1] for k, v in cols.items()} if cols else {})
    return out


def list_runs(root):
    runs = []
    for p in glob.glob(os.path.join(root, "*", "run_params.json")):
        name = os.path.basename(os.path.dirname(p))
        try:
            runs.append(summarise(root, name, series=False))
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda r: r["last_change"], reverse=True)


PAGE = os.path.join(os.path.dirname(__file__), "monitor.html")


class Handler(BaseHTTPRequestHandler):
    root = "runs"

    def log_message(self, *a):   # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(PAGE, "rb") as f:   # read per request: page edits need no restart
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/runs":
            return self._send(200, json.dumps(list_runs(self.root)).encode())
        if u.path == "/api/run":
            name = parse_qs(u.query).get("name", [""])[0]
            # only direct children of the runs root: no path tricks
            if not name or name != os.path.basename(name) or name.startswith("."):
                return self._send(400, b'{"error": "bad run name"}')
            try:
                return self._send(200, json.dumps(summarise(self.root, name)).encode())
            except FileNotFoundError:
                return self._send(404, b'{"error": "no such run"}')
        self._send(404, b'{"error": "not found"}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="runs", help="directory holding one sub-directory per run")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep local; tunnel over ssh)")
    args = ap.parse_args(argv)
    Handler.root = os.path.abspath(args.runs)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fusor-monitor: {Handler.root} -> http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
