"""
grid3d.py -- 3D vacuum field of the real wire grids, and the azimuthal "grid kicks" they give ions
=================================================================================================

2D-RZ turns each grid into axisymmetric rings, so the field never exerts a torque about the z-axis:
an ion born at rest keeps l_z = x v_y - y v_x = 0 and passes exactly through the axis. That is the
bright line on the axis of the RZ maps. The real grids are great circles of wire (an equator plus
n-1 meridians); the meridians break the symmetry and twist ions out of their meridional plane, so
in 3D they miss the axis.

This module measures that twist once, for one geometry:
  * the 3D vacuum field of cathode + anode inside the grounded chamber, by the thin-wire charge
    simulation method [singer1974charge]: an unknown uniform line charge on every wire segment and
    an unknown charge on every wall panel, chosen so the potential is right at one collocation point
    per element (wire surface / panel centre);
  * test D+ ions traced in that field (velocity Verlet, step limited near wires);
  * the change of l_z between successive cathode crossings, tabulated by crossing direction,
    polar angle and energy -> data/grid_kicks_3d.dat, which the RZ driver samples at each cathode
    crossing (fp.load_grid_kicks / fp.sample_grid_kick / fp.apply_azimuthal_kick).

Vacuum field only: at the grids the 60 kV applied field dominates the ion space charge, which
mostly reshapes the converged core, not the lens strength of the wire gaps.

    python -m fusorsim.physics.grid3d --case cases/R1_bound_5mT.yaml --out data/grid_kicks_3d.dat
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np

from fusorsim.physics import core as fp

K_E = 1.0 / (4.0 * np.pi * fp.eps0)
QUANTILES = np.linspace(0.0, 1.0, 41)
COS_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])             # |cos(polar angle)| at the crossing
E_FRAC_EDGES = np.array([0.0, 0.5, 0.85, 1.2])                 # crossing energy / e|V_c|
GEOM_KEYS = ("R_c", "d_c", "R_a", "d_a", "great_circles", "V_c", "R_ch", "H_ch")


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def great_circles(R, n_circles, seg_len):
    """Closed polylines of an equator plus (n_circles-1) meridians on a sphere of radius R, poles on z;
    meridian k at azimuth k*pi/(n_circles-1) (each one is a full great circle, so 2(n-1) half-meridians
    30 deg apart for 7 circles). n_circles = 1 gives the equator alone (axisymmetric test case)."""
    n_seg = max(8, int(np.ceil(2 * np.pi * R / seg_len)))
    t = np.linspace(0.0, 2.0 * np.pi, n_seg + 1)
    out = [np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1)]
    nm = n_circles - 1
    for k in range(nm):
        ph = k * np.pi / nm
        out.append(np.stack([R * np.sin(t) * np.cos(ph), R * np.sin(t) * np.sin(ph), R * np.cos(t)], axis=1))
    return out


def wall_panels(R, H, h):
    """Centres, areas and outward normals of ~h x h panels on the closed cylinder r = R, |z| <= H."""
    C, A = [], []
    nphi = max(8, int(round(2 * np.pi * R / h)))
    nz = max(2, int(round(2 * H / h)))
    ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    zc = -H + (np.arange(nz) + 0.5) * 2 * H / nz
    P, Z = np.meshgrid(ph, zc, indexing="ij")
    C.append(np.stack([R * np.cos(P).ravel(), R * np.sin(P).ravel(), Z.ravel()], axis=1))
    A.append(np.full(P.size, (2 * np.pi * R / nphi) * (2 * H / nz)))
    nr = max(2, int(round(R / h)))
    re = np.linspace(0.0, R, nr + 1)
    for zcap in (-H, H):
        for j in range(nr):
            rm = 0.5 * (re[j] + re[j + 1])
            n = max(1, int(round(2 * np.pi * rm / h)))
            phj = (np.arange(n) + 0.5) * 2 * np.pi / n
            C.append(np.stack([rm * np.cos(phj), rm * np.sin(phj), np.full(n, zcap)], axis=1))
            A.append(np.full(n, np.pi * (re[j + 1] ** 2 - re[j] ** 2) / n))
    return np.vstack(C), np.concatenate(A)


class GridField:
    """Vacuum field of two great-circle wire grids (cathode at V_c, anode at 0) in a grounded cylinder."""

    def __init__(self, R_c, d_c, R_a, d_a, n_circles, V_c, R_ch, H_ch, seg_c=3e-3, seg_a=4e-3, panel=10e-3,
                 n_circles_anode=None):
        self.geom = dict(R_c=R_c, d_c=d_c, R_a=R_a, d_a=d_a, great_circles=n_circles, V_c=V_c,
                         R_ch=R_ch, H_ch=H_ch)
        A, B, rad, V, grid, circ = [], [], [], [], [], []
        for R, d, Vg, gid, seg, nc in ((R_c, d_c, V_c, 0, seg_c, n_circles),
                                       (R_a, d_a, 0.0, 1, seg_a, n_circles if n_circles_anode is None
                                        else n_circles_anode)):
            for pts in great_circles(R, nc, seg):
                A.append(pts[:-1]); B.append(pts[1:])
                n = len(pts) - 1
                rad.append(np.full(n, d / 2)); V.append(np.full(n, Vg)); grid.append(np.full(n, gid))
                circ.append(np.full(n, len(circ)))
        self.A, self.B = np.vstack(A), np.vstack(B)
        self.a = np.concatenate(rad)
        self.grid = np.concatenate(grid)
        circ = np.concatenate(circ)
        V = [np.concatenate(V)]
        # Where wires cross (all meridians at the poles, every meridian at the equator) segments overlap;
        # overlapping unknowns make the system near-singular (huge alternating charges). Keep the first wire
        # through each crossing and drop later segments whose midpoint lies inside it: same conductor,
        # same potential, so the kept segments carry that charge.
        keep = np.ones(len(self.A), bool)
        M = 0.5 * (self.A + self.B)
        U = self.B - self.A
        L2 = np.einsum("ij,ij->i", U, U)
        for i in range(len(self.A)):
            prev = np.flatnonzero(keep[:i] & (self.grid[:i] == self.grid[i]) & (circ[:i] != circ[i]))
            if not prev.size:
                continue
            tt = np.clip(np.einsum("ij,ij->i", M[i] - self.A[prev], U[prev]) / L2[prev], 0.0, 1.0)
            dist = np.linalg.norm(M[i] - (self.A[prev] + tt[:, None] * U[prev]), axis=1)
            if np.any(dist < self.a[prev] + self.a[i]):
                keep[i] = False
        self.A, self.B, self.a, self.grid = self.A[keep], self.B[keep], self.a[keep], self.grid[keep]
        V = [V[0][keep]]
        self.U = self.B - self.A
        self.L = np.linalg.norm(self.U, axis=1)
        self.U /= self.L[:, None]
        self.Cp, self.Ap = wall_panels(R_ch, H_ch, panel)
        self.rho_p = np.sqrt(self.Ap / np.pi)                   # radius of the equal-area disc
        nw, npn = len(self.A), len(self.Cp)
        # collocation: each segment's midpoint pushed out by the wire radius (radially from the centre),
        # each panel's centre
        M = 0.5 * (self.A + self.B)
        Mc = M + self.a[:, None] * M / np.linalg.norm(M, axis=1)[:, None]
        Pc = np.vstack([Mc, self.Cp])
        rhs = np.concatenate([np.concatenate(V), np.zeros(npn)])
        G = np.empty((nw + npn, nw + npn))
        for s in range(0, len(Pc), 1024):                       # row blocks: bounded memory
            G[s:s + 1024] = self._influence(Pc[s:s + 1024], self_rows=np.arange(s, min(s + 1024, len(Pc))))
        q = np.linalg.solve(G, rhs)
        self.lam, self.Qp = q[:nw], q[nw:]

    # potential of unit line charge on each segment / unit charge on each panel, at points P
    def _influence(self, P, self_rows=None):
        nw = len(self.A)
        rA = np.linalg.norm(P[:, None, :] - self.A[None], axis=2)
        rB = np.linalg.norm(P[:, None, :] - self.B[None], axis=2)
        s = rA + rB
        Gw = K_E * np.log((s + self.L) / np.maximum(s - self.L, 1e-15))
        d = np.linalg.norm(P[:, None, :] - self.Cp[None], axis=2)
        Gp = K_E / np.maximum(d, 1e-12)
        if self_rows is not None:                               # panel self-term: uniform disc, centre
            ip = self_rows - nw
            m = ip >= 0
            Gp[np.flatnonzero(m), ip[m]] = 2.0 * K_E / self.rho_p[ip[m]]
        return np.hstack([Gw, Gp])

    def potential(self, P):
        P = np.atleast_2d(P)
        out = np.empty(len(P))
        for s in range(0, len(P), 512):
            out[s:s + 512] = self._influence(P[s:s + 512]) @ np.concatenate([self.lam, self.Qp])
        return out

    def field(self, P):
        """E (V/m) at points P, and each point's clearance from the nearest wire surface (m, < 0 inside)."""
        P = np.atleast_2d(P)
        D = P[:, None, :] - self.A[None]                        # from segment start to point
        sp = np.einsum("ijk,jk->ij", D, self.U)                 # position along the segment
        rho_v = D - sp[..., None] * self.U[None]
        rho2 = np.maximum(np.einsum("ijk,ijk->ij", rho_v, rho_v), (self.a ** 2)[None])
        rA = np.sqrt(sp ** 2 + rho2)
        rB = np.sqrt((sp - self.L[None]) ** 2 + rho2)
        lam = self.lam[None]
        E_par = K_E * lam * (1.0 / rB - 1.0 / rA)
        E_perp = K_E * lam * (sp / rA - (sp - self.L[None]) / rB) / rho2
        E = np.einsum("ij,jk->ik", E_par, self.U) + np.einsum("ij,ijk->ik", E_perp, rho_v)
        Dp = P[:, None, :] - self.Cp[None]
        dp = np.maximum(np.linalg.norm(Dp, axis=2), 1e-9)
        E += np.einsum("ij,ijk->ik", K_E * self.Qp[None] / dp ** 3, Dp)
        # clearance: distance to the segment (clamped to its ends) minus the wire radius
        t = np.clip(sp, 0.0, self.L[None])
        dist = np.linalg.norm(D - t[..., None] * self.U[None], axis=2) - self.a[None]
        return E, dist.min(axis=1)


# ---------------------------------------------------------------------------
# test ions
# ---------------------------------------------------------------------------
def birth_positions(n, R_c, d_c, R_a, d_a, rng, f_anode=0.7):
    """f_anode at rest just inside the anode (the driver's bounding makeup ions), the rest uniform in
    volume between the grids; isotropic directions."""
    na = int(round(f_anode * n))
    r = np.concatenate([np.full(na, R_a - d_a - 1.5e-3),
                        ((R_c + 3 * d_c) ** 3 + rng.random(n - na) * ((R_a - 3 * d_a) ** 3 - (R_c + 3 * d_c) ** 3))
                        ** (1 / 3)])
    mu = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(1 - mu ** 2)
    return np.stack([r * st * np.cos(ph), r * st * np.sin(ph), r * mu], axis=1)


def trace_kicks(gf, X0, ds=0.25e-3, max_cross=8, t_max=1.5e-6, max_steps=60000, report=None):
    """Follow test D+ ions from rest; at every cathode crossing record (direction, |cos polar|, energy eV,
    and the change of l_z until the NEXT crossing). Intervals cut short by a wire hit are dropped.
    Returns (events, stats)."""
    R_c, R_ch, H_ch = gf.geom["R_c"], gf.geom["R_ch"], gf.geom["H_ch"]
    q_m = fp.e / fp.m_d
    X = X0.copy(); V = np.zeros_like(X)
    E, clr_cur = gf.field(X)
    keep = clr_cur > 0                                          # born inside a wire: drop
    X, V, E, clr_cur = X[keep], V[keep], E[keep], clr_cur[keep]
    n0 = len(X)
    ids = np.arange(n0)
    t = np.zeros(n0)
    rs_prev = np.linalg.norm(X, axis=1)
    last = {}                                                   # id -> (inward, cos, E_eV, lz) at last crossing
    ncross = np.zeros(n0, int)
    events, hits = [], {0: 0, 1: 0, "wall": 0}
    step = 0
    while len(X) and step < max_steps:
        step += 1
        a = q_m * E
        vmag = np.linalg.norm(V, axis=1); amag = np.linalg.norm(a, axis=1)
        dsi = np.minimum(ds, np.maximum(0.3 * clr_cur, 2e-5))   # shorter steps close to a wire
        dt =np.minimum(dsi / np.maximum(vmag, 1.0), np.sqrt(2.0 * dsi / np.maximum(amag, 1e-30)))
        V += 0.5 * a * dt[:, None]                              # velocity Verlet: kick, drift, kick
        X += V * dt[:, None]
        E, clr_cur = gf.field(X)
        V += 0.5 * q_m * E * dt[:, None]
        t += dt
        rs = np.linalg.norm(X, axis=1)
        cross = (rs_prev - R_c) * (rs - R_c) < 0
        for j in np.flatnonzero(cross):
            i = ids[j]
            lz = X[j, 0] * V[j, 1] - X[j, 1] * V[j, 0]
            Ek = 0.5 * fp.m_d * (V[j] @ V[j]) / fp.e
            if i in last:
                inw, cs, E0, lz0 = last[i]
                events.append((inw, cs, E0, lz - lz0))
            last[i] = (bool(rs[j] < rs_prev[j]), abs(X[j, 2]) / rs[j], Ek, lz)
            ncross[j] += 1
        rs_prev = rs
        rcyl = np.hypot(X[:, 0], X[:, 1])
        hit = clr_cur <= 0
        out = (rcyl >= R_ch) | (np.abs(X[:, 2]) >= H_ch)
        for j in np.flatnonzero(hit):
            hits[int(gf.grid[np.argmin(np.abs(np.linalg.norm(X[j] - 0.5 * (gf.A + gf.B), axis=1)))])] += 1
        hits["wall"] += int(out.sum())
        alive = ~hit & ~out & (ncross < max_cross) & (t < t_max)
        if not alive.all():
            for j in np.flatnonzero(~alive):
                last.pop(ids[j], None)
            X, V, E, clr_cur, t, rs_prev, ids, ncross = (X[alive], V[alive], E[alive], clr_cur[alive],
                                                          t[alive], rs_prev[alive], ids[alive], ncross[alive])
        if report and step % 500 == 0:
            report(f"  step {step}: {len(X)} ions flying, {len(events)} events")
    ev = np.array(events, dtype=float).reshape(-1, 4)
    return ev, dict(n_ions=n0, n_events=len(ev), hits_cathode=hits[0], hits_anode=hits[1], lost_wall=hits["wall"],
                    steps=step)


def kick_table(ev, V_c):
    """Quantiles of delta l_z per (direction, |cos| bin, energy bin). Cells with < 20 events take the
    quantiles pooled over energy for their (direction, |cos|) -- flagged pooled = 1."""
    rows = []
    Eedges = E_FRAC_EDGES * abs(V_c)
    for inw in (1, 0):
        d = ev[ev[:, 0] == inw]
        for i in range(len(COS_EDGES) - 1):
            dc = d[(d[:, 1] >= COS_EDGES[i]) & (d[:, 1] <= COS_EDGES[i + 1])]
            for k in range(len(Eedges) - 1):
                de = dc[(dc[:, 2] >= Eedges[k]) & (dc[:, 2] < Eedges[k + 1])]
                src, pooled = (de, 0) if len(de) >= 20 else (dc, 1)
                qv = np.quantile(src[:, 3], QUANTILES) if len(src) else np.zeros(len(QUANTILES))
                rows.append([inw, COS_EDGES[i], COS_EDGES[i + 1], Eedges[k], Eedges[k + 1], len(src), pooled, *qv])
    return np.array(rows)


def write_table(path, rows, geom, stats, note):
    with open(path, "w") as f:
        f.write("# grid_kicks_3d.dat -- change of l_z = x v_y - y v_x (m^2/s) of D+ ions between successive\n")
        f.write("# cathode crossings, traced in the 3D vacuum field of the real great-circle wire grids\n")
        f.write("# (fusorsim.physics.grid3d). The RZ driver adds one sample at each cathode crossing.\n")
        f.write(f"# geometry: {json.dumps(geom)}\n")
        f.write(f"# made: {note}\n")
        f.write(f"# stats: {json.dumps(stats)}\n")
        f.write(f"# quantiles: {' '.join(f'{p:.3f}' for p in QUANTILES)}\n")
        f.write("# columns: inward cos_lo cos_hi E_lo_eV E_hi_eV n pooled q[quantiles]...\n")
        for r in rows:
            f.write(" ".join([f"{int(r[0])}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.1f}", f"{r[4]:.1f}",
                              f"{int(r[5])}", f"{int(r[6])}"] + [f"{v:.6e}" for v in r[7:]]) + "\n")


def _case_geometry(case_path):
    """Geometry from a case file, falling back to the driver's defaults (kept in step with driver.py)."""
    g = dict(R_c=0.069, d_c=0.002, R_a=0.129, d_a=0.001, great_circles=7, V_c=-60e3, R_ch=0.144, H_ch=0.144)
    keys = dict(cathode_radius="R_c", cathode_wire="d_c", anode_radius="R_a", anode_wire="d_a",
                great_circles="great_circles", voltage="V_c", chamber_radius="R_ch", chamber_half_length="H_ch")
    if case_path:
        for ln in open(case_path):
            ln = ln.split("#", 1)[0].strip()
            if ":" in ln:
                k, v = (s.strip() for s in ln.split(":", 1))
                if k in keys:
                    g[keys[k]] = int(v) if keys[k] == "great_circles" else float(v)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description="3D grid-kick calibration for the RZ driver")
    ap.add_argument("--case", default=None, help="case YAML for the geometry (driver defaults otherwise)")
    ap.add_argument("--out", default=os.path.join(fp.DATA_DIR, "grid_kicks_3d.dat"))
    ap.add_argument("--n-ions", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--ds", type=float, default=0.25e-3, help="max step length, m")
    args = ap.parse_args(argv)
    g = _case_geometry(args.case)
    t0 = time.time()
    print(f"geometry {g}", flush=True)
    gf = GridField(n_circles=g["great_circles"], **{k: v for k, v in g.items() if k != "great_circles"})
    print(f"field solved: {len(gf.A)} wire segments, {len(gf.Cp)} wall panels ({time.time()-t0:.0f} s); "
          f"centre potential {gf.potential(np.zeros((1, 3)))[0]/1e3:.2f} kV", flush=True)
    rng = np.random.default_rng(args.seed)
    X0 = birth_positions(args.n_ions, g["R_c"], g["d_c"], g["R_a"], g["d_a"], rng)
    ev, st = trace_kicks(gf, X0, ds=args.ds, report=lambda s: print(s, flush=True))
    st["centre_potential_V"] = float(gf.potential(np.zeros((1, 3)))[0])
    st["dlz_rms_m2_s"] = float(np.sqrt(np.mean(ev[:, 3] ** 2))) if len(ev) else 0.0
    try:
        git = subprocess.run(["git", "describe", "--always", "--dirty", "--tags"], capture_output=True, text=True,
                             cwd=os.path.dirname(__file__)).stdout.strip() or "unknown"
    except OSError:
        git = "unknown"
    note = (f"python -m fusorsim.physics.grid3d --case {args.case} --n-ions {args.n_ions} --seed {args.seed} "
            f"--ds {args.ds}; git {git}; {datetime.date.today().isoformat()}; {time.time()-t0:.0f} s")
    write_table(args.out, kick_table(ev, g["V_c"]), g, st, note)
    print(f"{st}\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
