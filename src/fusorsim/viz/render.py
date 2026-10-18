#!/usr/bin/env python3
"""
viz_v6.py -- pictures and animations of a fusor_v6 run
=======================================================
    python viz_v6.py <run_dir>            # all still images  (~1 min)
    python viz_v6.py <run_dir> --movie    # + ion animation and ion tracks

Writes into <run_dir>/viz/:
  potential.png        electrostatic potential averaged over the snapshots in the
                       steady-state window, full cross-section,
                       equipotentials and electrodes
  ion_density.png      time-averaged D+ density (log), full cross-section
  neutron_emission.png where the D(d,n) neutrons are made (log), full cross-section
  camera_view.png      the same two quantities line-of-sight integrated, i.e.
                       what a camera or collimated detector at a side viewport
                       would see (forward Abel projection of the axisymmetric maps)
  phase_space.png      ion radial velocity against radius: recirculating beams,
                       charge-exchange-born ions, the turning points
  spectra.png          cathode ion impact energies, wall fast-neutral energies,
                       electron energies at the wall, reacting-deuteron energies
                       and the neutron spectrum
  ion_tracks.png       paths of --n-tracks ions, sampled every callback (~1.5 ns),
                       coloured by energy; x = ion lost (needs tracks.npz)
  ions_movie.mp4       every ion as a dot coloured by energy, side view, over the whole run,
                       1080x1080, from the driver's diags/movie/ frames         (--movie)
  ion_tracks_movie.mp4 the tracked ions moving, with short tails                (--movie)
                       (older runs with neither: ions_movie from the ~20 snapshots)

Maps: the 2D-RZ axis band (first 3 radial bins) is hatched and left out of the colour
scale -- RZ focuses everything aimed at the centre onto the axis, overstating it.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm

from fusorsim.physics import core as fp  # noqa: E402
from fusorsim.warpx import tracks as ftracks  # noqa: E402  (pure numpy, no pywarpx)

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--movie", action="store_true", help="also make the ion movie (from tracks.npz)")
ap.add_argument("--max-frame-particles", type=int, default=40000, help="snapshot movie only (older runs)")
ap.add_argument("--n-tracks", type=int, default=60, help="paths drawn in ion_tracks.png")
ap.add_argument("--fps", type=int, default=30)
args = ap.parse_args()

run = os.path.abspath(args.run_dir)
out = os.path.join(run, "viz")
os.makedirs(out, exist_ok=True)
P = json.load(open(os.path.join(run, "run_params.json")))
T = np.load(os.path.join(run, "tallies.npz"))
R_ch, H_ch, R_c, R_a = P["R_ch"], P["H_ch"], P["R_c"], P["R_a"]
MM = 1e3
plt.rcParams.update({"figure.dpi": 140, "font.size": 10})
DARK = {"axes.facecolor": "black", "figure.facecolor": "black", "text.color": "white",
        "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white", "axes.edgecolor": "#888"}


# ---------------------------------------------------------------------------
# geometry helpers (same ring model as fusor_v6.py)
# ---------------------------------------------------------------------------
def rings(T_, nb):
    period = np.pi / nb
    hw = (1 - T_) * period / 2
    return [(i + 0.5) * period for i in range(nb)], hw


def solid(r, z):
    rs, th = np.hypot(r, z), np.arctan2(np.abs(r), z)
    m = np.zeros(np.shape(r), bool)
    for R, d, T_, nb in ((R_c, P["d_c"], P["T_c"], P.get("cathode_bands", 11)),
                         (R_a, P["d_a"], P["T_a"], P.get("anode_bands", 5))):
        cs, hw = rings(T_, nb)
        band = np.zeros_like(m)
        for c in cs:
            band |= np.abs(th - c) < hw
        m |= (np.abs(rs - R) < max(d / 2, 0.0006)) & band
    return m


_xg = np.linspace(-R_ch, R_ch, 1200)
_zg = np.linspace(-H_ch, H_ch, 1200)
_X, _Z = np.meshgrid(_xg, _zg)
_SOLID = solid(_X, _Z)


def draw_electrodes(ax, colour="white"):
    ax.contourf(_X * MM, _Z * MM, _SOLID.astype(float), levels=[0.5, 1.5], colors=[colour])
    ax.set_xlim(-R_ch * MM, R_ch * MM); ax.set_ylim(-H_ch * MM, H_ch * MM)
    ax.set_aspect("equal")


def draw_projected_rings(ax, colour="#bbbbbb"):
    """Side view of the 3D ring grids: each latitude ring projects to a line."""
    for R, T_, nb in ((R_c, P["T_c"], P.get("cathode_bands", 11)), (R_a, P["T_a"], P.get("anode_bands", 5))):
        cs, _ = rings(T_, nb)
        for c in cs:
            x = R * np.sin(c)
            ax.plot([-x * MM, x * MM], [R * np.cos(c) * MM] * 2, color=colour, lw=0.6, alpha=0.7)
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(R * np.cos(t) * MM, R * np.sin(t) * MM, color=colour, lw=0.5, ls=":", alpha=0.6)
    ax.add_patch(plt.Rectangle((-R_ch * MM, -H_ch * MM), 2 * R_ch * MM, 2 * H_ch * MM, fill=False,
                               ec=colour, lw=1.0))


def mirror(a):
    """(nr, nz) half-plane array -> (2nr, nz) full cross-section."""
    return np.concatenate([a[::-1], a], axis=0)


# ---------------------------------------------------------------------------
# tallied maps
# ---------------------------------------------------------------------------
r_e, z_e = T["r_edges"], T["z_edges"]
tW = float(T["t_window"][0])
Vcell = (np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2))[:, None] * np.diff(z_e)[None, :]
n_ion = T["ion_wdt_rz"] / max(tW, 1e-30) / Vcell
emis = (T["fus_ion_rz"] + T["fus_neu_rz"]) / max(tW, 1e-30) / Vcell      # n / s / m^3
xe_full = np.concatenate([-r_e[::-1], r_e[1:]]) * MM
# 2D-RZ focuses everything aimed at the centre onto the axis, and the 1/r cell volume amplifies it, so
# the first few radial bins are far brighter than a 3D device (see the report's core ion-ion note).
# Colour scales come from the cells outside this band; inside it the maps saturate and are hatched.
AXIS_R = r_e[min(3, len(r_e) - 1)]
xc_full = 0.5 * (xe_full[1:] + xe_full[:-1])
OFF_AXIS = np.abs(xc_full) >= AXIS_R * MM


def off_axis_max(d):
    v = d[OFF_AXIS]
    v = v[np.isfinite(v) & (v > 0)]
    return v.max() if v.size else 1.0


def mark_axis(ax):
    ax.axvspan(-AXIS_R * MM, AXIS_R * MM, facecolor="none", edgecolor="white", hatch="////", lw=0, alpha=0.35)
    ax.text(AXIS_R * MM + 1.5, -H_ch * MM * 0.93, "RZ axis band: saturated, overstated in 2D", color="white",
            fontsize=7, alpha=0.8)


def cross_section(data, title, label, fname, cmap="inferno", floor=1e-4):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        d = mirror(data)
        vmax = off_axis_max(d)
        im = ax.pcolormesh(xe_full, z_e * MM, np.clip(d, vmax * floor, None).T, norm=LogNorm(vmax * floor, vmax),
                           cmap=cmap, shading="flat")
        draw_electrodes(ax, "#9ecbff")
        mark_axis(ax)
        fig.colorbar(im, ax=ax, label=label, shrink=0.85, extend="max")
        ax.set(xlabel="x (mm)", ylabel="z, symmetry axis (mm)", title=title)
        fig.savefig(os.path.join(out, fname)); plt.close(fig)


if tW > 0:
    cross_section(n_ion, "Time-averaged D+ density", "n_i (m$^{-3}$)", "ion_density.png", "viridis")
    cross_section(emis, "Where the D(d,n) neutrons are made", "emission (n s$^{-1}$ m$^{-3}$)",
                  "neutron_emission.png", "inferno")

    # forward Abel projection: line-of-sight integral through the axisymmetric map
    rc = r_e
    y = 0.5 * (r_e[1:] + r_e[:-1])
    Y = np.concatenate([-y[::-1], y])
    L = np.zeros((len(Y), len(rc) - 1))
    for j, yy in enumerate(np.abs(Y)):
        a = np.sqrt(np.maximum(rc[1:] ** 2 - yy ** 2, 0)); b = np.sqrt(np.maximum(rc[:-1] ** 2 - yy ** 2, 0))
        L[j] = 2 * (a - b)
    with plt.rc_context(DARK):
        fig, axs = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
        for ax, f, ttl, lab, cmap in ((axs[0], emis, "Neutron emission seen from the side",
                                       "line-integrated (n s$^{-1}$ m$^{-2}$)", "inferno"),
                                      (axs[1], n_ion, "D+ column density seen from the side",
                                       "line-integrated (m$^{-2}$)", "viridis")):
            img = L @ f
            vmax = off_axis_max(img)       # chords |y| < AXIS_R cross the overstated axis cells
            im = ax.pcolormesh(xe_full, z_e * MM, np.clip(img, vmax * 1e-3, None).T,
                               norm=LogNorm(vmax * 1e-3, vmax), cmap=cmap, shading="flat")
            draw_projected_rings(ax)
            mark_axis(ax)
            ax.set(xlim=(-R_ch * MM, R_ch * MM), ylim=(-H_ch * MM, H_ch * MM), aspect="equal",
                   xlabel="horizontal (mm)", ylabel="z (mm)", title=ttl)
            fig.colorbar(im, ax=ax, label=lab, shrink=0.85, extend="max")
        fig.text(0.5, 0.01, "Horizontal bands: each 3D great-circle wire is modelled as latitude rings in 2D-RZ.",
                 ha="center", color="#aaaaaa", fontsize=8)
        fig.savefig(os.path.join(out, "camera_view.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# time-averaged potential (WarpX TimeAveragedFieldDiagnostic)
# ---------------------------------------------------------------------------
try:
    import yt
    yt.set_log_level(50)
    HAVE_YT = True
except ImportError:
    HAVE_YT = False
    print("yt not installed: skipping potential map, phase space and movie (conda install -c conda-forge yt)")


def load_field(path, name):
    ds = yt.load(path)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return np.array(cg["boxlib", name])[:, :, 0], ds


diags = os.path.join(run, "diags")
if HAVE_YT:
    # Average the ordinary field snapshots inside the steady-state window.
    # (WarpX 26.01's TimeAveraged diagnostic returned -2x the potential in our RZ
    # tests, so it is not used.)
    snapsF = [f for f in sorted(glob.glob(os.path.join(diags, "fields0*")))]
    use = []
    for f in snapsF:
        ph, ds = load_field(f, "phi")
        if float(ds.current_time) >= P["t_avg"] * 0.999:
            use.append(ph)
    if not use and snapsF:
        use = [load_field(snapsF[-1], "phi")[0]]
    if use:
        phi = np.mean(use, axis=0)
        nr, nz = phi.shape
        re_ = np.linspace(0, R_ch, nr + 1); ze_ = np.linspace(-H_ch, H_ch, nz + 1)
        rcn = 0.5 * (re_[1:] + re_[:-1]); zcn = 0.5 * (ze_[1:] + ze_[:-1])
        full = mirror(phi) / 1e3
        xc = np.concatenate([-rcn[::-1], rcn]) * MM
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
            im = ax.pcolormesh(np.concatenate([-re_[::-1], re_[1:]]) * MM, ze_ * MM, full.T, cmap="magma",
                               shading="flat")
            cs = ax.contour(xc, zcn * MM, full.T, levels=np.arange(-60, 1, 5), colors="white",
                            linewidths=0.5, alpha=0.6)
            ax.clabel(cs, cs.levels[::2], fontsize=6, fmt="%d")
            draw_electrodes(ax, "#9ecbff")
            fig.colorbar(im, ax=ax, label="potential (kV)", shrink=0.85)
            ax.set(xlabel="x (mm)", ylabel="z (mm)",
                   title=f"Potential ({len(use)} snapshots averaged), centre {phi[0, nz // 2] / 1e3:.1f} kV")
            fig.savefig(os.path.join(out, "potential.png")); plt.close(fig)


def load_ions(path):
    ds = yt.load(path)
    if ("ions", "particle_position_x") not in ds.field_list:     # e.g. step 0: no ions yet
        z0 = np.zeros(0)
        return dict(t=float(ds.current_time), r=z0, z=z0, th=z0, w=z0, px=z0, py=z0, pz=z0, E=z0,
                    id=np.zeros(0, np.int64))
    ad = ds.all_data()
    g = lambda k: np.array(ad["ions", k])
    r, z, th, w = g("particle_position_x"), g("particle_position_y"), g("particle_theta"), g("particle_weight")
    px, py, pz = g("particle_momentum_x"), g("particle_momentum_y"), g("particle_momentum_z")
    ids = g("particle_id").astype(np.int64)
    E = (px**2 + py**2 + pz**2) / (2 * fp.m_d) / fp.e            # eV (non-relativistic is fine)
    return dict(t=float(ds.current_time), r=r, z=z, th=th, w=w, px=px, py=py, pz=pz, E=E, id=ids)


snaps = sorted(glob.glob(os.path.join(diags, "ions_snap*"))) if HAVE_YT else []

# ---------------------------------------------------------------------------
# phase space of the last snapshot
# ---------------------------------------------------------------------------
if snaps:
    s = load_ions(snaps[-1])
    if s["r"].size:
        rs = np.hypot(s["r"], s["z"])
        pr = s["px"] * np.cos(s["th"]) + s["py"] * np.sin(s["th"])
        vr = (pr * s["r"] + s["pz"] * s["z"]) / np.maximum(rs, 1e-9) / fp.m_d
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            h = ax.hist2d(rs * MM, vr / 1e6, bins=[300, 240], weights=s["w"], norm=LogNorm(), cmap="turbo")
            for R, lab in ((R_c, "cathode"), (R_a, "anode")):
                ax.axvline(R * MM, color="white", lw=0.8, ls="--"); ax.text(R * MM + 1, 2.3, lab, color="white")
            fig.colorbar(h[3], ax=ax, label="real ions per bin")
            ax.set(xlabel="distance from centre (mm)", ylabel="radial velocity (10$^6$ m/s)",
                   title=f"Ion phase space at t = {s['t']*1e6:.2f} us  (inward < 0 < outward)")
            fig.savefig(os.path.join(out, "phase_space.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------
if tW > 0:
    Ee = T["E_edges"]; Ec = 0.5 * (Ee[1:] + Ee[:-1]) / 1e3
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for key, lab in (("imp_ion_E__cathode", "D+ hitting the cathode"), ("imp_neu_E__wall", "fast D atoms hitting the wall"),
                     ("imp_neu_E__cathode", "fast D atoms hitting the cathode"), ("e_wall_E", "electrons hitting the wall")):
        if key in T and T[key].sum() > 0:
            axs[0].step(Ec, T[key] / tW / np.diff(Ee / 1e3), where="mid", label=lab)
    axs[0].set(yscale="log", xlabel="impact energy (keV)", ylabel="particles s$^{-1}$ keV$^{-1}$",
               title="What hits the surfaces")
    axs[0].legend(fontsize=8)
    wf = T["fus_ion_E"] + T["fus_neu_E"]
    axs[1].step(Ec, wf / wf.sum(), where="mid", label="energy of the reacting deuteron")
    axs[1].set(xlabel="deuteron energy (keV)", ylabel="fraction of D(d,n) reactions")
    sel = wf > 0
    b, sp = fp.dd_neutron_spectrum(Ec[sel] * 1e3, wf[sel], bins_MeV=np.linspace(2.0, 3.0, 51), n_dir=2000)
    ax2 = axs[1].inset_axes([0.55, 0.5, 0.42, 0.42])
    ax2.step(0.5 * (b[1:] + b[:-1]), sp / sp.sum(), where="mid", color="C3")
    ax2.set_title("neutron energy (MeV)", fontsize=8); ax2.tick_params(labelsize=7)
    axs[1].set_title("Which deuterons make the neutrons")
    fig.savefig(os.path.join(out, "spectra.png")); plt.close(fig)

# ---------------------------------------------------------------------------
# ion tracks and movie. Paths come from tracks.npz (the driver follows --track-n ions every callback,
# ~1.5 ns); snapshots are ~0.2 us apart, longer than an ion lives, so they cannot give paths.
# ---------------------------------------------------------------------------
def save_anim(anim, stem):
    try:
        anim.save(os.path.join(out, stem + ".mp4"), writer=animation.FFMpegWriter(fps=args.fps, bitrate=16000))
        print(f"wrote {stem}.mp4")
    except Exception as ex:
        anim.save(os.path.join(out, stem + ".gif"), writer=animation.PillowWriter(fps=args.fps))
        print(f"ffmpeg unavailable ({ex}); wrote {stem}.gif")


ts = None
tsf = os.path.join(run, "timeseries.csv")
if os.path.exists(tsf):
    ts = np.genfromtxt(tsf, delimiter=",", names=True)


def run_state(t):
    if ts is None or not len(ts):
        return ""
    j = np.argmin(np.abs(ts["t"] - t))
    return f"   I = {ts['I_ema'][j]*1e3:.1f} mA   Yn = {ts['fus_ion_n_rate'][j]+ts['fus_neu_n_rate'][j]:.2e}/s"


E_NORM = plt.Normalize(0, abs(P["V_c"]) / 1e3)          # ion energy colour scale, keV
MERID = "along each ion's own meridional plane (mm)"
TR = {}
trk_file = os.path.join(run, "tracks.npz")
if os.path.exists(trk_file):
    Ztr = np.load(trk_file)
    trk_t0, trk_dt = float(Ztr["t0"]), float(Ztr["dt_cb"])
    for i, d in ftracks.split(Ztr).items():
        if d["t"].size < 3:
            continue
        # an ion moves almost in one plane through the axis: project onto ITS (signed r, z) plane, which
        # stays continuous through the centre (plain x foreshortens; |r| folds the path back on itself)
        th0 = np.arctan2(d["y"][0], d["x"][0])
        u = d["x"] * np.cos(th0) + d["y"] * np.sin(th0)
        TR[i] = dict(t=d["t"], u=u * MM, z=d["z"] * MM, E=d["E"] / 1e3)
    trk_end = max(v["t"][-1] for v in TR.values()) if TR else 0.0
else:
    print("no tracks.npz (run made before ion tracking was added): ion_tracks.png skipped")
    stale = os.path.join(out, "ion_tracks.png")      # snapshot-based version: not real paths
    if os.path.exists(stale):
        os.remove(stale)

if TR:
    from matplotlib.collections import LineCollection
    rng = np.random.default_rng(0)
    keys = list(TR)
    show = rng.choice(keys, min(args.n_tracks, len(keys)), replace=False)
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
        draw_electrodes(ax, "#9ecbff")
        n_lost = 0
        for i in show:
            v = TR[int(i)]
            p = np.column_stack([v["u"], v["z"]])
            lc = LineCollection(np.stack([p[:-1], p[1:]], axis=1), cmap="plasma", norm=E_NORM, lw=0.7)
            lc.set_array(0.5 * (v["E"][1:] + v["E"][:-1]))
            ax.add_collection(lc)
            if v["t"][-1] < trk_end - 1.5 * trk_dt:      # gone before the window closed
                ax.plot(*p[-1], "x", ms=3.5, mew=0.9, color="#ff6b6b"); n_lost += 1
            else:
                ax.plot(*p[-1], "o", ms=2, color="white")
        fig.colorbar(plt.cm.ScalarMappable(norm=E_NORM, cmap="plasma"), ax=ax, label="ion energy (keV)", shrink=0.85)
        ax.set(xlabel=MERID, ylabel="z (mm)",
               title=f"{len(show)} ion paths, {(trk_end - trk_t0)*1e9:.0f} ns (every {trk_dt*1e9:.1f} ns)")
        ax.text(0.01, 0.01, f"o still flying at the end    x lost ({n_lost}): grid/wall hit or removed",
                transform=ax.transAxes, fontsize=7, color="#cccccc")
        fig.savefig(os.path.join(out, "ion_tracks.png")); plt.close(fig)

if args.movie and TR:
    from matplotlib.collections import LineCollection
    times = np.unique(np.concatenate([v["t"] for v in TR.values()]))
    TRAIL = 12                                            # samples of tail behind each ion (~18 ns)
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        fig.subplots_adjust(0.1, 0.08, 0.86, 0.94)
        cax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
        draw_electrodes(ax, "#9ecbff")
        tails = LineCollection([], cmap="plasma", norm=E_NORM, lw=0.9, alpha=0.8)
        ax.add_collection(tails)
        heads = ax.scatter([], [], s=7, c=[], cmap="plasma", norm=E_NORM, zorder=3)
        fig.colorbar(heads, cax=cax, label="ion energy (keV)")
        ax.set(xlabel=MERID, ylabel="z (mm)")
        title = ax.set_title("", fontsize=9)

        def track_frame(k):
            tk = times[k]
            segs, cols, hp, hc = [], [], [], []
            for v in TR.values():
                j = np.searchsorted(v["t"], tk, side="right")
                if j == 0 or v["t"][j - 1] < tk - 0.5 * trk_dt:    # not born yet, or already gone
                    continue
                a = max(0, j - TRAIL)
                p = np.column_stack([v["u"][a:j], v["z"][a:j]])
                if len(p) > 1:
                    segs.extend(np.stack([p[:-1], p[1:]], axis=1)); cols.extend(v["E"][a + 1:j])
                hp.append(p[-1]); hc.append(v["E"][j - 1])
            tails.set_segments(segs); tails.set_array(np.asarray(cols))
            heads.set_offsets(np.asarray(hp) if hp else np.zeros((0, 2))); heads.set_array(np.asarray(hc))
            title.set_text(f"t = {tk*1e9:6.1f} ns   {len(hp)} tracked ions{run_state(tk)}")
            return tails, heads, title

        save_anim(animation.FuncAnimation(fig, track_frame, frames=len(times), blit=False), "ion_tracks_movie")
        plt.close(fig)

MFR = sorted(glob.glob(os.path.join(diags, "movie", "frame_*.npz")))
if args.movie and not MFR and snaps:             # older runs: one frame per snapshot (~0.2 us apart)
    print(f"no tracks.npz: movie from {len(snaps)} snapshots, each a different set of ions")
    rng = np.random.default_rng(0)
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        fig.subplots_adjust(0.1, 0.08, 0.86, 0.94)
        cax = fig.add_axes([0.88, 0.15, 0.025, 0.7])
        sc = ax.scatter([], [], s=0.6, c=[], cmap="plasma", vmin=0, vmax=abs(P["V_c"]) / 1e3)
        fig.colorbar(sc, cax=cax, label="ion energy (keV)")
        draw_projected_rings(ax)
        ax.set(xlim=(-R_ch * MM, R_ch * MM), ylim=(-H_ch * MM, H_ch * MM), aspect="equal",
               xlabel="x (mm)", ylabel="z (mm)")
        title = ax.set_title("")

        def frame(i):
            s = load_ions(snaps[i])
            n = s["r"].size
            k = rng.choice(n, min(n, args.max_frame_particles), replace=False) if n else np.array([], int)
            x = s["r"][k] * np.cos(s["th"][k]) * MM
            sc.set_offsets(np.column_stack([x, s["z"][k] * MM]) if n else np.zeros((0, 2)))
            sc.set_array(s["E"][k] / 1e3)
            title.set_text(f"t = {s['t']*1e9:6.0f} ns   ions shown {len(k)}{run_state(s['t'])}")
            return sc, title

        save_anim(animation.FuncAnimation(fig, frame, frames=len(snaps), blit=False), "ions_movie")
        plt.close(fig)

# ---------------------------------------------------------------------------
# full-run movie: every ion, every frame (diags/movie/, binned by the driver each callback). Side view
# like a camera: brightness = ion density (log), colour = mean ion energy.
# ---------------------------------------------------------------------------
if args.movie and MFR:
    E_kev_max = abs(P["V_c"]) / 1e3                               # colour scale 0..e|V_c| (as before)
    LUT = (plt.get_cmap("plasma")(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(7.2, 7.2), dpi=150)             # 1080 x 1080 px
        ax = fig.add_axes([0.10, 0.07, 0.74, 0.86])
        cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, E_kev_max), "plasma"), cax=cax, label="ion energy (keV)")
        ax.set(xlim=(-R_ch * MM, R_ch * MM), ylim=(-H_ch * MM, H_ch * MM), aspect="equal",
               xlabel="x (mm)", ylabel="z (mm)")
        fig.canvas.draw()
        W, Hpx = int(round(ax.bbox.width)), int(round(ax.bbox.height))   # raster = the axes' own pixels
        im = ax.imshow(np.zeros((Hpx, W, 3), np.uint8), extent=(-R_ch * MM, R_ch * MM, -H_ch * MM, H_ch * MM),
                       origin="lower", interpolation="nearest", zorder=1)
        draw_projected_rings(ax)
        ax.set(xlim=(-R_ch * MM, R_ch * MM), ylim=(-H_ch * MM, H_ch * MM))
        title = ax.set_title("", fontsize=10)

        def raster(F):
            """Every ion as a 2x2-pixel dot coloured by energy; hotter ions drawn last (on top)."""
            img = np.zeros((Hpx, W, 3), np.uint8)
            e = F["E"]
            o = np.argsort(e, kind="stable")
            ix = (F["x"][o].astype(np.int64) * (W - 1)) // 65535
            iz = (F["z"][o].astype(np.int64) * (Hpx - 1)) // 65535
            ci = np.minimum(e[o].astype(np.int64) * float(F["E_top_eV"]) / 1e3 / E_kev_max, 255).astype(np.int64)
            col = LUT[ci]
            for dx_, dz_ in ((0, 0), (1, 0), (0, 1), (1, 1)):
                img[np.minimum(iz + dz_, Hpx - 1), np.minimum(ix + dx_, W - 1)] = col
            return img

        def full_frame(i):
            F = np.load(MFR[i])
            im.set_data(raster(F))
            tt = float(F["t"])
            title.set_text(f"t = {tt*1e9:6.0f} ns   ions shown {F['E'].size}{run_state(tt)}")
            return im, title

        save_anim(animation.FuncAnimation(fig, full_frame, frames=len(MFR), blit=False), "ions_movie")
        plt.close(fig)

print(f"Images in {out}:")
for f in sorted(os.listdir(out)):
    print("  ", f)
