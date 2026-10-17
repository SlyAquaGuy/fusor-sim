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
  ions_movie.mp4/.gif  every ion snapshot as a frame, 3D positions projected on a
                       side view, coloured by energy           (--movie)
  ion_tracks.png       paths of individual ions followed by particle id  (--movie)

The production run writes only 10 snapshots. For a smooth movie and tracks run a
short dedicated movie run (about the same cost as the timing test x 7):
    python fusor_v6.py --chamber-half-length 0.144 --t-end 1.2e-6 --t-avg 0.6e-6 \\
        --n-macro 1e5 --n-snapshots 240 --outdir movie_run
    python viz_v6.py movie_run --movie
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

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--movie", action="store_true", help="also make the ion animation and ion tracks")
ap.add_argument("--max-frame-particles", type=int, default=40000)
ap.add_argument("--n-tracks", type=int, default=60)
ap.add_argument("--fps", type=int, default=12)
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


def cross_section(data, title, label, fname, cmap="inferno", floor=1e-4):
    with plt.rc_context(DARK):
        fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        d = mirror(data)
        vmax = np.nanmax(d[np.isfinite(d)]) if np.any(d > 0) else 1.0
        im = ax.pcolormesh(xe_full, z_e * MM, np.maximum(d, vmax * floor).T, norm=LogNorm(vmax * floor, vmax),
                           cmap=cmap, shading="flat")
        draw_electrodes(ax, "#9ecbff")
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)
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
            vmax = img.max()
            im = ax.pcolormesh(xe_full, z_e * MM, np.maximum(img, vmax * 1e-3).T,
                               norm=LogNorm(vmax * 1e-3, vmax), cmap=cmap, shading="flat")
            draw_projected_rings(ax)
            ax.set(xlim=(-R_ch * MM, R_ch * MM), ylim=(-H_ch * MM, H_ch * MM), aspect="equal",
                   xlabel="horizontal (mm)", ylabel="z (mm)", title=ttl)
            fig.colorbar(im, ax=ax, label=lab, shrink=0.85)
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
# movie and tracks
# ---------------------------------------------------------------------------
if args.movie and snaps:
    if len(snaps) < 20:
        print(f"only {len(snaps)} snapshots: the movie will be choppy. See the docstring for a movie run.")
    ts = None
    tsf = os.path.join(run, "timeseries.csv")
    if os.path.exists(tsf):
        ts = np.genfromtxt(tsf, delimiter=",", names=True)
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
            extra = ""
            if ts is not None and len(ts):
                j = np.argmin(np.abs(ts["t"] - s["t"]))
                extra = f"   I = {ts['I_ema'][j]*1e3:.1f} mA   Yn = {ts['fus_ion_n_rate'][j]+ts['fus_neu_n_rate'][j]:.2e}/s"
            title.set_text(f"t = {s['t']*1e9:6.0f} ns   ions shown {len(k)}{extra}")
            return sc, title

        anim = animation.FuncAnimation(fig, frame, frames=len(snaps), blit=False)
        try:
            anim.save(os.path.join(out, "ions_movie.mp4"), writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000))
            print("wrote ions_movie.mp4")
        except Exception as ex:
            anim.save(os.path.join(out, "ions_movie.gif"), writer=animation.PillowWriter(fps=args.fps))
            print(f"ffmpeg unavailable ({ex}); wrote ions_movie.gif")
        plt.close(fig)

    # tracks: follow particle ids through consecutive snapshots
    first = load_ions(snaps[len(snaps) // 2])
    if first["id"].size:
        pick = rng.choice(first["id"], min(args.n_tracks, first["id"].size), replace=False)
        paths = {int(p): [] for p in pick}
        for f in snaps[len(snaps) // 2:]:
            s = load_ions(f)
            idx = {int(v): j for j, v in enumerate(s["id"]) if int(v) in paths}
            for p, j in idx.items():
                paths[p].append((s["r"][j] * np.sign(np.cos(s["th"][j])), s["z"][j], s["E"][j]))
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
            draw_electrodes(ax, "#9ecbff")
            for p, pts in paths.items():
                if len(pts) < 3:
                    continue
                a = np.array(pts)
                ax.plot(a[:, 0] * MM, a[:, 1] * MM, lw=0.6, alpha=0.8)
                ax.plot(a[-1, 0] * MM, a[-1, 1] * MM, "o", ms=1.5, color="white")
            ax.set(xlabel="x (mm)", ylabel="z (mm)",
                   title=f"Paths of {len(paths)} ions over {len(snaps) - len(snaps) // 2} snapshots")
            fig.savefig(os.path.join(out, "ion_tracks.png")); plt.close(fig)

print(f"Images in {out}:")
for f in sorted(os.listdir(out)):
    print("  ", f)
