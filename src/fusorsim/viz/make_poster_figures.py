#!/usr/bin/env python3
"""
make_poster_figures.py -- the four 2D-RZ figures of the A2 research poster, at their printed size
=================================================================================================
    python make_poster_figures.py <run_dir> [--out poster_figs]

Reads the same run outputs as render.py (run_params.json, tallies.npz, diags/fields*, diags/ions_snap*);
nothing is re-simulated. Writes <out>/fig3a_potential, fig3b_phase_space, fig3c_neutron_emission and
fig3d_deuteron_energy, each as .pdf (vector) and .png (600 dpi).

Every figure is made at the 3.7 x 3.6 in of its poster tile, so the font sizes are the printed sizes.
White background, render.py's colour maps (magma potential, turbo phase space, inferno emission), no
titles (the poster captions carry them).
Maps: the 2D-RZ axis band (render.py's hatched first 3 radial bins) is left out of the colour scale -- RZ
focuses everything aimed at the centre onto the axis, overstating it -- so it saturates at the top colour.
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fusorsim.physics import core as fp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--out", default="poster_figs", help="output folder (default ./poster_figs)")
ap.add_argument("--inset", action="store_true", help="fig 3c: add the angle-averaged emission-vs-radius inset")
args = ap.parse_args()

run = os.path.abspath(args.run_dir)
out = os.path.abspath(args.out)
os.makedirs(out, exist_ok=True)
P = json.load(open(os.path.join(run, "run_params.json")))
T = np.load(os.path.join(run, "tallies.npz"))
R_ch, H_ch, R_c, R_a = P["R_ch"], P["H_ch"], P["R_c"], P["R_a"]
MM = 1e3
LIM = 140.0                 # mm, common extent of the three maps (chamber is 144 mm)
BLUE = "#006DAE"            # the poster's blue

plt.rcParams.update({
    "figure.figsize": (3.7, 3.6),        # inches, the printed tile
    "savefig.dpi": 600, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,                  # TrueType in the PDF: text stays text for the printer
})
# each map keeps render.py's colour map, as on the earlier poster drafts
CMAP_PHI, CMAP_PHASE, CMAP_EMIS = "magma", "turbo", "inferno"
WIRE = dict(ls="none", marker="o", ms=2.4, mfc="white", mec="#333333", mew=0.5, zorder=4)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def sci(x, digits=1):
    """2.655e6 -> '2.7\\times10^{6}' for mathtext."""
    n = int(np.floor(np.log10(abs(x))))
    return f"{x / 10**n:.{digits}f}\\times10^{{{n}}}"


# ---------------------------------------------------------------------------
# geometry: wire cross-sections of the latitude-ring grid model (as render.py / the driver)
# ---------------------------------------------------------------------------
def ring_centres(T_, nb):
    return [(i + 0.5) * np.pi / nb for i in range(nb)]


def draw_wires(ax):
    """Each 3D great-circle wire is a set of latitude rings in 2D-RZ; a ring crosses the x-z plane at
    (+-R sin c, R cos c)."""
    for R, T_, nb in ((R_c, P["T_c"], P.get("cathode_bands", 11)), (R_a, P["T_a"], P.get("anode_bands", 5))):
        c = np.array(ring_centres(T_, nb))
        x, z = R * np.sin(c) * MM, R * np.cos(c) * MM
        ax.plot(np.r_[x, -x], np.r_[z, z], **WIRE)


def map_axes(ax):
    ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), xlabel="x (mm)", ylabel="z (mm)")
    ax.set_aspect("equal")
    ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_yticks([-100, -50, 0, 50, 100])


def mirror(a):
    """(nr, nz) half-plane array -> (2nr, nz) full cross-section."""
    return np.concatenate([a[::-1], a], axis=0)


# ---------------------------------------------------------------------------
# tallied maps (as render.py)
# ---------------------------------------------------------------------------
r_e, z_e = T["r_edges"], T["z_edges"]
tW = float(T["t_window"][0])
Vcell = (np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2))[:, None] * np.diff(z_e)[None, :]   # exact annulus volume
emis = (T["fus_ion_rz"] + T["fus_neu_rz"]) / max(tW, 1e-30) / Vcell                 # n / s / m^3
# 2D-RZ focuses everything aimed at the centre onto the axis and the 1/r cell volume amplifies it, so the
# first few radial bins are far brighter than a 3D device. Same band render.py hatched.
AXIS_R = r_e[min(3, len(r_e) - 1)]
xe_full = np.concatenate([-r_e[::-1], r_e[1:]]) * MM
xc_full = 0.5 * (xe_full[1:] + xe_full[:-1])
OFF_AXIS = np.abs(xc_full) >= AXIS_R * MM


# ---------------------------------------------------------------------------
# fig 3a: time-averaged potential (fields snapshots in the steady-state window, as render.py)
# ---------------------------------------------------------------------------
import yt  # noqa: E402
yt.set_log_level(50)


def load_field(path, name):
    ds = yt.load(path)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    return np.array(cg["boxlib", name])[:, :, 0], ds


diags = os.path.join(run, "diags")
use = []
for f in sorted(glob.glob(os.path.join(diags, "fields0*"))):
    ph, ds = load_field(f, "phi")
    if float(ds.current_time) >= P["t_avg"] * 0.999:
        use.append(ph)
phi = np.mean(use, axis=0)
nr, nz = phi.shape
re_ = np.linspace(0, R_ch, nr + 1); ze_ = np.linspace(-H_ch, H_ch, nz + 1)
rcn = 0.5 * (re_[1:] + re_[:-1]); zcn = 0.5 * (ze_[1:] + ze_[:-1])
full = mirror(phi) / 1e3                                            # kV
xc = np.concatenate([-rcn[::-1], rcn]) * MM
V_core = phi[0, nz // 2] / 1e3
V_C = P["V_c"] / 1e3

fig, ax = plt.subplots(constrained_layout=True)
im = ax.pcolormesh(np.concatenate([-re_[::-1], re_[1:]]) * MM, ze_ * MM, full.T, vmin=-60, vmax=0,
                   cmap=CMAP_PHI, shading="flat", rasterized=True)
cs = ax.contour(xc, zcn * MM, full.T, levels=np.arange(-60, 0, 10), colors="white", linewidths=0.5,
                alpha=0.6)                                            # render.py's contour colours
ax.clabel(cs, fontsize=7.5, fmt=lambda v: f"{v:.0f}".replace("-", "−"), inline=True)
draw_wires(ax)
ax.text(0, 0, f"{V_core:.1f} kV".replace("-", "−"), ha="center", va="center", color="white", fontsize=9)
map_axes(ax)
cb = fig.colorbar(im, ax=ax, label="Potential (kV)", shrink=0.8, pad=0.02, ticks=np.arange(-60, 1, 10))
save(fig, "fig3a_potential")

# ---------------------------------------------------------------------------
# fig 3b: ion phase space of the last snapshot (as render.py), inside the anode only
# ---------------------------------------------------------------------------
def load_ions(path):
    ds = yt.load(path)
    ad = ds.all_data()
    g = lambda k: np.array(ad["ions", k])
    return dict(t=float(ds.current_time), r=g("particle_position_x"), z=g("particle_position_y"),
                th=g("particle_theta"), w=g("particle_weight"),
                px=g("particle_momentum_x"), py=g("particle_momentum_y"), pz=g("particle_momentum_z"))


snaps = sorted(glob.glob(os.path.join(diags, "ions_snap*")))
s = load_ions(snaps[-1])
rs = np.hypot(s["r"], s["z"])
pr = s["px"] * np.cos(s["th"]) + s["py"] * np.sin(s["th"])
vr = (pr * s["r"] + s["pz"] * s["z"]) / np.maximum(rs, 1e-9) / fp.m_d      # spherical radial velocity
inside = rs <= R_a
V_TOP = np.sqrt(2 * abs(P["V_c"]) * fp.e / fp.m_d) / 1e6                    # 10^6 m/s for e|V_c|
VLIM = np.ceil(max(V_TOP, np.abs(vr[inside]).max() / 1e6) * 10 + 1) / 10    # no ion clipped off the plot

fig, ax = plt.subplots(constrained_layout=True)
h = ax.hist2d(rs[inside] * MM, vr[inside] / 1e6, bins=[280, 240], range=[[0, LIM], [-VLIM, VLIM]],
              weights=s["w"][inside], norm=LogNorm(), cmap=CMAP_PHASE, rasterized=True)   # empty bins stay white
for R, lab in ((R_c, "cathode"), (R_a, "anode")):
    ax.axvline(R * MM, color="#444444", lw=0.7, ls="--")
    ax.text(R * MM, 1.01, lab, color="#444444", fontsize=7, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())                       # above the frame, clear of the data
ax.axhline(0, color="#999999", lw=0.4)
ax.set(xlim=(0, LIM), ylim=(-VLIM, VLIM), xlabel="Radius (mm)", ylabel="Radial velocity (10$^6$ m/s)")
ax.set_box_aspect(1)
# right axis: the deuteron kinetic energy that radial speed alone carries, E = m_D v^2 / 2
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim()); ax2.set_box_aspect(1)
E_t = np.array([10, 20, 40, 60])
v_t = np.sqrt(2 * E_t * 1e3 * fp.e / fp.m_d) / 1e6
ax2.set_yticks(np.r_[-v_t[::-1], 0, v_t]); ax2.set_yticklabels([*map(str, E_t[::-1]), "0", *map(str, E_t)])
ax2.set_ylabel("Deuteron energy (keV)")
fig.colorbar(h[3], ax=[ax, ax2], label="Real ions per bin", shrink=0.8, pad=0.04)
save(fig, "fig3b_phase_space")
v_max_in = np.abs(vr[inside]).max()

# ---------------------------------------------------------------------------
# fig 3c: where the D(d,n) neutrons are made
# ---------------------------------------------------------------------------
Y_all = float(np.sum(emis * Vcell))                                 # n/s, whole RZ volume
Y_off = float(np.sum((emis * Vcell)[r_e[:-1] >= AXIS_R]))           # axis band excluded

d = mirror(emis)
vmax = np.nanmax(np.where(OFF_AXIS[:, None] & (d > 0), d, np.nan))
vmin = vmax * 1e-4
d_show = np.where(d > 0, np.clip(d, vmin, None), np.nan)            # no reactions -> white

fig, ax = plt.subplots(constrained_layout=True)
# axis cells drawn as they are: above vmax they saturate at the top colour (colour bar's "max" arrow)
im = ax.pcolormesh(xe_full, z_e * MM, d_show.T, norm=LogNorm(vmin, vmax), cmap=CMAP_EMIS,
                   shading="flat", rasterized=True)
draw_wires(ax)
# bottom-left corner, clear of the wire markers (lowest anode ring is at |x| = 40 mm)
ax.text(0.015, 0.015, f"Gas-volume\ntotal:\n${sci(Y_all)}$ n/s", transform=ax.transAxes, fontsize=7,
        ha="left", va="bottom", zorder=6, bbox=dict(fc="white", ec="none", pad=1.5))
map_axes(ax)
fig.colorbar(im, ax=ax, label="Emission (n s$^{-1}$ m$^{-3}$)", shrink=0.8, pad=0.02, extend="max")
if args.inset:
    # emission against distance from the centre, averaged over angle, off-axis cells only
    rc_, zc_ = 0.5 * (r_e[1:] + r_e[:-1]), 0.5 * (z_e[1:] + z_e[:-1])
    RHO = np.hypot(rc_[:, None], zc_[None, :]) * MM
    off = (r_e[:-1] >= AXIS_R)[:, None] & np.ones_like(RHO, bool)
    bins = np.arange(0, LIM + 4, 4.0)
    num = np.histogram(RHO[off], bins, weights=(emis * Vcell)[off])[0]
    den = np.histogram(RHO[off], bins, weights=np.broadcast_to(Vcell, RHO.shape)[off])[0]
    prof = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    ins = ax.inset_axes([0.62, 0.66, 0.35, 0.31], zorder=7)
    ins.semilogy(0.5 * (bins[1:] + bins[:-1]), prof, color=BLUE, lw=0.9)
    for R in (R_c, R_a):
        ins.axvline(R * MM, color="#444444", lw=0.5, ls="--")
    ins.set_xlim(0, LIM); ins.set_xticks([0, 69, 129]); ins.tick_params(labelsize=6, length=2, pad=1)
    ins.set_xlabel("r (mm)", fontsize=6, labelpad=0)
save(fig, "fig3c_neutron_emission")

# ---------------------------------------------------------------------------
# fig 3d: which deuterons make the neutrons (energy at reaction, fraction per keV)
# ---------------------------------------------------------------------------
Ee = T["E_edges"] / 1e3; Ec = 0.5 * (Ee[1:] + Ee[:-1])
wf = T["fus_ion_E"] + T["fus_neu_E"]
pdf = wf / (wf.sum() * np.diff(Ee))                                 # area = 1
E_peak = Ec[np.argmax(pdf)]

fig, ax = plt.subplots(constrained_layout=True)
ax.stairs(pdf, Ee, color=BLUE, lw=1.1)
ytop = pdf[Ec <= 70].max() * 1.08
for E, lab, side in ((-V_core, "core potential", "right"), (abs(V_C), "cathode bias", "left")):
    ax.axvline(E, color="#444444", lw=0.7, ls="--")
    ax.text(E + (-0.8 if side == "right" else 0.8), ytop * 0.97, lab, rotation=90, fontsize=12, color="#444444",
            ha=side, va="top")
ax.set(xlim=(0, 70), ylim=(0, ytop))
# larger axis text than the maps (no colour bar here to make room for); y label on two lines so it fits
ax.set_xlabel("Deuteron energy at reaction (keV)", fontsize=12)
ax.set_ylabel("Fraction of D(d,n)\nreactions per keV", fontsize=12)
ax.tick_params(labelsize=10.5)
save(fig, "fig3d_deuteron_energy")

# ---------------------------------------------------------------------------
print()
ts = np.genfromtxt(os.path.join(run, "timeseries.csv"), delimiter=",", names=True)
win = ts["t"] >= P["t_avg"]
Y_ts = float(np.mean(ts["fus_ion_n_rate"][win] + ts["fus_neu_n_rate"][win]))
print(f"Neutron yield, emission map integrated over the 3D volume (sum emis * 2 pi r dr dz):")
print(f"  all cells              {Y_all:.4e} n/s")
print(f"  axis band (r < {AXIS_R*MM:.0f} mm) excluded {Y_off:.4e} n/s  -> the band carries "
      f"{(Y_all - Y_off) / Y_all * 100:.2f} %")
print(f"  cross-check, timeseries beam-gas + fast-neutral rate over the window: {Y_ts:.4e} n/s")
print(f"  (gas-volume channels only: the run's 1.7e7 n/s peak is mostly D implanted in the wall/grids,"
      f" a surface term not in this map)")
print(f"Peak deuteron energy at reaction (fig 3d): {E_peak:.1f} keV   "
      f"(fusion-weighted mean {np.sum(Ec * wf) / wf.sum():.1f} keV)")
print(f"Max radial ion speed inside the anode (fig 3b, t = {s['t']*1e6:.2f} us): {v_max_in:.3e} m/s "
      f"= {0.5 * fp.m_d * v_max_in**2 / fp.e / 1e3:.1f} keV;  all ions {np.abs(vr).max():.3e} m/s")
print(f"Core potential (fig 3a, {len(use)} snapshots averaged): {V_core:.1f} kV")
