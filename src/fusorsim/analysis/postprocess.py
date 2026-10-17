#!/usr/bin/env python3
"""
postprocess_v6.py -- turn a fusor_v6 run into certification numbers
====================================================================
    python postprocess_v6.py <run_dir> [--duration 15] [options]

Writes into <run_dir>/post/:
  report.md                 human-readable results with uncertainties and checks
  source_terms.json         every number OpenMC (or anyone) needs
  openmc_sources.py         importable: neutron_sources(), electron_sources(),
                            photon_sources_kramers()  (OpenMC units: cm, eV)
  *.png                     time series, fusion map, cathode heating, wire T(t)

Steady-state -> 15 s argument (see methods doc): the plasma relaxes in ~1 us,
so the PIC window average IS the 15 s source rate as long as the slow
variables (gas density, wire temperature, surface D loading) do not raise it.
Gas heating/depletion and thermionic emission only LOWER the rate at fixed
supply current, so they are bounded by ignoring them. Surface D loading
RAISES the embedded-target rate with time; it is integrated explicitly here.
"""

import argparse
import json
import os
import sys

import numpy as np

from fusorsim.physics import core as fp  # noqa: E402
from fusorsim.physics.core import _trapz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--duration", type=float, default=15.0, help="shot length, s")
ap.add_argument("--cathode-material", default="ThW", choices=list(fp.MATERIALS))
ap.add_argument("--anode-material", default="Fe", choices=list(fp.MATERIALS))
ap.add_argument("--eps-steel", type=float, default=0.2,
                help="total emissivity of the steel anode (low = conservative). Oxidised mild steel 0.6-0.8, bright 0.1-0.3")
ap.add_argument("--f-cathode-sat", type=float, default=0.4,
                help="saturation D/W in the implanted layer (cold W, keV D: 0.1-0.4 in the literature)")
ap.add_argument("--f-wall", type=float, default=0.08,
                help="D per metal atom in the implanted layer of the anode and chamber wall, saturated over many "
                     "previous shots. 12 keV D in 18Cr10NiTi steel at 295 K saturates at 7-8 at.%% above "
                     "1e17 D/cm^2 (Nanoscale Res. Lett. 11 (2016) 1251); 0.08 is that saturation value")
ap.add_argument("--surface-field", type=float, default=5e6, help="cathode surface field for Schottky lowering, V/m")
ap.add_argument("--no-plots", action="store_true")
args = ap.parse_args()

run = os.path.abspath(args.run_dir)
post = os.path.join(run, "post")
os.makedirs(post, exist_ok=True)
P = json.load(open(os.path.join(run, "run_params.json")))
ts_raw = np.genfromtxt(os.path.join(run, "timeseries.csv"), delimiter=",", names=True)
T = np.load(os.path.join(run, "tallies.npz"))
cols = ts_raw.dtype.names
ts = {k: np.asarray(ts_raw[k], dtype=float) for k in cols}
V_abs = abs(P["V_c"])
P_supply = P["P_supply"]
dur = args.duration
lines = []


def out(s=""):
    print(s)
    lines.append(s)


# =============================================================================
# 1. Steady state: window statistics, drift, convergence
# =============================================================================
win = ts["t"] >= P["t_avg"]
if win.sum() < 20:
    sys.exit("fewer than 20 callback rows inside the averaging window -- run longer")
tw = ts["t"][win]
Dtw = ts["Dt"][win]


def stats(key):
    """Time-weighted mean, batch-means standard error (10 batches; batches
    are ~0.25 us, longer than the ~0.1 us ion lifetime, so roughly
    independent), and relative drift over the window from a linear fit."""
    y = ts[key][win]
    m = np.sum(y * Dtw) / np.sum(Dtw)
    nb = 10
    idx = np.array_split(np.arange(y.size), nb)
    bm = np.array([np.sum(y[i] * Dtw[i]) / np.sum(Dtw[i]) for i in idx])
    se = bm.std(ddof=1) / np.sqrt(nb)
    slope = np.polyfit(tw - tw[0], y, 1)[0]
    drift = slope * (tw[-1] - tw[0]) / m if m != 0 else 0.0
    return m, se, drift


S = {k: stats(k) for k in cols if k not in ("t", "step", "Dt")}

out("# Fusor v6 results\n")
out(f"Run: `{run}`  ")
out(f"Operating point: {P['V_c']/1e3:.0f} kV, {P['I_target']*1e3:.0f} mA ({P_supply:.0f} W), "
    f"{P['pressure_mTorr']} mTorr D2, gamma_i = {P['gamma_i']}, reflection {'on' if P['reflection'] else 'off'}, "
    f"makeup '{P.get('makeup','shape')}', mesh {P['nr']}x{P['nz']}  ")
out(f"Averaging window: {tw[0]*1e6:.2f}-{tw[-1]*1e6:.2f} us ({win.sum()} samples)\n")

out("## 1. Steady-state and validity checks\n")
checks = []


def chk(name, ok, detail):
    checks.append((name, ok))
    out(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


Im, Ise, Idr = S["I_meas"]
chk("supply current held", abs(Im / P["I_target"] - 1) < 0.03,
    f"{Im*1e3:.2f} +/- {Ise*1e3:.2f} mA vs {P['I_target']*1e3:.1f} mA target")
for key, label in (("fus_ion_n_rate", "ion beam-gas neutron rate"), ("fus_neu_n_rate", "fast-neutral neutron rate")):
    m, se, dr = S[key]
    # a rate still FALLING through the window means the window mean over-estimates the
    # true steady state (conservative); a RISING rate would under-estimate it
    chk(f"{label} not rising", dr < 0.05,
        f"drift {dr*100:+.1f} % over window ({'falling: window mean is an upper bound' if dr < -0.05 else 'stationary' if abs(dr) < 0.05 else 'RISING: run longer'}), "
        f"stat. error {se/m*100:.1f} %")
m, se, dr = S["N_real"]
out(f"- [INFO] ion inventory drift {dr*100:+.1f} % over window")
chk("bootstrap source off in window", np.max(ts["boot_frac"][win]) == 0.0, "artificial start-up source fully faded")
cfl = np.max(ts["vmax_cfl"])
chk("particle CFL", cfl < 1.0, f"max ion displacement {cfl:.2f} cell/step")
nmac = S["n_macro"][0]
chk("macroparticle count", nmac > max(1e4, 0.3 * P["n_macro_target"]),
    f"{nmac:.3g} ions in window (target {P['n_macro_target']:.3g})")
# ion plasma frequency resolution from the time-averaged density map
r_e, z_e = T["r_edges"], T["z_edges"]
rc = 0.5 * (r_e[1:] + r_e[:-1])
Vcell = (np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2))[:, None] * np.diff(z_e)[None, :]
tW = float(T["t_window"][0])
n_ion = T["ion_wdt_rz"] / tW / Vcell
Ebar = np.where(T["ion_wdt_rz"] > 0, T["ion_wEdt_rz"] / np.maximum(T["ion_wdt_rz"], 1e-300), 0.0)
n_peak = np.max(n_ion[2:, :])          # exclude the two axis rings (volume -> 0, noisy)
wpi = np.sqrt(n_peak * fp.e**2 / (fp.eps0 * fp.m_d))
chk("ion plasma frequency resolved", wpi * P["dt"] < 0.2, f"omega_pi*dt = {wpi*P['dt']:.3g} (n_i,peak = {n_peak:.3g} m^-3)")
alpha_m = S["alpha"][0]
makeup = 1 - 1 / alpha_m if alpha_m > 1 else 0.0
out(f"- [INFO] controller alpha = {alpha_m:.3f}: "
    + (f"{makeup*100:.0f} % of ion births are makeup (ionisation not captured by the model), born "
       + ("at rest just inside the anode (bounding)" if P.get("makeup") == "anode"
          else "with the same spatial distribution as the modelled births") if alpha_m > 1 else
       f"modelled ionisation exceeds what {P['I_target']*1e3:.0f} mA needs; supply would current-limit "
       f"(holding full voltage is an upper bound)"))

# energy balance
P_in = {k: S[k][0] for k in ("P_ion_cathode", "P_ion_anode", "P_ion_wall", "P_neu_cathode", "P_neu_anode",
                             "P_neu_wall", "Pe_anode", "Pe_wall")}
P_out = sum(P_in.values())
P_drawn = V_abs * Im
chk("energy balance", abs(P_out / P_drawn - 1) < 0.10,
    f"particle power to all surfaces {P_out:.0f} W vs V x I_measured = {P_drawn:.0f} W "
    f"(difference = elastic gas heating, ions in flight, statistics)")

# =============================================================================
# 2. Neutrons
# =============================================================================
out("\n## 2. Neutron source\n")
Sn_ion, Sn_ion_se, _ = S["fus_ion_n_rate"]
Sn_neu, Sn_neu_se, _ = S["fus_neu_n_rate"]
Sp_ion = S["fus_ion_p_rate"][0]
Sp_neu = S["fus_neu_p_rate"][0]
E_e = T["E_edges"]
Ec = 0.5 * (E_e[1:] + E_e[:-1])


def impact_rate(kind, surf):
    return T[f"imp_{kind}_E__{surf}"] / tW         # particles/s per energy bin


ytt = {}
for mat in ("W", "Fe", "SS304"):
    for f in (0.01, 0.1, 0.4, 0.5, 1.0):
        ytt[(mat, f)] = np.array([fp.thick_target_yield(E, mat, f) if E > 1e3 else 0.0 for E in Ec])


def tt_rate(rate_E, mat, f):
    key = (mat, f)
    if key not in ytt:
        ytt[key] = np.array([fp.thick_target_yield(E, mat, f) if E > 1e3 else 0.0 for E in Ec])
    return float(np.sum(rate_E * ytt[key]))


# cathode: fluence-limited loading during the shot (100 % retention, half the wire area
# exposed, electronic-only range -> all three choices raise the loading = conservative)
cath_mat = "W"
rate_c = impact_rate("ion", "cathode") + impact_rate("neu", "cathode")
G_c = rate_c.sum()
E_mean_c = np.sum(rate_c * Ec) / max(G_c, 1e-30)
R_rng, n_host = fp.projected_range_upper(max(E_mean_c, 2e3), cath_mat)
A_exp = 0.5 * P["geo_c"]["area"]
t_grid = np.linspace(0, dur, 301)
f_t = np.minimum(args.f_cathode_sat, G_c * t_grid / A_exp / (n_host * R_rng))
Sn_cath_t = np.array([tt_rate(rate_c, cath_mat, max(ff, 1e-6)) if ff > 0 else 0.0 for ff in f_t])
rate_a = impact_rate("ion", "anode") + impact_rate("neu", "anode")
rate_w = impact_rate("ion", "wall") + impact_rate("neu", "wall")
Sn_anode = tt_rate(rate_a, "Fe", args.f_wall)
Sn_wall = tt_rate(rate_w, "SS304", args.f_wall)

Sn_vol = Sn_ion + Sn_neu
Sn_t = Sn_vol + Sn_anode + Sn_wall + Sn_cath_t
N15 = _trapz(Sn_t, t_grid)
Sn_max = Sn_t.max()

# beam-beam (converged core) upper estimate: all ion pairs head-on at local mean energy
bb = 0.5 * n_ion**2 * fp.sigma_DD(2 * Ebar / 1e3, "n") * 2 * fp.speed_from_energy(Ebar * fp.e, fp.m_d) * Vcell
Sn_bb = float(np.sum(bb[1:, :]))

stat_rel = np.hypot(Sn_ion_se, Sn_neu_se) / Sn_vol
out("| channel | rate (n/s) | note |")
out("|---|---|---|")
out(f"| D+ ions on D2 gas (beam-gas) | {Sn_ion:.3e} +/- {Sn_ion_se:.1e} | PIC, expected-value tally |")
out(f"| fast neutrals on D2 gas | {Sn_neu:.3e} +/- {Sn_neu_se:.1e} | CX + reflected neutrals, ray traced |")
out(f"| implanted D in cathode | {Sn_cath_t[0]:.2e} -> {Sn_cath_t[-1]:.2e} | fluence-limited loading over {dur:g} s, cap f={args.f_cathode_sat} |")
out(f"| implanted D in anode | {Sn_anode:.2e} | f = {args.f_wall} D/Fe (history-saturated) |")
out(f"| implanted D in wall | {Sn_wall:.2e} | f = {args.f_wall} D/metal |")
out(f"| ion-ion in converged core | {Sn_bb:.1e} | upper estimate, NOT added (see note) |")
out(f"\n**Peak neutron emission rate: {Sn_max:.3e} n/s** (statistical +/-{stat_rel*100:.1f} %)  ")
out(f"**Neutrons in a {dur:g} s shot: {N15:.3e}** (rate x time, with the time-dependent cathode loading integrated)  ")
out(f"D(d,p)T rate {Sp_ion + Sp_neu:.3e}/s -> tritium made per shot {((Sp_ion+Sp_neu)/max(Sn_vol,1e-30))*N15:.3e} atoms "
    f"= {((Sp_ion+Sp_neu)/max(Sn_vol,1e-30))*N15*np.log(2)/(12.32*3.156e7):.3e} Bq (negligible)  ")
out(f"Core ion-ion estimate is {Sn_bb/Sn_vol*100:.2g} % of beam-gas; it is reported, not added, because the "
    f"2D-RZ axis focusing over-states core density. If it is > 5 % treat it as an additional bound term.")
out(f"\nSensitivity of embedded-target terms to the assumed D loading f (n/s):")
out("| f | cathode (saturated) | anode | wall |")
out("|---|---|---|---|")
for f in (0.01, 0.1, 0.4, 1.0):
    out(f"| {f} | {tt_rate(rate_c, 'W', f):.2e} | {tt_rate(rate_a, 'Fe', f):.2e} | {tt_rate(rate_w, 'SS304', f):.2e} |")

# neutron energy spectrum
bins = np.linspace(2.0, 3.0, 101)
w_fus = T["fus_ion_E"] + T["fus_neu_E"]
sel = w_fus > 0
_, spec = fp.dd_neutron_spectrum(Ec[sel], w_fus[sel], bins_MeV=bins)
spec = spec / max(spec.sum(), 1e-300)
E_n_mean = np.sum(0.5 * (bins[1:] + bins[:-1]) * spec)
out(f"\nNeutron energy: mean {E_n_mean:.3f} MeV, 5-95 % range "
    f"{bins[np.searchsorted(np.cumsum(spec), 0.05)]:.2f}-{bins[np.searchsorted(np.cumsum(spec), 0.95)]:.2f} MeV "
    f"(isotropic-CM kinematics of the reacting deuterons)")
E_react_mean = np.sum(Ec * w_fus) / max(w_fus.sum(), 1e-30)
out(f"Fusion-weighted mean deuteron energy: {E_react_mean/1e3:.1f} keV")

# =============================================================================
# 3. X-rays (electrons onto grounded surfaces)
# =============================================================================
out("\n## 3. X-ray source (electrons striking grounded metal)\n")
Pe_a, Pe_w = S["Pe_anode"][0], S["Pe_wall"][0]
ea_E = T["e_anode_E"] / tW
ew_E = T["e_wall_E"] / tW
Re_a, Re_w = ea_E.sum(), ew_E.sum()
bound = fp.xray_absolute_bound(V_abs, P_supply)
eta_Fe = fp.thick_target_brems_efficiency(V_abs, 26)
out(f"Modelled: {Re_a:.3e} e/s onto the anode ({Pe_a:.0f} W), {Re_w:.3e} e/s onto the wall ({Pe_w:.0f} W)  ")
out(f"Absolute bound (independent of gamma): {bound['electron_rate']:.3e} e/s at {V_abs/1e3:.0f} keV "
    f"= {P_supply:.0f} W, every electron given the full gap energy  ")
out(f"Cross-check, thick-target bremsstrahlung in Fe: modelled {eta_Fe*(Pe_a+Pe_w):.2f} W of photons, "
    f"bound {eta_Fe*P_supply:.2f} W (Kramers, unfiltered)")
out("Production path for OpenMC: electron sources inside the wall/anode steel with "
    "`settings.electron_treatment = 'ttb'` and photon transport on.")

# =============================================================================
# 4. Wire temperatures over the shot
# =============================================================================
out("\n## 4. Grid wire heating over the shot\n")
Pc = S["P_ion_cathode"][0] + S["P_neu_cathode"][0]
Pa = S["P_ion_anode"][0] + S["P_neu_anode"][0] + Pe_a
Pa_bound = S["P_ion_anode"][0] + S["P_neu_anode"][0] + (1 - P["T_a"]) * P_supply
gc, ga = P["geo_c"], P["geo_a"]
mc = fp.MATERIALS[args.cathode_material]["rho"] * gc["volume"]
ma = fp.MATERIALS[args.anode_material]["rho"] * ga["volume"]
th_e = T["th_edges"]
pw = T["cathode_power_theta"] / tW
# peaking: power per unit WIRE area on each RZ ring vs the grid average. All rings
# have the same polar width, so ring area ~ sin(theta_ring); every angular bin is
# assigned to its nearest ring centre.
nb = P.get("cathode_bands", 11)
th_c = (np.arange(nb) + 0.5) * np.pi / nb
thb = 0.5 * (th_e[1:] + th_e[:-1])
ring = np.argmin(np.abs(thb[:, None] - th_c[None, :]), axis=1)
P_ring = np.bincount(ring, weights=pw, minlength=nb)
A_ring = np.sin(th_c)
q_ring = P_ring / A_ring
peak = float(q_ring.max() / (P_ring.sum() / A_ring.sum())) if pw.sum() > 0 else 1.0
dOmega = 2 * np.pi * (np.cos(th_e[:-1]) - np.cos(th_e[1:]))
flux = pw / dOmega
tc, Tc, Tc_ss = fp.lumped_wire_transient(Pc, args.cathode_material, gc["area"], mc, t_end=dur)
tl, Tl, Tl_ss = fp.lumped_wire_transient(Pc * peak, args.cathode_material, gc["area"], mc, t_end=dur)
ta, Ta, Ta_ss = fp.lumped_wire_transient(Pa, args.anode_material, ga["area"], ma, t_end=dur, eps_const=args.eps_steel)
tab, Tab, Tab_ss = fp.lumped_wire_transient(Pa_bound, args.anode_material, ga["area"], ma, t_end=dur,
                                            eps_const=args.eps_steel)
out(f"Cathode ({args.cathode_material}, {gc['length']:.2f} m of {P['d_c']*1e3:.1f} mm wire, {mc*1e3:.0f} g): "
    f"heat input {Pc:.0f} W ({Pc/P_supply*100:.0f} % of supply)")
out(f"- grid-average temperature at {dur:g} s: **{Tc[-1]:.0f} K** (steady state would be {Tc_ss:.0f} K)")
out(f"- angular peaking factor from PIC (RZ, azimuthally averaged): {peak:.2f} -> hottest latitude "
    f"{Tl[-1]:.0f} K at {dur:g} s (no conduction along the wire: conservative)")
out(f"- melting point {fp.MATERIALS[args.cathode_material]['T_melt']:.0f} K. 3D wire joints/hubs concentrate "
    f"flux further; apply the Maxwell 3D field-line focusing factor on top of this number.")
for mat in ("ThW", "W"):
    J = fp.richardson_current_density(Tc[-1], mat, args.surface_field)
    Il = fp.richardson_current_density(Tl[-1], mat, args.surface_field) * gc["area"]
    out(f"- thermionic emission if surface is {'activated thoriated W' if mat=='ThW' else 'pure W'}: "
        f"{J*gc['area']*1e3:.3g} mA at grid-average T, up to {Il*1e3:.3g} mA at the hottest latitude "
        f"(vs {P['I_target']*1e3:.0f} mA supply). Thermionic current displaces ion current at fixed I, so "
        f"it LOWERS neutron output; the neutron numbers above remain an upper bound.")
out(f"\nAnode ({args.anode_material}, {ga['length']:.2f} m of {P['d_a']*1e3:.1f} mm wire, {ma*1e3:.0f} g, "
    f"eps = {args.eps_steel}): heat input {Pa:.0f} W modelled, {Pa_bound:.0f} W if all supply power were electrons")
out(f"- temperature at {dur:g} s: **{Ta[-1]:.0f} K** modelled, {Tab[-1]:.0f} K bound "
    f"(melting ~{fp.MATERIALS[args.anode_material]['T_melt']:.0f} K)")

# =============================================================================
# 5. OpenMC source terms
# =============================================================================
out("\n## 5. OpenMC source terms (written to source_terms.json / openmc_sources.py)\n")
# neutron volume source on a coarse (r,z) grid
fr = 4
fus_rz = (T["fus_ion_rz"] + T["fus_neu_rz"])
NR, NZ = fus_rz.shape
fus_c = fus_rz[: NR // fr * fr, : NZ // fr * fr].reshape(NR // fr, fr, NZ // fr, fr).sum(axis=(1, 3))
rce = r_e[::fr][: NR // fr + 1]
zce = z_e[::fr][: NZ // fr + 1]
frac_vol = fus_c / max(fus_c.sum(), 1e-300)
vol_bins = [dict(r0=float(rce[i]), r1=float(rce[i + 1]), z0=float(zce[j]), z1=float(zce[j + 1]), frac=float(frac_vol[i, j]))
            for i in range(fus_c.shape[0]) for j in range(fus_c.shape[1]) if frac_vol[i, j] > 1e-5]
s_kept = sum(b["frac"] for b in vol_bins)
for b in vol_bins:
    b["frac"] /= s_kept
cath_cos = np.cos(0.5 * (th_e[1:] + th_e[:-1]))
src = dict(
    units="SI here; openmc_sources.py converts to cm",
    duration_s=dur,
    neutron=dict(
        peak_rate_per_s=float(Sn_max), shot_total=float(N15),
        components_per_s=dict(volume_beam_gas=float(Sn_vol), cathode_embedded_end_of_shot=float(Sn_cath_t[-1]),
                              anode_embedded=float(Sn_anode), wall_embedded=float(Sn_wall),
                              core_ion_ion_estimate_not_added=float(Sn_bb)),
        statistical_rel_error=float(stat_rel),
        energy_bins_MeV=bins.tolist(), energy_pdf=spec.tolist(),
        volume_bins=vol_bins,
        cathode_shell=dict(radius=P["R_c"], frac_of_cathode_term_vs_cos_theta=dict(
            cos=cath_cos.tolist(), w=(pw / max(pw.sum(), 1e-300)).tolist())),
        wall=dict(radius=P["R_ch"], half_length=P["H_ch"]),
    ),
    electrons=dict(
        energy_bins_eV=E_e.tolist(),
        anode=dict(rate_per_s=float(Re_a), radius=P["R_a"], spectrum=(ea_E / max(Re_a, 1e-300)).tolist()),
        wall=dict(rate_per_s=float(Re_w), spectrum=(ew_E / max(Re_w, 1e-300)).tolist(),
                  side_z_edges=T["wz_edges"].tolist(), side_z_rate=(T["e_wall_side_z"] / tW).tolist(),
                  cap_r_edges=T["wr_edges"].tolist(), cap_lo_r_rate=(T["e_wall_cap_r"][0] / tW).tolist(),
                  cap_hi_r_rate=(T["e_wall_cap_r"][1] / tW).tolist()),
        absolute_bound=dict(rate_per_s=float(bound["electron_rate"]), energy_eV=float(V_abs)),
    ),
    geometry=dict(R_c=P["R_c"], R_a=P["R_a"], R_ch=P["R_ch"], H_ch=P["H_ch"], T_c=P["T_c"], T_a=P["T_a"]),
    run=dict(V=P["V_c"], I=P["I_target"], p_mTorr=P["pressure_mTorr"], gamma=P["gamma_i"], reflection=P["reflection"]),
)
json.dump(src, open(os.path.join(post, "source_terms.json"), "w"), indent=1)
out(f"Neutron: {len(vol_bins)} volume bins + cathode shell + wall terms; electrons: anode shell + "
    f"{len(T['wz_edges'])-1} wall-side and 2x{len(T['wr_edges'])-1} end-cap bins; Kramers photon alternative.")

OPENMC_PY = r'''"""
openmc_sources.py -- generated by postprocess_v6.py. Do not edit numbers here;
re-run the post-processor. Units: OpenMC uses cm and eV.

    import openmc, openmc_sources as fs
    settings.source = fs.neutron_sources(scenario="peak")       # n/s strength
    # photons via electrons (production):
    settings.source = fs.electron_sources(bound=False)          # e/s strength
    settings.photon_transport = True
    settings.electron_treatment = "ttb"
    # cross-check without TTB:
    settings.source = fs.photon_sources_kramers(bound=False)

Source strengths are in particles per second, so tally results (per source
particle) multiplied by total_strength(sources) give per-second quantities;
multiply neutron results by the shot total / peak rate ratio for per-shot dose.
TTB electrons deposit where they are born, so electron sources are placed
10 um INSIDE the steel: your geometry must have steel there (wall inner
radius R_ch, and an anode shell -- model the anode as a homogenised steel
shell of the wire diameter's thickness at density (1-T_a)*rho; thick-target
bremsstrahlung yield does not depend on density).
"""
import json, os
import numpy as np
import openmc
import openmc.stats as st

_D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "source_terms.json")))
CM = 100.0
TWO_PI = 2 * np.pi


def total_strength(sources):
    return sum(s.strength for s in sources)


def _nE():
    b = np.array(_D["neutron"]["energy_bins_MeV"]) * 1e6
    p = np.array(_D["neutron"]["energy_pdf"])
    return st.Tabular(b, np.append(p, 0.0), interpolation="histogram")


def neutron_sources(scenario="peak"):
    """scenario 'peak': rate at end of shot (max); 'average': shot total / duration."""
    n = _D["neutron"]
    comp = n["components_per_s"]
    S_peak = n["peak_rate_per_s"]
    scale = 1.0 if scenario == "peak" else n["shot_total"] / _D["duration_s"] / S_peak
    E = _nE()
    out = []
    for b in n["volume_bins"]:
        sp = st.CylindricalIndependent(r=st.PowerLaw(b["r0"] * CM, b["r1"] * CM, 1), phi=st.Uniform(0, TWO_PI),
                                       z=st.Uniform(b["z0"] * CM, b["z1"] * CM))
        out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(), energy=E,
                                            strength=scale * comp["volume_beam_gas"] * b["frac"]))
    c = n["cathode_shell"]
    mu = np.array(c["frac_of_cathode_term_vs_cos_theta"]["cos"])[::-1]
    w = np.array(c["frac_of_cathode_term_vs_cos_theta"]["w"])[::-1]
    if w.sum() > 0:
        sp = st.SphericalIndependent(r=st.Discrete([c["radius"] * CM], [1.0]),
                                     cos_theta=st.Discrete(mu, w / w.sum()), phi=st.Uniform(0, TWO_PI))
        out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(), energy=E,
                                            strength=scale * comp["cathode_embedded_end_of_shot"]))
    wl = n["wall"]
    sp = st.CylindricalIndependent(r=st.Discrete([wl["radius"] * CM - 1e-3], [1.0]), phi=st.Uniform(0, TWO_PI),
                                   z=st.Uniform(-wl["half_length"] * CM, wl["half_length"] * CM))
    out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(), energy=E,
                                        strength=scale * (comp["wall_embedded"])))
    a = _D["geometry"]
    sp = st.SphericalIndependent(r=st.Discrete([a["R_a"] * CM], [1.0]), cos_theta=st.Uniform(-1, 1),
                                 phi=st.Uniform(0, TWO_PI))
    out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(), energy=E,
                                        strength=scale * comp["anode_embedded"]))
    return [s for s in out if s.strength > 0]


def _espec(spec):
    b = np.array(_D["electrons"]["energy_bins_eV"])
    p = np.array(spec)
    return st.Tabular(b, np.append(p, 0.0), interpolation="histogram")


def _wall_bins(scale, make_energy, particle, inside, n_phi=16, outward_only_angle=None):
    e = _D["electrons"]; g = _D["geometry"]
    out = []
    R, Hh = g["R_ch"] * CM, g["H_ch"] * CM
    ze = np.array(e["wall"]["side_z_edges"]) * CM
    zr = np.array(e["wall"]["side_z_rate"])
    phis = np.linspace(0, TWO_PI, n_phi + 1)
    for i in range(len(zr)):
        if zr[i] <= 0:
            continue
        for k in range(n_phi):
            pc = 0.5 * (phis[k] + phis[k + 1])
            nrm = (np.cos(pc), np.sin(pc), 0.0)
            sp = st.CylindricalIndependent(r=st.Discrete([R + (1e-3 if inside else -1e-3)], [1.0]),
                                           phi=st.Uniform(phis[k], phis[k + 1]), z=st.Uniform(ze[i], ze[i + 1]))
            ang = st.Monodirectional(nrm) if outward_only_angle is None else outward_only_angle(nrm)
            out.append(openmc.IndependentSource(space=sp, angle=ang, energy=make_energy(),
                                                strength=scale * zr[i] / n_phi, particle=particle))
    re_ = np.array(e["wall"]["cap_r_edges"]) * CM
    for key, zc, nz in (("cap_lo_r_rate", -Hh, -1.0), ("cap_hi_r_rate", Hh, 1.0)):
        rr = np.array(e["wall"][key])
        for i in range(len(rr)):
            if rr[i] <= 0:
                continue
            sp = st.CylindricalIndependent(r=st.PowerLaw(re_[i], re_[i + 1], 1), phi=st.Uniform(0, TWO_PI),
                                           z=st.Discrete([zc + nz * (1e-3 if inside else -1e-3)], [1.0]))
            nrm = (0.0, 0.0, nz)
            ang = st.Monodirectional(nrm) if outward_only_angle is None else outward_only_angle(nrm)
            out.append(openmc.IndependentSource(space=sp, angle=ang, energy=make_energy(),
                                                strength=scale * rr[i], particle=particle))
    return out


def electron_sources(bound=False):
    """Electrons born inside the steel with their impact energy, moving
    outward (normal incidence). bound=True rescales to the absolute bound
    (all supply power as full-energy electrons)."""
    e = _D["electrons"]; g = _D["geometry"]
    tot = e["anode"]["rate_per_s"] + e["wall"]["rate_per_s"]
    if bound:
        b = np.array(e["energy_bins_eV"])
        mono = np.zeros(len(b) - 1); mono[np.searchsorted(b, e["absolute_bound"]["energy_eV"]) - 1] = 1.0
        mk = lambda: _espec(mono)
        scale = e["absolute_bound"]["rate_per_s"] / tot
    else:
        mk = lambda: _espec(e["wall"]["spectrum"])
        scale = 1.0
    out = _wall_bins(scale * e["wall"]["rate_per_s"] / max(sum(e["wall"]["side_z_rate"]) +
                     sum(e["wall"]["cap_lo_r_rate"]) + sum(e["wall"]["cap_hi_r_rate"]), 1e-300),
                     mk, "electron", inside=True)
    sp = st.SphericalIndependent(r=st.Discrete([g["R_a"] * CM], [1.0]), cos_theta=st.Uniform(-1, 1),
                                 phi=st.Uniform(0, TWO_PI))
    # radial-outward direction on a sphere cannot be expressed globally; for the
    # thin anode shell the incidence direction is immaterial to TTB yield.
    out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(),
                                        energy=mk() if bound else _espec(e["anode"]["spectrum"]),
                                        strength=scale * e["anode"]["rate_per_s"], particle="electron"))
    return [s for s in out if s.strength > 0]


def photon_sources_kramers(bound=False, Z=26, n_bins=120):
    """Cross-check: thick-target Kramers photons (unfiltered) emitted into
    the outward hemisphere from the wall and anode surfaces."""
    e = _D["electrons"]
    b = np.array(e["energy_bins_eV"]); bc = 0.5 * (b[1:] + b[:-1])
    if bound:
        Eel = np.array([e["absolute_bound"]["energy_eV"]]); Rel = np.array([e["absolute_bound"]["rate_per_s"]])
    else:
        Rel_w = np.array(e["wall"]["spectrum"]) * e["wall"]["rate_per_s"]
        Rel_a = np.array(e["anode"]["spectrum"]) * e["anode"]["rate_per_s"]
        Rel = Rel_w + Rel_a; Eel = bc
    kb = np.linspace(1e3, Eel.max(), n_bins + 1); kc = 0.5 * (kb[1:] + kb[:-1])
    ph = np.zeros(n_bins)
    for E0, R in zip(Eel, Rel):
        if R <= 0 or E0 <= 1e3:
            continue
        shape = np.maximum(E0 - kc, 0) / kc
        Esum = np.sum(shape * kc * np.diff(kb))
        if Esum <= 0:          # electron below the first photon bin: no photon above 1 keV
            continue
        norm = 1.1e-9 * Z * E0 * E0 / Esum
        ph += R * shape * norm * np.diff(kb)
    total = ph.sum()
    if total <= 0:
        return []
    mk = lambda: st.Tabular(kb, np.append(ph / total, 0.0), interpolation="histogram")
    hemi = lambda nrm: st.PolarAzimuthal(mu=st.Uniform(0.0, 1.0), phi=st.Uniform(0, TWO_PI), reference_uvw=nrm)
    wall_frac = e["wall"]["rate_per_s"] / (e["wall"]["rate_per_s"] + e["anode"]["rate_per_s"])
    norm_rate = sum(e["wall"]["side_z_rate"]) + sum(e["wall"]["cap_lo_r_rate"]) + sum(e["wall"]["cap_hi_r_rate"])
    out = _wall_bins(total * wall_frac / max(norm_rate, 1e-300), mk, "photon", inside=False, outward_only_angle=hemi)
    g = _D["geometry"]
    sp = st.SphericalIndependent(r=st.Discrete([g["R_a"] * CM], [1.0]), cos_theta=st.Uniform(-1, 1),
                                 phi=st.Uniform(0, TWO_PI))
    out.append(openmc.IndependentSource(space=sp, angle=st.Isotropic(), energy=mk(),
                                        strength=total * (1 - wall_frac), particle="photon"))
    return [s for s in out if s.strength > 0]
'''
open(os.path.join(post, "openmc_sources.py"), "w").write(OPENMC_PY)

# =============================================================================
# 6. Summary table
# =============================================================================
out("\n## 6. Headline numbers\n")
out("| quantity | value |")
out("|---|---|")
out(f"| peak neutron rate | {Sn_max:.2e} n/s |")
out(f"| neutrons per {dur:g} s shot | {N15:.2e} |")
out(f"| electrons onto ground (modelled / bound) | {Re_a+Re_w:.2e} / {bound['electron_rate']:.2e} e/s |")
out(f"| bremsstrahlung power, Fe thick target (modelled / bound) | {eta_Fe*(Pe_a+Pe_w):.2f} / {eta_Fe*P_supply:.2f} W |")
out(f"| cathode T at {dur:g} s (average / hottest latitude) | {Tc[-1]:.0f} / {Tl[-1]:.0f} K |")
out(f"| anode T at {dur:g} s (modelled / bound) | {Ta[-1]:.0f} / {Tab[-1]:.0f} K |")
out(f"| tritium per shot | {((Sp_ion+Sp_neu)/max(Sn_vol,1e-30))*N15:.1e} atoms |")
n_fail = sum(1 for _, ok in checks if not ok)
out(f"\nChecks: {len(checks)-n_fail}/{len(checks)} passed." + (" Resolve FAILs before using these numbers." if n_fail else ""))
open(os.path.join(post, "report.md"), "w").write("\n".join(lines) + "\n")

# =============================================================================
# 7. Plots
# =============================================================================
if not args.no_plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    tt = ts["t"] * 1e6
    ax[0, 0].plot(tt, ts["I_meas"] * 1e3, lw=0.5, label="I supply (inst.)")
    ax[0, 0].plot(tt, ts["I_ema"] * 1e3, label="I supply (EMA)")
    ax[0, 0].axhline(P["I_target"] * 1e3, color="k", ls=":")
    ax[0, 0].set(xlabel="t (us)", ylabel="mA"); ax[0, 0].legend()
    ax[0, 1].semilogy(tt, ts["fus_ion_n_rate"], lw=0.6, label="ions")
    ax[0, 1].semilogy(tt, ts["fus_neu_n_rate"], lw=0.6, label="fast neutrals")
    ax[0, 1].axvline(P["t_avg"] * 1e6, color="k", ls=":")
    ax[0, 1].set(xlabel="t (us)", ylabel="D(d,n) rate (1/s)"); ax[0, 1].legend()
    ax[1, 0].plot(tt, ts["alpha"]); ax[1, 0].set(xlabel="t (us)", ylabel="controller alpha")
    ax[1, 1].plot(tt, ts["n_macro"]); ax[1, 1].set(xlabel="t (us)", ylabel="ion macroparticles")
    fig.savefig(os.path.join(post, "timeseries.png"), dpi=130)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    dens = fus_rz / Vcell / tW
    im = ax[0].pcolormesh(z_e * 1e3, r_e * 1e3, np.log10(np.maximum(dens, dens.max() * 1e-4)), shading="flat")
    fig.colorbar(im, ax=ax[0], label="log10 neutron emission density (n/s/m^3)")
    ax[0].set(xlabel="z (mm)", ylabel="r (mm)", aspect="equal", title="D(d,n) emission (beam-gas)")
    im = ax[1].pcolormesh(z_e * 1e3, r_e * 1e3, np.log10(np.maximum(n_ion, n_peak * 1e-4)), shading="flat")
    fig.colorbar(im, ax=ax[1], label="log10 n_i (m^-3)")
    ax[1].set(xlabel="z (mm)", ylabel="r (mm)", aspect="equal", title="time-averaged D+ density")
    fig.savefig(os.path.join(post, "fusion_map.png"), dpi=130)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(tc, Tc, label="cathode, grid average"); ax[0].plot(tl, Tl, label="cathode, hottest latitude")
    ax[0].plot(ta, Ta, label="anode (modelled)"); ax[0].plot(tab, Tab, ":", label="anode (bound)")
    ax[0].set(xlabel="t (s)", ylabel="T (K)"); ax[0].legend()
    ax[1].bar(np.degrees(th_c), q_ring / (P_ring.sum() / A_ring.sum()), width=0.6 * 180 / nb)
    ax[1].set(xlabel="ring polar angle (deg)", ylabel="cathode ring heat flux / grid mean")
    fig.savefig(os.path.join(post, "wire_heating.png"), dpi=130)

print(f"\nWrote {post}/report.md, source_terms.json, openmc_sources.py")
