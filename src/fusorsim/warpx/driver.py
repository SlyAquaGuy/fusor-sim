#!/usr/bin/env python3
"""
fusor_v6.py -- steady-state source-term simulation of the Monash IEC fusor
==========================================================================
WarpX 26.01 (MPI + OpenMP), 2D-RZ electrostatic PIC. Run under srun/mpirun for MPI;
each rank owns its WarpX boxes AND runs the Python physics on its own particles.

WHAT THIS RUN PRODUCES (all written to --outdir):
  timeseries.csv      per-interval rates: currents, fusion rates by channel,
                      CX / ionisation / injection rates, power to each surface,
                      controller state, validity monitors
  tallies.npz         steady-state-window histograms: fusion locations (r,z),
                      reacting-projectile energies, ion density/energy maps,
                      surface impact energy spectra, cathode angular power,
                      electron impact spectra & wall positions
  run_params.json     every input and derived quantity
  tracks.npz          --track-n ions followed every callback (~1.5 ns) over the
                      last --track-len of the run: positions and energies (viz)
  diags/             WarpX field snapshots, time-averaged phi/rho, reduced
                      diagnostics, checkpoints (restart with --restart)
Then run  postprocess_v6.py <outdir>  for the 15 s totals, OpenMC source
terms, wire temperatures and certification numbers.

MODEL (full justification in the methods document):
  * D+ ions are PIC macroparticles in the self-consistent electrostatic field
    of the real 2D-RZ electrode geometry (transparency-matched shells).
  * Electrons are NOT PIC particles. Their space charge is < ~2 % of the ion
    space charge (they leave the gap at ~1e8 m/s), so they are treated
    analytically: secondary electrons from cathode impacts fly radially to
    ground with e|V| (X-ray source, anode heating, ~0.2 % ionisation each).
  * D+ / D2 elastic scattering: WarpX MCC (background_mass = m_D2 set
    explicitly -- WarpX otherwise uses the ion mass; v5 bug).
  * Charge exchange, ion-impact ionisation, beam-gas D-D fusion: Python
    callback every --ncb steps, Monte Carlo (CX, ionisation) and expected-
    value (fusion) estimators.
  * Fast neutrals (from CX and from reflection at surfaces) are ray traced
    in 3D: grid interception, stripping (re-injected as fast ions),
    target ionisation, beam-gas fusion along the path, wall/wire impact.
  * The supply current I = e[(1+gamma) Gamma_ion,cathode + gamma_n
    Gamma_neutral,cathode] is held at --current by scaling the physically
    sampled ion-birth set by a controller factor alpha (reported). alpha>1
    means the modelled ionisation is insufficient and the missing ions are
    assumed to be born with the same spatial distribution (makeup); alpha<1
    means the discharge would draw more than the set current at this V and
    pressure (the supply would current-limit; holding V is then an upper
    bound).

Run:   python fusor_v6.py --preflight          (seconds; checks everything)
       python fusor_v6.py --outdir run_5mTorr   (production)
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

from fusorsim.physics import core as fp  # noqa: E402
from fusorsim.warpx import mpi as fmpi  # noqa: E402
from fusorsim.warpx import tracks as ftracks  # noqa: E402

RANK, SIZE = fmpi.RANK, fmpi.SIZE
ROOT = RANK == 0

# =============================================================================
# Inputs
# =============================================================================
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
g = ap.add_argument_group("operating point")
g.add_argument("--voltage", type=float, default=-60e3, help="cathode voltage, V (negative)")
g.add_argument("--current", type=float, default=0.050, help="supply current held by controller, A")
g.add_argument("--pressure", type=float, default=5.0, help="TRUE D2 fill pressure, mTorr")
g.add_argument("--gas-temp", type=float, default=300.0, help="gas temperature, K")
g = ap.add_argument_group("geometry (user spec, 2025)")
g.add_argument("--chamber-radius", type=float, default=0.144, help="inner radius of grounded SS cylinder, m")
g.add_argument("--chamber-half-length", type=float, default=0.144,
               help="half inner length of the cylinder, m  ** NOT YET CONFIRMED -- set this **")
g.add_argument("--cathode-radius", type=float, default=0.069)
g.add_argument("--cathode-wire", type=float, default=0.002, help="cathode wire diameter, m")
g.add_argument("--anode-radius", type=float, default=0.129)
g.add_argument("--anode-wire", type=float, default=0.001)
g.add_argument("--great-circles", type=int, default=7, help="1 equatorial ring + 6 meridian great circles")
g.add_argument("--cathode-T", type=float, default=None, help="override cathode transparency (default: from wires)")
g.add_argument("--anode-T", type=float, default=None, help="override anode transparency (default: from wires)")
g.add_argument("--cathode-bands", type=int, default=11, help="RZ rings representing the cathode (odd -> ring on the equator, like the real grid)")
g.add_argument("--anode-bands", type=int, default=5, help="RZ rings representing the anode (odd -> equatorial ring)")
g = ap.add_argument_group("surface physics scenario")
g.add_argument("--gamma", type=float, default=1.0,
               help="ion-induced SEE yield for a D+ at the full gap energy e|V|. Scaled to lower impact "
                    "energies with the electronic stopping power (kinetic emission). 1.0 = low end = "
                    "NEUTRON BOUND (more of the supply current carried by ions)")
g.add_argument("--gamma-neutral", type=float, default=None, help="fast-neutral SEE yield (default = --gamma)")
g.add_argument("--reflection", choices=["on", "off"], default="on",
               help="fast-particle reflection at surfaces (on = neutron bound)")
g.add_argument("--RN-W", type=float, default=fp.REFLECTION_DEFAULTS["W"][0])
g.add_argument("--RE-W", type=float, default=fp.REFLECTION_DEFAULTS["W"][1])
g.add_argument("--RN-Fe", type=float, default=fp.REFLECTION_DEFAULTS["Fe"][0])
g.add_argument("--RE-Fe", type=float, default=fp.REFLECTION_DEFAULTS["Fe"][1])
g.add_argument("--makeup", choices=["shape", "anode"], default="anode",
               help="where controller makeup ions are born: 'shape' = same distribution as modelled births; "
                    "'anode' = at rest just inside the anode (falls through full V: neutron upper bound)")
g = ap.add_argument_group("numerics")
g.add_argument("--nr", type=int, default=200, help="radial cells (200 -> 0.72 mm; use 400 for convergence check)")
g.add_argument("--cfl", type=float, default=0.5, help="max fraction of a cell a 60 keV ion moves per step")
g.add_argument("--t-end", type=float, default=4e-6, help="simulated time, s")
g.add_argument("--t-avg", type=float, default=2e-6, help="start of steady-state averaging window, s (test runs settle by ~1 us)")
g.add_argument("--t-boot", type=float, default=0.4e-6, help="bootstrap ion source fades out by this time, s")
g.add_argument("--n-macro", type=float, default=4e5, help="target number of ion macroparticles")
g.add_argument("--ncb", type=int, default=10, help="callback interval, steps")
g.add_argument("--ctrl-tau", type=float, default=0.25e-6,
               help="integral time of the current controller, s (a few ion lifetimes)")
g.add_argument("--max-grid-size", type=int, default=64)
g.add_argument("--seed", type=int, default=20261007)
g.add_argument("--n-snapshots", type=int, default=20)
g.add_argument("--track-n", type=int, default=200, help="ions followed every callback for tracks.npz (0 = off)")
g.add_argument("--track-len", type=float, default=300e-9, help="tracking window length, s")
g.add_argument("--track-start", type=float, default=None,
               help="tracking window start, s (default: the last --track-len of the run, not before --t-avg)")
g.add_argument("--checkpoint-every", type=float, default=1e-6, help="checkpoint interval, s")
g = ap.add_argument_group("run control")
g.add_argument("--outdir", default=None)
g.add_argument("--restart", default=None, help="checkpoint directory to restart from (inside --outdir)")
g.add_argument("--preflight", action="store_true", help="check inputs/geometry/data, do not start WarpX")
args = ap.parse_args()

# =============================================================================
# Derived quantities
# =============================================================================
V_c = args.voltage
assert V_c < 0, "cathode voltage must be negative"
V_abs = abs(V_c)
I_target = args.current
gamma_i = args.gamma
gamma_n = args.gamma if args.gamma_neutral is None else args.gamma_neutral
n_gas = fp.gas_density(args.pressure, args.gas_temp)
R_ch, H_ch = args.chamber_radius, args.chamber_half_length
R_c, R_a = args.cathode_radius, args.anode_radius
R_mid = 0.5 * (R_c + R_a)

geo_c = fp.grid_geometry(R_c, args.cathode_wire, args.great_circles)
geo_a = fp.grid_geometry(R_a, args.anode_wire, args.great_circles)
T_c = geo_c["transparency"] if args.cathode_T is None else args.cathode_T
T_a = geo_a["transparency"] if args.anode_T is None else args.anode_T

refl = args.reflection == "on"
RN = {"cathode": args.RN_W if refl else 0.0, "anode": args.RN_Fe if refl else 0.0, "wall": args.RN_Fe if refl else 0.0}
RE = {"cathode": args.RE_W if refl else 0.0, "anode": args.RE_Fe if refl else 0.0, "wall": args.RE_Fe if refl else 0.0}

if args.nr % 8:
    args.nr = int(round(args.nr / 8.0)) * 8   # AMReX blocking factor 8
    print(f"  note: --nr rounded to {args.nr} (must be a multiple of 8)")
dx = R_ch / args.nr
nz = int(round(2 * H_ch / dx / 8.0)) * 8
dz = 2 * H_ch / nz
v_max = fp.speed_from_energy(V_abs * fp.e, fp.m_d)
dt = args.cfl * min(dx, dz) / v_max
max_steps = int(np.ceil(args.t_end / dt))
step_avg = int(args.t_avg / dt)
dt_cb = args.ncb * dt

# reference macroparticle weight: N_real ~ (I_ion/e) * tau_life, tau_life ~ 150 ns
I_ion_target = I_target / (1.0 + gamma_i)
w_ref = (I_ion_target / fp.e) * 150e-9 / args.n_macro

# electron ionisation profile (per secondary electron leaving the cathode)
r_eprof, cum_eprof = fp.electron_ionisation_profile(V_c, R_c, R_a, np.hypot(R_ch, H_ch), n_gas)
nu_e = cum_eprof[-1]

outdir = args.outdir or time.strftime("run_%Y%m%d_%H%M%S")
outdir = os.path.abspath(outdir)
if args.restart is None and not args.preflight and os.path.exists(os.path.join(outdir, "timeseries.csv")):
    sys.exit(f"{outdir} already contains a run. Use a fresh --outdir (old diags must never mix "
             f"with new ones -- v5 bug) or --restart.")
os.makedirs(outdir, exist_ok=True)   # (exist_ok: all ranks race here harmlessly)

# =============================================================================
# Geometry: implicit function (WarpX convention: >0 inside solid, <0 in vacuum,
# 0 on the surface -- verified in v4/v5: inverted sign fills the cathode).
# Each grid = spherical shell (thickness = wire diameter) intersected with
# n axisymmetric rings whose polar-angle coverage equals the grid opacity.
# =============================================================================
def _nest(op, terms):
    expr = terms[0]
    for t in terms[1:]:
        expr = f"{op}({expr}, {t})"
    return expr


def ring_centres(transparency, n_bands):
    period = np.pi / n_bands
    hw = (1.0 - transparency) * period / 2.0
    return [i * period + period / 2.0 for i in range(n_bands)], hw


def shell_expr(R, thick, transparency, n_bands):
    centres, hw = ring_centres(transparency, n_bands)
    shell = f"({thick / 2.0:.8f} - abs(sqrt(x**2+z**2) - {R:.8f}))"
    bands = _nest("max", [f"({R:.8f}*({hw:.8f} - abs(atan2(x,z) - {cc:.8f})))" for cc in centres])
    return f"min({shell}, {bands})"


def solid_numpy(r, z):
    """numpy twin of the WarpX implicit function (True inside a solid)."""
    out = np.zeros(np.shape(r), dtype=bool)
    rs = np.hypot(r, z)
    th = np.arctan2(r, z)
    for R, thick, T, nb in ((R_c, args.cathode_wire, T_c, args.cathode_bands),
                            (R_a, args.anode_wire, T_a, args.anode_bands)):
        centres, hw = ring_centres(T, nb)
        inshell = np.abs(rs - R) < thick / 2
        inband = np.zeros_like(inshell)
        for cc in centres:
            inband |= np.abs(th - cc) < hw
        out |= inshell & inband
    return out


implicit_function = (f"max({shell_expr(R_c, args.cathode_wire, T_c, args.cathode_bands)}, "
                     f"{shell_expr(R_a, args.anode_wire, T_a, args.anode_bands)})")
potential_expr = f"if(sqrt(x**2+z**2) < {R_mid:.6f}, {V_c:.3f}, 0.0)"


def implicit_numpy(r, z):
    """numpy twin of the WarpX implicit-function string (same CSG, same sign)."""
    rs = np.hypot(r, z)
    th = np.arctan2(r, z)
    vals = []
    for R, thick, T, nb in ((R_c, args.cathode_wire, T_c, args.cathode_bands),
                            (R_a, args.anode_wire, T_a, args.anode_bands)):
        centres, hw = ring_centres(T, nb)
        shell = thick / 2.0 - np.abs(rs - R)
        bands = np.max([R * (hw - np.abs(th - cc)) for cc in centres], axis=0)
        vals.append(np.minimum(shell, bands))
    return np.maximum(vals[0], vals[1])


def realised_opacity(R, thick, T, nb, ncell):
    """Fraction of RADIAL rays (uniform in solid angle) that hit the shell:
    (a) for the exact ring geometry, (b) as WarpX's particle scraper sees it,
    i.e. with the implicit function sampled on the mesh NODES and bilinearly
    interpolated (WarpX scrapes on the interpolated nodal level set)."""
    mu = np.linspace(-1, 1, 20001)[1:-1]
    th = np.arccos(mu)
    rs = np.arange(R - thick / 2 - 2 * dx, R + thick / 2 + 2 * dx, dx / 8)
    TH, RS = np.meshgrid(th, rs, indexing="ij")
    rr, zz = RS * np.sin(TH), RS * np.cos(TH)
    exact = (implicit_numpy(rr, zz) > 0).any(axis=1).mean()
    rn = np.arange(args.nr + 1) * dx
    zn = -H_ch + np.arange(nz + 1) * dz
    Fn = implicit_numpy(*np.meshgrid(rn, zn, indexing="ij"))
    fi, fj = rr / dx, (zz + H_ch) / dz
    i0 = np.clip(fi.astype(int), 0, args.nr - 1); j0 = np.clip(fj.astype(int), 0, nz - 1)
    a, b = fi - i0, fj - j0
    F = ((1 - a) * (1 - b) * Fn[i0, j0] + a * (1 - b) * Fn[i0 + 1, j0]
         + (1 - a) * b * Fn[i0, j0 + 1] + a * b * Fn[i0 + 1, j0 + 1])
    mesh = (F > 0).any(axis=1).mean()
    return exact, mesh


# =============================================================================
# Preflight report
# =============================================================================
op_c = realised_opacity(R_c, args.cathode_wire, T_c, args.cathode_bands, args.nr)
op_a = realised_opacity(R_a, args.anode_wire, T_a, args.anode_bands, args.nr)
band_arc_c = (1 - T_c) * np.pi / args.cathode_bands * R_c
band_arc_a = (1 - T_a) * np.pi / args.anode_bands * R_a
params = dict(
    V_c=V_c, I_target=I_target, P_supply=V_abs * I_target, pressure_mTorr=args.pressure,
    T_gas=args.gas_temp, n_D2=n_gas, gamma_i=gamma_i, gamma_n=gamma_n, reflection=refl,
    RN=RN, RE=RE, R_ch=R_ch, H_ch=H_ch, R_c=R_c, R_a=R_a, d_c=args.cathode_wire, d_a=args.anode_wire,
    great_circles=args.great_circles, T_c=T_c, T_a=T_a, geo_c=geo_c, geo_a=geo_a,
    cathode_bands=args.cathode_bands, anode_bands=args.anode_bands, makeup=args.makeup,
    opacity_cathode_analytic=op_c[0], opacity_cathode_mesh=op_c[1],
    opacity_anode_analytic=op_a[0], opacity_anode_mesh=op_a[1],
    nr=args.nr, nz=nz, dx=dx, dz=dz, dt=dt, dt_cb=dt_cb, ncb=args.ncb, max_steps=max_steps,
    t_end=args.t_end, t_avg=args.t_avg, step_avg=step_avg, t_boot=args.t_boot, w_ref=w_ref,
    n_macro_target=args.n_macro, I_ion_target=I_ion_target, nu_e=nu_e, seed=args.seed,
    v_max=v_max, implicit_function=implicit_function, potential=potential_expr,
    warpx_version_expected="26.01",
)

print("=" * 78)
print("fusor v6 -- inputs and derived quantities")
print("=" * 78)
print(f"  V = {V_c/1e3:.1f} kV   I = {I_target*1e3:.1f} mA   P = {V_abs*I_target:.0f} W")
print(f"  D2 {args.pressure} mTorr @ {args.gas_temp} K -> n = {n_gas:.4e} m^-3")
print(f"  SEE yield at e|V|: ions {gamma_i}, neutrals {gamma_n} (scaled with stopping power below e|V|); "
      f"makeup ions born at: {args.makeup}")
print(f"  reflection {'ON' if refl else 'OFF'}: RN={RN} RE={RE}")
print(f"  cathode: R={R_c*1e3:.0f} mm d={args.cathode_wire*1e3:.1f} mm T={T_c:.4f} "
      f"(ring model opacity analytic {op_c[0]:.4f}, on mesh {op_c[1]:.4f}; ring arc {band_arc_c*1e3:.2f} mm "
      f"= {band_arc_c/dx:.1f} cells)")
print(f"  anode  : R={R_a*1e3:.0f} mm d={args.anode_wire*1e3:.1f} mm T={T_a:.4f} "
      f"(ring model opacity analytic {op_a[0]:.4f}, on mesh {op_a[1]:.4f}; ring arc {band_arc_a*1e3:.2f} mm "
      f"= {band_arc_a/dx:.1f} cells)")
print(f"  chamber: R={R_ch*1e3:.0f} mm, half-length {H_ch*1e3:.0f} mm "
      f"{'<-- DEFAULT, CONFIRM REAL LENGTH' if args.chamber_half_length == 0.144 else ''}")
print(f"  mesh {args.nr} x {nz}, dx = {dx*1e3:.3f} mm, dz = {dz*1e3:.3f} mm")
print(f"  dt = {dt:.3e} s ({args.cfl} cell per step at {V_abs/1e3:.0f} keV), steps = {max_steps}, "
      f"averaging from step {step_avg}")
print(f"  w_ref = {w_ref:.3e} real ions per macroparticle")
print(f"  ionisation by one secondary electron on its way out: {nu_e:.4f} ion pairs")
for name, op, d in (("cathode", op_c, args.cathode_wire), ("anode", op_a, args.anode_wire)):
    if d / dx < 1.3:
        print(f"  WARNING: {name} shell thickness {d*1e3:.2f} mm is only {d/dx:.2f} cells; "
              f"use --nr {int(np.ceil(1.5*R_ch/d))} or more for a resolved shell")
    if abs(op[1] - op[0]) > 0.3 * op[0]:
        print(f"  WARNING: {name} opacity on mesh ({op[1]:.4f}) differs >30 % from target ({op[0]:.4f})")
p_cx_max = 1 - np.exp(-n_gas * fp.sigma_cx(np.array([5e3]))[0] * fp.speed_from_energy(5e3 * fp.e, fp.m_d) * dt_cb)
print(f"  max CX probability per callback interval: {p_cx_max:.3f} (must be << 1)")
assert p_cx_max < 0.1, "reduce --ncb"
print("=" * 78)

trk_t0 = args.track_start if args.track_start is not None else max(args.t_avg, args.t_end - args.track_len)
params.update(track_n=args.track_n, track_t0=trk_t0, track_t1=trk_t0 + args.track_len)
params.update(nranks=SIZE, omp_threads=os.environ.get("OMP_NUM_THREADS"),
              fusorsim_git=os.environ.get("FUSORSIM_GIT"), sif_digest=os.environ.get("FUSORSIM_SIF_SHA256"))
if ROOT:
    with open(os.path.join(outdir, "run_params.json"), "w") as f:
        json.dump(params, f, indent=1, default=float)

# cross-section file for WarpX MCC (uniform grid, no comments)
xs_path = os.path.join(outdir, "D+_D2_elastic_momentum_transfer.dat")
if ROOT:
    fp.write_elastic_table_for_warpx(xs_path)
fmpi.barrier()
_xs = np.loadtxt(xs_path)
assert np.allclose(np.diff(_xs[:, 0]), _xs[1, 0] - _xs[0, 0], rtol=1e-6), "non-uniform grid"

if args.preflight:
    print("physics self-test:")
    ok = fp._selftest()
    print(f"\nPreflight complete ({'OK' if ok else 'PROBLEMS ABOVE'}). Files in {outdir}")
    sys.exit(0 if ok else 1)

# =============================================================================
# WarpX setup
# =============================================================================
os.chdir(outdir)
from pywarpx import callbacks, picmi  # noqa: E402
import pywarpx  # noqa: E402
from pywarpx._libwarpx import libwarpx  # noqa: E402
from pywarpx import particle_containers  # noqa: E402

grid = picmi.CylindricalGrid(
    number_of_cells=[args.nr, nz],
    lower_bound=[0.0, -H_ch],
    upper_bound=[R_ch, H_ch],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_max_grid_size=args.max_grid_size,
    warpx_blocking_factor=8,
)
solver = picmi.ElectrostaticSolver(
    grid=grid, method="Multigrid", required_precision=1e-5, warpx_self_fields_verbosity=2,
    warpx_absolute_tolerance=1e-2, maximum_iterations=200,
)
ions = picmi.Species(
    name="ions", charge=fp.e, mass=fp.m_d,
    warpx_save_particles_at_eb=1,
    warpx_save_particles_at_xhi=1, warpx_save_particles_at_zlo=1, warpx_save_particles_at_zhi=1,
)
mcc = picmi.MCCCollisions(
    name="ion_elastic", species=ions,
    background_density=n_gas, background_temperature=args.gas_temp,
    background_mass=fp.m_D2,
    scattering_processes={"elastic": {"cross_section": xs_path}},
)
eb = picmi.EmbeddedBoundary(implicit_function=implicit_function, potential=potential_expr,
                            cover_multiple_cuts=True)

snap = max(1, max_steps // args.n_snapshots)
chk = max(args.ncb, int(args.checkpoint_every / dt) // args.ncb * args.ncb)   # multiple of ncb
sim = picmi.Simulation(
    solver=solver, time_step_size=dt, max_steps=max_steps, verbose=1,
    warpx_collisions=[mcc], warpx_embedded_boundary=eb,
    warpx_random_seed=args.seed,
    warpx_amr_restart=args.restart,
    warpx_reduced_diags_path="diags/reduced/",
)
sim.add_species(ions, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid))
sim.add_diagnostic(picmi.FieldDiagnostic(name="fields", grid=grid, period=snap,
                                         data_list=["phi", "rho", "Er", "Ez"], write_dir="diags"))
# (WarpX's TimeAveragedFieldDiagnostic was tried: in 26.01 RZ it returned -2x the
#  potential and -1x the charge density, so it is not used. viz_v6.py averages
#  the ordinary snapshots inside the steady-state window instead.)
sim.add_diagnostic(picmi.ParticleDiagnostic(name="ions_snap", period=snap, species=[ions],
                                            data_list=["position", "momentum", "weighting"], write_dir="diags"))
sim.add_diagnostic(picmi.Checkpoint(name="chk", period=chk, write_dir="diags"))
for typ in ("ParticleNumber", "ParticleEnergy", "FieldEnergy"):
    sim.add_diagnostic(picmi.ReducedDiagnostic(diag_type=typ, name=typ, period=50))

sim.initialize_inputs()
# belt and braces: v2-v4 lost these when only set through PICMI
pywarpx.warpx.self_fields_required_precision = 1e-5
pywarpx.warpx.self_fields_absolute_tolerance = 1e-2
sim.initialize_warpx()
_av = pywarpx.warpx.argvattrs
print(f"[v6] solver check: self_fields_required_precision = {_av.get('self_fields_required_precision')}, "
      f"absolute_tolerance = {_av.get('self_fields_absolute_tolerance')}", flush=True)

# =============================================================================
# Python-side physics state
# =============================================================================
rng = fmpi.rank_rng(args.seed + (0 if args.restart is None else 7919))
buf = particle_containers.ParticleBoundaryBufferWrapper()
BOUNDARIES = ("eb", "x_hi", "z_lo", "z_hi")
REAL = ("r", "theta", "z", "ux", "uy", "uz", "w")

# --- histogram definitions (steady-state window only) -----------------------
NRH, NZH = 72, 144
r_edges = np.linspace(0, R_ch, NRH + 1)
z_edges = np.linspace(-H_ch, H_ch, NZH + 1)
E_edges = np.linspace(0, 2.2 * V_abs, 221)                    # eV, 1 % of 2|V| bins
th_edges = np.linspace(0, np.pi, 91)                           # cathode polar angle
wz_edges = np.linspace(-H_ch, H_ch, 49)                        # wall position (side)
wr_edges = np.linspace(0, R_ch, 25)                            # end caps
SURF = ("cathode", "anode", "wall")
H = dict(
    fus_ion_rz=np.zeros((NRH, NZH)), fus_neu_rz=np.zeros((NRH, NZH)),
    fus_ion_E=np.zeros(len(E_edges) - 1), fus_neu_E=np.zeros(len(E_edges) - 1),
    fusp_total=np.zeros(2),                                    # [ion, neutral] D(d,p)T counts
    ion_wdt_rz=np.zeros((NRH, NZH)), ion_wEdt_rz=np.zeros((NRH, NZH)),
    birth_rz=np.zeros((NRH, NZH)),
    cathode_power_theta=np.zeros(len(th_edges) - 1),
    imp_ion_E={s: np.zeros(len(E_edges) - 1) for s in SURF},
    imp_neu_E={s: np.zeros(len(E_edges) - 1) for s in SURF},
    e_anode_E=np.zeros(len(E_edges) - 1), e_wall_E=np.zeros(len(E_edges) - 1),
    e_wall_side_z=np.zeros(len(wz_edges) - 1), e_wall_cap_r=np.zeros((2, len(wr_edges) - 1)),
    e_wall_side_zE=np.zeros(len(wz_edges) - 1), e_wall_cap_rE=np.zeros((2, len(wr_edges) - 1)),
    t_window=np.zeros(1),
)
state = dict(alpha=1.0, alpha_I=1.0, I_ema=0.0, t_last=0.0, ncall=0)
pending = dict(P=np.zeros((0, 3)), D=np.zeros((0, 3)), E=np.zeros(0), w=np.zeros(0))
# ion tracks: own RNG stream, so turning tracking on/off leaves the physics draws unchanged
trk = ftracks.Recorder()
trk_rng = np.random.default_rng([args.seed, 77, RANK])
TRK = dict(picked=False, saved=False)
ts_rows = []
TS_COLS = None

def state_name(step):
    # one file per rank: tallies and queued neutrals are rank-local until the final reduction
    return f"python_state_{step:08d}.npz" if SIZE == 1 else f"python_state_{step:08d}_r{RANK:04d}.npz"


if args.restart is not None:
    # Python state is saved at exactly the checkpoint steps (see callback), so
    # restoring the file with the checkpoint's step number is consistent.
    import re
    chk_step = int(re.findall(r"(\d+)\s*$", args.restart.rstrip("/"))[-1])
    fn = os.path.join(outdir, state_name(chk_step))
    if not os.path.exists(fn):
        sys.exit(f"restart: {fn} not found (needed to restore tallies consistently)")
    z_ = np.load(fn, allow_pickle=True)
    H = z_["H"].item(); state = z_["state"].item(); pending = z_["pending"].item()
    tsf = os.path.join(outdir, "timeseries.csv")
    if ROOT and os.path.exists(tsf):   # drop rows written after the checkpoint
        lines = open(tsf).read().splitlines()
        keep = [lines[0]] + [ln for ln in lines[1:] if float(ln.split(",")[1]) <= chk_step]
        open(tsf, "w").write("\n".join(keep) + "\n")
    fmpi.barrier()
    print(f"restored Python state at step {chk_step} (t_window = {H['t_window'][0]:.3e} s, "
          f"alpha = {state['alpha']:.3f})")


def concat(lst):
    lst = [np.asarray(a) for a in lst if np.size(a) > 0]
    return np.concatenate(lst) if lst else np.zeros(0)


def ion_tiles():
    """Writable numpy views of the ion SoA arrays, tile by tile (same access
    path as pywarpx.particle_containers.get_particle_real_arrays)."""
    pc = sim.particles.get("ions")
    idx = {k: pc.get_real_comp_index(k) for k in REAL}
    for pti in libwarpx.libwarpx_so.WarpXParIter(pc, 0):
        soa = pti.soa()
        d = {k: np.array(soa.get_real_data(i), copy=False) for k, i in idx.items()}
        d["idcpu"] = np.array(soa.get_idcpu_data(), copy=False)
        yield d


def u_to_E_v(ux, uy, uz, m=fp.m_d):
    u2 = ux * ux + uy * uy + uz * uz
    gam = np.sqrt(1 + u2 / fp.c**2)
    E = (gam - 1) * m * fp.c**2 / fp.e          # eV
    v = np.sqrt(u2) / gam
    return E, v


def thermal_velocities(n):
    s = np.sqrt(fp.kB * args.gas_temp / fp.m_D2)
    return rng.normal(0, s, (n, 3))


def classify_eb(r, z):
    return np.where(np.hypot(r, z) < R_mid, 0, 1)            # 0 cathode, 1 anode


def ray_exit_cylinder(P, D):
    """Distance along unit direction D from P (inside) to the chamber wall,
    and which surface (0 side, 1 low cap, 2 high cap)."""
    a = D[:, 0] ** 2 + D[:, 1] ** 2
    b = P[:, 0] * D[:, 0] + P[:, 1] * D[:, 1]
    cc = P[:, 0] ** 2 + P[:, 1] ** 2 - R_ch**2
    with np.errstate(divide="ignore", invalid="ignore"):
        s_side = np.where(a > 1e-14, (-b + np.sqrt(np.maximum(b * b - a * cc, 0))) / a, np.inf)
        s_hi = np.where(D[:, 2] > 1e-14, (H_ch - P[:, 2]) / D[:, 2], np.inf)
        s_lo = np.where(D[:, 2] < -1e-14, (-H_ch - P[:, 2]) / D[:, 2], np.inf)
    S = np.stack([s_side, s_lo, s_hi], axis=1)
    k = np.argmin(S, axis=1)
    return np.maximum(S[np.arange(len(P)), k], 0.0), k


def tally_electrons_to_ground(P0, w, E_eV, inside_anode):
    """Electrons leaving point P0 radially (spherical) to ground: fraction
    (1-T_a) of those crossing the anode stop there; the rest reach the wall.
    Returns electron power to anode and wall (W*s, i.e. energy) and fills the
    X-ray source histograms when inside the averaging window."""
    if len(w) == 0:
        return 0.0, 0.0
    rr = np.linalg.norm(P0, axis=1)
    D = P0 / np.maximum(rr, 1e-9)[:, None]
    w_an = np.where(inside_anode, w * (1 - T_a), 0.0)
    w_wall = w - w_an
    Ean = np.sum(w_an * E_eV) * fp.e
    Ewall = np.sum(w_wall * E_eV) * fp.e
    if in_window:
        H["e_anode_E"] += np.histogram(E_eV, E_edges, weights=w_an)[0]
        H["e_wall_E"] += np.histogram(E_eV, E_edges, weights=w_wall)[0]
        s, k = ray_exit_cylinder(P0, D)
        X = P0 + s[:, None] * D
        side = k == 0
        H["e_wall_side_z"] += np.histogram(X[side, 2], wz_edges, weights=w_wall[side])[0]
        H["e_wall_side_zE"] += np.histogram(X[side, 2], wz_edges, weights=(w_wall * E_eV)[side])[0]
        for j, kk in enumerate((1, 2)):
            m = k == kk
            rcap = np.hypot(X[m, 0], X[m, 1])
            H["e_wall_cap_r"][j] += np.histogram(rcap, wr_edges, weights=w_wall[m])[0]
            H["e_wall_cap_rE"][j] += np.histogram(rcap, wr_edges, weights=(w_wall * E_eV)[m])[0]
    return Ean, Ewall


def lambertian(normals):
    """Cosine-distributed unit vectors about the given unit normals."""
    n = len(normals)
    u1, u2 = rng.random(n), rng.random(n)
    ct = np.sqrt(u1)
    st = np.sqrt(1 - u1)
    ph = 2 * np.pi * u2
    a = np.where(np.abs(normals[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(normals, a)
    t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(normals, t1)
    return (st * np.cos(ph))[:, None] * t1 + (st * np.sin(ph))[:, None] * t2 + ct[:, None] * normals


def add_reflections(P, Dinc, E, w, normal, surface):
    """Queue reflected fast neutrals from an impact (normal sign-agnostic)."""
    if RN[surface] <= 0 or len(w) == 0:
        return
    sgn = -np.sign(np.sum(Dinc * normal, axis=1))
    sgn[sgn == 0] = 1.0
    nout = normal * sgn[:, None]
    keep = E * RE[surface] / RN[surface] > 1e3          # < 1 keV: no fusion relevance
    # roulette very light reflected neutrals (unbiased) to bound the neutral count
    w_new = w * RN[surface]
    w_min = 0.1 * w_ref
    p_keep = np.minimum(1.0, w_new / w_min)
    keep &= rng.random(len(w)) < p_keep
    if not np.any(keep):
        return
    Dn = lambertian(nout[keep])
    pending["P"] = np.vstack([pending["P"], P[keep] + 1.5e-3 * nout[keep]])
    pending["D"] = np.vstack([pending["D"], Dn])
    pending["E"] = np.concatenate([pending["E"], E[keep] * RE[surface] / RN[surface]])
    pending["w"] = np.concatenate([pending["w"], w_new[keep] / p_keep[keep]])


def trace_neutrals(P, D, E, w, acc):
    """Straight-line fast-neutral transport (neutrals do not see the field).
    Events: grid interception at each shell crossing (prob. 1-T), stripping
    (exponential, re-injected as a fast ion), wall. Along the path: expected
    beam-gas fusion and sampled target ionisation (ion births)."""
    if len(w) == 0:
        return
    N = len(w)
    s_wall, kwall = ray_exit_cylinder(P, D)
    s_end = s_wall.copy()
    surf = np.full(N, 2)                              # 0 cathode,1 anode,2 wall
    b = np.sum(P * D, axis=1)
    p2 = np.sum(P * P, axis=1)
    for R, T, code in ((R_c, T_c, 0), (R_a, T_a, 1)):
        disc = b * b - (p2 - R * R)
        ok = disc > 0
        sq = np.sqrt(np.maximum(disc, 0))
        for s_c in (-b - sq, -b + sq):
            hit = ok & (s_c > 1.5e-3) & (s_c < s_end) & (rng.random(N) < (1 - T))
            s_end = np.where(hit, s_c, s_end)
            surf = np.where(hit, code, surf)
    sig_s = fp.sigma_strip(E)
    s_strip = rng.exponential(1.0, N) / np.maximum(n_gas * sig_s, 1e-30)
    strip = s_strip < s_end
    s_end = np.where(strip, s_strip, s_end)
    # beam-gas fusion along the path (expected value)
    K_over_v_n = 2 * fp.sigma_DD(E / 2e3, "n")
    K_over_v_p = 2 * fp.sigma_DD(E / 2e3, "p")
    Yn = w * n_gas * K_over_v_n * s_end
    Yp = w * n_gas * K_over_v_p * s_end
    acc["fus_neu_n"] += Yn.sum()
    acc["fus_neu_p"] += Yp.sum()
    if in_window:
        for _ in range(4):
            X = P + (rng.random(N) * s_end)[:, None] * D
            H["fus_neu_rz"] += np.histogram2d(np.hypot(X[:, 0], X[:, 1]), X[:, 2], [r_edges, z_edges],
                                              weights=Yn / 4)[0]
        H["fus_neu_E"] += np.histogram(E, E_edges, weights=Yn)[0]
        H["fusp_total"][1] += Yp.sum()
    # target ionisation births along the path
    nb = rng.poisson(n_gas * fp.sigma_neutral_target_ionisation(E) * s_end)
    if nb.sum() > 0:
        idx = np.repeat(np.arange(N), nb)
        X = P[idx] + (rng.random(idx.size) * s_end[idx])[:, None] * D[idx]
        births.append((X, w[idx]))
        acc["birth_neu"] += w[idx].sum()
    # stripped -> fast ions. Russian roulette back to the reference weight:
    # without it, reflection (x R_N per bounce) keeps splitting weights and the
    # macroparticle count grows without bound (seen in the v6 test runs:
    # mean weight fell 40x in 2 us). Unbiased: kept with p = w/w_ref, weight w/p.
    if np.any(strip):
        acc["strip"] += w[strip].sum()
        ws = w[strip]
        p_keep = np.minimum(1.0, ws / w_ref)
        keep = rng.random(ws.size) < p_keep
        idx = np.flatnonzero(strip)[keep]
        X = P[idx] + s_end[idx, None] * D[idx]
        v = fp.speed_from_energy(E[idx] * fp.e, fp.m_d)
        gam = 1 / np.sqrt(1 - (v / fp.c) ** 2)
        U = (gam * v)[:, None] * D[idx]
        fast_ions.append((X, U, ws[keep] / p_keep[keep]))
    # surface impacts
    X = P + s_end[:, None] * D
    for code, name in ((0, "cathode"), (1, "anode"), (2, "wall")):
        m = (surf == code) & ~strip
        if not np.any(m):
            continue
        acc[f"P_neu_{name}"] += np.sum(w[m] * E[m]) * fp.e
        acc[f"G_neu_{name}"] += w[m].sum()
        if in_window:
            H["imp_neu_E"][name] += np.histogram(E[m], E_edges, weights=w[m])[0]
        if code == 2:
            nrm = np.zeros((m.sum(), 3))
            km = kwall[m]
            Xm = X[m]
            rr = np.maximum(np.hypot(Xm[:, 0], Xm[:, 1]), 1e-9)
            nrm[km == 0, 0] = Xm[km == 0, 0] / rr[km == 0]
            nrm[km == 0, 1] = Xm[km == 0, 1] / rr[km == 0]
            nrm[km == 1, 2] = 1.0
            nrm[km == 2, 2] = -1.0
        else:
            nrm = X[m] / np.linalg.norm(X[m], axis=1)[:, None]
        if code == 0:
            # neutral-induced SEE: current + electrons to ground + their ionisation
            we = gamma_of(E[m], gamma_n) * w[m]
            acc["G_see_neu"] += we.sum()
            Ea, Ew = tally_electrons_to_ground(X[m], we, np.full(m.sum(), V_abs), np.ones(m.sum(), bool))
            acc["Pe_anode"] += Ea; acc["Pe_wall"] += Ew
            electron_births_from_cathode(X[m], we, acc)
            if in_window:
                th = np.arctan2(np.hypot(X[m, 0], X[m, 1]), X[m, 2])
                H["cathode_power_theta"] += np.histogram(th, th_edges, weights=w[m] * E[m] * fp.e)[0]
        add_reflections(X[m], D[m], E[m], w[m], nrm, name)


def electron_births_from_cathode(Xc, we, acc):
    """Ion pairs made by secondary electrons on their radial flight out."""
    if len(we) == 0 or nu_e <= 0:
        return
    nb = rng.poisson(we * nu_e / w_ref)
    if nb.sum() == 0:
        return
    idx = np.repeat(np.arange(len(we)), nb)
    u = rng.random(idx.size) * nu_e
    rb = np.interp(u, cum_eprof, r_eprof)
    dirs = Xc[idx] / np.linalg.norm(Xc[idx], axis=1)[:, None]
    X = dirs * rb[:, None]
    births.append((X, np.full(idx.size, w_ref)))
    acc["birth_e"] += idx.size * w_ref


PENDING_INJ = []


def inject(X, U, W):
    """Queue ions for this callback's single add_particles call (flush_injections)."""
    if len(W):
        PENDING_INJ.append((X, U, W))
    return len(W)


def flush_injections():
    """Exactly ONE add_particles per callback on EVERY rank, even with nothing to add: WarpX's
    AddNParticles ends in a collective Redistribute, so a rank that skips the call (no births in its
    boxes this interval) or makes a different number of calls deadlocks the others (seen with 2 ranks)."""
    if PENDING_INJ:
        X = np.vstack([p[0] for p in PENDING_INJ])
        U = np.vstack([p[1] for p in PENDING_INJ])
        W = np.concatenate([p[2] for p in PENDING_INJ])
        PENDING_INJ.clear()
    elif SIZE == 1:
        return
    else:
        X = U = np.zeros((0, 3)); W = np.zeros(0)
    sim.particles.get("ions").add_particles(
        x=X[:, 0].copy(), y=X[:, 1].copy(), z=X[:, 2].copy(),
        ux=U[:, 0].copy(), uy=U[:, 1].copy(), uz=U[:, 2].copy(), w=W.copy(), unique_particles=True)


# --- vacuum potential map (read from WarpX's first solve, before any charge) ---
PHI = {"map": None}


def load_vacuum_phi():
    fn = os.path.join(outdir, "phi_vacuum.npy")
    if args.restart is not None and os.path.exists(fn):
        PHI["map"] = np.load(fn); src = "saved map"
    else:
        try:
            from pywarpx import fields as wx_fields
            a = np.squeeze(np.asarray(wx_fields.PhiFPWrapper(0)[...]))
            if a.shape != (args.nr + 1, nz + 1):
                raise ValueError(f"unexpected shape {a.shape}")
            PHI["map"] = a; src = "WarpX vacuum solve"
        except Exception as ex:      # fall back to ideal spheres
            rn = np.arange(args.nr + 1) * dx; zn = -H_ch + np.arange(nz + 1) * dz
            Rg, Zg = np.meshgrid(rn, zn, indexing="ij")
            PHI["map"] = fp.vacuum_potential_spherical(np.hypot(Rg, Zg), V_c, R_c, R_a)
            src = f"ideal spheres ({ex})"
        PHI["map"] = fmpi.bcast(PHI["map"])
        if ROOT:
            np.save(fn, PHI["map"])
    if ROOT:
        print(f"[v6] vacuum potential from {src}: centre {phi_at(np.zeros(1), np.zeros(1))[0]/1e3:.2f} kV "
          f"(cathode {V_c/1e3:.1f} kV)", flush=True)


def phi_at(r, z):
    m = PHI["map"]
    fi = np.clip(r / dx, 0, args.nr - 1e-9); fj = np.clip((z + H_ch) / dz, 0, nz - 1e-9)
    i0 = fi.astype(int); j0 = fj.astype(int); a = fi - i0; b = fj - j0
    return ((1 - a) * (1 - b) * m[i0, j0] + a * (1 - b) * m[i0 + 1, j0]
            + (1 - a) * b * m[i0, j0 + 1] + a * b * m[i0 + 1, j0 + 1])


_E_grid = np.linspace(0, 2.2 * V_abs, 2201)
_S_grid = fp.stopping_D(np.maximum(_E_grid, 1.0), "W")
_S_ref = fp.stopping_D(np.array([V_abs]), "W")[0]


def gamma_of(E_eV, g_ref):
    """Kinetic electron emission scales with the electronic stopping power
    (Sternglass 1957; Baragiola et al. 1979): gamma(E) = gamma_ref S(E)/S(e|V|)."""
    return g_ref * np.interp(E_eV, _E_grid, _S_grid) / _S_ref


def deliver_interior(X, W, acc):
    """Ions created at rest INSIDE the hollow cathode (charge-exchange products,
    ionisation) cannot leave it: every exit is at a lower potential than their
    birth point, and the electrons created with them are trapped too, so the
    interior is a quasi-neutral, long-lived (~us) population. In steady state
    each such ion ends on the cathode, arriving with e(phi(birth) - V_c) of
    energy (a few keV). Tracking them would add a ~5-10 us equilibration and
    an unphysical ions-only space-charge hump (seen in the v6 test runs), so
    they are delivered to the cathode immediately: identical steady-state
    current and power, negligible fusion (keV energies)."""
    if len(W) == 0:
        return
    rr = np.hypot(X[:, 0], X[:, 1])
    E = np.maximum(phi_at(rr, X[:, 2]) - V_c, 0.0)
    acc["G_ion_cathode"] += W.sum()
    acc["G_int_cathode"] += W.sum()
    acc["P_ion_cathode"] += np.sum(W * E) * fp.e
    we = gamma_of(E, gamma_i) * W
    acc["G_see_ion"] += we.sum()
    nrm = X / np.maximum(np.linalg.norm(X, axis=1), 1e-12)[:, None]
    Xc = nrm * R_c
    Ea, Ew = tally_electrons_to_ground(Xc, we, np.full(len(W), V_abs), np.ones(len(W), bool))
    acc["Pe_anode"] += Ea; acc["Pe_wall"] += Ew
    electron_births_from_cathode(Xc, we, acc)
    if in_window:
        H["imp_ion_E"]["cathode"] += np.histogram(E, E_edges, weights=W)[0]
        th = np.arctan2(rr, X[:, 2])
        H["cathode_power_theta"] += np.histogram(th, th_edges, weights=W * E * fp.e)[0]


def valid_birth_positions(X):
    r = np.hypot(X[:, 0], X[:, 1])
    ok = (r < R_ch - dx) & (np.abs(X[:, 2]) < H_ch - dz) & ~solid_numpy(r, X[:, 2])
    return ok


def flush_timeseries():
    global TS_COLS
    if not ts_rows:
        return
    if not ROOT:
        ts_rows.clear()      # rows are identical on every rank (built from reduced sums)
        return
    path = os.path.join(outdir, "timeseries.csv")
    new = not os.path.exists(path)
    if TS_COLS is None:
        TS_COLS = list(ts_rows[0].keys())
    with open(path, "a") as f:
        if new:
            f.write(",".join(TS_COLS) + "\n")
        for row in ts_rows:
            f.write(",".join(f"{row[k]:.8e}" for k in TS_COLS) + "\n")
    ts_rows.clear()


def save_state(step=None):
    if step is not None:   # exact snapshot matching a WarpX checkpoint (rank-local, see state_name)
        np.savez(os.path.join(outdir, state_name(step)), H=H, state=state, pending=pending,
                 allow_pickle=True)
        return
    Hr = fmpi.reduce_tree(H, skip=("t_window",))      # collective: every rank must call
    if not ROOT:
        return
    out = {}
    for k, v in Hr.items():
        if isinstance(v, dict):
            for s, a in v.items():
                out[f"{k}__{s}"] = a
        else:
            out[k] = v
    np.savez(os.path.join(outdir, "tallies.npz"), r_edges=r_edges, z_edges=z_edges, E_edges=E_edges,
             th_edges=th_edges, wz_edges=wz_edges, wr_edges=wr_edges, **out)


def pick_tracked_ions():
    """Choose --track-n ions, each rank in proportion to the ions it holds, and give every rank the
    whole id list (an ion may move to another rank's boxes). Collective: called by every rank at the
    same callback (the window test uses t, which is identical on all ranks)."""
    ids = concat([a["idcpu"][a["w"] > 0] for a in ion_tiles()]).astype(np.uint64)
    n_all = max(sum(fmpi.allgather(ids.size)), 1)
    k = min(ids.size, int(round(args.track_n * ids.size / n_all)))
    mine = trk_rng.choice(ids, k, replace=False) if k else ids[:0]
    trk.set_ids(np.concatenate(fmpi.allgather(mine)))
    TRK["picked"] = True


def save_tracks():
    """Gather every rank's track rows to rank 0 -> tracks.npz. Collective."""
    parts = fmpi.gather(trk.rows())
    TRK["saved"] = True
    if ROOT:
        d = ftracks.merge(parts)
        if d["t"].size:
            np.savez(os.path.join(outdir, "tracks.npz"), **d, t0=trk_t0, t1=trk_t0 + args.track_len,
                     dt_cb=dt_cb, n_picked=trk.ids.size)
        print(f"[v6] tracks: {trk.ids.size} ions, {d['t'].size} samples -> tracks.npz", flush=True)


# =============================================================================
# The callback
# =============================================================================
in_window = False
births = []
fast_ions = []
t_wall0 = time.time()


def physics_callback():
    global in_window, births, fast_ions
    step = sim.extension.warpx.getistep(0)
    if step % args.ncb != 0:
        return
    t = sim.extension.warpx.gett_new(0)
    if PHI["map"] is None:
        load_vacuum_phi()
    Dt = t - state["t_last"] if state["t_last"] > 0 else dt_cb
    state["t_last"] = t
    in_window = t >= args.t_avg
    births, fast_ions = [], []
    acc = {k: 0.0 for k in (
        "fus_ion_n", "fus_ion_p", "fus_neu_n", "fus_neu_p", "cx", "birth_ion", "birth_neu", "birth_e",
        "strip", "G_ion_cathode", "G_ion_anode", "G_ion_wall", "P_ion_cathode", "P_ion_anode", "P_ion_wall",
        "G_neu_cathode", "G_neu_anode", "G_neu_wall", "P_neu_cathode", "P_neu_anode", "P_neu_wall",
        "G_see_ion", "G_see_neu", "Pe_anode", "Pe_wall", "Pe_gasborn", "N_real", "E_ion_sum", "G_int_cathode",
        "birth_outer")}
    n_macro = 0
    vmax_seen = 0.0

    # ---- 1. scraped ions (EB and domain walls), then clear the buffer ------
    for bnd in BOUNDARIES:
        if buf.get_particle_boundary_buffer_size("ions", bnd) == 0:
            continue
        d = {k: concat(buf.get_particle_boundary_buffer("ions", bnd, k, 0)) for k in REAL}
        if bnd == "eb":
            for k in ("nx", "ny", "nz"):
                d[k] = concat(buf.get_particle_boundary_buffer("ions", bnd, k, 0))
        if d["w"].size == 0:
            continue
        x = d["r"] * np.cos(d["theta"]); y = d["r"] * np.sin(d["theta"])
        P = np.stack([x, y, d["z"]], axis=1)
        E, v = u_to_E_v(d["ux"], d["uy"], d["uz"])
        Dinc = np.stack([d["ux"], d["uy"], d["uz"]], axis=1) / np.maximum(np.sqrt(d["ux"]**2 + d["uy"]**2 + d["uz"]**2), 1e-30)[:, None]
        w = d["w"]
        if bnd == "eb":
            cls = classify_eb(d["r"], d["z"])
            nrm = np.stack([d["nx"], d["ny"], d["nz"]], axis=1)
            bad = np.linalg.norm(nrm, axis=1) < 0.5
            nrm[bad] = P[bad] / np.linalg.norm(P[bad], axis=1)[:, None]
            for code, name in ((0, "cathode"), (1, "anode")):
                m = cls == code
                if not np.any(m):
                    continue
                acc[f"G_ion_{name}"] += w[m].sum()
                acc[f"P_ion_{name}"] += np.sum(w[m] * E[m]) * fp.e
                if in_window:
                    H["imp_ion_E"][name] += np.histogram(E[m], E_edges, weights=w[m])[0]
                add_reflections(P[m], Dinc[m], E[m], w[m], nrm[m], name)
                if code == 0:
                    we = gamma_of(E[m], gamma_i) * w[m]
                    acc["G_see_ion"] += we.sum()
                    Ea, Ew = tally_electrons_to_ground(P[m], we, np.full(m.sum(), V_abs), np.ones(m.sum(), bool))
                    acc["Pe_anode"] += Ea; acc["Pe_wall"] += Ew
                    electron_births_from_cathode(P[m], we, acc)
                    if in_window:
                        th = np.arctan2(d["r"][m], d["z"][m])
                        H["cathode_power_theta"] += np.histogram(th, th_edges, weights=w[m] * E[m] * fp.e)[0]
        else:
            acc["G_ion_wall"] += w.sum()
            acc["P_ion_wall"] += np.sum(w * E) * fp.e
            if in_window:
                H["imp_ion_E"]["wall"] += np.histogram(E, E_edges, weights=w)[0]
            nrm = np.zeros_like(P)
            if bnd == "x_hi":
                nrm[:, 0] = np.cos(d["theta"]); nrm[:, 1] = np.sin(d["theta"])
            else:
                nrm[:, 2] = 1.0
            add_reflections(P, Dinc, E, w, nrm, "wall")
    buf.clear_buffer()

    # ---- 2. live ions: fusion (expected value), CX and ionisation (MC) ------
    tracking = args.track_n > 0 and trk_t0 <= t <= trk_t0 + args.track_len
    if tracking and not TRK["picked"]:
        pick_tracked_ions()                          # collective, once
    new_neutrals = []
    for a in ion_tiles():
        npart = a["w"].size
        if npart == 0:
            continue
        n_macro += npart
        E, v = u_to_E_v(a["ux"], a["uy"], a["uz"])
        w = a["w"]
        vmax_seen = max(vmax_seen, float(v.max()))
        acc["N_real"] += w.sum()
        acc["E_ion_sum"] += np.sum(w * E)
        Kn = 2 * fp.sigma_DD(E / 2e3, "n") * v
        Kp = 2 * fp.sigma_DD(E / 2e3, "p") * v
        Rn = w * n_gas * Kn * Dt
        acc["fus_ion_n"] += Rn.sum()
        acc["fus_ion_p"] += np.sum(w * n_gas * Kp * Dt)
        r, z = a["r"], a["z"]
        if tracking:                                 # before CX below changes velocities
            trk.record(t, a["idcpu"], r * np.cos(a["theta"]), r * np.sin(a["theta"]), z, E)
        if in_window:
            H["fus_ion_rz"] += np.histogram2d(r, z, [r_edges, z_edges], weights=Rn)[0]
            H["fus_ion_E"] += np.histogram(E, E_edges, weights=Rn)[0]
            H["fusp_total"][0] += np.sum(w * n_gas * Kp * Dt)
            H["ion_wdt_rz"] += np.histogram2d(r, z, [r_edges, z_edges], weights=w * Dt)[0]
            H["ion_wEdt_rz"] += np.histogram2d(r, z, [r_edges, z_edges], weights=w * E * Dt)[0]
        # charge exchange
        p_cx = -np.expm1(-n_gas * fp.sigma_cx(E) * v * Dt)
        cx = rng.random(npart) < p_cx
        # ion-impact ionisation (independent; the ion survives it)
        p_io = -np.expm1(-n_gas * fp.sigma_ion_impact(E) * v * Dt)
        io = rng.random(npart) < p_io
        th = a["theta"]
        X = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
        if np.any(io):
            births.append((X[io], w[io].copy()))
            acc["birth_ion"] += w[io].sum()
        if np.any(cx):
            U = np.stack([a["ux"][cx], a["uy"][cx], a["uz"][cx]], axis=1)
            Dn = U / np.linalg.norm(U, axis=1)[:, None]
            new_neutrals.append((X[cx], Dn, E[cx], w[cx].copy()))
            acc["cx"] += w[cx].sum()
            vt = thermal_velocities(cx.sum())         # slow product ion (D2+ in reality)
            a["ux"][cx] = vt[:, 0]; a["uy"][cx] = vt[:, 1]; a["uz"][cx] = vt[:, 2]
            # slow products born inside the hollow cathode: delivered (see deliver_interior)
            ins_idx = np.flatnonzero(cx)[np.hypot(r[cx], z[cx]) < R_c]
            if ins_idx.size:
                deliver_interior(X[ins_idx], w[ins_idx].copy(), acc)
                a["w"][ins_idx] = 0.0
                a["idcpu"][ins_idx] &= np.uint64(0x7FFFFFFFFFFFFFFF)   # AMReX invalid flag -> removed

    # ---- 3. neutral transport: new CX neutrals + queued reflections ----------
    if new_neutrals or len(pending["w"]):
        P = np.vstack([nn[0] for nn in new_neutrals] + [pending["P"]])
        D = np.vstack([nn[1] for nn in new_neutrals] + [pending["D"]])
        E = np.concatenate([nn[2] for nn in new_neutrals] + [pending["E"]])
        W = np.concatenate([nn[3] for nn in new_neutrals] + [pending["w"]])
        for k in pending:
            pending[k] = pending[k][:0]
        trace_neutrals(P, D, E, W, acc)

    boot_frac = float(np.clip(1.0 - (t - 0.5 * args.t_boot) / (0.5 * args.t_boot), 0.0, 1.0))
    # ---- 4. ion births: physical set scaled by alpha (+ fading bootstrap) ----
    n_inj = 0
    phys_real = acc["birth_ion"] + acc["birth_neu"] + acc["birth_e"]
    Xb_list, Wb_list = [], []
    if births:
        Xb = np.vstack([b_[0] for b_ in births]); Wb = np.concatenate([b_[1] for b_ in births])
        a_shape = state["alpha"] if (args.makeup == "shape" or state["alpha"] <= 1) else 1.0
        nmac = rng.poisson(a_shape * Wb.sum() / w_ref)
        if nmac > 0:
            pick = rng.choice(len(Wb), size=nmac, p=Wb / Wb.sum())
            Xs = Xb[pick] + rng.uniform(-0.5, 0.5, (nmac, 3)) * dx
            Xb_list.append(Xs)
        if args.makeup == "anode" and state["alpha"] > 1:
            # bounding variant: every makeup ion starts at rest just inside the anode,
            # i.e. falls through the full potential
            nx_ = rng.poisson((state["alpha"] - 1) * Wb.sum() / w_ref)
            if nx_ > 0:
                mu = rng.uniform(-1, 1, nx_); ph = rng.uniform(0, 2 * np.pi, nx_)
                rr = R_a - args.anode_wire - 2 * dx
                st_ = np.sqrt(1 - mu**2)
                Xb_list.append(np.stack([rr * st_ * np.cos(ph), rr * st_ * np.sin(ph), rr * mu], axis=1))
    if boot_frac > 0:
        nboot = rng.poisson(boot_frac * (I_ion_target / fp.e) * Dt / w_ref / SIZE)
        if nboot > 0:   # uniform in volume between the grids
            rr = (R_c**3 + rng.random(nboot) * (R_a**3 - R_c**3)) ** (1 / 3)
            mu = rng.uniform(-1, 1, nboot); ph = rng.uniform(0, 2 * np.pi, nboot)
            st = np.sqrt(1 - mu**2)
            Xb_list.append(np.stack([rr * st * np.cos(ph), rr * st * np.sin(ph), rr * mu], axis=1))
    if Xb_list:
        Xs = np.vstack(Xb_list)
        # Ions born outside the anode sit in an almost field-free region; in the
        # real device they drift as part of a weak quasi-neutral plasma until they
        # reach the wall or wander through the anode. Bounding choice: all of them
        # enter, starting at rest just inside the anode (full fall, most fusion).
        rs_ = np.linalg.norm(Xs, axis=1)
        out_ = rs_ > R_a - args.anode_wire
        acc["birth_outer"] += out_.sum() * w_ref
        if np.any(out_):
            Xs[out_] *= ((R_a - args.anode_wire - 2 * dx) / rs_[out_])[:, None]
        ok = valid_birth_positions(Xs)
        Xs = Xs[ok]
        inside = np.linalg.norm(Xs, axis=1) < R_c
        Ws = np.full(len(Xs), w_ref)
        deliver_interior(Xs[inside], Ws[inside], acc)
        n_inj += inject(Xs[~inside], thermal_velocities((~inside).sum()), Ws[~inside])
        # electrons of these pairs: to ground with e|phi(birth)| (vacuum model,
        # interior of cathode taken at the full |V| -> conservative)
        rs = np.linalg.norm(Xs, axis=1)
        # electron of each pair reaches ground with e|phi(birth)|; those born inside
        # the cathode leave only over the gap saddle -> take the full e|V_c| (upper bound)
        Ee = np.where(inside, V_abs, -phi_at(np.hypot(Xs[:, 0], Xs[:, 1]), Xs[:, 2]))
        Ea, Ew = tally_electrons_to_ground(Xs, Ws, np.maximum(Ee, 0.0), rs < R_a)
        acc["Pe_anode"] += Ea; acc["Pe_wall"] += Ew; acc["Pe_gasborn"] += Ea + Ew
        if in_window:
            H["birth_rz"] += np.histogram2d(np.hypot(Xs[:, 0], Xs[:, 1]), Xs[:, 2], [r_edges, z_edges], weights=Ws)[0]
    injected_real = n_inj * w_ref
    # fast ions from stripping (physical, not controller-scaled)
    for X, U, W in fast_ions:
        ok = valid_birth_positions(X)
        n_inj += inject(X[ok], U[ok], W[ok])
    flush_injections()   # collective: every rank, every callback
    if TRK["picked"] and not TRK["saved"] and t > trk_t0 + args.track_len:
        save_tracks()    # collective, once: written at the window's end so a later crash keeps them

    # ---- 4b. MPI: one global picture of this interval. Everything below (controller,
    #      timeseries) is computed from identical reduced numbers on every rank, so
    #      alpha stays in lock-step without any broadcast. ---------------------------
    if SIZE > 1:
        keys = list(acc)
        vec = fmpi.allsum(np.array([acc[k] for k in keys] + [n_macro, injected_real, phys_real]))
        acc = dict(zip(keys, vec[:len(keys)]))
        n_macro, injected_real, phys_real = int(vec[-3]), vec[-2], vec[-1]
        vmax_seen = fmpi.allmax(vmax_seen)

    # ---- 5. supply current (incl. interior deliveries above) and controller
    #      (alpha set here is used for the NEXT interval's births) -----------------------------------
    G_i = acc["G_ion_cathode"] / Dt
    G_n = acc["G_neu_cathode"] / Dt
    # supply current = ions arriving at the cathode + secondary electrons leaving it
    # (energy-dependent yields, summed where each impact was processed)
    I_meas = fp.e * (G_i + (acc["G_see_ion"] + acc["G_see_neu"]) / Dt)
    state["I_ema"] += min(1.0, Dt / 30e-9) * (I_meas - state["I_ema"])
    if t > 0.5 * args.t_boot:
        # PI control in log space. The ion population multiplies itself
        # (births ~ alpha * ions), so a pure integrator would give an undamped
        # log-oscillation; the proportional term supplies the damping.
        err = float(np.log(I_target / max(state["I_ema"], 1e-3 * I_target)))
        state["alpha_I"] = float(np.clip(state["alpha_I"] * np.exp(np.clip(Dt / args.ctrl_tau * err, -0.05, 0.05)),
                                         1e-3, 1e4))
        state["alpha"] = state["alpha_I"] * float(np.exp(np.clip(1.0 * err, -1.0, 1.0)))

    if in_window:
        H["t_window"][0] += Dt

    # ---- 6. bookkeeping ----------------------------------------------------
    row = dict(t=t, step=step, Dt=Dt, n_macro=n_macro, N_real=acc["N_real"],
               E_ion_mean_eV=acc["E_ion_sum"] / max(acc["N_real"], 1e-30),
               I_meas=I_meas, I_ema=state["I_ema"], alpha=state["alpha"], boot_frac=boot_frac,
               I_ion_cathode=fp.e * G_i, I_see=fp.e * (acc["G_see_ion"] + acc["G_see_neu"]) / Dt,
               injected_real_rate=injected_real / Dt, phys_birth_rate=phys_real / Dt,
               vmax_cfl=vmax_seen * dt / min(dx, dz))
    for k, v in acc.items():
        if k in ("N_real", "E_ion_sum"):
            continue
        if k.startswith("fus") or k.startswith("G_") or k in ("cx", "birth_ion", "birth_neu", "birth_e", "strip", "birth_outer"):
            row[k + "_rate"] = v / Dt                      # per second
        else:
            row[k] = v / Dt                                # energies -> W
    ts_rows.append(row)
    state["ncall"] += 1
    if state["ncall"] % 20 == 0:
        flush_timeseries()
    if step % chk == 0:
        flush_timeseries()
        save_state(step)
    if state["ncall"] % 200 == 0:
        save_state()
        el = time.time() - t_wall0
        if ROOT: print(f"[v6] t={t*1e9:8.1f} ns step {step}/{max_steps} macro={n_macro} "
              f"I={state['I_ema']*1e3:6.2f} mA alpha={state['alpha']:.3g} "
              f"Yn_ion={row['fus_ion_n_rate']:.3e}/s Yn_neu={row['fus_neu_n_rate']:.3e}/s "
              f"CFL={row['vmax_cfl']:.2f} wall={el/60:.1f} min", flush=True)
        if ROOT and row["vmax_cfl"] > 0.95:
            print("[v6] WARNING: ions move ~1 cell per step; reduce --cfl", flush=True)


callbacks.installafterstep(physics_callback)

if ROOT: print(f"[v6] running {max_steps} steps on {SIZE} rank(s); output in {outdir}", flush=True)
remaining = max_steps - (sim.extension.warpx.getistep(0) if args.restart else 0)
sim.step(remaining)
flush_timeseries()
save_state()
if args.track_n > 0 and not TRK["saved"]:     # window ran to t_end (same on every rank: collective)
    save_tracks()
if ROOT: print(f"[v6] done in {(time.time()-t_wall0)/3600:.2f} h. Next: fusor-post {outdir}")
