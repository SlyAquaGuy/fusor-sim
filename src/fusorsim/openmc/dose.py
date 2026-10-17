#!/usr/bin/env python3
"""
fusor_openmc.py -- dose-rate model of the IEC fusor in room G131 (Eng 37, Clayton).

Takes the source terms written by postprocess_v6.py (run_X/post/source_terms.json
and openmc_sources.py), puts them in a model of the chamber, the poly shield and
the room, and tallies ambient dose equivalent H*(10) -- the quantity a survey
meter reads -- on a 3-D map and at named points.

    conda activate openmc
    python fusor_openmc.py run_R1/post --plot                 # geometry pictures only (seconds)
    python fusor_openmc.py run_R1/post --shield 20 --quick    # 10^5 neutrons, checks it all works
    python fusor_openmc.py run_R1/post --shield 0 10 20 30    # neutron dose for each poly thickness
    python fusor_openmc.py run_R1/post --mode xray --shield 0 20   # X-ray dose (TTB electrons)
    python fusor_openmc.py run_R1/post --post-only --shield 20     # redo tables/plots from finished runs

Outputs go to run_R1/openmc/<mode>_shield<t>cm/ : dose_points.md, dose_points.csv,
dose_plan.png (plan view at chamber height), dose_side.png (vertical slice),
dose_mesh.npz; plus run_R1/openmc/summary_<mode>.md comparing shield thicknesses.

Coordinate frame (same as the WarpX/source files): origin = chamber centre,
z = chamber axis = up (chamber is vertical). +y = north = towards the cinder-block
back wall; -y = south = the brick outside wall with the door and window;
-x = west = plaster wall to G130 (behind the table); +x = east = plaster wall.
All ROOM numbers below are in metres -- EDIT THEM when you have measurements.
"""
import argparse, csv, glob, json, os, sys
import numpy as np

# ============================== EDIT ME ==================================
ROOM = dict(            # distance from chamber CENTRE to the INSIDE face of each wall, m
    west=0.50,          # plaster wall to G130, behind the table   [estimated from photo -- MEASURE]
    south=1.00,         # brick outside wall (door + window)        [Elijah: ~1 m]
    east=2.50,          # plaster wall on the right from the door   [Elijah: 2.5 m]
    north=4.00,         # cinder-block back wall                    [Elijah: 4 m]
    floor=1.30,         # floor surface below chamber centre        [estimated from photo -- MEASURE]
    ceiling=3.00,       # floor-to-ceiling height                   [estimated from photo -- MEASURE]
)
WALLS = dict(           # (material, thickness m)
    west=("plaster", 0.026),     # stud wall: 2 x 13 mm plasterboard (air gap ignored)
    east=("plaster", 0.026),
    north=("block", 0.19),       # 190 mm hollow concrete block, homogenised
    south=("brick", 0.10),       # Elijah: ~10 cm brick
    floor=("concrete", 0.25),    # slab thicknesses assumed
    ceiling=("concrete", 0.25),
)
CHAMBER_WALL = 0.020    # m, stainless steel (Elijah: 2 cm); end plates assumed the same
VIEWPORT = dict(dir="north", diameter=0.10, glass=0.005)   # all viewports face the back wall;
                        # for several viewports set diameter so the total glass area matches
SHIELD_GAP = 0.02       # m between chamber outside and the poly
OPERATOR = (2.0, -4.6, 0.0)    # m, placeholder: outside the door in the lane, ~5 m away
MINUTES_PER_WEEK = 15.0  # plan: at most 15 min of beam a week (override with --minutes-per-week)
SHOTS_PER_YEAR = MINUTES_PER_WEEK * 60 / 15 * 52   # 15 s shots, 52 weeks
# =========================================================================

CM = 100.0
REFS = "Reference: public 1 mSv/yr, occupational 20 mSv/yr, proposed constraint 0.3 mSv/yr (Section 5.2.2 of the RMP)."


def points():
    w, s, e, n = ROOM["west"], ROOM["south"], ROOM["east"], ROOM["north"]
    tw, ts, te, tn = (WALLS[k][1] for k in ("west", "south", "east", "north"))
    up = ROOM["ceiling"] - ROOM["floor"] + WALLS["ceiling"][1] + 1.0
    dn = -(ROOM["floor"] + WALLS["floor"][1] + 1.5)
    return {
        "In G131, 1 m from chamber (east)": (1.0, 0.0, 0.0),
        "Operator position (placeholder)": OPERATOR,
        "G131 far corner (north-east)": (e - 0.3, n - 0.3, 0.0),
        "G130, 0.5 m past west wall": (-(w + tw + 0.5), 0.0, 0.0),
        "Lane outside, 0.5 m past south wall": (0.0, -(s + ts + 0.5), 0.0),
        "East neighbour, 0.5 m past east wall": (e + te + 0.5, 0.0, 0.0),
        "North of back wall, 0.5 m past it": (0.0, n + tn + 0.5, 0.0),
        "Floor above, 1 m over its floor": (0.0, 0.0, up),
        "Floor below, 1.5 m under the slab": (0.0, 0.0, dn),
    }


def world_box():
    w, s, e, n = ROOM["west"], ROOM["south"], ROOM["east"], ROOM["north"]
    tw, ts, te, tn = (WALLS[k][1] for k in ("west", "south", "east", "north"))
    zf, zc = -ROOM["floor"], ROOM["ceiling"] - ROOM["floor"]
    return (-(w + tw + 1.6), e + te + 2.0, -(s + ts + 5.0), n + tn + 1.6,
            zf - WALLS["floor"][1] - 2.0, zc + WALLS["ceiling"][1] + 2.0)


# --------------------------------------------------------------- materials
def materials(openmc, borated):
    M = {}
    ss = openmc.Material(name="stainless 304")
    for el, f in (("Fe", 0.6842), ("Cr", 0.19), ("Ni", 0.095), ("Mn", 0.02), ("Si", 0.01), ("C", 0.0008)):
        ss.add_element(el, f, "wo")
    ss.set_density("g/cm3", 8.0); M["steel"] = ss
    glass = openmc.Material(name="borosilicate glass")       # NIST 'Glass, Borosilicate (Pyrex)'
    for el, f in (("B", 0.040064), ("O", 0.539562), ("Na", 0.028191), ("Al", 0.011644),
                  ("Si", 0.377220), ("K", 0.003321)):
        glass.add_element(el, f, "wo")
    glass.set_density("g/cm3", 2.23); M["glass"] = glass
    if borated:
        pe = openmc.Material(name="borated polyethylene 5 wt% B")
        for el, f in (("H", 0.1365), ("C", 0.8135), ("B", 0.05)):
            pe.add_element(el, f, "wo")
        pe.set_density("g/cm3", 1.00)
    else:
        pe = openmc.Material(name="HDPE")
        pe.add_element("H", 0.1437, "wo"); pe.add_element("C", 0.8563, "wo")
        pe.set_density("g/cm3", 0.95)
    pe.add_s_alpha_beta("c_H_in_CH2"); M["poly"] = pe
    air = openmc.Material(name="air")
    for el, f in (("N", 0.7553), ("O", 0.2318), ("Ar", 0.0128), ("C", 0.0001)):
        air.add_element(el, f, "wo")
    air.set_density("g/cm3", 0.001205); M["air"] = air
    conc = openmc.Material(name="concrete (Portland, NIST)")
    conc_wo = (("H", 0.010), ("C", 0.001), ("O", 0.529107), ("Na", 0.016), ("Mg", 0.002), ("Al", 0.033872),
               ("Si", 0.337021), ("K", 0.013), ("Ca", 0.044), ("Fe", 0.014))
    for el, f in conc_wo:
        conc.add_element(el, f, "wo")
    conc.set_density("g/cm3", 2.30); M["concrete"] = conc
    block = openmc.Material(name="hollow concrete block (55 % solid)")
    for el, f in conc_wo:
        block.add_element(el, f, "wo")
    block.set_density("g/cm3", 2.30 * 0.55); M["block"] = block
    brick = openmc.Material(name="clay brick (approximate)")
    for el, f in (("O", 0.52), ("Si", 0.31), ("Al", 0.08), ("Fe", 0.03), ("Ca", 0.02), ("K", 0.02),
                  ("Mg", 0.01), ("Na", 0.01)):
        brick.add_element(el, f, "wo")
    brick.set_density("g/cm3", 1.80); M["brick"] = brick
    gyp = openmc.Material(name="gypsum plasterboard")
    gyp.add_elements_from_formula("CaSO6H4")                  # CaSO4.2H2O
    gyp.set_density("g/cm3", 0.80); M["plaster"] = gyp
    return M


def grid_shell(openmc, R, d, T, rho, el, name):
    """Wire grid homogenised into a spherical shell one wire diameter thick:
    wire volume fraction = (1 - T) * pi/4 (round wires on a d x d footprint)."""
    m = openmc.Material(name=name); m.add_element(el, 1.0)
    m.set_density("g/cm3", rho * (1 - T) * np.pi / 4)
    return m, openmc.Sphere(r=(R - d / 2) * CM), openmc.Sphere(r=(R + d / 2) * CM)


# ---------------------------------------------------------------- geometry
def build_geometry(openmc, D, run_params, t_shield, borated):
    g = D["geometry"]
    M = materials(openmc, borated)
    Rch, Hch = g["R_ch"] * CM, g["H_ch"] * CM
    Ro, Ho = Rch + CHAMBER_WALL * CM, Hch + CHAMBER_WALL * CM
    d_c = run_params.get("d_c", 0.002); d_a = run_params.get("d_a", 0.001)
    cath, c_in, c_out = grid_shell(openmc, g["R_c"], d_c, g["T_c"], 19.3, "W", "cathode grid (W, homogenised)")
    anod, a_in, a_out = grid_shell(openmc, g["R_a"], d_a, g["T_a"], 7.87, "Fe", "anode grid (steel, homogenised)")
    M["cathode"], M["anode"] = cath, anod

    cyl_i, cyl_o = openmc.ZCylinder(r=Rch), openmc.ZCylinder(r=Ro)
    zi_lo, zi_hi = openmc.ZPlane(-Hch), openmc.ZPlane(Hch)
    zo_lo, zo_hi = openmc.ZPlane(-Ho), openmc.ZPlane(Ho)
    inner = -cyl_i & +zi_lo & -zi_hi
    outer = -cyl_o & +zo_lo & -zo_hi
    cells = []
    cells.append(openmc.Cell(name="cathode grid", fill=cath, region=+c_in & -c_out))
    cells.append(openmc.Cell(name="anode grid", fill=anod, region=+a_in & -a_out))
    cells.append(openmc.Cell(name="chamber vacuum", fill=None,
                             region=inner & (-c_in | (+c_out & -a_in) | +a_out)))

    # viewport: a hole through the side wall, glass on the inside face (where the
    # electrons hit), open flange bore outside it
    rv = VIEWPORT["diameter"] / 2 * CM
    ax = {"north": (openmc.YCylinder, openmc.YPlane, +1), "south": (openmc.YCylinder, openmc.YPlane, -1),
          "east": (openmc.XCylinder, openmc.XPlane, +1), "west": (openmc.XCylinder, openmc.XPlane, -1)}
    Cyl, Pl, sgn = ax[VIEWPORT["dir"]]
    bore = Cyl(r=rv)
    half = +Pl(0.0) if sgn > 0 else -Pl(0.0)
    hole = -bore & half & outer & ~inner
    glass_r = openmc.ZCylinder(r=Rch + VIEWPORT["glass"] * CM)
    cells.append(openmc.Cell(name="viewport glass", fill=M["glass"], region=hole & -glass_r))
    cells.append(openmc.Cell(name="viewport bore", fill=M["air"], region=hole & +glass_r))
    cells.append(openmc.Cell(name="chamber wall", fill=M["steel"], region=outer & ~inner & ~hole))

    # shield: cylindrical poly can around the chamber, same thickness on top and bottom
    if t_shield > 0:
        rs_i, hs_i = Ro + SHIELD_GAP * CM, Ho + SHIELD_GAP * CM
        rs_o, hs_o = rs_i + t_shield, hs_i + t_shield
        s_in = -openmc.ZCylinder(r=rs_i) & +openmc.ZPlane(-hs_i) & -openmc.ZPlane(hs_i)
        s_out = -openmc.ZCylinder(r=rs_o) & +openmc.ZPlane(-hs_o) & -openmc.ZPlane(hs_o)
        cells.append(openmc.Cell(name="gap", fill=M["air"], region=s_in & ~outer))
        cells.append(openmc.Cell(name="poly shield", fill=M["poly"], region=s_out & ~s_in))
        core = s_out
        reach = (rs_o, hs_o)
    else:
        core = outer
        reach = (Ro, Ho)
    if reach[0] >= min(ROOM["west"], ROOM["south"], ROOM["east"], ROOM["north"]) * CM or \
            reach[1] >= ROOM["floor"] * CM:
        sys.exit(f"Shield ({reach[0]:.0f} cm radius) would go through a wall or the floor -- "
                 f"reduce --shield or check ROOM.")

    # room, walls, slabs
    w, s, e, n = (ROOM[k] * CM for k in ("west", "south", "east", "north"))
    tw, ts, te, tn = (WALLS[k][1] * CM for k in ("west", "south", "east", "north"))
    zf, zc = -ROOM["floor"] * CM, (ROOM["ceiling"] - ROOM["floor"]) * CM
    tf, tc = WALLS["floor"][1] * CM, WALLS["ceiling"][1] * CM
    X0, X1, Y0, Y1, Z0, Z1 = (v * CM for v in world_box())
    px = {v: openmc.XPlane(v) for v in (X0, X1, -w - tw, -w, e, e + te)}
    py = {v: openmc.YPlane(v) for v in (Y0, Y1, -s - ts, -s, n, n + tn)}
    pz = {v: openmc.ZPlane(v) for v in (Z0, Z1, zf - tf, zf, zc, zc + tc)}
    for p in (px[X0], px[X1], py[Y0], py[Y1], pz[Z0], pz[Z1]):
        p.boundary_type = "vacuum"

    def box(x0, x1, y0, y1, z0, z1):
        return +px[x0] & -px[x1] & +py[y0] & -py[y1] & +pz[z0] & -pz[z1]

    room = box(-w, e, -s, n, zf, zc)
    cells.append(openmc.Cell(name="room air", fill=M["air"], region=room & ~core))
    walls = {
        "west": box(-w - tw, -w, -s - ts, n + tn, zf, zc), "east": box(e, e + te, -s - ts, n + tn, zf, zc),
        "south": box(-w, e, -s - ts, -s, zf, zc), "north": box(-w, e, n, n + tn, zf, zc)}
    for k, reg in walls.items():
        cells.append(openmc.Cell(name=f"{k} wall", fill=M[WALLS[k][0]], region=reg))
    cells.append(openmc.Cell(name="floor slab", fill=M[WALLS["floor"][0]], region=box(X0, X1, Y0, Y1, zf - tf, zf)))
    cells.append(openmc.Cell(name="ceiling slab", fill=M[WALLS["ceiling"][0]], region=box(X0, X1, Y0, Y1, zc, zc + tc)))
    shell = box(-w - tw, e + te, -s - ts, n + tn, zf, zc)
    cells.append(openmc.Cell(name="air around room", fill=M["air"], region=box(X0, X1, Y0, Y1, zf, zc) & ~shell))
    cells.append(openmc.Cell(name="air below", fill=M["air"], region=box(X0, X1, Y0, Y1, Z0, zf - tf)))
    cells.append(openmc.Cell(name="air above", fill=M["air"], region=box(X0, X1, Y0, Y1, zc + tc, Z1)))

    geom = openmc.Geometry(cells)
    used = {c.fill.name: c.fill for c in cells if c.fill is not None}
    return geom, openmc.Materials(list(used.values())), M


# ----------------------------------------------------------------- tallies
VOXEL = 20.0      # cm, dose-map voxel
POINT = 40.0      # cm, side of the cube scored around each named point


def dose_filters(openmc):
    out = {}
    for p in ("neutron", "photon"):
        E, h = openmc.data.dose_coefficients(p, data_source="icrp74", dose_quantity="ambient")  # H*(10), pSv cm2
        out[p] = openmc.EnergyFunctionFilter(E, h, interpolation="linear-linear")
    return out


def build_tallies(openmc):
    X0, X1, Y0, Y1, Z0, Z1 = (v * CM for v in world_box())
    mesh = openmc.RegularMesh(name="dose map")
    mesh.lower_left = (X0, Y0, Z0)
    mesh.dimension = [int(round((X1 - X0) / VOXEL)), int(round((Y1 - Y0) / VOXEL)), int(round((Z1 - Z0) / VOXEL))]
    mesh.upper_right = (X0 + mesh.dimension[0] * VOXEL, Y0 + mesh.dimension[1] * VOXEL, Z0 + mesh.dimension[2] * VOXEL)
    DF = dose_filters(openmc)
    tallies = []
    for p in ("neutron", "photon"):
        t = openmc.Tally(name=f"map_{p}")
        t.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter([p]), DF[p]]
        t.scores = ["flux"]; tallies.append(t)
    for i, (name, (x, y, z)) in enumerate(points().items()):
        pm = openmc.RegularMesh(name=name)
        c = np.array([x, y, z]) * CM
        pm.lower_left, pm.upper_right, pm.dimension = c - POINT / 2, c + POINT / 2, [1, 1, 1]
        for p in ("neutron", "photon"):
            t = openmc.Tally(name=f"pt{i}_{p}")
            t.filters = [openmc.MeshFilter(pm), openmc.ParticleFilter([p]), DF[p]]
            t.scores = ["flux"]; tallies.append(t)
    return openmc.Tallies(tallies), mesh


def build_plots(openmc, M, used):
    colors = {M["steel"]: (120, 120, 130), M["glass"]: (120, 200, 255), M["poly"]: (250, 240, 200),
              M["air"]: (255, 255, 255), M["concrete"]: (150, 150, 150), M["block"]: (170, 160, 150),
              M["brick"]: (170, 80, 60), M["plaster"]: (230, 230, 210), M["cathode"]: (255, 150, 0),
              M["anode"]: (0, 120, 0)}
    X0, X1, Y0, Y1, Z0, Z1 = (v * CM for v in world_box())
    plots = []
    for name, basis, origin, width in (
            ("geometry_plan", "xy", ((X0 + X1) / 2, (Y0 + Y1) / 2, 0.0), (X1 - X0, Y1 - Y0)),
            ("geometry_side", "xz", ((X0 + X1) / 2, 0.0, (Z0 + Z1) / 2), (X1 - X0, Z1 - Z0)),
            ("geometry_chamber", "xy", (0.0, 0.0, 0.0), (120.0, 120.0)),
            ("geometry_chamber_side", "yz", (0.0, 0.0, 0.0), (120.0, 120.0))):
        pl = openmc.SlicePlot(name=name)
        pl.filename, pl.basis, pl.origin, pl.width = name, basis, origin, width
        pl.pixels = (int(1000 * width[0] / max(width)), int(1000 * width[1] / max(width)))
        pl.color_by = "material"; pl.colors = {m: c for m, c in colors.items() if m in used}
        plots.append(pl)
    return openmc.Plots(plots)


# ------------------------------------------------------------------ results
def strength_and_shot(D, sources, mode, fs):
    S = fs.total_strength(sources)
    if mode == "neutron":       # peak-rate run; per shot = rate x (shot total / peak rate) seconds
        sec_per_shot = D["neutron"]["shot_total"] / D["neutron"]["peak_rate_per_s"]
    else:
        sec_per_shot = D["duration_s"]
    return S, sec_per_shot


def post(case_dir, D, mode, S, sec_per_shot, t_shield):
    import openmc
    sps = sorted(glob.glob(os.path.join(case_dir, "statepoint.*.h5")))
    if not sps:
        sys.exit(f"no statepoint in {case_dir}")
    sp = openmc.StatePoint(sps[-1])
    nps = sp.n_particles * sp.n_batches
    # OpenMC >= 0.15 already multiplies fixed-source tallies by the total source
    # strength (Tally::accumulate), so results are per second; older versions
    # give per source particle and need x S.
    ver = tuple(int(x) for x in openmc.__version__.split(".")[:2])
    per_h = (1.0 if ver >= (0, 15) else S) * 3600 * 1e-6      # pSv/s (or pSv/particle x S) -> uSv/h

    def val(name, vol):
        t = sp.get_tally(name=name)
        return t.mean.ravel() / vol * per_h, t.std_dev.ravel() / vol * per_h

    rows = []
    for i, (name, xyz) in enumerate(points().items()):
        vol = POINT ** 3
        n, dn = val(f"pt{i}_neutron", vol); g, dg = val(f"pt{i}_photon", vol)
        tot = n[0] + g[0]; err = np.hypot(dn[0], dg[0])
        shot = tot / 3600 * sec_per_shot
        rows.append(dict(point=name, x_m=xyz[0], y_m=xyz[1], z_m=xyz[2], dist_m=float(np.linalg.norm(xyz)),
                         neutron_uSv_h=n[0], photon_uSv_h=g[0], total_uSv_h=tot,
                         rel_err=err / tot if tot > 0 else float("nan"),
                         uSv_per_shot=shot, mSv_per_year=shot * SHOTS_PER_YEAR / 1000))
    with open(os.path.join(case_dir, "dose_points.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    tm = sp.get_tally(name="map_neutron")
    mesh = tm.find_filter(openmc.MeshFilter).mesh
    nx, ny, nz = mesh.dimension
    vol = VOXEL ** 3
    maps = {}
    for p in ("neutron", "photon"):
        m, s = val(f"map_{p}", vol)
        maps[p] = m.reshape(nz, ny, nx).transpose(2, 1, 0); maps[p + "_sd"] = s.reshape(nz, ny, nx).transpose(2, 1, 0)
    total = maps["neutron"] + maps["photon"]
    ll, ur = np.array(mesh.lower_left), np.array(mesh.upper_right)
    np.savez_compressed(os.path.join(case_dir, "dose_mesh.npz"), total_uSv_h=total, lower_left_cm=ll,
                        upper_right_cm=ur, voxel_cm=VOXEL, **{k + "_uSv_h": v for k, v in maps.items()})

    lines = [f"# Fusor dose rates -- {mode}, poly shield {t_shield:g} cm\n",
             f"Source: {S:.3e} {'n' if mode == 'neutron' else 'e'}/s ({'peak' if mode == 'neutron' else 'TTB electrons'}); "
             f"{nps:.2e} histories; H*(10) with ICRP 74 coefficients. Per shot = {sec_per_shot:.1f} s at this rate; "
             f"{SHOTS_PER_YEAR:.0f} shots/year ({SHOTS_PER_YEAR * 15 / 3600:.1f} h of beam).\n",
             "| Location | x, y, z (m) | dist (m) | neutron µSv/h | photon µSv/h | total µSv/h | ±% | µSv/shot | mSv/yr |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['point']} | {r['x_m']:.1f}, {r['y_m']:.1f}, {r['z_m']:.1f} | {r['dist_m']:.1f} | "
                     f"{r['neutron_uSv_h']:.3g} | {r['photon_uSv_h']:.3g} | {r['total_uSv_h']:.3g} | "
                     f"{100 * r['rel_err']:.0f} | {r['uSv_per_shot']:.3g} | {r['mSv_per_year']:.3g} |")
    lines += ["", REFS, "±% is the 1-sigma Monte Carlo error; above ~20 % run more particles (--particles)."]
    open(os.path.join(case_dir, "dose_points.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_maps(case_dir, total, ll, ur, mode, t_shield)
    return rows


def plot_maps(case_dir, total, ll, ur, mode, t_shield):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("matplotlib missing -- skipping dose maps"); return
    nx, ny, nz = total.shape
    xs = np.linspace(ll[0], ur[0], nx + 1) / CM; ys = np.linspace(ll[1], ur[1], ny + 1) / CM
    zs = np.linspace(ll[2], ur[2], nz + 1) / CM
    k0 = np.searchsorted(zs, 0.0) - 1; j0 = np.searchsorted(ys, 0.0) - 1
    pos = total[total > 0]
    if pos.size == 0:
        print("dose map empty (nothing scored) -- skipping plots"); return
    norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-6), vmax=pos.max())
    w, s, e, n = (ROOM[k] for k in ("west", "south", "east", "north"))
    tw, ts, te, tn = (WALLS[k][1] for k in ("west", "south", "east", "north"))
    zf, zc = -ROOM["floor"], ROOM["ceiling"] - ROOM["floor"]
    for fname, data, ex, ey, xl, yl, outline in (
            ("dose_plan.png", total[:, :, k0].T, xs, ys, "x (m, east)", "y (m, north)",
             [((-w, -s), e + w, n + s), ((-w - tw, -s - ts), e + te + w + tw, n + tn + s + ts)]),
            ("dose_side.png", total[:, j0, :].T, xs, zs, "x (m, east)", "z (m, up)",
             [((-w, zf), e + w, zc - zf), ((-w - tw, zf - WALLS['floor'][1]), e + te + w + tw,
                                             zc - zf + WALLS['floor'][1] + WALLS['ceiling'][1])])):
        fig, ax = plt.subplots(figsize=(8, 8 * (ey[-1] - ey[0]) / (ex[-1] - ex[0]) + 0.6))
        im = ax.pcolormesh(ex, ey, np.where(data > 0, data, np.nan), norm=norm, cmap="inferno", shading="flat")
        for (x0, y0), wd, ht in outline:
            ax.add_patch(plt.Rectangle((x0, y0), wd, ht, fill=False, ec="cyan", lw=1))
        for name, (x, y, z) in points().items():
            if fname == "dose_plan.png" and abs(z) < 0.3:
                ax.plot(x, y, "c+", ms=8)
            if fname == "dose_side.png" and abs(y) < 0.3:
                ax.plot(x, z, "c+", ms=8)
        ax.set_aspect("equal"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"H*(10) rate, {mode}, {t_shield:g} cm poly ({'chamber height' if 'plan' in fname else 'through chamber'})")
        fig.colorbar(im, ax=ax, label="µSv/h (peak)", shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(case_dir, fname), dpi=130); plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("post", help="run_X/post folder (has source_terms.json and openmc_sources.py)")
    ap.add_argument("--mode", choices=["neutron", "xray", "xray-bound", "xray-kramers"], default="neutron",
                    help="neutron (with secondary gammas), xray (TTB electrons), xray-bound (all power as 60 keV "
                         "electrons), xray-kramers (photon source cross-check)")
    ap.add_argument("--shield", type=float, nargs="+", default=[20.0], help="poly thickness(es), cm")
    ap.add_argument("--borated", action="store_true", help="5 wt%% borated poly instead of plain HDPE")
    ap.add_argument("--particles", type=float, default=1e6, help="particles per batch")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="10 batches x 1e4: check everything works")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="only make geometry pictures")
    ap.add_argument("--post-only", action="store_true", help="only re-read finished runs")
    ap.add_argument("--out", default=None, help="output folder (default: <run>/openmc)")
    ap.add_argument("--minutes-per-week", type=float, default=MINUTES_PER_WEEK,
                    help="beam-on time per week for the yearly dose (default %(default)s)")
    a = ap.parse_args()
    global SHOTS_PER_YEAR
    SHOTS_PER_YEAR = a.minutes_per_week * 60 / 15 * 52

    post_dir = os.path.abspath(a.post)
    sys.path.insert(0, post_dir)
    import openmc
    import openmc_sources as fs
    D = json.load(open(os.path.join(post_dir, "source_terms.json")))
    rp_path = os.path.join(os.path.dirname(post_dir), "run_params.json")
    run_params = json.load(open(rp_path)) if os.path.exists(rp_path) else {}
    out_root = a.out or os.path.join(os.path.dirname(post_dir), "openmc")
    os.makedirs(out_root, exist_ok=True)
    if a.quick:
        a.particles, a.batches = 1e4, 10

    if a.mode == "neutron":
        sources = fs.neutron_sources(scenario="peak")
    elif a.mode == "xray":
        sources = fs.electron_sources(bound=False)
    elif a.mode == "xray-bound":
        sources = fs.electron_sources(bound=True)
    else:
        sources = fs.photon_sources_kramers(bound=False)
    S, sec_per_shot = strength_and_shot(D, sources, "neutron" if a.mode == "neutron" else "xray", fs)

    summary = {}
    for t in a.shield:
        case = os.path.join(out_root, f"{a.mode}_shield{t:g}cm" + ("_borated" if a.borated else ""))
        os.makedirs(case, exist_ok=True)
        if not a.post_only:
            openmc.reset_auto_ids()
            geom, mats, M = build_geometry(openmc, D, run_params, t, a.borated)
            st = openmc.Settings()
            st.run_mode = "fixed source"
            st.particles, st.batches = int(a.particles), a.batches
            st.source = sources
            st.photon_transport = True
            st.electron_treatment = "ttb"
            st.output = {"tallies": False}
            tallies, _ = build_tallies(openmc)
            model = openmc.Model(geometry=geom, materials=mats, settings=st, tallies=tallies,
                                 plots=build_plots(openmc, M, list(mats)))
            model.export_to_xml(directory=case)
            if a.plot:
                openmc.plot_geometry(cwd=case)
                print(f"geometry pictures in {case}/geometry_*.png"); continue
            for f in glob.glob(os.path.join(case, "statepoint.*.h5")):
                os.remove(f)
            print(f"\n=== {a.mode}, shield {t:g} cm: {st.particles * st.batches:.1e} particles -> {case}")
            openmc.run(cwd=case, threads=a.threads)
        summary[t] = post(case, D, a.mode, S, sec_per_shot, t)

    if len(summary) > 1:
        names = list(points())
        th = list(summary)
        lines = [f"# Shield comparison -- {a.mode}{' (borated)' if a.borated else ''}\n",
                 "Peak dose rate µSv/h (mSv/yr at " + f"{SHOTS_PER_YEAR:.0f} shots/yr = {SHOTS_PER_YEAR * 15 / 3600:.1f} h of beam)\n",
                 "| Location | " + " | ".join(f"{t:g} cm" for t in th) + " |",
                 "|---|" + "---|" * len(th)]
        for i, nm in enumerate(names):
            lines.append(f"| {nm} | " + " | ".join(
                f"{summary[t][i]['total_uSv_h']:.3g} ({summary[t][i]['mSv_per_year']:.2g})" for t in th) + " |")
        lines += ["", REFS]
        p = os.path.join(out_root, f"summary_{a.mode}{'_borated' if a.borated else ''}.md")
        open(p, "w").write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines) + f"\n\nWrote {p}")


if __name__ == "__main__":
    main()
