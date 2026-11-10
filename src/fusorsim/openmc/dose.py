#!/usr/bin/env python3
"""
fusor_openmc.py -- dose-rate model of the IEC fusor in room G131 (Eng 37, Clayton).

Takes the source terms written by fusor-post (run_X/post/source_terms.json and
openmc_sources.py), puts them in a model of the chamber, the poly wall lining and
G131 with its neighbours, and tallies ambient dose equivalent H*(10) -- the
quantity a survey meter reads -- on maps and at named points.

    fusor-openmc run_R1/post --plot                     # geometry pictures only (seconds)
    fusor-openmc run_R1/post --shield box20 --quick         # 10^5 neutrons, checks it all works
    fusor-openmc run_R1/post --shield 0 box20 box20b        # bare, 20 cm HDPE box, borated (b suffix)
    fusor-openmc run_R1/post --shield boxlid20 lining10     # box with a lid; poly lining on walls 3 and 2
    fusor-openmc run_R1/post --mode xray --shield 0 box20   # X-ray dose (TTB electrons)
    fusor-openmc run_R1/post --post-only --shield box20     # redo tables/plots from finished runs

Outputs go to run_R1/openmc/<mode>_<bare|box20cm|...>[_borated]/ : dose_points.md/.csv,
dose_plan.png (plan at chamber height), dose_side_xz.png / dose_side_yz.png (vertical
cuts through the chamber), dose_mesh.npz, openmc_params.json, geometry_*.png; plus
run_R1/openmc/summary_<mode>.md and compare_plan_<mode>.png across the shield cases.

Coordinate frame (same as the WarpX/source files): origin = chamber centre, z = chamber
axis = up. In the floor-plan frame: +x = towards wall 5 / G132, -x = towards wall 2 / G130,
+y = towards wall 4 / the road, -y = towards wall 3 / the lane (wall numbers as marked on
Elijah's plan, 2026-10-26). All ROOM numbers below are in metres.
"""
import argparse, csv, glob, hashlib, json, os, re, subprocess, sys
import numpy as np

# ============================== EDIT ME ==================================
# Distances to walls are to the INSIDE face of G131's walls from the chamber CENTRE, except wall 3 (LANE,
# to its lane side). If they were measured from the chamber surface every wall is ~16 cm further away,
# so this reading is conservative.
TABLE_TOP = 0.80        # m, floor to table top, scaled off the photo (pumps ~40 cm tall)     [MEASURE]
CHAMBER_STAND = 0.10    # m, table top to the chamber's bottom flange, from the sketch         [MEASURE]
CHAMBER_TALL = 0.50     # m, the new chamber, ~36 cm OD [Elijah]. The chamber modelled in OpenMC is still the
                        # one the source terms came from (R1); only its centre height follows the new one
FLOOR = TABLE_TOP + CHAMBER_STAND + CHAMBER_TALL / 2   # m, floor below chamber centre
CEILING = 2.50          # m, floor to ceiling; no storey above             [Elijah]
W2, W4, W5 = 0.65, 6.75, 3.55   # m to the inside faces of walls 2 (G130), 4 (road), 5 (G132)   [Elijah]
LANE = 1.40             # m to the lane side of wall 3: the closest anyone outside can get [Elijah 2026-10-27];
                        # wall 3's inside face (W3, below WALL_LAYERS) is its thickness nearer the chamber
G130_WIDTH = 2.84       # m, wall 2 to wall 1, scaled off the plan (1.56 cm/px from W2 + W5)   [MEASURE]
G132_DEPTH = 5.75       # m, wall 5 to the far side of G132, scaled off the plan
G132_TOP = 1.80         # m, y of G132's wall to G132A, scaled off the plan; wall 5 runs -W3..here
ROAD = 6.0              # m, road edge beyond wall 4's outside face        [Elijah]
WALL1_TOP = 2.10        # m above the floor; wall 1 is open above it        [Elijah]
DOOR = dict(x=(2.65, 3.55), height=2.10, wood=0.04)   # 4 cm solid wood [Elijah]; 0.9 x 2.1 m at the
                        # east end of wall 3 (next to wall 5), as read off the plan -- CHECK
WALL_LAYERS = dict(     # inside -> outside: (material, thickness m, solid fraction of that thickness)
    w1=[("block", 0.24, 1.0)],                       # cinder block (homogenised hollow block)    [Elijah]
    w2=[("plaster", 0.10, 0.04 / 0.10)],             # 2 x 2 cm plasterboard; 10 cm stud assumed  [Elijah]
    w3=[("plaster", 0.013, 1.0), ("brick", 0.11, 1.0)],                  # 1 sheet + 11 cm brick [Elijah]
    w4=[("plaster", 0.013, 1.0), ("brick", 0.11, 1.0), ("steel", 0.0005, 1.0)],   # + corrugated iron
    w5=[("plaster", 0.15, 0.04 / 0.15)],             # 2 sheets in 15 cm; 2 cm sheets as wall 2  [Elijah]
    ceiling=[("plaster", 0.013, 1.0)],               # roof structure above ignored (conservative)
    ground=[("concrete", 1.0, 1.0)],                 # slab on ground, soil below taken as concrete
)
W3 = LANE - sum(t for _, t, _ in WALL_LAYERS["w3"])   # m, inside face of wall 3
# Walls the plan shows but nobody measured (G132A, G133, the building's end walls) are left out:
# less attenuation, so conservative for every dose point beyond them.
LINING = dict(          # --shield lining<t>: poly flat on the walls inside G131, floor to ceiling [Elijah]
    wall3_to=DOOR["x"][0],   # along wall 3 from the wall-2 corner to the door
    wall2_len=2.0,           # m along wall 2 from the wall-3 corner (corner piece counted)
)
BOX = dict(             # --shield box...: poly box on the table around the chamber            [Elijah 2026-10-27]
    inner=0.75,              # m, inside width both ways (36 cm OD chamber + room for its ports)
    height=1.00,             # m, walls from the table top up
    under=0.20,              # m, slab under the table top, box footprint (pumps, floor), unless the token sets u
    lid=0.20,                # m, lid on 'boxlid' cases, unless the token sets l
    gap=0.02,                # m kept between the box and a room wall: the box slides off wall 2 / wall 3
    clear=0.05,              # m, least room between the chamber (CHAMBER_OD) and the box's inside faces
)                            # 'box' leaves the top open; 'boxlid' adds the lid
CHAMBER_OD = 0.36       # m, the new chamber [Elijah]; used only for the box clearance check
CHAMBER_WALL = 0.020    # m, stainless steel (Elijah: 2 cm); end plates assumed the same
VIEWPORT = dict(dir="north", diameter=0.10, glass=0.005)   # all viewports face +y;
                        # for several viewports set diameter so the total glass area matches
WORLD = (-15.0, 20.0, -12.0, 25.0, 30.0)   # m, x0 x1 y0 y1 z1 of the air around the building:
                        # 30 m of air overhead so roof skyshine onto the lane and road is scored
NEUTRON_FOS = 5.0       # factor of safety on the R1 peak neutron rate. The largest uncertainty is the
                        # implanted-D loading f of the wall/cathode targets; x5 on 1.75e7 n/s (8.75e7 n/s)
                        # equals the f = 0.4 saturation row of the R1 report (override with --fos)
BEAM_HOURS = 960.0      # h of beam a year [Elijah 2026-10-27] (override with --hours-per-year). The yearly
                        # dose takes the PEAK rate for every hour: no credit for the cathode loading ramp
DOSE_TARGET = 1.0       # mSv/yr design target at the named points outside G131 [Elijah 2026-10-27]
# =========================================================================

CM = 100.0
REFS = f"Design target {DOSE_TARGET:g} mSv/yr outside G131. Reference: public 1 mSv/yr, occupational 20 mSv/yr, proposed constraint 0.3 mSv/yr (Section 5.2.2 of the RMP)."


def thick(key):
    return sum(t for _, t, _ in WALL_LAYERS[key])


def frame():
    """Derived planes of the building, m."""
    zf = -FLOOR
    x2o = -W2 - thick("w2")                 # G130 face of wall 2
    x1 = x2o - G130_WIDTH                   # G130 face of wall 1
    return dict(zf=zf, zc=CEILING - FLOOR, x2o=x2o, x1=x1, xW=x1 - thick("w1"),
                x5o=W5 + thick("w5"), xE=W5 + thick("w5") + G132_DEPTH,
                y3o=-W3 - thick("w3"), y4o=W4 + thick("w4"), zg=zf - thick("ground"))


SIDES = dict(w="G130 (west)", e="G132 (east)", s="lane (south)", n="back (north)")


def parse_shield(tok, borated=False):
    """Shield token -> dict(kind, t (m) or sides {w e s n u l: m}, borated). Forms (cm, suffix b = borated):
    0 | lining10 | box20 (all four sides) | boxw36e10s30n10[u20][l20] (per side) | boxlid... (with a lid)."""
    s = tok.strip().lower()
    b = s.endswith("b")
    s = s.removesuffix("b")
    num = r"\d+(?:\.\d+)?"
    if re.fullmatch(num, s) and float(s) == 0:
        return dict(kind="none", t=0.0, borated=False)
    m = re.fullmatch(rf"(lining|boxlid|box)?({num})", s)
    if m and m.group(1) == "lining":
        return dict(kind="lining", t=float(m.group(2)) / CM, borated=b or borated)
    if m:
        sides = dict.fromkeys("wesn", float(m.group(2)) / CM)
    else:
        m = re.fullmatch(rf"(boxlid|box)((?:[wesnul]{num})+)", s)
        if not m:
            sys.exit(f"--shield {tok!r}: expected 0, box20, boxw36e10s30n10, boxlid20, lining10 (suffix b = borated)")
        sides = {k: float(v) / CM for k, v in re.findall(rf"([wesnul])({num})", m.group(2))}
        if set("wesn") - set(sides):
            sys.exit(f"--shield {tok!r}: give all four sides w, e, s, n")
    kind = m.group(1) or "box"
    sides.setdefault("u", BOX["under"])
    if kind == "boxlid":
        sides.setdefault("l", BOX["lid"])
    return dict(kind=kind, sides=sides, borated=b or borated)


def _cm(v):
    return f"{v * CM:g}"


def case_label(sh):
    if sh["kind"] == "none":
        return "no shield"
    mat = "borated PE" if sh["borated"] else "HDPE"
    if sh["kind"] == "lining":
        return f"{_cm(sh['t'])} cm {mat} wall lining"
    S = sh["sides"]
    walls = f"{_cm(S['w'])} cm" if len({S[k] for k in 'wesn'}) == 1 else \
        "W/E/S/N " + "/".join(_cm(S[k]) for k in "wesn") + " cm"
    return f"{mat} box {walls}, under {_cm(S['u'])}" + (f", lid {_cm(S['l'])}" if "l" in S else ", open top")


def case_dir_name(mode, sh):
    if sh["kind"] == "none":
        return f"{mode}_bare"
    if sh["kind"] == "lining":
        name = f"lining{_cm(sh['t'])}cm"
    else:
        name = sh["kind"] + "_" + "".join(k + _cm(v) for k, v in sh["sides"].items())
    return f"{mode}_{name}" + ("_borated" if sh["borated"] else "")


def box_frame(sh):
    """Inside faces (x0, x1, y0, y1) of the box, m. Centred on the chamber, unless a thick side would
    reach wall 2 or wall 3: then the box slides away from that wall and the chamber sits off-centre."""
    S, I = sh["sides"], BOX["inner"] / 2
    cx = max(0.0, -W2 + BOX["gap"] + I + S["w"])
    cy = max(0.0, -W3 + BOX["gap"] + I + S["s"])
    return cx - I, cx + I, cy - I, cy + I


def shield_mass(sh):
    """kg of poly in a case."""
    rho = 1000.0 if sh.get("borated") else 950.0
    return sum(np.prod([s["box"][2 * i + 1] - s["box"][2 * i] for i in range(3)]) * rho
               for s in layout(sh) if s["kind"] == "shield")


def layout(sh):
    """Every solid outside the chamber: dicts of name, material key, solid fraction, box
    (x0, x1, y0, y1, z0, z1) in m, and an optional box cut out of it. sh from parse_shield."""
    F = frame()
    zf, zc = F["zf"], F["zc"]
    out = []

    def stack(key, axis, face, sign, span, cut=None):
        p = face
        for mat, t, frac in WALL_LAYERS[key]:
            q = p + sign * t
            lim = dict(span); lim[axis] = (min(p, q), max(p, q))
            out.append(dict(name=f"{key} {mat}", mat=mat, frac=frac, kind="wall", cut=cut,
                            box=lim["x"] + lim["y"] + lim["z"]))
            p = q

    door_hole = (*DOOR["x"], F["y3o"], -W3, zf, zf + DOOR["height"])
    stack("w3", "y", -W3, -1, dict(x=(F["xW"], F["xE"]), z=(zf, zc)), cut=door_hole)
    stack("w4", "y", W4, +1, dict(x=(F["xW"], F["xE"]), z=(zf, zc)))
    stack("w2", "x", -W2, -1, dict(y=(-W3, W4), z=(zf, zc)))
    stack("w1", "x", F["x1"], -1, dict(y=(-W3, W4), z=(zf, zf + WALL1_TOP)))
    stack("w5", "x", W5, +1, dict(y=(-W3, G132_TOP), z=(zf, zc)))
    stack("ceiling", "z", zc, +1, dict(x=(F["xW"], F["xE"]), y=(F["y3o"], F["y4o"])))
    stack("ground", "z", zf, -1, dict(x=WORLD[0:2], y=WORLD[2:4]))
    out.append(dict(name="door (wood)", mat="wood", frac=1.0, kind="door", cut=None,
                    box=(*DOOR["x"], F["y3o"], F["y3o"] + DOOR["wood"], zf, zf + DOOR["height"])))
    def poly(name, b):
        out.append(dict(name=name, mat="poly", frac=1.0, kind="shield", cut=None, box=b))

    if sh["kind"] == "lining":
        t = sh["t"]
        poly("poly lining, wall 3", (-W2, LINING["wall3_to"], -W3, -W3 + t, zf, zc))
        poly("poly lining, wall 2", (-W2, -W2 + t, -W3 + t, -W3 + LINING["wall2_len"], zf, zc))
    elif sh["kind"] in ("box", "boxlid"):
        S = sh["sides"]
        x0, x1, y0, y1 = box_frame(sh)
        X0, X1, Y0, Y1 = x0 - S["w"], x1 + S["e"], y0 - S["s"], y1 + S["n"]
        zt = TABLE_TOP - FLOOR
        zb = zt + BOX["height"]
        for name, b in (("west", (X0, x0, Y0, Y1)), ("east", (x1, X1, Y0, Y1)),      # x sides full depth,
                        ("south", (x0, x1, Y0, y0)), ("north", (x0, x1, y1, Y1))):   # y sides between them
            poly(f"poly box, {name} side", (*b, zt, zb))
        if S["u"] > 0:
            poly("poly box, under the table top", (X0, X1, Y0, Y1, zt - S["u"], zt))
        if S.get("l", 0) > 0:
            poly("poly box, lid", (X0, X1, Y0, Y1, zb, zb + S["l"]))
    return out


def points():
    """Named dose points: name -> ((x, y, z) m, scoring box size m: a cube side or (dx, dy, dz))."""
    F = frame()
    lane = (0.4, 0.1, 0.4)              # a thin box flat on the lane side of wall 3: dose at the closest approach
    y_lane = F["y3o"] - lane[1] / 2
    P = {"G131, 1 m from chamber": ((1.0, 0.0, 0.0), 0.4),
         "G130, 0.3 m behind wall 2": ((F["x2o"] - 0.3, 0.0, 0.0), 0.4),
         "G130, middle": (((F["x2o"] + F["x1"]) / 2, (W4 - W3) / 2, 0.0), 0.4),
         "G132, middle": (((F["x5o"] + F["xE"]) / 2, (G132_TOP - W3) / 2, 0.0), 0.4)}
    for x in (-0.5, 0.0, 1.0, 2.0):
        P[f"Lane, against wall 3, x {x:+.1f} m"] = ((x, y_lane, 0.0), lane)
    P["Lane, against the door"] = ((sum(DOOR["x"]) / 2, y_lane, 0.0), lane)
    P[f"Road, {ROAD:g} m past wall 4"] = ((0.0, F["y4o"] + ROAD, 0.0), 1.0)   # bigger cube: far, flat field
    # bare dirt between wall 4 and the road: nothing there, but people can stand on it [Elijah 2026-10-27];
    # scored as fully occupied (no occupancy factor), against the wall opposite the chamber
    P["Dirt strip, against wall 4"] = ((0.0, F["y4o"] + lane[1] / 2, 0.0), lane)
    return P


# --------------------------------------------------------------- materials
COLORS = dict(steel=(120, 120, 130), glass=(120, 200, 255), poly=(250, 230, 150), air=(255, 255, 255),
              concrete=(150, 150, 150), block=(170, 160, 150), brick=(170, 80, 60), plaster=(215, 215, 190),
              wood=(160, 110, 60), cathode=(255, 150, 0), anode=(0, 120, 0))


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
    wood = openmc.Material(name="wood (cellulose, light)")    # low end for solid timber: conservative
    wood.add_elements_from_formula("C6H10O5")
    wood.set_density("g/cm3", 0.60); M["wood"] = wood
    return M


def grid_shell(openmc, R, d, T, rho, el, name):
    """Wire grid homogenised into a spherical shell one wire diameter thick:
    wire volume fraction = (1 - T) * pi/4 (round wires on a d x d footprint)."""
    m = openmc.Material(name=name); m.add_element(el, 1.0)
    m.set_density("g/cm3", rho * (1 - T) * np.pi / 4)
    return m, openmc.Sphere(r=(R - d / 2) * CM), openmc.Sphere(r=(R + d / 2) * CM)


# ---------------------------------------------------------------- geometry
def build_geometry(openmc, D, run_params, sh):
    g = D["geometry"]
    M = materials(openmc, sh["borated"])
    key_of = {id(m): k for k, m in M.items()}
    Rch, Hch = g["R_ch"] * CM, g["H_ch"] * CM
    Ro, Ho = Rch + CHAMBER_WALL * CM, Hch + CHAMBER_WALL * CM
    d_c = run_params.get("d_c", 0.002); d_a = run_params.get("d_a", 0.001)
    cath, c_in, c_out = grid_shell(openmc, g["R_c"], d_c, g["T_c"], 19.3, "W", "cathode grid (W, homogenised)")
    anod, a_in, a_out = grid_shell(openmc, g["R_a"], d_a, g["T_a"], 7.87, "Fe", "anode grid (steel, homogenised)")
    M["cathode"], M["anode"] = cath, anod
    key_of[id(cath)], key_of[id(anod)] = "cathode", "anode"

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

    # the shield must stay clear of the chamber and of the building
    def overlap(a, b):
        return all(a[2 * i] < b[2 * i + 1] and b[2 * i] < a[2 * i + 1] for i in range(3))

    solid_list = layout(sh)
    r_new = CHAMBER_OD / 2 + BOX["clear"]          # the real (new) chamber plus working room, not the R1 one
    chamber_bb = (-max(Ro / CM, r_new), max(Ro / CM, r_new), -max(Ro / CM, r_new), max(Ro / CM, r_new),
                  -max(Ho / CM, CHAMBER_TALL / 2), max(Ho / CM, CHAMBER_TALL / 2))
    for s in (s for s in solid_list if s["kind"] == "shield"):
        hit = [o["name"] for o in solid_list if o["kind"] != "shield" and overlap(s["box"], o["box"])]
        if overlap(s["box"], chamber_bb) or hit:
            sys.exit(f"{s['name']} ({case_label(sh)}) runs into {hit or 'the chamber'} -- change --shield or BOX.")

    # building: boxes of solid, everything else air out to the vacuum boundary
    F = frame()
    planes = {}

    def plane(axis, v):
        k = (axis, round(v * CM, 6))
        if k not in planes:
            planes[k] = {"x": openmc.XPlane, "y": openmc.YPlane, "z": openmc.ZPlane}[axis](k[1])
        return planes[k]

    def box(b):
        return (+plane("x", b[0]) & -plane("x", b[1]) & +plane("y", b[2]) & -plane("y", b[3]) &
                +plane("z", b[4]) & -plane("z", b[5]))

    world_b = (*WORLD[:4], F["zg"], WORLD[4])
    world = box(world_b)
    for axis, v in zip("xxyyzz", world_b):
        plane(axis, v).boundary_type = "vacuum"
    mats = {}
    solids = []
    for s in solid_list:
        if s["frac"] < 1.0:     # a stud wall: its sheets spread over the wall thickness (same areal mass)
            k = (s["mat"], s["frac"])
            if k not in mats:
                m = M[s["mat"]].clone(); m.name = f"{M[s['mat']].name} ({100 * s['frac']:.0f} % of wall)"
                m.set_density("g/cm3", M[s["mat"]].density * s["frac"])
                mats[k] = m; key_of[id(m)] = s["mat"]
            fill = mats[k]
        else:
            fill = M[s["mat"]]
        reg = box(s["box"])
        if s["cut"] is not None:      # e.g. the doorway: what the door leaf doesn't fill stays air
            reg = reg & ~box(s["cut"])
        solids.append(reg)
        cells.append(openmc.Cell(name=s["name"], fill=fill, region=reg))
    cells.append(openmc.Cell(name="air", fill=M["air"], region=world & ~outer & ~openmc.Union(solids)))

    geom = openmc.Geometry(cells)
    used = {c.fill.name: c.fill for c in cells if c.fill is not None}
    colors = {m: COLORS[key_of[id(m)]] for m in used.values()}
    return geom, openmc.Materials(list(used.values())), colors


# ----------------------------------------------------------------- tallies
VOXEL = 0.20      # m, 3-D dose map
PIXEL = 0.05      # m, slice maps
SLAB = 0.40       # m, thickness each slice map averages over (centred on the chamber)
MAP = dict(x=(-5.0, 11.0), y=(-3.5, 14.0), z=(-FLOOR - 1.0, 4.0))   # m, extent of all maps
SLICES = {"plan": ("xy", "z"), "side_xz": ("xz", "y"), "side_yz": ("yz", "x")}   # shown axes, fixed axis


def dose_filters(openmc):
    out = {}
    for p in ("neutron", "photon"):
        E, h = openmc.data.dose_coefficients(p, data_source="icrp74", dose_quantity="ambient")  # H*(10), pSv cm2
        out[p] = openmc.EnergyFunctionFilter(E, h, interpolation="linear-linear")
    return out


def mesh_spec(name):
    """(lower_left, upper_right, dimension) in m of a named map mesh: '3d' or a SLICES key."""
    if name == "3d":
        dims = {a: int(round((MAP[a][1] - MAP[a][0]) / VOXEL)) for a in "xyz"}
        lo = {a: MAP[a][0] for a in "xyz"}
        hi = {a: MAP[a][0] + dims[a] * VOXEL for a in "xyz"}
    else:
        shown, fixed = SLICES[name]
        dims = {a: int(round((MAP[a][1] - MAP[a][0]) / PIXEL)) for a in shown}
        lo = {a: MAP[a][0] for a in shown}
        hi = {a: MAP[a][0] + dims[a] * PIXEL for a in shown}
        lo[fixed], hi[fixed], dims[fixed] = -SLAB / 2, SLAB / 2, 1
    return [lo[a] for a in "xyz"], [hi[a] for a in "xyz"], [dims[a] for a in "xyz"]


def build_tallies(openmc):
    DF = dose_filters(openmc)
    tallies = []
    for name in ("3d", *SLICES):
        lo, hi, dim = mesh_spec(name)
        mesh = openmc.RegularMesh(name=f"map {name}")
        mesh.lower_left = [v * CM for v in lo]; mesh.dimension = dim; mesh.upper_right = [v * CM for v in hi]
        for p in ("neutron", "photon"):
            t = openmc.Tally(name=f"map_{name}_{p}")
            t.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter([p]), DF[p]]
            t.scores = ["flux"]; tallies.append(t)
    for i, (name, ((x, y, z), side)) in enumerate(points().items()):
        pm = openmc.RegularMesh(name=name)
        c, h = np.array([x, y, z]) * CM, np.broadcast_to(side, 3) * CM / 2
        pm.lower_left, pm.upper_right, pm.dimension = c - h, c + h, [1, 1, 1]
        for p in ("neutron", "photon"):
            t = openmc.Tally(name=f"pt{i}_{p}")
            t.filters = [openmc.MeshFilter(pm), openmc.ParticleFilter([p]), DF[p]]
            t.scores = ["flux"]; tallies.append(t)
    return openmc.Tallies(tallies)


def build_plots(openmc, colors):
    plots = []
    zc = CEILING - FLOOR
    for name, basis, origin, width in (
            ("geometry_plan", "xy", (sum(MAP["x"]) / 2, sum(MAP["y"]) / 2, 0.0), (np.ptp(MAP["x"]), np.ptp(MAP["y"]))),
            ("geometry_plan_above_wall1", "xy", (sum(MAP["x"]) / 2, sum(MAP["y"]) / 2, (WALL1_TOP - FLOOR + zc) / 2),
             (np.ptp(MAP["x"]), np.ptp(MAP["y"]))),
            ("geometry_side_xz", "xz", (sum(MAP["x"]) / 2, 0.0, sum(MAP["z"]) / 2), (np.ptp(MAP["x"]), np.ptp(MAP["z"]))),
            ("geometry_side_yz", "yz", (0.0, sum(MAP["y"]) / 2, sum(MAP["z"]) / 2), (np.ptp(MAP["y"]), np.ptp(MAP["z"]))),
            ("geometry_corner", "xy", (0.0, -0.4, 0.0), (2.6, 2.6)),
            ("geometry_shield_side", "xz", (0.0, 0.0, (CEILING - 2 * FLOOR) / 2), (2.6, CEILING)),
            ("geometry_chamber_side", "yz", (0.0, 0.0, 0.0), (0.6, 0.6))):
        w = [v * CM for v in width]
        pl = openmc.SlicePlot(name=name)
        pl.filename, pl.basis, pl.origin, pl.width = name, basis, tuple(v * CM for v in origin), w
        pl.pixels = (int(3000 * w[0] / max(w)), int(3000 * w[1] / max(w)))
        pl.color_by = "material"; pl.colors = colors
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


def post(case_dir, D, mode, S, sec_per_shot, sh, fos=1.0):
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
    year = BEAM_HOURS / 1000                                  # mSv/yr per uSv/h of peak rate

    def val(name, vol_m3):
        t = sp.get_tally(name=name)
        vol = vol_m3 * CM ** 3
        return t.mean.ravel() / vol * per_h, t.std_dev.ravel() / vol * per_h

    rows = []
    for i, (name, (xyz, side)) in enumerate(points().items()):
        vol = float(np.prod(np.broadcast_to(side, 3)))
        n, dn = val(f"pt{i}_neutron", vol); g, dg = val(f"pt{i}_photon", vol)
        tot = n[0] + g[0]; err = np.hypot(dn[0], dg[0])
        rows.append(dict(point=name, x_m=xyz[0], y_m=xyz[1], z_m=xyz[2], dist_m=float(np.linalg.norm(xyz)),
                         neutron_uSv_h=n[0], photon_uSv_h=g[0], total_uSv_h=tot,
                         rel_err=err / tot if tot > 0 else float("nan"),
                         uSv_per_shot=tot / 3600 * sec_per_shot, mSv_per_year=tot * year))
    with open(os.path.join(case_dir, "dose_points.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    maps = {}
    for name in ("3d", *SLICES):
        lo, hi, (nx, ny, nz) = mesh_spec(name)
        vol = np.prod((np.array(hi) - np.array(lo)) / np.array([nx, ny, nz]))
        for p in ("neutron", "photon"):
            m, s = val(f"map_{name}_{p}", vol)
            m, s = (a.reshape(nz, ny, nx).transpose(2, 1, 0) for a in (m, s))
            if name != "3d":
                m, s = (a.squeeze(axis="xyz".index(SLICES[name][1])) for a in (m, s))
            maps[f"{name}_{p}"], maps[f"{name}_{p}_sd"] = m, s
        maps[f"{name}_total"] = maps[f"{name}_neutron"] + maps[f"{name}_photon"]
    np.savez_compressed(os.path.join(case_dir, "dose_mesh.npz"), units="uSv/h at peak source rate",
                        mSv_per_year_per_uSv_h=year, voxel_m=VOXEL, pixel_m=PIXEL, slab_m=SLAB,
                        **{f"extent_{a}_m": np.array(MAP[a]) for a in "xyz"}, **maps)

    lines = [f"# Fusor dose rates -- {mode}, {case_label(sh)}\n",
             f"Shield mass {shield_mass(sh):.0f} kg. Source:{S:.3e} {'n' if mode == 'neutron' else 'e'}/s "
             f"({f'peak x FOS {fos:g}' if mode == 'neutron' else 'TTB electrons'}); "
             f"{nps:.2e} histories; H*(10) with ICRP 74 coefficients. Per shot = {sec_per_shot:.1f} s at this rate; "
             f"mSv/yr = {BEAM_HOURS:g} h of beam a year, all at the peak rate.\n",
             "| Location | x, y, z (m) | dist (m) | neutron µSv/h | photon µSv/h | total µSv/h | ±% | µSv/shot | mSv/yr |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['point']} | {r['x_m']:.1f}, {r['y_m']:.1f}, {r['z_m']:.1f} | {r['dist_m']:.1f} | "
                     f"{r['neutron_uSv_h']:.3g} | {r['photon_uSv_h']:.3g} | {r['total_uSv_h']:.3g} | "
                     f"{100 * r['rel_err']:.0f} | {r['uSv_per_shot']:.3g} | {r['mSv_per_year']:.3g} |")
    lines += ["", REFS, "±% is the 1-sigma Monte Carlo error; above ~20 % run more particles (--particles)."]
    open(os.path.join(case_dir, "dose_points.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_maps(case_dir, maps, rows, mode, sh, fos, year)
    return rows, maps


# --------------------------------------------------------------- dose maps
NORM = (1e-2, 1e4)    # µSv/h, fixed colour range so every case is on the same scale
LEVELS = (0.3, 1.0)   # mSv/yr contours: proposed constraint, public limit


def _smooth(a):
    """3 x 3 mean in log space -- only for drawing contours through Monte Carlo noise."""
    la = np.log(np.where(a > 0, a, np.nan))
    p = np.pad(la, 1, mode="edge")
    st = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
    return np.exp(np.nanmean(st, axis=0))


def draw_slice(ax, plt, data, name, sh, year, rows=None):
    from matplotlib.colors import LogNorm
    import warnings
    shown, fixed = SLICES[name]
    lo, hi, dim = mesh_spec(name)
    i, j, k = ("xyz".index(shown[0]), "xyz".index(shown[1]), "xyz".index(fixed))
    ea = np.linspace(lo[i], hi[i], dim[i] + 1); eb = np.linspace(lo[j], hi[j], dim[j] + 1)
    im = ax.pcolormesh(ea, eb, np.where(data > 0, data, np.nan).T, norm=LogNorm(*NORM), cmap="inferno",
                       shading="flat", rasterized=True)
    ca, cb = (ea[1:] + ea[:-1]) / 2, (eb[1:] + eb[:-1]) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ax.contour(ca, cb, _smooth(data).T, levels=[L / year for L in LEVELS],
                        colors=["#7CFC00", "#00E5FF"], linewidths=1.2)
    ax.clabel(cs, fmt={L / year: f"{L:g} mSv/yr" for L in LEVELS}, fontsize=7)
    for s in layout(sh):           # outline every solid the slice plane cuts
        b = s["box"]
        if not b[2 * k] <= 0.0 <= b[2 * k + 1] or s["name"].startswith("ground"):
            continue
        ec = {"shield": "#7CFC00", "door": "#FFD700"}.get(s["kind"], "cyan")
        ax.add_patch(plt.Rectangle((b[2 * i], b[2 * j]), b[2 * i + 1] - b[2 * i], b[2 * j + 1] - b[2 * j],
                                   fill=False, ec=ec, lw=0.8))
    if "z" in shown:
        ax.axhline(-FLOOR, color="cyan", lw=0.8)
    if rows is not None:
        for n, r in enumerate(rows, 1):
            c = (r["x_m"], r["y_m"], r["z_m"])
            if abs(c[k]) < SLAB:
                ax.plot(c[i], c[j], "c+", ms=7)
                ax.annotate(f"{n}: {r['total_uSv_h']:.2g}", (c[i], c[j]), xytext=(4, 4),
                            textcoords="offset points", color="white", fontsize=7)
    ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j]); ax.set_aspect("equal")
    lab = dict(x="x (m) → G132 / wall 5", y="y (m) → road / wall 4", z="z (m, up)")
    ax.set_xlabel(lab[shown[0]]); ax.set_ylabel(lab[shown[1]])
    return im


def plot_maps(case_dir, maps, rows, mode, sh, fos, year):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing -- skipping dose maps"); return
    where = dict(plan=f"plan at chamber height (z = ±{SLAB / 2:g} m)",
                 side_xz=f"side cut through the chamber, y = ±{SLAB / 2:g} m",
                 side_yz=f"side cut through the chamber, x = ±{SLAB / 2:g} m")
    for name in SLICES:
        data = maps[f"{name}_total"]
        if not np.any(data > 0):
            print(f"{name} map empty (nothing scored) -- skipping"); continue
        lo, hi, _ = mesh_spec(name)
        shown = SLICES[name][0]
        wa, wb = (hi["xyz".index(a)] - lo["xyz".index(a)] for a in shown)
        fig, ax = plt.subplots(figsize=(11, 11 * wb / wa + 1.2))
        im = draw_slice(ax, plt, data, name, sh, year, rows)
        ax.set_title(f"H*(10) rate, {mode}{f' (FOS {fos:g})' if fos != 1 else ''}, {case_label(sh)}\n"
                     f"{where[name]}; points numbered as in dose_points.md, labels in µSv/h", fontsize=10)
        fig.colorbar(im, ax=ax, label="µSv/h at peak rate", shrink=0.8)
        fig.tight_layout(); fig.savefig(os.path.join(case_dir, f"dose_{name}.png"), dpi=220); plt.close(fig)


def compare_plans(out_root, mode, results, year):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    nc = min(n, 3); nr = -(-n // nc)
    lo, hi, _ = mesh_spec("plan")
    fig, axs = plt.subplots(nr, nc, figsize=(6.5 * nc, nr * (6.5 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 1.0)),
                            squeeze=False)
    for ax, (sh, rows, maps) in zip(axs.flat, results):
        im = draw_slice(ax, plt, maps["plan_total"], "plan", sh, year)
        ax.set_title(case_label(sh))
    for ax in list(axs.flat)[n:]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axs.ravel().tolist(), label="µSv/h at peak rate", shrink=0.6)
    p = os.path.join(out_root, f"compare_plan_{mode}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


def provenance():
    here = os.path.abspath(__file__)
    git = os.environ.get("FUSORSIM_GIT", "unknown")
    if git == "unknown":
        try:
            git = subprocess.run(["git", "-C", os.path.dirname(here), "describe", "--always", "--dirty"],
                                 capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
        except Exception:
            pass
    return dict(git=git, dose_py_sha256=hashlib.sha256(open(here, "rb").read()).hexdigest())


# --------------------------------------------------------------------- main
def main():
    global BEAM_HOURS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("post", help="run_X/post folder (has source_terms.json and openmc_sources.py)")
    ap.add_argument("--mode", choices=["neutron", "xray", "xray-bound", "xray-kramers"], default="neutron",
                    help="neutron (with secondary gammas), xray (TTB electrons), xray-bound (all power as 60 keV "
                         "electrons), xray-kramers (photon source cross-check)")
    ap.add_argument("--shield", nargs="+", default=["box20"],
                    help="shield case(s): 0 = none; box<cm> = poly box on the table, open top (BOX); boxlid<cm> = "
                         "the same with a lid; lining<cm> = wall 3 to the door + 2 m of wall 2 (LINING); "
                         "suffix b = 5 wt%% borated (e.g. 0 box20 box20b boxlid20)")
    ap.add_argument("--borated", action="store_true", help="every shield is 5 wt%% borated poly")
    ap.add_argument("--particles", type=float, default=1e6, help="particles per batch")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="10 batches x 1e4: check everything works")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--plot", action="store_true", help="only make geometry pictures")
    ap.add_argument("--post-only", action="store_true", help="only re-read finished runs")
    ap.add_argument("--out", default=None, help="output folder (default: <run>/openmc)")
    ap.add_argument("--hours-per-year", type=float, default=BEAM_HOURS,
                    help="beam-on time per year for the yearly dose, h (default %(default)s)")
    ap.add_argument("--fos", type=float, default=NEUTRON_FOS,
                    help="factor of safety on the neutron source strength, dimensionless; neutron mode only "
                         "(default %(default)s)")
    a = ap.parse_args()
    BEAM_HOURS = a.hours_per_year
    cases = [parse_shield(tok, a.borated) for tok in a.shield]

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
    fos = a.fos if a.mode == "neutron" else 1.0
    for s in sources:                  # scales the rate, not the shot length: dose per shot scales too
        s.strength *= fos
    S, sec_per_shot = strength_and_shot(D, sources, "neutron" if a.mode == "neutron" else "xray", fs)
    year = BEAM_HOURS / 1000

    results = []
    for sh in cases:
        case = os.path.join(out_root, case_dir_name(a.mode, sh))
        os.makedirs(case, exist_ok=True)
        if not a.post_only:
            openmc.reset_auto_ids()
            geom, mats, colors = build_geometry(openmc, D, run_params, sh)
            st = openmc.Settings()
            st.run_mode = "fixed source"
            st.particles, st.batches = int(a.particles), a.batches
            st.source = sources
            st.photon_transport = True
            st.electron_treatment = "ttb"
            st.output = {"tallies": False}
            model = openmc.Model(geometry=geom, materials=mats, settings=st, tallies=build_tallies(openmc),
                                 plots=build_plots(openmc, colors))
            model.export_to_xml(directory=case)
            json.dump(dict(mode=a.mode, fos=fos, source_per_s=S, shield=sh, box=BOX, shield_kg=shield_mass(sh),
                           box_inside_faces_m=box_frame(sh) if "sides" in sh else None,
                           particles=st.particles, batches=st.batches, floor_m=FLOOR, ceiling_m=CEILING,
                           table_top_m=TABLE_TOP, chamber_stand_m=CHAMBER_STAND, chamber_tall_m=CHAMBER_TALL,
                           walls_m=dict(W2=W2, W3=W3, W4=W4, W5=W5), g130_width_m=G130_WIDTH,
                           g132=dict(depth_m=G132_DEPTH, top_m=G132_TOP), road_m=ROAD, wall1_top_m=WALL1_TOP,
                           door=DOOR, wall_layers=WALL_LAYERS, lining=LINING, world_m=WORLD,
                           chamber_wall_m=CHAMBER_WALL, viewport=VIEWPORT, points=points(),
                           hours_per_year=BEAM_HOURS, dose_target_mSv=DOSE_TARGET, openmc=openmc.__version__,
                           cross_sections=os.environ.get("OPENMC_CROSS_SECTIONS"), **provenance()),
                      open(os.path.join(case, "openmc_params.json"), "w"), indent=1)
            if a.plot:
                openmc.plot_geometry(cwd=case)
                print(f"geometry pictures in {case}/geometry_*.png"); continue
            for f in glob.glob(os.path.join(case, "statepoint.*.h5")):
                os.remove(f)
            print(f"\n=== {a.mode}, {case_label(sh)}: {st.particles * st.batches:.1e} particles -> {case}", flush=True)
            openmc.run(cwd=case, threads=a.threads)
            openmc.plot_geometry(cwd=case, output=False)
        results.append((sh, *post(case, D, a.mode, S, sec_per_shot, sh, fos)))

    if len(results) > 1:
        names = list(points())
        lines = [f"# Shield comparison -- {a.mode}{f', FOS {fos:g}' if fos != 1 else ''}\n",
                 f"Peak dose rate µSv/h (mSv/yr at {BEAM_HOURS:g} h of beam a year)\n",
                 "| # | Location | " + " | ".join(case_label(sh) for sh, _, _ in results) + " |",
                 "|---|---|" + "---|" * len(results),
                 "| | Shield mass | " + " | ".join(f"{shield_mass(sh):.0f} kg" for sh, _, _ in results) + " |"]
        for i, nm in enumerate(names):
            lines.append(f"| {i + 1} | {nm} | " + " | ".join(
                f"{rows[i]['total_uSv_h']:.3g} ±{100 * rows[i]['rel_err']:.0f}% ({rows[i]['mSv_per_year']:.2g})"
                for _, rows, _ in results) + " |")
        lines += ["", REFS]
        p = os.path.join(out_root, f"summary_{a.mode}.md")
        open(p, "w").write("\n".join(lines) + "\n")
        fig = compare_plans(out_root, a.mode, results, year)
        print("\n" + "\n".join(lines) + f"\n\nWrote {p} and {fig}")


if __name__ == "__main__":
    main()
