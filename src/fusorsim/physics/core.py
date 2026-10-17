#!/usr/bin/env python3
"""
fusor_physics.py -- physics data and reduced models for the fusor v6 pipeline
==============================================================================

Pure numpy/scipy. No WarpX dependency, so every function here can be unit-
tested on its own (run ``python fusor_physics.py`` for the self-test).

Units: SI unless a name says otherwise (``_eV``, ``_keV``). Energies of fast
particles are LAB-frame kinetic energies of the projectile unless the name
says ``_cm`` (centre of mass).

Every number taken from the literature is cited in place. Where a model is an
approximation, the docstring says why it is acceptable and which direction it
errs in (conservative = over-predicts neutron/X-ray output or wire heating).
"""

from __future__ import annotations

import os
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x and 2.x

# ---------------------------------------------------------------------------
# Constants (CODATA 2018)
# ---------------------------------------------------------------------------
e = 1.602176634e-19          # C
kB = 1.380649e-23            # J/K
amu = 1.66053906660e-27      # kg
m_e = 9.1093837015e-31       # kg
m_p = 1.67262192369e-27      # kg  (proton)
m_d = 3.3435837724e-27       # kg  (deuteron, i.e. the D+ ion)
m_n = 1.67492749804e-27      # kg  (neutron)
m_He3 = 5.0064127796e-27     # kg  (helion)
m_D_atom = 2.01410177812 * amu
m_D2 = 2.0 * m_D_atom        # molecular binding (4.5 eV) is negligible here
c = 299792458.0              # m/s
eps0 = 8.8541878128e-12
a0 = 5.29177210903e-11       # Bohr radius, m
sigma_SB = 5.670374419e-8    # W m^-2 K^-4
I_H_eV = 13.598434           # hydrogen ionisation energy, eV
Ry_eV = 13.605693            # Rydberg energy, eV
MTORR_TO_PA = 0.133322368
Q_DDn_MeV = 3.2689           # D(d,n)3He Q value
Q_DDp_MeV = 4.0327           # D(d,p)T  Q value

def _find_data_dir():
    """FUSORSIM_DATA if set (containers), else <repo>/data (src layout), else ./data."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("FUSORSIM_DATA"),
             os.path.normpath(os.path.join(here, "..", "..", "..", "data")),
             os.path.join(os.getcwd(), "data")]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError("fusorsim data directory not found; set FUSORSIM_DATA")


DATA_DIR = _find_data_dir()


def gas_density(p_mTorr: float, T_K: float = 300.0) -> float:
    """D2 molecule number density (m^-3) from pressure and gas temperature.

    The pressure must be the TRUE D2 pressure. A capacitance manometer reads
    this directly; a Pirani or ion gauge calibrated for N2/air must be
    corrected with its D2 gas-correction factor before use.
    """
    return p_mTorr * MTORR_TO_PA / (kB * T_K)


def speed_from_energy(E_J, m):
    """Relativistically correct speed for kinetic energy E (J), mass m."""
    gam = 1.0 + np.asarray(E_J) / (m * c * c)
    return c * np.sqrt(1.0 - 1.0 / gam**2)


# ---------------------------------------------------------------------------
# 1. D-D fusion cross sections: Bosch & Hale, Nucl. Fusion 32 (1992) 611
#    Coefficients cross-checked against WarpX's own implementation
#    (Source/Particles/Collision/BinaryCollision/NuclearFusion/
#     BoschHaleFusionCrossSection.H, WarpX 26.01): identical.
#    Fit validity: E_cm = 0.5-5000 keV (D(d,n)), 0.5-4900 keV (D(d,p)),
#    stated accuracy ~2 % vs the R-matrix evaluation.
# ---------------------------------------------------------------------------
_BH = {
    "n": dict(BG=31.3970, A=(5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9)),
    "p": dict(BG=31.3970, A=(5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10)),
}


def sigma_DD(E_cm_keV, channel: str = "n"):
    """D+D fusion cross section (m^2) at centre-of-mass energy E_cm (keV).

    channel 'n': D(d,n)3He ; channel 'p': D(d,p)T. Returns 0 for E<=0.
    """
    E = np.asarray(E_cm_keV, dtype=float)
    out = np.zeros_like(E)
    ok = E > 0
    p = _BH[channel]
    A1, A2, A3, A4, A5 = p["A"]
    Ek = E[ok]
    S = A1 + Ek * (A2 + Ek * (A3 + Ek * (A4 + Ek * A5)))      # keV mb
    out[ok] = 1e-31 * S / Ek * np.exp(-p["BG"] / np.sqrt(Ek))   # mb -> m^2
    return out


def dd_beam_gas_rate_coeff(E_lab_eV, channel: str = "n"):
    """Reactions per second per fast D, per unit D2 MOLECULE density (m^3/s).

    A fast D (ion or atom) of lab energy E on cold D2: each molecule offers two
    target deuterons; at E >> 4.5 eV binding the molecule acts as two free
    deuterons. For equal masses E_cm = E_lab/2. Thermal motion of the gas
    (0.04 eV) is negligible against keV beams.
    rate per fast D = n_D2 * K(E);  K = 2 * sigma(E/2) * v
    """
    E = np.asarray(E_lab_eV, dtype=float)
    v = speed_from_energy(E * e, m_d)
    return 2.0 * sigma_DD(E / 2e3, channel) * v


# ---------------------------------------------------------------------------
# 2. Heavy-particle atomic cross sections for D+/D0 on D2
#    Data are for hydrogen (H+/H0 on H2). Isotope substitution rule:
#    electronic (charge-exchange, ionisation, stripping) cross sections are
#    functions of the collision VELOCITY (Barnett, ORNL-6086 "Red Book",
#    1990, and standard practice in IAEA/ALADDIN data). So a D projectile at
#    lab energy E_D uses the H value at E_H = E_D * m_p/m_d (~ E_D/2).
#    Elastic (momentum-transfer) scattering at low energy is polarisation
#    (Langevin) dominated and depends on the CM energy only, so it is
#    mapped at equal E_cm (exact for Langevin; the D+/D2 and H+/H2 mass
#    ratios are both 1:2, so E_cm = (2/3) E_lab in both systems).
# ---------------------------------------------------------------------------
_H_TO_CM = 2.01588 / (1.00794 + 2.01588)   # E_cm = E_lab * m_H2/(m_H + m_H2)


def _load_table(fname):
    d = np.loadtxt(os.path.join(DATA_DIR, fname))
    return d[:, 0], d[:, 1]


_TAB = {}


def _tab(name):
    if name not in _TAB:
        _TAB[name] = _load_table(name)
    return _TAB[name]


def _H_table_at_D_velocity(E_D_eV, fname):
    """Look up an H+H2 table (stored vs E_cm of the H system) at the H energy
    that has the same velocity as a D of lab energy E_D."""
    E_H_lab = np.asarray(E_D_eV, dtype=float) * (m_p / m_d)
    E_cm_H = E_H_lab * _H_TO_CM
    x, y = _tab(fname)
    return np.interp(E_cm_H, x, y, left=0.0, right=y[-1])


def sigma_cx(E_D_eV):
    """D+ + D2 -> D + D2+ charge exchange (m^2).
    Tabata & Shirai, At. Data Nucl. Data Tables 76 (2000) 1, reaction 6
    (H+ + H2), as tabulated in BLAST-WarpX/warpx-data (H/Hion_on_H2_charge_
    exchange.dat). Zero below the 1.83 eV endothermic threshold."""
    return _H_table_at_D_velocity(E_D_eV, "TS2000_R6_Hp_H2_charge_exchange_Ecm.dat")


def sigma_strip(E_D_eV):
    """Fast D0 + D2 -> D+ + e + D2 (projectile stripping, m^2).
    Tabata & Shirai (2000) reaction 31 (H + H2), warpx-data H_on_H2_ionization."""
    return _H_table_at_D_velocity(E_D_eV, "TS2000_R31_H_H2_stripping_Ecm.dat")


def sigma_ion_impact(E_D_eV):
    """D+ + D2 -> D+ + D2+ + e  (target ionisation, m^2).
    Rudd, DuBois, Toburen, Ratcliffe & Goffe, Phys. Rev. A 28 (1983) 3244;
    recommended fit Rudd et al., Rev. Mod. Phys. 57 (1985) 965, for H2:
    A=0.71, B=1.63, C=0.51, D=1.24 (as quoted by Padovani et al.,
    A&A 501 (2009) 619, eqs. 5-6).  x = (m_e/m_p) E_p / I_H.
    Equal-velocity scaling: E_p -> E_D m_p/m_d, so x = m_e E_D/(m_d I_H).
    Stated accuracy of the fit ~ +/-20 % over 5-4000 keV/u."""
    E = np.asarray(E_D_eV, dtype=float)
    x = (m_e / m_d) * E / I_H_eV
    A, B, C, D = 0.71, 1.63, 0.51, 1.24
    out = np.zeros_like(E)
    ok = x > 0
    sl = C * x[ok] ** D
    sh = (A * np.log1p(x[ok]) + B) / x[ok]
    out[ok] = 4 * np.pi * a0**2 / (1.0 / sl + 1.0 / sh)
    return out


def sigma_neutral_target_ionisation(E_D_eV):
    """Fast D0 + D2 -> D0 + D2+ + e (m^2). Proxy: the proton fit above.
    For v >~ 1 a.u. (E_D >~ 50 keV) the screened neutral and the bare proton
    give target-ionisation cross sections within ~30 % (Barnett, ORNL-6086);
    below that the neutral is lower, so the proxy over-predicts ion
    production there (conservative for the current balance only)."""
    return sigma_ion_impact(E_D_eV)


def write_elastic_table_for_warpx(path, dE_cm=1.0, E_cm_max=45e3):
    """Write the D+ + D2 momentum-transfer table on the UNIFORM grid that
    WarpX's ScatteringProcess requires (no comment lines: a '#' line makes
    WarpX 26.01 read zero rows and segfault -- v4/v5 bug).

    WarpX looks the table up at E_coll = mu v_rel^2/2 (centre of mass; see
    ParticleUtils::getCollisionEnergy) and REQUIRES background_mass = m_D2
    to be passed explicitly (otherwise it silently uses the ion mass; v5 bug).
    Source: Tabata & Shirai (2000) reaction 10, 0.1 eV - 10 keV (H lab),
    i.e. 0.0667-6667 eV E_cm. Above that: power-law continuation of the last
    decade (sigma <= 1.4e-24 m^2, 4 orders below CX, dynamically irrelevant).
    Mapping at equal E_cm (Langevin regime, see section header).
    """
    x, y = _tab("TS2000_R10_Hp_H2_momentum_transfer_Ecm.dat")
    E = np.arange(0.0, E_cm_max + dE_cm / 2, dE_cm)
    E[0] = 0.0
    lx, ly = np.log(x), np.log(y)
    sig = np.exp(np.interp(np.log(np.maximum(E, x[0])), lx, ly))
    # power-law tail
    k = (ly[-1] - ly[-11]) / (lx[-1] - lx[-11])
    hi = E > x[-1]
    sig[hi] = y[-1] * (E[hi] / x[-1]) ** k
    np.savetxt(path, np.column_stack([E, sig]), fmt="%.10e")
    return path


# ---------------------------------------------------------------------------
# 3. Electron-impact ionisation of D2 (= H2 electronically)
#    BEB model, Kim & Rudd, Phys. Rev. A 50 (1994) 3954, H2 parameters
#    B = 15.43 eV, U = 25.68 eV, N = 2, Q = 1.  Agrees with Rapp &
#    Englander-Golden (1965) / Straub et al. (1996) to ~10 % from threshold
#    to 1 keV. Non-relativistic: at 60 keV it is ~10 % LOW vs relativistic
#    Bethe; the channel supplies only a few % of ion births, so the effect on
#    any output is < 1 %.
# ---------------------------------------------------------------------------
def sigma_e_ionisation(T_eV):
    T = np.asarray(T_eV, dtype=float)
    B, U, N = 15.43, 25.68, 2.0
    t = T / B
    u = U / B
    S = 4 * np.pi * a0**2 * N * (Ry_eV / B) ** 2
    out = np.zeros_like(T)
    ok = t > 1.0
    tt = t[ok]
    lt = np.log(tt)
    out[ok] = S / (tt + u + 1.0) * (0.5 * lt * (1 - 1 / tt**2) + 1 - 1 / tt - lt / (tt + 1))
    return out


def vacuum_potential_spherical(r, V_c, R_c, R_a):
    """Ideal concentric-sphere potential (V). Cathode sphere at V_c, anode and
    everything outside at 0. Used ONLY for the analytic electron path
    (ionisation profile and gas-born electron energies): electrons follow the
    radial vacuum field to <~10 % accuracy because electron space charge is
    negligible (n_e/n_i ~ 1e-2, see methods doc)."""
    r = np.asarray(r, dtype=float)
    phi = np.where(r <= R_c, V_c, 0.0)
    gap = (r > R_c) & (r < R_a)
    phi = np.where(gap, V_c * (1 / np.maximum(r, 1e-12) - 1 / R_a) / (1 / R_c - 1 / R_a), phi)
    return phi


def electron_ionisation_profile(V_c, R_c, R_a, R_wall, n_gas, n_r=4000):
    """Ion births along the radial path of ONE electron leaving the cathode
    surface with ~0 eV and flying straight out to the wall.

    Returns (r_grid, cumulative_births(r)). cumulative[-1] = total births per
    secondary electron (typically 0.01-0.05 at 5 mTorr).
    """
    r = np.linspace(R_c, R_wall, n_r)
    T = e * (vacuum_potential_spherical(r, V_c, R_c, R_a) - V_c) / e   # eV
    T = np.maximum(T, 0.0)
    dN = n_gas * sigma_e_ionisation(T)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dN[1:] + dN[:-1]) * np.diff(r))])
    return r, cum


# ---------------------------------------------------------------------------
# 4. Surface physics (inputs, not first-principles)
# ---------------------------------------------------------------------------
# Ion-induced secondary electron yield gamma: strongly surface-condition
# dependent. Clean metals, H+ at 10-50 keV: 1-2 (Hasselkamp, Springer Tracts
# Mod. Phys. 123 (1992)); gas-covered cathodes in hydrogen discharges: up to
# 3-6 near 100 keV (Hill et al. 1939; Thomas 1985, as used by Phelps-type
# Paschen modelling, arXiv:2404.01187). Treated as a scenario input:
#   neutron bound -> LOW gamma (more of the 50 mA is carried by ions)
#   X-ray bound   -> independent of gamma (see xray_absolute_bound)
GAMMA_DEFAULTS = dict(low=1.0, nominal=2.0, high=4.0)

# Particle/energy reflection of D from surfaces (TRIM-type values for H
# isotopes on W at 10-60 keV: R_N ~ 0.3-0.5, R_E ~ 0.15-0.35; on Fe about
# half of that -- Eckstein, IPP 9/132 (2002)). Reflected particles leave as
# fast neutrals. Scenario inputs:
#   neutron bound -> reflection ON with high values (more fast neutrals)
#   wire-heating bound -> energy reflection OFF (all energy stays in wire)
REFLECTION_DEFAULTS = {"W": (0.5, 0.35), "Fe": (0.25, 0.15)}


# ---------------------------------------------------------------------------
# 5. Electronic stopping (for the implanted-deuterium beam-target channel)
#    ICRU Report 49 (1993) proton parametrisation (Ziegler-type, eV per
#    1e15 atoms/cm^2), coefficients copied from Geant4 G4hICRU49p.cc
#    (geant4 master). Deuterons at equal velocity: T(keV/amu)=E_D/2.01355.
#    Nuclear stopping is deliberately omitted: it lengthens the computed range
#    and so OVER-predicts the thick-target yield (conservative).
# ---------------------------------------------------------------------------
_ICRU49 = {
    1: (1.254, 1.440, 242.6, 12000.0, 0.1159),
    22: (4.858, 5.489, 5260.0, 651.1, 0.00893),
    24: (3.983, 4.489, 5616.0, 1336.0, 0.008413),   # Cr
    26: (3.519, 3.963, 6065.0, 1243.0, 0.007782),
    28: (3.553, 4.004, 6205.0, 555.1, 0.008763),    # Ni
    42: (6.424, 7.248, 9545.0, 480.2, 0.005376),
    74: (4.587, 5.160, 15410.0, 415.3, 0.003406),
    75: (5.201, 5.851, 16120.0, 441.6, 0.003122),
}
# atomic fractions
MATERIAL_Z = {
    "W": {74: 1.0},
    "WRe": {74: 0.74, 75: 0.26},
    "Fe": {26: 1.0},                                  # mild steel
    "SS304": {26: 0.70, 24: 0.20, 28: 0.10},          # approx. atomic
}


def stopping_H_ICRU49(T_keV_per_amu, Z):
    """Electronic stopping (eV per 1e15 atoms/cm^2) for H at T keV/amu."""
    T = np.atleast_1d(np.asarray(T_keV_per_amu, dtype=float)).copy()
    a = _ICRU49[Z]
    fac = np.ones_like(T)
    lo = T < 10.0
    fac[lo] = np.sqrt(np.maximum(T[lo], 0) * 0.1)      # free-electron-gas v-scaling
    T[lo] = 10.0
    slow = a[1] * T**0.45
    shigh = np.log(1.0 + a[3] / T + a[4] * T) * a[2] / T
    return slow * shigh * fac / (slow + shigh)


def stopping_D(E_D_eV, material):
    """Electronic stopping of a deuteron in a host, eV*m^2 per host atom."""
    T = np.asarray(E_D_eV, dtype=float) / 1e3 / (m_d / amu)
    S = sum(f * stopping_H_ICRU49(T, Z) for Z, f in MATERIAL_Z[material].items())
    return S * 1e-19          # 1 eV/(1e15 at/cm^2) = 1e-19 eV m^2


def thick_target_yield(E0_eV, material, f_D, channel="n", nE=400):
    """Fusion reactions per incident D of energy E0 stopping in a host loaded
    with f_D deuterium atoms per host atom (Bragg additivity for stopping).
    Y = int_0^E0 f sigma(E/2) / (S_host + f S_D) dE."""
    E0 = float(E0_eV)
    if E0 <= 0 or f_D <= 0:
        return 0.0
    E = np.linspace(E0 / nE, E0, nE)
    T = E / 1e3 / (m_d / amu)
    S_host = stopping_D(E, material)
    S_D = stopping_H_ICRU49(T, 1) * 1e-19
    integrand = f_D * sigma_DD(E / 2e3, channel) / (S_host + f_D * S_D)
    return float(_trapz(integrand, E))


def projected_range_upper(E0_eV, material, nE=400):
    """Electronic-only path length (m) -- an UPPER bound on the implantation
    depth, used to convert fluence into a loading fraction."""
    E = np.linspace(E0_eV / nE, E0_eV, nE)
    n_host = {"W": 6.31e28, "WRe": 6.4e28, "Fe": 8.49e28, "SS304": 8.6e28}[material]
    S = stopping_D(E, material) * n_host        # eV/m
    return float(_trapz(1.0 / S, E)), n_host


# ---------------------------------------------------------------------------
# 6. Thermal properties for the lumped wire transient
# ---------------------------------------------------------------------------
# Tungsten total hemispherical emissivity: Forsythe & Worthing, ApJ 61 (1925)
# 146 (1000-3000 K, reproduced in the CRC Handbook); below 1000 K linear to
# 0.03 at 300 K (Matsumoto et al., Int. J. Thermophys. 20 (1999) 1539 report
# 0.03-0.11 over 300-1000 K). Underestimating emissivity over-predicts T
# (conservative).
_W_EPS_T = np.array([300, 600, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000.0])
_W_EPS = np.array([0.030, 0.064, 0.114, 0.143, 0.175, 0.207, 0.236, 0.260, 0.281, 0.299, 0.312, 0.324, 0.334])
# Tungsten specific heat (J/kg/K): White & Minges, Int. J. Thermophys. 18
# (1997) 1269 recommended values (rounded, +/-3 %).
_W_CP_T = np.array([300, 500, 1000, 1500, 2000, 2500, 3000.0])
_W_CP = np.array([132.0, 138.0, 148.0, 158.0, 170.0, 188.0, 215.0])
# Low-carbon steel specific heat, smoothed THROUGH the Curie anomaly
# (which would add heat capacity, i.e. omitting it is conservative).
_FE_CP_T = np.array([300, 500, 700, 900, 1100, 1300, 1600.0])
_FE_CP = np.array([470.0, 530.0, 600.0, 690.0, 650.0, 650.0, 700.0])

MATERIALS = {
    # rho kg/m^3, cp(T), eps(T), melting K, thermionic (A [A/m^2/K^2], W [eV])
    "W": dict(rho=19250.0, cp=(_W_CP_T, _W_CP), eps=(_W_EPS_T, _W_EPS), T_melt=3695.0,
              rich=(6.0e5, 4.54)),
    # 2 % thoriated W: activated surface (Dushman & Ewald 1923: A=3e4, W=2.63 eV).
    # Ion bombardment sputters the Th monolayer, so the true value lies between
    # this and pure W; both are reported.
    "ThW": dict(rho=19250.0, cp=(_W_CP_T, _W_CP), eps=(_W_EPS_T, _W_EPS), T_melt=3650.0,
                rich=(3.0e4, 2.63)),
    "WRe": dict(rho=19700.0, cp=(_W_CP_T, _W_CP), eps=(_W_EPS_T, _W_EPS), T_melt=3390.0,
                rich=(6.0e5, 4.6)),
    "Fe": dict(rho=7850.0, cp=(_FE_CP_T, _FE_CP), eps=None, T_melt=1750.0,
               rich=(6.0e5, 4.4)),
    "SS304": dict(rho=7900.0, cp=(_FE_CP_T, _FE_CP), eps=None, T_melt=1670.0,
                  rich=(6.0e5, 4.4)),
}


def emissivity(material, T, eps_const=None):
    m = MATERIALS[material]
    if eps_const is not None or m["eps"] is None:
        return eps_const if eps_const is not None else 0.2
    return np.interp(T, *m["eps"])


def cp(material, T):
    return np.interp(T, *MATERIALS[material]["cp"])


def richardson_current_density(T, material, E_surf=0.0):
    """Thermionic emission J (A/m^2), Richardson-Dushman with optional Schottky
    lowering for surface field E_surf (V/m)."""
    A, W = MATERIALS[material]["rich"]
    dW = np.sqrt(e * E_surf / (4 * np.pi * eps0)) if E_surf > 0 else 0.0   # eV
    return A * T**2 * np.exp(-(W - dW) / (kB * T / e))


def grid_geometry(R, d, n_great_circles):
    """Wire length, surface area, volume and geometric transparency of a
    'ring-style' grid of n great circles of round wire (diameter d) on a
    sphere of radius R. Crossing overlaps (poles, ring/meridian joints) are
    ignored: this slightly UNDER-states transparency and over-states mass."""
    L = n_great_circles * 2 * np.pi * R
    A = np.pi * d * L
    V = np.pi * d**2 / 4 * L
    T = 1.0 - n_great_circles * d / (2.0 * R)
    return dict(length=L, area=A, volume=V, transparency=T)


def load_grid_kicks(path=None):
    """Azimuthal grid-kick table written by fusorsim.physics.grid3d: quantiles of the change of
    l_z = x v_y - y v_x (m^2/s) between successive cathode crossings of ions in the 3D field of the real
    great-circle grids, per (inward?, |cos polar|, energy) cell. 2D-RZ has no such torque (its rings are
    axisymmetric), which is why every RZ ion passes through the axis."""
    import hashlib
    import json
    path = path or os.path.join(DATA_DIR, "grid_kicks_3d.dat")
    geom, probs, rows = None, None, []
    with open(path) as f:
        for ln in f:
            if ln.startswith("# geometry:"):
                geom = json.loads(ln.split(":", 1)[1])
            elif ln.startswith("# quantiles:"):
                probs = np.array(ln.split(":", 1)[1].split(), float)
            elif ln.strip() and not ln.startswith("#"):
                rows.append([float(v) for v in ln.split()])
    if geom is None or probs is None or not rows:
        raise ValueError(f"{path}: not a grid-kick table")
    R = np.array(rows)
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return dict(path=os.path.abspath(path), sha256=sha, geometry=geom, probs=probs, inward=R[:, 0].astype(bool),
                cos_lo=R[:, 1], cos_hi=R[:, 2], E_lo=R[:, 3], E_hi=R[:, 4], n=R[:, 5].astype(int), q=R[:, 7:])


def check_grid_kicks_geometry(table, **geom):
    """Raise ValueError unless the table was made for this geometry (R_c, d_c, R_a, d_a, great_circles,
    V_c, R_ch, H_ch): kicks from another grid would be silently wrong."""
    bad = [f"{k}: table {table['geometry'].get(k)} vs run {v}" for k, v in geom.items()
           if table["geometry"].get(k) is None or not np.isclose(float(table["geometry"][k]), float(v), rtol=1e-6)]
    if bad:
        raise ValueError(f"grid-kick table {table['path']} was made for another geometry: " + "; ".join(bad)
                         + "  (re-run: python -m fusorsim.physics.grid3d --case <case>)")


def sample_grid_kick(table, inward, cos_abs, E_eV, rng):
    """Change of l_z (m^2/s) for each crossing, by inverse-CDF interpolation in its table cell. Energies
    outside the tabulated range use the nearest cell."""
    inward = np.asarray(inward, bool); cos_abs = np.clip(np.asarray(cos_abs, float), 0.0, 1.0)
    E_eV = np.clip(np.asarray(E_eV, float), table["E_lo"].min(), np.nextafter(table["E_hi"].max(), 0))
    out = np.zeros(inward.shape)
    if not out.size:
        return out
    nq = table["q"].shape[1]
    u = rng.random(out.size) * (nq - 1)
    k = np.minimum(u.astype(int), nq - 2); f = u - k
    for c in range(len(table["n"])):
        m = ((inward == table["inward"][c]) & (cos_abs >= table["cos_lo"][c]) & (cos_abs <= table["cos_hi"][c])
             & (E_eV >= table["E_lo"][c]) & (E_eV < table["E_hi"][c]))
        if np.any(m):
            q = table["q"][c]
            out[m] = q[k[m]] * (1 - f[m]) + q[k[m] + 1] * f[m]
    return out


def apply_azimuthal_kick(ux, uy, uz, theta, r, dlz, max_frac=0.5):
    """Add dlz/r to the azimuthal velocity (e_phi = (-sin th, cos th, 0)) and rescale to the old speed, so
    the kick only turns the velocity (no energy change). |dlz/r| is capped at max_frac*|v| for crossings
    very near the axis (r -> 0), where the 3D torque is small anyway. Returns new (ux, uy, uz)."""
    v = np.sqrt(ux * ux + uy * uy + uz * uz)
    dv = np.clip(dlz / np.maximum(r, 1e-6), -max_frac * v, max_frac * v)
    nx, ny, nz = ux - dv * np.sin(theta), uy + dv * np.cos(theta), uz
    s = v / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-30)
    return nx * s, ny * s, nz * s


def lumped_wire_transient(P_W, material, area, mass, t_end=15.0, T0=300.0,
                          T_env=300.0, eps_const=None, dt=1e-3):
    """Lumped (isothermal) radiative transient of a wire grid:
        m c(T) dT/dt = P - eps(T) sigma A (T^4 - T_env^4)
    Conduction to the stalk/feedthrough is neglected (conservative: higher T).
    Biot number for a 2 mm W wire radiating at 2000 K: h_rad ~ 4 eps sigma T^3
    ~ 100 W/m^2/K, Bi = h r/k ~ 100*1e-3/100 = 1e-3 << 0.1, so the wire is
    isothermal across its section; along the wire, non-uniform heating is
    handled with the angular peaking factor reported by the PIC run.
    Returns times, temperatures, steady-state temperature."""
    n = int(round(t_end / dt))
    t = np.linspace(0, t_end, n + 1)
    T = np.empty(n + 1)
    T[0] = T0

    def f(TT):
        return (P_W - emissivity(material, TT, eps_const) * sigma_SB * area * (TT**4 - T_env**4)) / (mass * cp(material, TT))

    for i in range(n):   # RK4
        k1 = f(T[i]); k2 = f(T[i] + 0.5 * dt * k1); k3 = f(T[i] + 0.5 * dt * k2); k4 = f(T[i] + dt * k3)
        T[i + 1] = T[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # steady state by bisection
    lo, hi = T_env, 5000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if P_W - emissivity(material, mid, eps_const) * sigma_SB * area * (mid**4 - T_env**4) > 0:
            lo = mid
        else:
            hi = mid
    return t, T, 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 7. X-ray helpers (cross-check only; OpenMC TTB is the production path)
# ---------------------------------------------------------------------------
def thick_target_brems_efficiency(E0_eV, Z):
    """Fraction of electron kinetic energy radiated as bremsstrahlung in a
    thick target: eta ~ 1.1e-9 Z V (V in volts). Classic Kramers/Compton &
    Allison result, +/-30 %; e.g. Fe at 60 kV -> 1.7e-3."""
    return 1.1e-9 * Z * np.asarray(E0_eV)


def kramers_spectrum(E0_eV, Z, n_bins=120, k_min_eV=1e3):
    """Photon NUMBER spectrum per incident electron (photons/eV), Kramers
    shape dN/dk ~ (E0-k)/k, normalised so total radiated energy = eta*E0.
    Unfiltered, i.e. before any self-absorption in the target (conservative)."""
    k = np.linspace(k_min_eV, E0_eV, n_bins + 1)
    kc = 0.5 * (k[1:] + k[:-1])
    shape = np.maximum(E0_eV - kc, 0) / kc
    Etot = np.sum(shape * kc * np.diff(k))
    norm = thick_target_brems_efficiency(E0_eV, Z) * E0_eV / Etot
    return k, shape * norm


def xray_absolute_bound(V_abs, P_supply):
    """Absolute upper bound on electron beam power and current onto grounded
    surfaces: all supply power carried by electrons of the full gap energy.
    Every electron reaching ground has energy <= e|V| (no point in the
    chamber is below the cathode potential), and total particle power cannot
    exceed the supply power. Brems yield per unit deposited energy rises
    with electron energy, so mono-energetic e|V| electrons are the bound."""
    return dict(electron_rate=P_supply / (e * V_abs), electron_energy_eV=V_abs, power_W=P_supply)


# ---------------------------------------------------------------------------
# 8. D-D neutron kinematics (beam on stationary target)
# ---------------------------------------------------------------------------
def dd_neutron_energy(E_d_eV, cos_lab):
    """Lab neutron energy (eV) for D(d,n)3He with projectile energy E_d and
    lab emission cosine (non-relativistic two-body kinematics)."""
    Ed = np.asarray(E_d_eV, dtype=float)
    Q = Q_DDn_MeV * 1e6
    m1, m2, m3, m4 = m_d, m_d, m_n, m_He3
    a = np.sqrt(m1 * m3 * Ed) * cos_lab
    b = a**2 + (m3 + m4) * (m4 * Q + (m4 - m1) * Ed)
    return ((a + np.sqrt(b)) / (m3 + m4)) ** 2


def dd_neutron_spectrum(E_d_eV_samples, weights, bins_MeV=None, n_dir=64, rng=None):
    """Angle-integrated neutron spectrum for a set of reacting projectile
    energies, assuming isotropic emission in the CM frame. (The real
    D(d,n) CM distribution is forward/backward peaked by up to ~1+0.5cos^2
    at 50 keV; for shielding the angle-integrated spectrum is what matters
    and its spread is only +/-0.3 MeV.)"""
    rng = np.random.default_rng(1) if rng is None else rng
    if bins_MeV is None:
        bins_MeV = np.linspace(2.0, 3.0, 101)
    Ed = np.repeat(np.asarray(E_d_eV_samples, dtype=float), n_dir)
    w = np.repeat(np.asarray(weights, dtype=float), n_dir) / n_dir
    # isotropic in CM -> lab: sample CM cosine, convert via kinematics
    Q = Q_DDn_MeV * 1e6
    mu = rng.uniform(-1, 1, Ed.size)
    # CM velocities
    vcm = np.sqrt(2 * Ed * e / m_d) * m_d / (2 * m_d)
    Ecm_tot = Ed * e / 2 + Q * e
    vn_cm = np.sqrt(2 * Ecm_tot * m_He3 / (m_n * (m_n + m_He3)))
    vx = vn_cm * mu + vcm
    vy = vn_cm * np.sqrt(1 - mu**2)
    En = 0.5 * m_n * (vx**2 + vy**2) / e / 1e6
    h, _ = np.histogram(En, bins=bins_MeV, weights=w)
    return bins_MeV, h


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def _selftest():
    ok = True

    def check(name, val, lo, hi):
        nonlocal ok
        good = lo <= val <= hi
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {name}: {val:.4g} (expected {lo:.3g}..{hi:.3g})")

    # Bosch-Hale: D(d,n) at E_cm = 50 keV (100 keV deuteron) ~ 16-17 mb;
    # at E_cm = 10 keV ~ 0.28 mb (Bosch & Hale table 3 / ENDF/B-VIII)
    check("sigma_DDn(50 keV cm) [mb]", sigma_DD(50.0, "n").item() / 1e-31, 15.5, 17.5)
    check("sigma_DDn(10 keV cm) [mb]", sigma_DD(10.0, "n").item() / 1e-31, 0.25, 0.31)
    check("sigma_DDp/sigma_DDn at 30 keV", (sigma_DD(30.0, "p") / sigma_DD(30.0, "n")).item(), 0.9, 1.1)
    # CX: H+ on H2 at 30 keV ~ 4e-20 m^2 -> D at 60 keV same
    check("sigma_cx(D 60 keV) [1e-20 m2]", sigma_cx(60e3).item() / 1e-20, 3.0, 5.5)
    check("sigma_cx(D 1 eV) == 0 (endothermic)", sigma_cx(1.0).item(), 0.0, 0.0)
    # Rudd: H+ on H2 at 30 keV ~ 1.5-2e-20 m^2
    check("sigma_ion_impact(D 60 keV) [1e-20 m2]", sigma_ion_impact(60e3).item() / 1e-20, 1.2, 2.2)
    # BEB H2 peak ~ 0.96-1.02e-20 m^2 near 70 eV (Straub 1996)
    TT = np.linspace(16, 1000, 2000)
    sb = sigma_e_ionisation(TT)
    check("BEB H2 peak [1e-20 m2]", sb.max() / 1e-20, 0.9, 1.1)
    check("BEB H2 peak energy [eV]", TT[sb.argmax()], 50, 100)
    check("BEB at threshold == 0", sigma_e_ionisation(15.43).item(), 0, 0)
    # stopping: H in Fe at 50 keV ~ 20-23 eV/(1e15 at/cm2) (ICRU49)
    check("S_e(H in Fe, 50 keV)", stopping_H_ICRU49(50 / 1.00728, 26).item(), 19, 24)
    check("S_e(H in W, 50 keV)", stopping_H_ICRU49(50 / 1.00728, 74).item(), 25, 40)
    # gas density 5 mTorr 300 K = 1.609e20
    check("n_D2(5 mTorr) [1e20]", gas_density(5.0) / 1e20, 1.60, 1.62)
    # D+ speed at 60 keV
    check("v(D 60 keV) [1e6 m/s]", speed_from_energy(60e3 * e, m_d) / 1e6, 2.39, 2.41)
    # cathode grid transparency (user spec 7 great circles, 2 mm, R 69 mm)
    check("cathode transparency", grid_geometry(0.069, 0.002, 7)["transparency"], 0.898, 0.900)
    check("anode transparency", grid_geometry(0.129, 0.001, 7)["transparency"], 0.972, 0.974)
    # DD neutron energy at 0 keV projectile = 2.45 MeV
    check("E_n(E_d->0) [MeV]", dd_neutron_energy(1.0, 0.0).item() / 1e6, 2.44, 2.46)
    check("brems eff Fe 60 kV", thick_target_brems_efficiency(60e3, 26), 1.6e-3, 1.8e-3)
    # thick-target: D 60 keV on fully loaded W (f=1) ~ 1e-9..1e-8 n/D
    y = thick_target_yield(60e3, "W", 1.0)
    check("thick-target yield D->WD (f=1) [1e-9]", y / 1e-9, 0.3, 30)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
