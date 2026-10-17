> COARSE TEST RUN (1.44 mm mesh, 20 000 ions, 3 us) made while developing v6.
> It shows what a report looks like and the order of magnitude. NOT for certification:
> run R1 (defaults) and R3 (--nr 400) on your PC and use those reports.

# Fusor v6 results

Run: `/tmp/claude-0/-home-claude/2383e32c-1f0b-5152-9057-e1eb79edc282/scratchpad/test6`  
Operating point: -60 kV, 50 mA (3000 W), 5.0 mTorr D2, gamma_i = 1.0, reflection on, makeup 'anode', mesh 104x208  
Averaging window: 1.50-3.00 us (520 samples)

## 1. Steady-state and validity checks

- [PASS] supply current held: 50.05 +/- 0.18 mA vs 50.0 mA target
- [PASS] ion beam-gas neutron rate not rising: drift +1.8 % over window (stationary), stat. error 0.5 %
- [PASS] fast-neutral neutron rate not rising: drift +1.4 % over window (stationary), stat. error 0.5 %
- [INFO] ion inventory drift +1.5 % over window
- [PASS] bootstrap source off in window: artificial start-up source fully faded
- [PASS] particle CFL: max ion displacement 0.62 cell/step
- [PASS] macroparticle count: 2.06e+04 ions in window (target 2e+04)
- [PASS] ion plasma frequency resolved: omega_pi*dt = 0.00319 (n_i,peak = 1.41e+14 m^-3)
- [INFO] controller alpha = 1.940: 48 % of ion births are makeup (ionisation not captured by the model), born at rest just inside the anode (bounding)
- [PASS] energy balance: particle power to all surfaces 2823 W vs V x I_measured = 3003 W (difference = elastic gas heating, ions in flight, statistics)

## 2. Neutron source

| channel | rate (n/s) | note |
|---|---|---|
| D+ ions on D2 gas (beam-gas) | 1.076e+06 +/- 5.6e+03 | PIC, expected-value tally |
| fast neutrals on D2 gas | 1.487e+06 +/- 7.5e+03 | CX + reflected neutrals, ray traced |
| implanted D in cathode | 0.00e+00 -> 1.11e+06 | fluence-limited loading over 15 s, cap f=0.4 |
| implanted D in anode | 3.48e+05 | f = 0.08 D/Fe (history-saturated) |
| implanted D in wall | 1.23e+07 | f = 0.08 D/metal |
| ion-ion in converged core | 5.3e-01 | upper estimate, NOT added (see note) |

**Peak neutron emission rate: 1.631e+07 n/s** (statistical +/-0.4 %)  
**Neutrons in a 15 s shot: 2.363e+08** (rate x time, with the time-dependent cathode loading integrated)  
D(d,p)T rate 2.540e+06/s -> tritium made per shot 2.342e+08 atoms = 4.175e-01 Bq (negligible)  
Core ion-ion estimate is 2.1e-05 % of beam-gas; it is reported, not added, because the 2D-RZ axis focusing over-states core density. If it is > 5 % treat it as an additional bound term.

Sensitivity of embedded-target terms to the assumed D loading f (n/s):
| f | cathode (saturated) | anode | wall |
|---|---|---|---|
| 0.01 | 5.32e+05 | 4.45e+04 | 1.57e+06 |
| 0.1 | 5.20e+06 | 4.31e+05 | 1.53e+07 |
| 0.4 | 1.93e+07 | 1.57e+06 | 5.56e+07 |
| 1.0 | 4.22e+07 | 3.31e+06 | 1.18e+08 |

Neutron energy: mean 2.471 MeV, 5-95 % range 2.26-2.67 MeV (isotropic-CM kinematics of the reacting deuterons)
Fusion-weighted mean deuteron energy: 41.2 keV

## 3. X-ray source (electrons striking grounded metal)

Modelled: 6.708e+15 e/s onto the anode (43 W), 2.405e+17 e/s onto the wall (1526 W)  
Absolute bound (independent of gamma): 3.121e+17 e/s at 60 keV = 3000 W, every electron given the full gap energy  
Cross-check, thick-target bremsstrahlung in Fe: modelled 2.69 W of photons, bound 5.15 W (Kramers, unfiltered)
Production path for OpenMC: electron sources inside the wall/anode steel with `settings.electron_treatment = 'ttb'` and photon transport on.

## 4. Grid wire heating over the shot

Cathode (ThW, 3.03 m of 2.0 mm wire, 184 g): heat input 398 W (13 % of supply)
- grid-average temperature at 15 s: **539 K** (steady state would be 1250 K)
- angular peaking factor from PIC (RZ, azimuthally averaged): 1.85 -> hottest latitude 732 K at 15 s (no conduction along the wire: conservative)
- melting point 3650 K. 3D wire joints/hubs concentrate flux further; apply the Maxwell 3D field-line focusing factor on top of this number.
- thermionic emission if surface is activated thoriated W: 2.59e-13 mA at grid-average T, up to 9.1e-07 mA at the hottest latitude (vs 50 mA supply). Thermionic current displaces ion current at fixed I, so it LOWERS neutron output; the neutron numbers above remain an upper bound.
- thermionic emission if surface is pure W: 7.06e-30 mA at grid-average T, up to 1.28e-18 mA at the hottest latitude (vs 50 mA supply). Thermionic current displaces ion current at fixed I, so it LOWERS neutron output; the neutron numbers above remain an upper bound.

Anode (Fe, 5.67 m of 1.0 mm wire, 35 g, eps = 0.2): heat input 66 W modelled, 105 W if all supply power were electrons
- temperature at 15 s: **358 K** modelled, 392 K bound (melting ~1750 K)

## 5. OpenMC source terms (written to source_terms.json / openmc_sources.py)

Neutron: 648 volume bins + cathode shell + wall terms; electrons: anode shell + 48 wall-side and 2x24 end-cap bins; Kramers photon alternative.

## 6. Headline numbers

| quantity | value |
|---|---|
| peak neutron rate | 1.63e+07 n/s |
| neutrons per 15 s shot | 2.36e+08 |
| electrons onto ground (modelled / bound) | 2.47e+17 / 3.12e+17 e/s |
| bremsstrahlung power, Fe thick target (modelled / bound) | 2.69 / 5.15 W |
| cathode T at 15 s (average / hottest latitude) | 539 / 732 K |
| anode T at 15 s (modelled / bound) | 358 / 392 K |
| tritium per shot | 2.3e+08 atoms |

Checks: 8/8 passed.
