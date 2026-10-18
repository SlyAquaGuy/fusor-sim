# Data tables (read-only)

Tabata & Shirai (2000), At. Data Nucl. Data Tables 76, 1 -- H+/H0 on H2, energies in the H+H2 centre-of-mass frame.

| file | reaction | origin |
|---|---|---|
| TS2000_R6_Hp_H2_charge_exchange_Ecm.dat | 6, charge exchange | BLAST-WarpX/warpx-data MCC_cross_sections/H/Hion_on_H2_charge_exchange.dat |
| TS2000_R31_H_H2_stripping_Ecm.dat | 31, ionisation (stripping) | BLAST-WarpX/warpx-data MCC_cross_sections/H/H_on_H2_ionization.dat |
| TS2000_R10_Hp_H2_momentum_transfer_Ecm.dat | 10, momentum transfer | digitised in v4 |

SHA-256 (verify before trusting a result):

    a6268d004fbaafd31ec3736d80ad20148374230a2dcec0e0b335bb42e3bf459f *TS2000_R10_Hp_H2_momentum_transfer_Ecm.dat
    37750fcd26b3f3c56d6ee7d6283aa9f56ceacb9a3f6db80f4ffe8900688f56c8 *TS2000_R31_H_H2_stripping_Ecm.dat
    049942748712aea47a42c3cfaa9c9f0069d98ae1917f05cb0160462ea3d0dc3c *TS2000_R6_Hp_H2_charge_exchange_Ecm.dat

## Derived: 3D grid kicks

`grid_kicks_3d.dat`: change of l_z between successive cathode crossings of test D+ ions in the 3D vacuum
field of the real great-circle grids (equator + 6 meridians), for the R1 geometry. Made by
`python -m fusorsim.physics.grid3d --case cases/R1_bound_5mT.yaml` (command, commit and date in its header).
Used by the driver with `--grid-kicks on`; regenerate after any grid, voltage or chamber change.

    b82a2f315d32f85a7698bea5f2e3758edb0ffc69e199e1af81c613e9a5bcf88a *grid_kicks_3d.dat
