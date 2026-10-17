# Practice behind the repository layout

| convention in this repo | source |
|---|---|
| code / data / config / results separated; raw data read-only; results never committed | Noble (2009) [3]; Wilson et al. (2017) [2] |
| every run records its inputs, code version and image digest in `run_params.json` | Sandve et al. (2013) [4] |
| one importable package shared by all components, small tested units | Wilson et al. (2014) [1]; Taschuk & Wilson (2017) [9] |
| pinned, versioned container recipes in git; no `latest` | Kurtzer et al. (2017) [5]; Nüst et al. (2020) [6] |
| tests split into unit / verification / validation | Oberkampf & Roy (2010) [7]; Roache (2002) [10]; PIC-MCC benchmarks: Turner et al. (2013) [11] |
| CITATION.cff, archived tagged releases, FAIR4RS metadata | Barker et al. (2022) [8] |
| hybrid-model MPI in Apptainer | [Apptainer MPI docs](https://apptainer.org/docs/user/1.5/mpi.html) |
| M3: Apptainer, no image builds, ANSYS licensing | [M3 containers](https://docs.erc.monash.edu/M3/Software/Containers), [M3 ANSYS](https://docs.erc.monash.edu/M3/Software/ANSYS) |

1. Wilson G. et al. (2014) Best Practices for Scientific Computing. *PLoS Biol* 12(1):e1001745. doi:10.1371/journal.pbio.1001745
2. Wilson G. et al. (2017) Good enough practices in scientific computing. *PLoS Comput Biol* 13(6):e1005510. doi:10.1371/journal.pcbi.1005510
3. Noble W.S. (2009) A Quick Guide to Organizing Computational Biology Projects. *PLoS Comput Biol* 5(7):e1000424.
4. Sandve G.K. et al. (2013) Ten Simple Rules for Reproducible Computational Research. *PLoS Comput Biol* 9(10):e1003285.
5. Kurtzer G.M., Sochat V., Bauer M.W. (2017) Singularity: Scientific containers for mobility of compute. *PLoS ONE* 12(5):e0177459.
6. Nüst D. et al. (2020) Ten simple rules for writing Dockerfiles for reproducible data science. *PLoS Comput Biol* 16(11):e1008316. doi:10.1371/journal.pcbi.1008316
7. Oberkampf W.L., Roy C.J. (2010) *Verification and Validation in Scientific Computing*. Cambridge University Press.
8. Barker M. et al. (2022) Introducing the FAIR Principles for research software. *Sci Data* 9:622.
9. Taschuk M., Wilson G. (2017) Ten simple rules for making research software more robust. *PLoS Comput Biol* 13(4):e1005412.
10. Roache P.J. (2002) Code Verification by the Method of Manufactured Solutions. *J Fluids Eng* 124(1):4–10.
11. Turner M.M. et al. (2013) Simulation benchmarks for low-pressure plasmas: Capacitive discharges. *Phys Plasmas* 20:013507.

Citation details other than [2] and [6] were recalled from memory, not re-checked against the publishers; verify before citing in a paper.
