# Summary of results
Library: `full`; temperature 293.6 K; thermal cut-off 0.625 eV; groups: fast E >= 0.1 MeV, slowing down 0.625 eV - 0.1 MeV, thermal < 0.625 eV.
## 1. Multiplication factor  k_inf  (± 1 standard deviation)
| Case | Medium | k_inf, N = 1,000,000 | k_inf, N = 1,000,000,000 | Can go critical? |
|---|---|---|---|---|
| 1 | pure U-238 | 0.23772 ± 0.00078 | 0.23784 ± 0.00002 | no |
| 2 | natural U | 0.33034 ± 0.00088 | 0.33072 ± 0.00003 | no |
| 3 | 2% U + H2O | 1.24884 ± 0.00123 | 1.24717 ± 0.00004 | **YES** |
| 4 | natural U + D2O | 1.18442 ± 0.00122 | 1.18674 ± 0.00004 | **YES** |

Moderator amounts used: case 3 = 3.40 H2O molecules per U atom (V_mod/V_U = 4.91); case 4 = 300.0 D2O molecules per U atom (V_mod/V_U = 435.2). Uranium metal 19.05 g/cm3, H2O 0.998, D2O 1.105 g/cm3, smeared homogeneously.

## 2. Comparison with the two reference reports (N = 10^9 unless stated)
| Case | This work | Ref. A (GitHub report, CH23B086/CH23B036) | Ref. B (MATLAB report, CH23B025/CH23B043) |
|---|---|---|---|
| 1 | 0.23784 ± 0.00002 | 0.22992 | 0.30202 |
| 2 | 0.33072 ± 0.00003 | 0.33816 | 0.40556 |
| 3 | 1.24717 ± 0.00004 | 1.27030 | 1.21885 |
| 4 | 1.18674 ± 0.00004 | 1.21403 | 1.14017 |

All three agree on the four yes/no answers. Differences in the actual k_inf come from cross-section data, inelastic-scattering and resonance treatment (see README).

## 3. Four factors (moderated cases)
| Case | eps | p | f | eta | k_inf | f (theory) | eta (theory) |
|---|---|---|---|---|---|---|---|
| 3 | 1.1101 | 0.7298 | 0.8793 | 1.7509 | 1.2472 | 0.8793 | 1.7509 |
| 4 | 1.0029 | 0.9171 | 0.9547 | 1.3516 | 1.1867 | 0.9547 | 1.3516 |

Definitions: p = fraction of source neutrons that reach 0.625 eV; f = fraction of thermal absorptions in uranium; eta = neutrons per thermal absorption in uranium; eps = k_inf/(eta f p) = total fission neutrons / thermal fission neutrons.

## 4. Neutron lifetimes (seconds, averaged over ALL neutrons, N = 1000000000)
| Case | time fast | time slowing down | time thermal | total lifetime | thermal lifetime of thermal neutrons (MC / 1/<vSigma_a>) | slowing-down time of thermalising neutrons (MC / theory) |
|---|---|---|---|---|---|---|
| 1 | 3.881e-08 | 1.720e-07 | 6.890e-12 | 2.108e-07 | 3.394e-05 / 3.508e-05 | 6.6e-06 / 0.000116 |
| 2 | 3.831e-08 | 1.627e-07 | 2.286e-12 | 2.011e-07 | 1.292e-05 / 1.242e-05 | 6.82e-07 / 0.000116 |
| 3 | 9.395e-09 | 1.198e-06 | 2.171e-05 | 2.291e-05 | 2.974e-05 / 2.975e-05 | 1.49e-06 / 1.52e-06 |
| 4 | 2.228e-08 | 9.450e-06 | 4.742e-03 | 4.751e-03 | 0.00517 / 0.00517 | 1.01e-05 / 1.02e-05 |

## 5. Slowing down: collisions needed to reach 0.625 eV
| Case | MC mean collisions | theory ln(E0/Eth)/xi_bar | xi_bar (1 eV-0.1 MeV) | fraction of neutrons reaching thermal (p) |
|---|---|---|---|---|
| 1 | 42.40 | 1724.83 | 0.008 | 0.0000 |
| 2 | 39.73 | 1724.76 | 0.008 | 0.0000 |
| 3 | 18.53 | 18.66 | 0.781 | 0.7298 |
| 4 | 29.85 | 28.98 | 0.503 | 0.9171 |

Elastic-scattering lethargy gain per collision xi = 1 + alpha ln(alpha)/(1-alpha):  U-235 0.0086, U-238 0.0085, H-1 1.0000, D (H-2) 0.7261, O-16 0.1210

## 6. Where neutrons are absorbed (fraction of all neutrons, N = 1000000000)
| Case | U-235 fission | U-235 capture | U-238 fission | U-238 capture | H-1 capture | D capture | O-16 capture | fuel total | moderator total | fission / capture (fuel) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00% | 0.00% | 8.59% | 91.41% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.0940 |
| 2 | 3.85% | 0.91% | 8.53% | 86.71% | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% | 0.1413 |
| 3 | 48.99% | 8.99% | 1.93% | 30.63% | 9.29% | 0.00% | 0.18% | 90.53% | 9.47% | 1.2852 |
| 4 | 48.69% | 8.24% | 0.03% | 38.64% | 0.00% | 3.51% | 0.89% | 95.60% | 4.40% | 1.0393 |

U-235 thermal fission/capture from the MC (thermal-pool absorptions only) vs the thermal-averaged data:

| Case | MC | data |
|---|---|---|
| 3 | 5.9273 | 5.9280 |
| 4 | 5.9278 | 5.9280 |

## 7. Birth energy
Mean sampled birth energy = 1.9329 MeV (exact for s(E): 1.5/0.776 = 1.9330 MeV).

## 8. Events per history
| Case | collisions/neutron (non-thermal) | thermal collisions/neutron |
|---|---|---|
| 1 | 38.09 | 0.00 |
| 2 | 36.77 | 0.00 |
| 3 | 17.25 | 10.37 |
| 4 | 29.30 | 481.72 |

## 9. Moderator scans (k_inf)
* **2% U + H2O**: maximum k_inf = 1.2477 at 3.43 molecules/U atom; k_inf > 1 for 0.74 ... 11.71 (scan points)
* **natural U + H2O**: maximum k_inf = 0.8959 at 1.86 molecules/U atom; k_inf > 1 for no ratio scanned
* **2% U + D2O**: maximum k_inf = 1.5882 at 632.53 molecules/U atom; k_inf > 1 for 20.00 ... 1500.00 (scan points)
* **natural U + D2O**: maximum k_inf = 1.1878 at 355.69 molecules/U atom; k_inf > 1 for 26.67 ... 1500.00 (scan points)
* **unmoderated uranium**: k_inf exceeds 1 only above ~10 wt% U-235 (scan point); natural U = 0.330

## 10. Sensitivity: only the 50-point assignment table (log-log interpolation) instead of the fine library
| Case | fine library | coarse 50-point table |
|---|---|---|
| 1 | 0.23784 ± 0.00002 | 0.21870 ± 0.00075 |
| 2 | 0.33072 ± 0.00003 | 0.31721 ± 0.00087 |
| 3 | 1.24717 ± 0.00004 | 1.41453 ± 0.00121 |
| 4 | 1.18674 ± 0.00004 | 1.27791 ± 0.00122 |
