# Deterministic cross-check of the Monte Carlo engine
Same cross-sections and scattering physics, but solved as a single high-to-low energy sweep of the collision-density equation (no random numbers).

| Case | k_inf deterministic | k_inf Monte Carlo (± 1σ) | difference (σ) | pool arrivals p (det.) | p (MC) |
|---|---|---|---|---|---|
| 1 | 0.23787 | 0.23784 ± 0.00002 | -0.91 | 0.00000 | 0.00000 |
| 2 | 0.33075 | 0.33072 ± 0.00003 | -0.90 | 0.00000 | 0.00000 |
| 3 | 1.24724 | 1.24717 ± 0.00004 | -1.78 | 0.72979 | 0.72976 |
| 4 | 1.18671 | 1.18674 ± 0.00004 | +0.80 | 0.91708 | 0.91708 |

The small remaining differences are the O(Δu) = 1e-4 energy-discretisation error of the deterministic sweep plus the statistical error of the MC.

## Absorption partition (fraction of all neutrons)

| Case | nuclide | reaction | deterministic | Monte Carlo |
|---|---|---|---|---|
| 1 | U238 | capture | 0.91406 | 0.91407 |
| 1 | U238 | fission | 0.08594 | 0.08593 |
| 2 | U235 | capture | 0.00911 | 0.00910 |
| 2 | U235 | fission | 0.03850 | 0.03850 |
| 2 | U238 | capture | 0.86707 | 0.86709 |
| 2 | U238 | fission | 0.08532 | 0.08531 |
| 3 | U235 | capture | 0.08985 | 0.08987 |
| 3 | U235 | fission | 0.48991 | 0.48989 |
| 3 | U238 | capture | 0.30628 | 0.30630 |
| 3 | U238 | fission | 0.01927 | 0.01926 |
| 3 | H1 | capture | 0.09294 | 0.09293 |
| 3 | O16 | capture | 0.00175 | 0.00175 |
| 4 | U235 | capture | 0.08240 | 0.08241 |
| 4 | U235 | fission | 0.48687 | 0.48688 |
| 4 | U238 | capture | 0.38641 | 0.38639 |
| 4 | U238 | fission | 0.00034 | 0.00034 |
| 4 | H2 | capture | 0.03507 | 0.03508 |
| 4 | O16 | capture | 0.00891 | 0.00890 |
