# CH 5960 – Assignment 1: Monte Carlo simulation of an infinite reactor

Python (NumPy + Numba) code that follows neutrons one at a time through four **infinite, homogeneous** media and
estimates the infinite multiplication factor `k_inf` (neutrons produced by fission per neutron born):

| # | Medium | Expected |
|---|--------|----------|
| 1 | pure U-238 | cannot go critical |
| 2 | natural uranium (0.7204 at% U-235) | cannot go critical |
| 3 | 2 wt% enriched uranium + light water | can go critical |
| 4 | natural uranium + heavy water | can go critical |

Each case is run with **10^6** and with **10^9** neutron histories (`~1 min` and `~1 h` on 4 cores).

## Quick start

```bash
pip install -r requirements.txt            # numpy scipy matplotlib numba
./run_all.sh                               # everything (or run the lines in it one by one)
# or only the essentials:
python mc_reactor.py --n 1e6 --out results_1e6          # 10^6 neutrons per case   (~15 s)
python mc_reactor.py --n 1e9 --out results_1e9 --resume # 10^9 neutrons per case   (~1 h, checkpointed, resumable)
python mc_reactor.py --scan --scan-n 1e6 --out results_scan
python analyze.py --main results_1e9 --small results_1e6 --scan results_scan/scan.json --out analysis
python crosscheck_deterministic.py --mc results_1e9 --out analysis
python validate_physics.py                              # seconds: data / sampling self-tests
```

## Where to find the results

| What | Where |
|------|-------|
| **all numbers in one place** (k_inf, four factors, lifetimes, absorption shares, fission/capture, ...) | `analysis/summary.md` |
| **all figures** (`fig01 … fig15`) | `analysis/*.png` |
| the cross-section table the assignment asks for (50 energies × 5 nuclides × 4 reactions) | `analysis/cross_section_table.md` / `.csv` |
| independent deterministic check of the Monte Carlo | `analysis/crosscheck_deterministic.md`, `fig15_deterministic_crosscheck.png` |
| raw tallies (NumPy) | `results_1e6/case{1..4}.npz`, `results_1e9/case{1..4}.npz`, `results_scan/scan.json` |
| console logs | `logs/` |

## Files

| File | Purpose |
|------|---------|
| `xs_data.py` | nuclear data: cross-sections, ν̄(E), U-238 resonance ladder, number densities, the required 50-point table |
| `mc_reactor.py` | the Monte Carlo engine (Numba, parallel) + command line driver (+ moderator scan) |
| `analyze.py` | all plots and `summary.md` |
| `crosscheck_deterministic.py` | non-random solution of the same problem (validates the Monte Carlo) |
| `validate_physics.py` | quick unit tests (thermal constants, resonance integrals, ξ, birth spectrum) |

## What the code does (one neutron)

1. **Birth** energy from `s(E)=0.771 √E exp(−0.776E)` (exact sampling `E = −T[ln r1 + ln r2 cos²(π r3/2)]`, `T=1/0.776`).
2. **Flight** `d = −ln ξ / Σt(E)`; time `d/v`, `v = 1.3831·10⁹ √E[MeV]` cm/s.
3. **Collision partner** ∝ `N_j σ_t,j(E)`; **reaction** ∝ elastic / inelastic / capture / fission.
4. **Elastic**: isotropic in the centre of mass → `E'` uniform in `[αE, E]`, `α=((A−1)/(A+1))²`.
   **Inelastic**: evaporation spectrum `E'e^{−E'/θ}`, `θ=√(8(E−Ex)/A)` (U-235/238); two-body kinematics with the
   level energy Q for O-16 (6.13 MeV) and D (2.22 MeV break-up).
5. **Capture** (incl. (n,α) in O-16) or **fission** ends the history; fission adds ν̄(E) neutrons to `k_inf`.
6. Below **0.625 eV** the neutron joins a Maxwellian (293.6 K) thermal pool.  Its remaining life is sampled in one
   step: geometric number of thermal collisions, exponential lifetime `1/⟨vΣa⟩`, absorbing nuclide/reaction/energy from
   the Maxwellian-weighted rates.  (This reproduces the analog result and is what makes 10⁹ histories possible even for
   D₂O where a neutron makes ~500 thermal collisions.)

`k_inf = (1/N) Σ ν_i`, `σ_k = √((⟨ν²⟩−k²)/N)`.  "Can go critical" ⇔ `k_inf − 3σ > 1`.
