"""
xs_data.py -- nuclear data for the infinite-medium Monte Carlo (units: barn, MeV, cm).

What is in here
---------------
* Microscopic cross-sections (elastic, inelastic, capture, fission) of
  U-235, U-238, H-1, H-2 (D) and O-16 as functions of neutron energy, 293.6 K.
* `required_table()`  -> the table the assignment asks for: E = (0.2,0.4,0.6,0.8,1.0)x10^y MeV,
  y = -8 ... 1 (50 energies) for every nuclide and every reaction.
* `Library`           -> the same data on a fine uniform-in-ln(E) grid ("pointwise"
  library) that the Monte Carlo code uses.  `mode="full"` uses the fine data (smooth
  evaluated values + Doppler-broadened U-238 resonances); `mode="coarse"` uses ONLY the
  50-point assignment table with log-log interpolation (sensitivity study).
* nu-bar(E), atom densities and thermal (Maxwellian) averages.

Where the numbers come from (please read)
-----------------------------------------
The KAERI nuclide chart (atom.kaeri.re.kr) / IAEA / NNDC servers are *not reachable* from the
sandbox this was written in, so no ENDF file could be downloaded.  Instead:
  - thermal constants (sigma_gamma, sigma_f at 0.0253 eV, 1/v laws, free-gas scattering) are the
    standard ENDF/B-VII.1 / Mughabghab values;
  - the smooth fast-energy curves (> ~1 keV, and the U-235 resolved-resonance average) are
    anchor points taken from the ENDF/B-VII.1 / JENDL-3.3 based tables printed in the Appendices
    of the two reference reports for this assignment, smoothed where they sat on a single
    resonance spike, and interpolated log-log;
  - the U-238 s-wave capture/scattering resonances 1 eV - 10 keV are generated here with single-level
    Breit-Wigner + Doppler (Voigt) broadening at 293.6 K: the 12 lowest resonances use
    measured parameters, higher ones come from a Wigner/Porter-Thomas statistical ladder
    (D = 20.8 eV, S0 = 1.03e-4, <Gamma_gamma> = 23 meV) whose resonance integral is checked
    against the evaluated value (~275 b).
If you can download the real KAERI/ENDF curves, put them in a folder as
    U235.csv U238.csv H1.csv H2.csv O16.csv     (columns: E_MeV, elastic, inelastic, capture, fission)
and run with `--xs-dir <folder>`; they then replace this built-in data (see `load_override`).
"""
import os
import numpy as np
from scipy.special import wofz, erf

# --------------------------------------------------------------------------- constants
K_B_MEV = 8.617333262e-11          # MeV / K
T_K = 293.6                        # K (as in the reference solutions)
KT = K_B_MEV * T_K                 # MeV  (0.0253 eV)
E_THERMAL_REF = 2.53e-8            # MeV  (0.0253 eV)
E_TH_CUT = 6.25e-7                 # MeV  (0.625 eV, cadmium cut-off = start of "thermal pool")
E_FAST = 0.1                       # MeV  (fast / slowing-down boundary)
N_AVO = 0.6022140857               # atoms/(b cm) per (g/mol) per (g/cm3)
VCONST = 1.3831e9                  # cm/s : v = VCONST * sqrt(E[MeV])
E_MAX = 20.0                       # MeV  (upper edge of the library)
E_MIN = 1.0e-11                    # MeV  (1e-5 eV)

NUCLIDES = ["U235", "U238", "H1", "H2", "O16"]
RX = ["elastic", "inelastic", "capture", "fission"]
# neutron-mass ratios (A in the kinematics)
A_RATIO = {"U235": 233.025, "U238": 236.006, "H1": 0.99917, "H2": 1.99681, "O16": 15.8576}
# inelastic scattering model: (excitation/ binding energy Ex in MeV, kind)  kind 0 = evaporation continuum, 1 = discrete Q
INEL_MODEL = {"U235": (0.013, 0), "U238": (0.045, 0), "H1": (0.0, 0), "H2": (2.2245, 1), "O16": (6.13, 1)}

PHYS = {  # g/mol and g/cm3
    "M_U235": 235.0439, "M_U238": 238.0508,
    "M_H2O": 18.015, "M_D2O": 20.0276,
    "rho_U": 19.05, "rho_H2O": 0.998, "rho_D2O": 1.105,
    "nat_U235_atom_frac": 0.007204,
}

# --------------------------------------------------------------------------- the assignment grid
def required_energies():
    """E = (0.2,0.4,0.6,0.8,1.0) x 10^y MeV,  y = -8..1  (50 energies)."""
    return np.array([m * 10.0 ** y for y in range(-8, 2) for m in (0.2, 0.4, 0.6, 0.8, 1.0)])


# --------------------------------------------------------------------------- interpolation helpers
def loglog(E, xs, ys):
    """Log-log interpolation; segments that touch a zero value are interpolated linearly in ln E.
    Outside the anchor range the end value is held constant."""
    E = np.asarray(E, dtype=float)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    lx, le = np.log(xs), np.log(np.clip(E, 1e-300, None))
    lin = np.interp(le, lx, ys)
    pos = ys > 0
    lg = np.exp(np.interp(le, lx, np.log(np.where(pos, ys, 1.0))))
    k = np.clip(np.searchsorted(lx, le) - 1, 0, len(xs) - 2)
    both = pos[k] & pos[k + 1]
    return np.where(both, lg, lin)


def freegas_elastic(E, sigma_free, A):
    """Elastic cross-section averaged over a Maxwellian target gas at 293.6 K (constant sigma_free)."""
    x = np.sqrt(A * np.asarray(E, float) / KT)
    return sigma_free * ((1.0 + 1.0 / (2.0 * x * x)) * erf(x) + np.exp(-x * x) / (np.sqrt(np.pi) * x))


# --------------------------------------------------------------------------- smooth evaluated anchors (E in MeV, sigma in b)
_A = {}
# ---- U-238 ------------------------------------------------------------------------------------
_A["U238", "elastic"] = ([1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                         [13.0, 13.3, 12.6, 11.95, 11.45, 11.05, 9.5, 7.6, 6.45, 5.37, 4.58, 3.55, 4.40, 3.92, 3.14, 2.70, 2.6])
_A["U238", "inelastic"] = ([0.045, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                           [0.0, 0.1535, 0.379, 0.5415, 1.018, 1.429, 1.708, 2.08, 2.46, 3.193, 2.877, 2.552, 2.112, 2.098, 2.0])
_A["U238", "capture"] = ([1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                         [0.62, 0.522, 0.383, 0.272, 0.212, 0.18, 0.129, 0.1086, 0.1145, 0.1251, 0.1251, 0.0505, 0.00402, 8.8e-4, 9.2e-4, 1.05e-3, 2e-3])
_A["U238", "fission"] = ([1e-9, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 14, 20],
                         [3e-5, 5e-5, 7.1e-5, 3.3e-4, 1.05e-3, 4.26e-3, 0.0148, 0.07, 0.28, 0.54, 0.55, 0.56, 0.57, 0.616, 0.76, 1.0, 0.99, 1.1, 1.1])
# ---- U-235 (resonance-averaged below ~10 keV) ---------------------------------------------------
_E5 = [2e-9, 4e-9, 6e-9, 8e-9, 1e-8, 2e-8, 4e-8, 6e-8, 8e-8, 1e-7, 2e-7, 4e-7, 6e-7, 8e-7, 1e-6,
       2e-6, 4e-6, 6e-6, 8e-6, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4,
       1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 14, 20]
_F5 = [2081, 1472, 1201, 1041, 930.7, 658.1, 465.3, 379.9, 329, 294.3, 208.1, 147.2, 120.1, 104.1, 93.07,
       65.81, 46.53, 37.99, 32.9, 29.43, 20.81, 14.72, 12.01, 10.41, 9.307, 6.581, 4.653, 3.799, 3.449,
       3.2, 2.9, 2.6, 2.45, 2.35, 2.3, 2.05, 1.85, 1.75, 1.65, 1.6, 1.4, 1.19, 1.12, 1.11, 1.2, 1.28, 1.13, 1.2, 1.78, 1.76, 2.0, 2.0]
_C5 = [351, 248.2, 202.7, 175.5, 157, 111, 78.5, 64.09, 55.51, 49.65, 35.1, 24.82, 20.27, 17.55, 15.7,
       11.1, 7.85, 6.409, 5.551, 4.965, 3.51, 2.482, 2.027, 1.755, 1.57, 1.11, 0.785, 0.6409, 0.55,
       0.9 * 0.8, 0.85, 0.8, 0.75, 0.72, 0.7, 0.58, 0.46, 0.4, 0.36, 0.35, 0.31, 0.17, 0.14, 0.12, 0.107, 0.06, 0.0115, 0.0032, 0.0021, 0.002, 0.002, 0.002]
_A["U235", "fission"] = (_E5, _F5)
_A["U235", "capture"] = (_E5, _C5)
_A["U235", "elastic"] = ([1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                         [15.0, 14.5, 12.6, 11.7, 11.5, 11.4, 11.3, 10.85, 10.4, 10.1, 9.95, 9.82, 8.28, 6.25, 5.06, 4.32, 3.87, 3.78, 4.58, 3.92, 3.16, 2.70, 2.5])
_A["U235", "inelastic"] = ([0.013, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                           [0.0, 0.004, 0.0154, 0.099, 0.1975, 0.3225, 0.826, 1.353, 1.614, 1.709, 1.721, 2.116, 2.266, 2.094, 1.327, 1.369, 1.3])
# ---- H-1 (free-atom sigma; thermal broadening applied separately) --------------------------------
_A["H1", "elastic"] = ([1e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                       [20.43, 20.30, 20.17, 19.91, 19.66, 19.42, 19.18, 18.1, 16.3, 14.87, 13.71, 12.74, 9.645, 6.879, 5.569, 4.785, 4.25, 2.908, 1.9, 1.423, 1.133, 0.937, 0.5])
_A["H1", "capture_hi"] = ([1e-3, 1e-2, 0.1, 0.2, 0.4, 1, 2, 10, 20],
                          [1.66e-3, 5.29e-4, 1.047e-4, 6e-5, 3.87e-5, 3.38e-5, 3.69e-5, 3.14e-5, 3.0e-5])
# ---- H-2 ----------------------------------------------------------------------------------------
_A["H2", "elastic"] = ([1e-4, 1e-3, 4e-3, 8e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                       [3.395, 3.395, 3.389, 3.374, 3.367, 3.342, 3.302, 3.27, 3.243, 3.22, 3.122, 3.012, 2.966, 2.922, 2.873, 2.54, 1.817, 1.4, 1.117, 0.916, 0.5])
_A["H2", "inelastic"] = ([3.34, 4, 6, 8, 10, 20], [0.0, 0.0135, 0.0602, 0.1027, 0.134, 0.15])
_A["H2", "capture_hi"] = ([1e-3, 1e-2, 0.1, 1, 10, 20], [2.5e-6, 1.0e-6, 1.9e-6, 5.9e-6, 1.1e-5, 1.1e-5])
# ---- O-16 ---------------------------------------------------------------------------------------
_A["O16", "elastic"] = ([1e-4, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 20],
                        [3.852, 3.83, 3.808, 3.765, 3.724, 3.683, 3.644, 3.477, 5.0, 3.1, 2.75, 4.5, 1.58, 2.04, 1.67, 0.9, 0.8, 0.8])
_A["O16", "inelastic"] = ([6.13, 6.5, 8, 10, 20], [0.0, 0.05, 0.234, 0.334, 0.4])
_A["O16", "capture_hi"] = ([1e-3, 1e-2, 1, 2, 3, 4, 6, 8, 10, 20], [9.6e-7, 3e-7, 1.0e-6, 1.0e-4, 0.008, 0.03, 0.05, 0.10, 0.20, 0.20])

# thermal (0.0253 eV) 1/v constants, barn
SIG_TH = {"U238_c": 2.683, "H1_c": 0.3326, "H2_c": 0.000506, "O16_c": 0.00019}

# --------------------------------------------------------------------------- nu-bar(E)
def nu235(E):
    E = np.asarray(E, float)
    return np.where(E <= 1.0, 2.4355 + 0.065 * E, 2.50 + 0.145 * (E - 1.0))


def nu238(E):
    return 2.30 + 0.15 * np.asarray(E, float)


# --------------------------------------------------------------------------- U-238 resonance ladder
# (E0 [eV], Gamma_n [meV]) measured s-wave resonances; Gamma_gamma = 23 meV for all
_KNOWN = [(6.674, 1.493), (20.87, 10.26), (36.68, 33.55), (66.03, 24.6), (80.75, 1.9), (102.56, 70.0),
          (116.9, 26.0), (145.66, 34.0), (165.28, 3.8), (189.67, 160.0), (208.5, 53.0), (210.6, 2.0)]
_GG = 0.023           # eV  capture width
_D0, _S0 = 20.8, 1.03e-4
_RADIUS = 9.48e-13    # cm (scattering radius a' of U-238, ENDF AP)
E_RES_MAX = 1.2e4     # eV  upper end of the explicit ladder


def build_ladder(seed=1, emax=E_RES_MAX):
    """returns arrays E0 [eV], Gn [eV], Gg [eV] (g = 1, s-wave, I = 0 target)."""
    E0 = [e for e, _ in _KNOWN]
    Gn = [g * 1e-3 for _, g in _KNOWN]
    rng = np.random.default_rng(seed)
    e = E0[-1] + 8.0
    while e < emax:
        e += _D0 * np.sqrt(-(4 / np.pi) * np.log(1 - rng.random()))  # Wigner surmise spacing
        z = rng.standard_normal()
        gn = _S0 * _D0 * np.sqrt(e) * z * z                         # Porter-Thomas, eV
        E0.append(e)
        Gn.append(max(gn, 1e-6))
    E0, Gn = np.array(E0), np.array(Gn)
    Gg = np.full_like(E0, _GG)
    return E0, Gn, Gg


def ladder_xs(E_eV, E0, Gn, Gg, T=T_K):
    """Doppler-broadened SLBW (s-wave): capture and elastic (resonance + interference) in barn.
    Potential scattering is NOT included (added by the caller)."""
    E_eV = np.asarray(E_eV, float)
    cap = np.zeros_like(E_eV)
    scat = np.zeros_like(E_eV)
    A = A_RATIO["U238"]
    kT_eV = KT * 1e6
    for e0, gn, gg in zip(E0, Gn, Gg):
        G = gn + gg
        k0 = 2.196771e9 * (A / (A + 1.0)) * np.sqrt(e0)        # cm^-1
        lam2 = 1.0e24 / (k0 * k0)                              # lambda-bar^2 at e0, barn
        sig0 = 4.0 * np.pi * lam2 * gn / G
        Delta = np.sqrt(4.0 * e0 * kT_eV / A)
        theta = G / Delta
        x = 2.0 * (E_eV - e0) / G
        w = wofz(0.5 * theta * (x + 1j))
        psi = 0.5 * theta * np.sqrt(np.pi) * w.real
        chi = 0.5 * theta * np.sqrt(np.pi) * w.imag
        sq = np.sqrt(e0 / E_eV)
        cap += sig0 * (gg / G) * sq * psi
        scat += sig0 * (gn / G) * psi          # resonance scattering; the (small, sign-changing) interference term is dropped
                                               # because single-level interference can drive sigma_el < 0 in the wings
    return cap, scat


def ladder_resonance_integral(E0, Gn, Gg, lo=0.5, hi=E_RES_MAX):
    """dilute capture resonance integral  int sigma_gamma dE/E  [b] from the ladder (fine quadrature)."""
    E = np.exp(np.arange(np.log(lo), np.log(hi), 2e-4))
    cap, _ = ladder_xs(E, E0, Gn, Gg)
    return np.sum(cap) * 2e-4


def _pick_ladder(target=275.0, seeds=range(1, 41)):
    """choose the statistical ladder whose resonance integral (incl. the smooth part above E_RES_MAX, ~ +1.5 b)
    is closest to the evaluated 275 b"""
    best = None
    for s in seeds:
        lad = build_ladder(s)
        # only the statistical part differs between seeds -> cheap enough on a coarse quadrature
        E = np.exp(np.arange(np.log(0.5), np.log(E_RES_MAX), 5e-4))
        cap, _ = ladder_xs(E, *lad)
        ri = np.sum(cap) * 5e-4 + 1.5
        if best is None or abs(ri - target) < abs(best[0] - target):
            best = (ri, s)
    return best[1]


LADDER_SEED = 4    # result of _pick_ladder() (kept fixed so that the library is reproducible): RI = 277 b

# --------------------------------------------------------------------------- the analytic "evaluated" functions
def micro(nuc, E, ladder=None, resonances=True):
    """Return dict rx -> sigma(E) [b] for one nuclide at energies E [MeV] (vectorised)."""
    E = np.asarray(E, float)
    out = {}
    z = np.zeros_like(E)
    eV = E * 1e6
    th = np.sqrt(E_THERMAL_REF / E)  # 1/v factor
    if nuc == "U238":
        el = loglog(E, *_A[nuc, "elastic"])
        inel = np.where(E >= 0.045, loglog(E, *_A[nuc, "inelastic"]), 0.0)
        fis = loglog(E, *_A[nuc, "fission"])
        capsm = loglog(E, *_A[nuc, "capture"])
        # resolved region E <= E_RES_MAX: 1/v background + Doppler-broadened ladder; above: smooth anchors
        lad = ladder if ladder is not None else _LADDER
        rescap, resel = ladder_xs(eV, *lad) if resonances else (z, z)
        thermal_part = ladder_xs(np.array([25.3e-3]), *lad)
        bg_c = max(SIG_TH["U238_c"] - thermal_part[0][0], 0.0) * th * np.exp(-eV / 3.0e3)
        bg_el = _u238_el_bg(eV, thermal_part[1][0])
        in_res = eV <= E_RES_MAX
        # smooth p-wave capture (not in the s-wave ladder): matches the evaluated average at the 12 keV junction
        sm = np.clip(np.log10(np.maximum(eV, 1.0) / 100.0), 0.0, 1.0)
        bg_p = 0.30 * (np.maximum(eV, 1.0) / E_RES_MAX) ** -0.3 * sm * sm * (3 - 2 * sm)
        cap = np.where(in_res, bg_c + rescap + bg_p, capsm)
        el = np.where(in_res, bg_el + resel, el)
        out = {"elastic": el, "inelastic": inel, "capture": cap, "fission": fis}
    elif nuc == "U235":
        out = {"elastic": loglog(E, *_A[nuc, "elastic"]),
               "inelastic": np.where(E >= 0.013, loglog(E, *_A[nuc, "inelastic"]), 0.0),
               "capture": loglog(E, *_A[nuc, "capture"]),
               "fission": loglog(E, *_A[nuc, "fission"])}
        # epithermal (resonance-region) capture of U-235: the smoothed anchors give RI_gamma ~ 47 b, the evaluated
        # value is ~140 b (alpha = sigma_c/sigma_f ~ 0.5 in the resonance region) -> raise by a smooth factor <= 4.
        u = np.log10(np.maximum(eV, 1e-3))
        ramp = np.clip(u / 0.7, 0, 1) * np.clip((4.0 - u) / 1.0, 0, 1)      # 0 at 1 eV -> 1 at 5 eV ... 1 until 1 keV -> 0 at 10 keV
        ramp = ramp * ramp * (3 - 2 * ramp)
        out["capture"] = out["capture"] * (1.0 + 3.0 * ramp)
        lo = E < 2e-9   # extrapolate below the lowest anchor as 1/v
        out["capture"] = np.where(lo, _C5[0] * np.sqrt(2e-9 / E), out["capture"])
        out["fission"] = np.where(lo, _F5[0] * np.sqrt(2e-9 / E), out["fission"])
    elif nuc in ("H1", "H2", "O16"):
        A = A_RATIO[nuc]
        free = loglog(E, *_A[nuc, "elastic"])
        el = freegas_elastic(E, free, A) if E.size else free
        el = np.where(E > 1e-4, free, el)      # broadening is irrelevant above 100 eV
        c1v = SIG_TH[f"{nuc}_c"] * th                      # 1/v capture
        cap = np.where(E < 1e-3, c1v, loglog(E, *_A[nuc, "capture_hi"]))
        inel = loglog(E, *_A[nuc, "inelastic"]) if (nuc, "inelastic") in _A else z
        inel = np.where(E >= _A[nuc, "inelastic"][0][0], inel, 0.0) if (nuc, "inelastic") in _A else z
        out = {"elastic": el, "inelastic": inel, "capture": cap, "fission": z}
    else:
        raise KeyError(nuc)
    return out


def _u238_el_bg(eV, ladder_el_thermal):
    """potential-scattering background: reproduces sigma_s(0.0253 eV)=9.3 b and rises to an effective 10 b by ~100 eV"""
    c0 = 9.3 - ladder_el_thermal
    c1 = 10.0     # effective potential scattering above ~100 eV (the ladder adds the resonance part)
    s = np.clip(np.log(np.maximum(eV, 1e-9) / 1.0) / np.log(100.0), 0.0, 1.0)
    s = s * s * (3 - 2 * s)
    return c0 + (c1 - c0) * s


_LADDER = build_ladder(LADDER_SEED)

# --------------------------------------------------------------------------- the required 50-energy table
def required_table():
    """dict nuc -> dict rx -> 50 values (barn) at the assignment energies."""
    E = required_energies()
    return {n: micro(n, E) for n in NUCLIDES}, E


def write_required_table(path_csv, path_md=None):
    tab, E = required_table()
    import csv
    with open(path_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["E_MeV"] + [f"{n}_{r}" for n in NUCLIDES for r in RX])
        for i, e in enumerate(E):
            w.writerow([f"{e:.1e}"] + [f"{tab[n][r][i]:.5g}" for n in NUCLIDES for r in RX])
    if path_md:
        with open(path_md, "w") as fh:
            for n in NUCLIDES:
                fh.write(f"\n### {n}  (barn)\n\n| E (MeV) | elastic | inelastic | capture | fission |\n|---|---|---|---|---|\n")
                for i, e in enumerate(E):
                    fh.write(f"| {e:.1e} | " + " | ".join(f"{tab[n][r][i]:.4g}" for r in RX) + " |\n")


# --------------------------------------------------------------------------- the pointwise library
D_LN = 1.0e-4                                   # grid step in ln(E) (0.01 %)
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".xs_cache")


class Library:
    """Cross-sections on a uniform ln(E) grid.  attributes: lnE0, d, n, sig[nuc][rx] (float64 arrays)."""

    def __init__(self, mode="full", xs_dir=None, cache=True):
        self.mode, self.d = mode, D_LN
        self.lnE0 = np.log(E_MIN)
        self.n = int(np.ceil((np.log(E_MAX) - self.lnE0) / self.d)) + 2
        self.E = np.exp(self.lnE0 + self.d * np.arange(self.n))
        self.sig = {}
        if xs_dir:
            self.sig = self._from_override(xs_dir)
            self.mode = "override"
            return
        tag = f"{mode}_{LADDER_SEED}_{D_LN:g}"
        fn = os.path.join(_CACHE, f"lib_{tag}.npz")
        if cache and os.path.exists(fn):
            z = np.load(fn)
            self.sig = {n: {r: z[f"{n}_{r}"] for r in RX} for n in NUCLIDES}
            return
        if mode == "full":
            self.sig = self._build_full()
        elif mode == "coarse":
            self.sig = self._build_coarse()
        else:
            raise ValueError(mode)
        if cache:
            os.makedirs(_CACHE, exist_ok=True)
            np.savez_compressed(fn, **{f"{n}_{r}": self.sig[n][r] for n in NUCLIDES for r in RX})

    def _build_full(self):
        sig = {}
        for n in NUCLIDES:
            sig[n] = {r: np.empty(self.n) for r in RX}
            for lo in range(0, self.n, 40000):               # chunk for memory
                sl = slice(lo, min(lo + 40000, self.n))
                d = micro(n, self.E[sl])
                for r in RX:
                    sig[n][r][sl] = d[r]
        return sig

    def _build_coarse(self):
        tab, Et = required_table()
        sig = {}
        for n in NUCLIDES:
            sig[n] = {}
            for r in RX:
                y = tab[n][r]
                sig[n][r] = loglog(self.E, Et, y)
                # below the lowest tabulated energy hold 1/v for capture/fission, constant otherwise
                lo = self.E < Et[0]
                if r in ("capture", "fission"):
                    sig[n][r] = np.where(lo, y[0] * np.sqrt(Et[0] / self.E), sig[n][r])
        return sig

    def _from_override(self, xs_dir):
        sig = {}
        for n in NUCLIDES:
            d = np.genfromtxt(os.path.join(xs_dir, f"{n}.csv"), delimiter=",", names=True)
            names = d.dtype.names
            col = lambda k: d[[m for m in names if m.lower().startswith(k)][0]]
            Eo = col("e")
            sig[n] = {}
            for r, key in zip(RX, ("el", "inel", "cap", "fis")):
                y = col(key)
                v = loglog(self.E, Eo, y)
                if r in ("capture", "fission"):
                    v = np.where(self.E < Eo[0], y[0] * np.sqrt(Eo[0] / self.E), v)
                sig[n][r] = v
        return sig

    # -------- point look-up (python side, used for thermal averages and plots)
    def at(self, nuc, rx, E):
        x = (np.log(np.asarray(E, float)) - self.lnE0) / self.d
        i = np.clip(np.floor(x).astype(int), 0, self.n - 2)
        f = x - i
        a = self.sig[nuc][rx]
        return a[i] * (1 - f) + a[i + 1] * f


# --------------------------------------------------------------------------- compositions
def number_densities(enrichment_wt=None, moderator=None, ratio=0.0, natural=False, pure238=False):
    """Atom densities [atoms/(b cm)] for a homogeneous mixture of uranium metal + moderator.
    ratio = moderator molecules per uranium atom.  Returns dict nuc -> N and the mixture density info."""
    P = PHYS
    if pure238:
        a5 = 0.0
    elif natural:
        a5 = P["nat_U235_atom_frac"]
    else:
        w = enrichment_wt / 100.0
        a5 = (w / P["M_U235"]) / (w / P["M_U235"] + (1 - w) / P["M_U238"])
    M_U = a5 * P["M_U235"] + (1 - a5) * P["M_U238"]
    v_U = M_U / (P["rho_U"] * N_AVO)                     # volume per U atom (1e-24 cm3)
    N = {n: 0.0 for n in NUCLIDES}
    v_m = 0.0
    if moderator in ("H2O", "D2O"):
        M_m, rho_m = P["M_" + moderator], P["rho_" + moderator]
        v_m = ratio * M_m / (rho_m * N_AVO)
    V = v_U + v_m
    N["U235"], N["U238"] = a5 / V, (1 - a5) / V
    if moderator == "H2O":
        N["H1"], N["O16"] = 2 * ratio / V, ratio / V
    elif moderator == "D2O":
        N["H2"], N["O16"] = 2 * ratio / V, ratio / V
    return N, {"a5": a5, "V_mod_over_V_U": v_m / v_U, "V": V}


if __name__ == "__main__":
    # quick self-check: print the assignment table and a few reference values
    t, E = required_table()
    print("U238 capture @0.0253eV:", micro("U238", np.array([2.53e-8]))["capture"])
    print("U235 fission/capture @0.0253eV:", micro("U235", np.array([2.53e-8]))["fission"], micro("U235", np.array([2.53e-8]))["capture"])
    print("H1 el @2meV:", micro("H1", np.array([2e-9]))["elastic"])
    print("RI of ladder:", ladder_resonance_integral(*_LADDER) + 1.5)
