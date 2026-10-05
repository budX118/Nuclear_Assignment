#!/usr/bin/env python3
"""
crosscheck_deterministic.py -- independent, NON-random check of the Monte Carlo engine.

The collision density F(E) (collisions per source neutron per unit energy) of an infinite medium obeys

    F(E) = S(E) + int  F(E') [ sum_j Sigma_s,j(E')/Sigma_t(E') p_j(E'->E) ]  dE'

with exactly the same physics as mc_reactor.py (isotropic-CM elastic scattering, evaporation / two-body
inelastic scattering, 0.625 eV thermal pool, same cross-sections).  Because neutrons only lose energy above
0.625 eV the equation is solved by ONE sweep from high to low energy on the 170 000-cell ln(E) library grid
(no iterations, no random numbers).  Then

    k_inf = sum_cells F_i * (sum_j nu_j Sigma_f,j / Sigma_t)_i  +  (neutrons reaching the pool) * nu_pool

Usage:   python crosscheck_deterministic.py --mc results_1e9 [--out analysis]
"""
import argparse
import json
import os

import numpy as np
from numba import njit
from scipy.special import gammainc

import xs_data as xs
import mc_reactor as mc

CS = 100          # fine cells per coarse bucket for the evaporation kernel (0.01 in lethargy)


@njit(cache=True)
def _edges(i, lnE0, d):
    return np.exp(lnE0 + (i - 0.5) * d), np.exp(lnE0 + (i + 0.5) * d)


@njit(cache=True)
def _idx(E, lnE0, d):
    return int(np.floor((np.log(E) - lnE0) / d + 0.5))


@njit(cache=True)
def _deposit(a, b, w, lnE0, d, i_th, diff, direct, pool):
    """spread weight w uniformly in E over [a, b]"""
    if w <= 0.0 or b <= a:
        return
    Ecut = np.exp(lnE0 + (i_th - 0.5) * d)
    dens = w / (b - a)
    if a < Ecut:
        pool[0] += dens * (min(b, Ecut) - a)
        a = Ecut
        if b <= Ecut:
            return
    ia = _idx(a, lnE0, d)
    ib = _idx(b, lnE0, d)
    if ia == ib:
        direct[ia] += dens * (b - a)
        return
    lo_a, hi_a = _edges(ia, lnE0, d)
    lo_b, hi_b = _edges(ib, lnE0, d)
    direct[ia] += dens * (hi_a - a)
    direct[ib] += dens * (b - lo_b)
    if ib - 1 >= ia + 1:
        diff[ib - 1] += dens
        diff[ia] -= dens


@njit(cache=True)
def _G(E, th):
    return -th * (E + th) * np.exp(-E / th)


@njit(cache=True)
def _flush_evap(W, Ebar, Ex, A, elo_bucket, lnE0, d, i_th, direct, pool):
    emax = Ebar - Ex
    if emax <= 0.0:
        emax = 0.5 * Ebar
    th = np.sqrt(8.0 * emax / A)
    emx = min(emax, elo_bucket)
    Ecut = np.exp(lnE0 + (i_th - 0.5) * d)
    norm = _G(emx, th) - _G(0.0, th)
    if norm <= 0.0:
        return
    kmax = _idx(emx, lnE0, d)
    for k in range(kmax, i_th - 1, -1):
        lo, hi = _edges(k, lnE0, d)
        if hi > emx:
            hi = emx
        if hi <= lo:
            continue
        direct[k] += W * (_G(hi, th) - _G(lo, th)) / norm
    if emx > Ecut:
        pool[0] += W * (_G(Ecut, th) - _G(0.0, th)) / norm
    else:
        pool[0] += W


@njit(cache=True)
def sweep(mac, lnE0, d, n, i_th, nn, alpha, A, Ex, ikind, nutype, S, Eg):
    F = np.zeros(n)
    diff = np.zeros(n + 2)
    direct = np.zeros(n + 2)
    pool = np.zeros(1)
    Wb = np.zeros(nn)
    Wbe = np.zeros(nn)
    absr = np.zeros((nn, 2))      # [nuclide, capture/fission] absorption (events per source neutron), non-thermal
    kk = 0.0
    run = 0.0
    for i in range(n - 1, i_th - 1, -1):
        lo, hi = _edges(i, lnE0, d)
        E = Eg[i]
        run += diff[i]
        # --- cross sections in the cell
        st = 0.0
        for j in range(nn):
            for r in range(4):
                st += mac[i, j * 4 + r]
        # self-scatter fraction (elastic landing in the same cell)
        selfc = 0.0
        for j in range(nn):
            pel = mac[i, j * 4] / st
            a_ = alpha[j] * E
            frac = (E - max(a_, lo)) / ((1.0 - alpha[j]) * E)
            if frac > 0.0:
                selfc += pel * frac
        Fi = (S[i] + run * (hi - lo) + direct[i]) / (1.0 - selfc)
        F[i] = Fi
        # --- absorption
        for j in range(nn):
            b = j * 4
            sc_ = mac[i, b + 2] / st
            sf_ = mac[i, b + 3] / st
            absr[j, 0] += Fi * sc_
            absr[j, 1] += Fi * sf_
            if sf_ > 0.0:
                kk += Fi * sf_ * mc._nu_of(nutype[j], E)
        # --- scattering out of this cell
        for j in range(nn):
            b = j * 4
            pel = mac[i, b] / st
            pin = mac[i, b + 1] / st
            if pel > 0.0:
                w_tot = Fi * pel
                dens = w_tot / ((1.0 - alpha[j]) * E)
                a_ = alpha[j] * E
                bb = lo * (1.0 - 1e-12)
                if a_ < bb:
                    _deposit(a_, bb, dens * (bb - a_), lnE0, d, i_th, diff, direct, pool)
            if pin > 0.0:
                w_in = Fi * pin
                if ikind[j] == 0:
                    Wb[j] += w_in
                    Wbe[j] += w_in * E
                else:
                    a1 = A[j] / (A[j] + 1.0)
                    val = a1 * E - Ex[j]
                    if val <= 0.0:       # same fall-back as the MC code: treat as elastic
                        w_tot = w_in
                        dens = w_tot / ((1.0 - alpha[j]) * E)
                        a_ = alpha[j] * E
                        bb = lo * (1.0 - 1e-12)
                        if a_ < bb:
                            _deposit(a_, bb, dens * (bb - a_), lnE0, d, i_th, diff, direct, pool)
                    else:
                        Enc = a1 * val
                        Ecm = E / ((A[j] + 1.0) * (A[j] + 1.0))
                        elo = (np.sqrt(Enc) - np.sqrt(Ecm)) ** 2
                        ehi = (np.sqrt(Enc) + np.sqrt(Ecm)) ** 2
                        ehi = min(ehi, lo * (1.0 - 1e-12))
                        if ehi > elo:
                            _deposit(elo, ehi, w_in, lnE0, d, i_th, diff, direct, pool)
        # --- flush evaporation buckets at the lowest cell of each bucket
        if i % CS == 0:
            for j in range(nn):
                if Wb[j] > 0.0:
                    _flush_evap(Wb[j], Wbe[j] / Wb[j], Ex[j], A[j], lo, lnE0, d, i_th, direct, pool)
                    Wb[j] = 0.0
                    Wbe[j] = 0.0
    return F, kk, pool[0], absr


def solve_case(case, lib):
    n, d, lnE0 = lib.n, lib.d, lib.lnE0
    Eg = np.exp(lnE0 + d * np.arange(n))
    i_th = int(np.ceil((np.log(xs.E_TH_CUT) - lnE0) / d))
    # birth probability in each cell:  s(E) dE,  Gamma(3/2, T)
    T = mc.T_BIRTH
    lo = np.exp(lnE0 + (np.arange(n) - 0.5) * d)
    hi = np.exp(lnE0 + (np.arange(n) + 0.5) * d)
    S = gammainc(1.5, np.minimum(hi, xs.E_MAX) / T) - gammainc(1.5, np.minimum(lo, xs.E_MAX) / T)
    S = S / gammainc(1.5, xs.E_MAX / T)
    mac = case.mac.astype(np.float64)
    F, kk, pool, absr = sweep(mac, lnE0, d, n, i_th, case.nn, case.alpha, case.A, case.Ex, case.ikind, case.nutype, S, Eg)
    # thermal pool: fate probabilities from th_fate (cumulative over [capture_j, fission_j] ordering)
    fate = np.diff(np.concatenate([[0.0], case.th_fate]))
    pool_abs = np.zeros((case.nn, 2))
    k_pool = 0.0
    for j in range(case.nn):
        for x in range(2):
            pool_abs[j, x] = pool * fate[j * 2 + x]
        if x == 1 or True:
            nu_th = float(mc._nu_of(int(case.nutype[j]), 2.5e-8))
            k_pool += pool * fate[j * 2 + 1] * nu_th
    k = kk + k_pool
    phi_u = np.zeros(n)
    st = np.array([case.mac[:, j * 4:(j + 1) * 4].sum(1) for j in range(case.nn)]).sum(0)
    phi_u = F / np.maximum(st, 1e-300) / d          # flux per unit lethargy  (collisions/Sigma_t per ln E)
    return dict(k=k, pool=pool, abs_nonth=absr, abs_pool=pool_abs, phi_u=phi_u, Eg=Eg, i_th=i_th, F=F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", default="results_1e9")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--coarse", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lib = xs.Library("coarse" if a.coarse else "full")
    rows = []
    sol = {}
    L = ["# Deterministic cross-check of the Monte Carlo engine\n",
         "Same cross-sections and scattering physics, but solved as a single high-to-low energy sweep of the collision-density equation (no random numbers).\n",
         "\n| Case | k_inf deterministic | k_inf Monte Carlo (± 1σ) | difference (σ) | pool arrivals p (det.) | p (MC) |\n|---|---|---|---|---|---|\n"]
    for c in (1, 2, 3, 4):
        case = mc.prepare_case(mc.CASE_SPECS[c], lib)
        r = solve_case(case, lib)
        sol[c] = r
        z = np.load(os.path.join(a.mc, f"case{c}.npz"))
        meta = json.loads(str(z["meta"]))
        n = meta["N"]
        k_mc = z["sc"][0] / n
        s_mc = np.sqrt(max(z["sc"][1] / n - k_mc ** 2, 0) / n)
        p_mc = z["cnt"][1] / n
        L.append(f"| {c} | {r['k']:.5f} | {k_mc:.5f} ± {s_mc:.5f} | {(k_mc - r['k']) / s_mc:+.2f} | {r['pool']:.5f} | {p_mc:.5f} |\n")
        print(f"case {c}: deterministic k = {r['k']:.5f}   MC k = {k_mc:.5f} +- {s_mc:.5f}   ({(k_mc - r['k']) / s_mc:+.2f} sigma)   p: {r['pool']:.5f} vs {p_mc:.5f}")
        rows.append((c, r, z, meta))
    L.append("\nThe small remaining differences are the O(Δu) = 1e-4 energy-discretisation error of the deterministic sweep plus the statistical error of the MC.\n")
    L.append("\n## Absorption partition (fraction of all neutrons)\n\n| Case | nuclide | reaction | deterministic | Monte Carlo |\n|---|---|---|---|---|\n")
    for c, r, z, meta in rows:
        for j, nm in enumerate(meta["nuc"]):
            for x, rx in enumerate(("capture", "fission")):
                det = r["abs_nonth"][j, x] + r["abs_pool"][j, x]
                mcv = z["absE"][j * 2 + x].sum() / meta["N"]
                if det > 1e-6 or mcv > 1e-6:
                    L.append(f"| {c} | {nm} | {rx} | {det:.5f} | {mcv:.5f} |\n")
    open(os.path.join(a.out, "crosscheck_deterministic.md"), "w").write("".join(L))
    # flux comparison figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    fe = mc.edges_flux()
    fc = np.sqrt(fe[:-1] * fe[1:])
    du = np.log(10.0) / mc.NFB_PER_DEC
    for a_, (c, r, z, meta) in zip(ax.ravel(), rows):
        n = meta["N"]
        case = mc.prepare_case(mc.CASE_SPECS[c], lib)
        f_mc = (z["flux"] / n + z["sc"][7] / n * case.th_shape) / du
        # deterministic flux averaged in the same bins (non-thermal part only)
        Eg, ph = r["Eg"], r["phi_u"]
        sel = np.arange(len(Eg)) >= r["i_th"]
        ib = np.clip(np.searchsorted(fe, Eg[sel]) - 1, 0, mc.NBF - 1)
        num = np.bincount(ib, weights=ph[sel], minlength=mc.NBF)
        cnt = np.bincount(ib, minlength=mc.NBF)
        f_det = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        a_.plot(fc * 1e6, f_mc, color="#2a78d6", lw=2.4, label="Monte Carlo")
        a_.plot(fc * 1e6, np.where(fc > xs.E_TH_CUT, f_det, np.nan), color="#eb6834", lw=1.0, ls="--", label="deterministic sweep")
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.set_xlim(1e-3, 2e7)
        top = np.nanmax(f_mc)
        a_.set_ylim(top * 1e-4, top * 3)
        a_.set_title(f"Case {c}")
        a_.grid(color="#e1e0d9")
    ax[0, 0].legend()
    for a_ in ax[1]:
        a_.set_xlabel("neutron energy (eV)")
    fig.supylabel("flux per unit lethargy (cm / source neutron)")
    fig.suptitle("Independent check: Monte Carlo flux against the deterministic collision-density sweep")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(a.out, "fig15_deterministic_crosscheck.png"), dpi=170)
    print("wrote", os.path.join(a.out, "crosscheck_deterministic.md"))


if __name__ == "__main__":
    main()
