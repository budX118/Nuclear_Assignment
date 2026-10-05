#!/usr/bin/env python3
"""
mc_reactor.py -- analog Monte Carlo of neutrons in an INFINITE HOMOGENEOUS medium (CH 5960, Assignment 1).

Each neutron is followed from birth (fission spectrum s(E) = 0.771 sqrt(E) exp(-0.776 E)) until it is
absorbed.  In an infinite medium nothing leaks, so every history ends in capture (n,gamma / n,alpha)
or fission, and

        k_inf = (1/N) * sum_i nu_i          nu_i = nu(E) if history i ends in fission, else 0

(neutrons produced per neutron born); an infinite medium "can go critical" if k_inf > 1.

Physics per history
  1. birth energy from s(E) (exact Maxwellian-type sampling)
  2. flight  d = -ln(xi)/Sigma_t(E), time d/v
  3. nuclide hit  ~ N_j sigma_t,j(E);  reaction ~ sigma_x,j / sigma_t,j   (elastic / inelastic / capture / fission)
  4. elastic: isotropic in the CM frame -> E' uniform in [alpha E, E];  inelastic: evaporation spectrum
     (U-235, U-238) or two-body kinematics with the level / binding energy Q (O-16, H-2 breakup)
  5. capture or fission ends the history; fission adds nu-bar(E) neutrons
  6. when a neutron scatters below 0.625 eV it joins the thermal pool: at 293.6 K it is in Maxwellian
     equilibrium with the medium, so the rest of its life is sampled in ONE step (geometric number of
     thermal collisions, exponential thermal lifetime 1/<v Sigma_a>, absorbing nuclide/reaction/energy from the
     Maxwellian-weighted rates).  This reproduces the analog result and makes 10^9 histories feasible.

Usage
  python mc_reactor.py --n 1e6 --out results_1e6          # all four cases, 10^6 neutrons each
  python mc_reactor.py --n 1e9 --out results_1e9 --resume # 10^9 neutrons (checkpointed, resumable)
  python mc_reactor.py --scan  --out results_scan         # k_inf vs amount of moderator
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit, prange

import xs_data as xs

# ----------------------------------------------------------------------------- tally layout
NMAXC = 400            # collisions followed for the "lethargy gained per collision" tally
NPATH = 700            # points stored for each sample neutron path
NFB_PER_DEC = 20       # flux / absorption-energy bins per decade
FB_LO, FB_HI = 1.0e-10, 20.0
NBF = int(np.ceil(np.log10(FB_HI / FB_LO) * NFB_PER_DEC))
NBB, EB_MAX = 240, 12.0                          # birth-energy histogram: 0..12 MeV
LT_LO, LT_HI, NLT_PER_DEC = 1.0e-12, 10.0, 20
NBL = int(np.ceil(np.log10(LT_HI / LT_LO) * NLT_PER_DEC))
NSC, NCNT = 16, 16
T_BIRTH = 1.0 / 0.776
NPATH_MAX = NPATH

# sc  : 0 nu  1 nu^2  2 t_fast  3 t_slow  4 t_thermal  5 life  6 life^2  7 thermal path length  8 nu from thermal fissions
#       9 E_birth  10 t_slow of thermalising neutrons  11 t_fast of thermalising neutrons
# cnt : 0 histories  1 reached thermal  2 thermal absorptions in fuel  3 collisions fast  4 collisions slowing
#       5 thermal collisions (pool)  6 sum of collisions-to-thermal  7 fissions  8 captures


def edges_flux():
    return np.logspace(np.log10(FB_LO), np.log10(FB_HI), NBF + 1)


def edges_life():
    return np.logspace(np.log10(LT_LO), np.log10(LT_HI), NBL + 1)


# ----------------------------------------------------------------------------- cases
CASE_SPECS = {
    1: dict(id=1, short="pure_U238", title="Pure U-238", pure238=True),
    2: dict(id=2, short="natural_U", title="Natural uranium", natural=True),
    3: dict(id=3, short="2pct_U_H2O", title="2 wt% enriched U + light water", enrichment=2.0, moderator="H2O", ratio=3.4),
    4: dict(id=4, short="natU_D2O", title="Natural U + heavy water", natural=True, moderator="D2O", ratio=300.0),
}


class Case:
    pass


def prepare_case(spec, lib, ratio=None):
    """Macroscopic tables, scattering parameters and thermal-pool data for one medium."""
    spec = dict(spec)
    if ratio is not None:
        spec["ratio"] = ratio
    if "N_override" in spec:                       # explicit densities (used by the validation tests)
        N = {n: float(spec["N_override"].get(n, 0.0)) for n in xs.NUCLIDES}
        info = {"a5": 0.0, "V_mod_over_V_U": 0.0, "V": 1.0}
    else:
        N, info = xs.number_densities(enrichment_wt=spec.get("enrichment"), moderator=spec.get("moderator"),
                                      ratio=spec.get("ratio", 0.0), natural=spec.get("natural", False),
                                      pure238=spec.get("pure238", False))
    act = [n for n in xs.NUCLIDES if N[n] > 0]
    nn = len(act)
    c = Case()
    c.spec, c.N, c.info, c.nuc, c.nn = spec, N, info, act, nn
    c.lib_mode = lib.mode
    # macroscopic table (cm^-1), [grid, nuclide*4 + reaction]
    mac = np.empty((lib.n, nn * 4), dtype=np.float32)
    for j, n in enumerate(act):
        for r, rx in enumerate(xs.RX):
            mac[:, j * 4 + r] = N[n] * lib.sig[n][rx]
    c.mac = mac
    c.A = np.array([xs.A_RATIO[n] for n in act])
    c.alpha = np.array([max(((a - 1.0) / (a + 1.0)) ** 2, 0.0) if a > 1.01 else 0.0 for a in c.A])
    c.Ex = np.array([xs.INEL_MODEL[n][0] for n in act])
    c.ikind = np.array([xs.INEL_MODEL[n][1] for n in act], dtype=np.int64)
    c.nutype = np.array([1 if n == "U235" else 2 if n == "U238" else 0 for n in act], dtype=np.int64)
    c.is_fuel = c.nutype > 0
    # ---- thermal pool
    NE = 800
    Eg = np.exp(np.linspace(np.log(1.0e-3 * xs.KT), np.log(xs.E_TH_CUT), NE))
    phi = Eg * np.exp(-Eg / xs.KT)                       # Maxwellian flux  E exp(-E/kT)
    dn = np.sqrt(Eg) * np.exp(-Eg / xs.KT)               # Maxwellian density sqrt(E) exp(-E/kT)
    w = Eg                                               # d(ln E) weight (grid is uniform in ln E)
    sig = {n: {rx: lib.at(n, rx, Eg) for rx in xs.RX} for n in act}
    nf = nn * 2
    R = np.zeros(nf)
    ecdf = np.zeros((nf + 1, NE))
    Rt = 0.0
    for j, n in enumerate(act):
        for x, rx in enumerate(("capture", "fission")):
            S = N[n] * sig[n][rx]
            R[j * 2 + x] = np.sum(phi * S * w)
            pdf = phi * S * w
            ecdf[j * 2 + x] = _cdf(pdf)
        Rt += np.sum(phi * N[n] * (sig[n]["elastic"] + sig[n]["capture"] + sig[n]["fission"] + sig[n]["inelastic"]) * w)
    pcoll = np.zeros(NE)
    for j, n in enumerate(act):
        pcoll += phi * N[n] * (sig[n]["elastic"] + sig[n]["capture"] + sig[n]["fission"] + sig[n]["inelastic"]) * w
    ecdf[nf] = _cdf(pcoll)
    Ra = R.sum()
    c.th_q = Ra / Rt
    c.th_fate = np.cumsum(R) / Ra
    c.th_fate[-1] = 1.0
    c.th_ecdf, c.th_egrid = ecdf, Eg
    # physical units: n(E) = dn, phi(E) = v n, v = VCONST sqrt(E)
    vE = xs.VCONST * np.sqrt(Eg)
    nphys = dn
    Ra_phys = np.sum(nphys * vE * sum_abs(N, sig, act) * w)     # sum over reactions of n v Sigma_a
    c.th_tau = np.sum(nphys * w) / Ra_phys                       # s   (mean thermal lifetime)
    c.th_vbar = np.sum(nphys * vE * w) / np.sum(nphys * w)        # cm/s (density-weighted mean speed)
    # thermal-flux shape in the flux bins (fraction of thermal track length per bin)
    fe = edges_flux()
    shape = np.zeros(NBF)
    Ef = np.exp(np.linspace(np.log(1.0e-3 * xs.KT), np.log(xs.E_TH_CUT), 20000))
    wt = Ef * Ef * np.exp(-Ef / xs.KT)                         # E*phi_M dlnE  = flux per dE x E ...
    ib = np.clip(np.searchsorted(fe, Ef) - 1, 0, NBF - 1)
    np.add.at(shape, ib, wt)
    c.th_shape = shape / shape.sum()
    # thermal-averaged cross sections (flux weighted) for the 4-factor theory
    c.th_avg = {n: {rx: np.sum(phi * sig[n][rx] * w) / np.sum(phi * w) for rx in xs.RX} for n in act}
    return c


def sum_abs(N, sig, act):
    return sum(N[n] * (sig[n]["capture"] + sig[n]["fission"]) for n in act)


def _cdf(pdf):
    c = np.cumsum(pdf)
    if c[-1] <= 0:
        return np.linspace(0.0, 1.0, len(pdf))
    c = c / c[-1]
    c[-1] = 1.0
    return c


# ----------------------------------------------------------------------------- Numba kernel
@njit(cache=True)
def _nu_of(kind, E):
    if kind == 1:
        if E <= 1.0:
            return 2.4355 + 0.065 * E
        return 2.50 + 0.145 * (E - 1.0)
    if kind == 2:
        return 2.30 + 0.15 * E
    return 0.0


@njit(cache=True)
def _evap(E, Ex, A):
    """outgoing energy from an evaporation spectrum  p(E') ~ E' exp(-E'/theta), E' < E - Ex,  theta = sqrt(8 (E-Ex)/A)"""
    emax = E - Ex
    if emax <= 0.0:
        emax = 0.5 * E
    th = np.sqrt(8.0 * emax / A)
    for _ in range(80):
        x = -th * (np.log(1.0 - np.random.random()) + np.log(1.0 - np.random.random()))
        if x < emax:
            return x
    return emax * np.random.random()


@njit(parallel=True, cache=True)
def run_batch(seed0, nchunk, m, dpar, ipar, mac, A, alpha, Ex, ikind, nutype, th_fate, th_ecdf, th_egrid,
              sc, cnt, evt, flux, absE, birth, lifeH, lsum, lcnt, cth, pathE, pathT, pathN):
    lnE0, inv_d, Eth, Efast = dpar[0], dpar[1], dpar[2], dpar[3]
    th_q, th_tau, th_vbar = dpar[4], dpar[5], dpar[6]
    fl_lo, fl_inv, lt_lo, lt_inv, bB_inv, Emax_b, Tb = dpar[7], dpar[8], dpar[9], dpar[10], dpar[11], dpar[12], dpar[13]
    ngrid, nn, nbF, nbB, nbL, nmaxc, rec_n = ipar[0], ipar[1], ipar[2], ipar[3], ipar[4], ipar[5], ipar[6]
    nf = nn * 2
    VC = 1.3831e9
    for c in prange(nchunk):
        np.random.seed(seed0 + c)
        stj = np.empty(nn)
        for h in range(m):
            rec = (rec_n > 0) and (c == 0) and (h < rec_n)
            # ---------------- birth
            while True:
                r1 = 1.0 - np.random.random()
                r2 = 1.0 - np.random.random()
                r3 = np.random.random()
                cc = np.cos(0.5 * np.pi * r3)
                E = -Tb * (np.log(r1) + np.log(r2) * cc * cc)
                if E < Emax_b:
                    break
            ib = int(E * bB_inv)
            if ib >= nbB:
                ib = nbB - 1
            birth[c, ib] += 1
            sc[c, 9] += E
            E0 = E
            tf = 0.0
            ts = 0.0
            tt = 0.0
            Lth = 0.0
            nc = 0
            nu_h = 0.0
            isth = False
            npts = 0
            if rec:
                pathE[h, 0] = E
                pathT[h, 0] = 0.0
                npts = 1
            if E < Eth:
                isth = True
            absorbed = False
            while (not isth) and (not absorbed):
                lnE = np.log(E)
                xg = (lnE - lnE0) * inv_d
                if xg < 0.0:
                    i = 0
                    f = 0.0
                elif xg >= ngrid - 1:
                    i = ngrid - 2
                    f = 1.0
                else:
                    i = int(xg)
                    f = xg - i
                st = 0.0
                for j in range(nn):
                    s = 0.0
                    for r in range(4):
                        k = j * 4 + r
                        s += mac[i, k] + f * (mac[i + 1, k] - mac[i, k])
                    stj[j] = s
                    st += s
                d = -np.log(1.0 - np.random.random()) / st
                ibf = int((lnE - fl_lo) * fl_inv)
                if ibf < 0:
                    ibf = 0
                elif ibf >= nbF:
                    ibf = nbF - 1
                flux[c, ibf] += d
                dt = d / (VC * np.sqrt(E))
                if E >= Efast:
                    tf += dt
                    cnt[c, 3] += 1
                else:
                    ts += dt
                    cnt[c, 4] += 1
                nc += 1
                # ---------------- which nuclide, which reaction
                rr = np.random.random() * st
                j = 0
                acc = stj[0]
                while rr >= acc and j < nn - 1:
                    j += 1
                    acc += stj[j]
                b = j * 4
                s0 = mac[i, b] + f * (mac[i + 1, b] - mac[i, b])
                s1 = mac[i, b + 1] + f * (mac[i + 1, b + 1] - mac[i, b + 1])
                s2 = mac[i, b + 2] + f * (mac[i + 1, b + 2] - mac[i, b + 2])
                r2 = np.random.random() * stj[j]
                if r2 < s0:
                    rx = 0
                elif r2 < s0 + s1:
                    rx = 1
                elif r2 < s0 + s1 + s2:
                    rx = 2
                else:
                    rx = 3
                evt[c, j, rx] += 1
                if rx >= 2:
                    absorbed = True
                    absE[c, j * 2 + (rx - 2), ibf] += 1
                    if rx == 3:
                        nu_h = _nu_of(nutype[j], E)
                        cnt[c, 7] += 1
                    else:
                        cnt[c, 8] += 1
                    break
                if rx == 0:
                    En = E * (alpha[j] + (1.0 - alpha[j]) * np.random.random())
                else:
                    if ikind[j] == 0:
                        En = _evap(E, Ex[j], A[j])
                    else:
                        a = A[j] / (A[j] + 1.0)
                        val = a * E - Ex[j]
                        if val <= 0.0:
                            En = E * (alpha[j] + (1.0 - alpha[j]) * np.random.random())
                        else:
                            Enc = a * val
                            Ecm = E / ((A[j] + 1.0) * (A[j] + 1.0))
                            mu = 2.0 * np.random.random() - 1.0
                            En = Enc + Ecm + 2.0 * np.sqrt(Enc * Ecm) * mu
                            if En < 1.0e-12:
                                En = 1.0e-12
                E = En
                if nc <= nmaxc:
                    lsum[c, nc - 1] += np.log(E0 / E)
                    lcnt[c, nc - 1] += 1
                if rec and npts < NPATH_MAX:
                    pathE[h, npts] = E
                    pathT[h, npts] = tf + ts
                    npts += 1
                if E < Eth:
                    isth = True
            # ---------------- thermal pool
            if isth:
                cnt[c, 1] += 1
                cnt[c, 6] += nc
                cth[c, min(nc, nmaxc)] += 1
                if th_q >= 1.0:
                    K = 1
                else:
                    K = 1 + int(np.log(1.0 - np.random.random()) / np.log(1.0 - th_q))
                cnt[c, 5] += K
                tt = -th_tau * np.log(1.0 - np.random.random())
                Lth = tt * th_vbar
                u = np.random.random()
                kk = 0
                while kk < nf - 1 and u >= th_fate[kk]:
                    kk += 1
                j = kk // 2
                x = kk - 2 * j
                row = th_ecdf[kk]
                idx = np.searchsorted(row, np.random.random())
                if idx >= row.shape[0]:
                    idx = row.shape[0] - 1
                Ea = th_egrid[idx]
                ibf = int((np.log(Ea) - fl_lo) * fl_inv)
                if ibf < 0:
                    ibf = 0
                elif ibf >= nbF:
                    ibf = nbF - 1
                absE[c, kk, ibf] += 1
                evt[c, j, 2 + x] += 1
                if nutype[j] > 0:
                    cnt[c, 2] += 1
                if x == 1:
                    nu_h = _nu_of(nutype[j], Ea)
                    sc[c, 8] += nu_h
                    cnt[c, 7] += 1
                else:
                    cnt[c, 8] += 1
                sc[c, 10] += ts
                sc[c, 11] += tf
                if rec:
                    t0 = tf + ts
                    nk = min(K, NPATH_MAX - npts - 1)
                    rowc = th_ecdf[nf]
                    for q in range(nk):
                        ic = np.searchsorted(rowc, np.random.random())
                        if ic >= rowc.shape[0]:
                            ic = rowc.shape[0] - 1
                        pathE[h, npts] = th_egrid[ic]
                        pathT[h, npts] = t0 + tt * (q + 1.0) / K
                        npts += 1
            life = tf + ts + tt
            sc[c, 0] += nu_h
            sc[c, 1] += nu_h * nu_h
            sc[c, 2] += tf
            sc[c, 3] += ts
            sc[c, 4] += tt
            sc[c, 5] += life
            sc[c, 6] += life * life
            sc[c, 7] += Lth
            cnt[c, 0] += 1
            for g in range(4):
                if g == 0:
                    tv = tf
                elif g == 1:
                    tv = ts
                elif g == 2:
                    tv = tt
                else:
                    tv = life
                if tv > 0.0:
                    il = int((np.log(tv) - lt_lo) * lt_inv)
                    if il < 0:
                        il = 0
                    elif il >= nbL:
                        il = nbL - 1
                    lifeH[c, g, il] += 1
            if rec:
                pathN[h] = npts


def kernel_params(case, lib, rec_n=0):
    dpar = np.array([lib.lnE0, 1.0 / lib.d, xs.E_TH_CUT, xs.E_FAST, case.th_q, case.th_tau, case.th_vbar,
                     np.log(FB_LO), NFB_PER_DEC / np.log(10.0), np.log(LT_LO), NLT_PER_DEC / np.log(10.0),
                     NBB / EB_MAX, xs.E_MAX, T_BIRTH])
    ipar = np.array([lib.n, case.nn, NBF, NBB, NBL, NMAXC, rec_n], dtype=np.int64)
    return dpar, ipar


def simulate(case, lib, N, seed=12345, nchunk=64, m=None, out_dir=None, tag="", resume=False, verbose=True, rec_n=8):
    """Run N histories. Returns a dict of summed tallies + chunk-level k data."""
    N = int(N)
    if m is None:
        m = int(max(1000, 10 ** round(np.log10(N / 4000.0))))
    m = int(min(m, N))
    nchunk_total = int(np.ceil(N / m))
    nb = case.nn
    ck = os.path.join(out_dir, f"{tag}_ckpt.npz") if out_dir else None
    acc = None
    done = 0                           # chunks done
    chunk_nu, chunk_nu2 = [], []
    if resume and ck and os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        acc = {k: z[k] for k in z.files if k not in ("chunk_nu", "chunk_nu2", "done")}
        chunk_nu, chunk_nu2 = list(z["chunk_nu"]), list(z["chunk_nu2"])
        done = int(z["done"])
        if verbose:
            print(f"   resuming {tag} at chunk {done}/{nchunk_total}", flush=True)
    dpar, ipar = None, None
    t0 = time.time()
    first = done == 0
    while done < nchunk_total:
        ncur = min(nchunk, nchunk_total - done)
        rn = rec_n if (done == 0) else 0
        dpar, ipar = kernel_params(case, lib, rn)
        sc = np.zeros((ncur, NSC))
        cnt = np.zeros((ncur, NCNT), dtype=np.int64)
        evt = np.zeros((ncur, nb, 4), dtype=np.int64)
        flux = np.zeros((ncur, NBF))
        absE = np.zeros((ncur, nb * 2, NBF), dtype=np.int64)
        birth = np.zeros((ncur, NBB), dtype=np.int64)
        lifeH = np.zeros((ncur, 4, NBL), dtype=np.int64)
        lsum = np.zeros((ncur, NMAXC))
        lcnt = np.zeros((ncur, NMAXC), dtype=np.int64)
        cth = np.zeros((ncur, NMAXC + 1), dtype=np.int64)
        pathE = np.zeros((max(rn, 1), NPATH))
        pathT = np.zeros((max(rn, 1), NPATH))
        pathN = np.zeros(max(rn, 1), dtype=np.int64)
        run_batch(seed + done, ncur, m, dpar, ipar, case.mac, case.A, case.alpha, case.Ex, case.ikind, case.nutype,
                  case.th_fate, case.th_ecdf, case.th_egrid, sc, cnt, evt, flux, absE, birth, lifeH, lsum, lcnt, cth,
                  pathE, pathT, pathN)
        part = dict(sc=sc.sum(0), cnt=cnt.sum(0), evt=evt.sum(0), flux=flux.sum(0), absE=absE.sum(0), birth=birth.sum(0),
                    lifeH=lifeH.sum(0), lsum=lsum.sum(0), lcnt=lcnt.sum(0), cth=cth.sum(0))
        if acc is None:
            acc = part
        else:
            for k in part:
                acc[k] = acc[k] + part[k]
        if rn > 0:
            acc["pathE"], acc["pathT"], acc["pathN"] = pathE, pathT, pathN
        chunk_nu.extend(sc[:, 0].tolist())
        chunk_nu2.extend(sc[:, 1].tolist())
        done += ncur
        nd = done * m
        if verbose and (done >= nchunk_total or int(10 * done / nchunk_total) > int(10 * (done - ncur) / nchunk_total)):
            k = acc["sc"][0] / min(nd, N)
            el = time.time() - t0
            print(f"   {tag}: {min(nd, N):>13,d}/{N:,d} histories   k_inf = {k:.5f}   [{el:7.1f} s]", flush=True)
        if ck:
            os.makedirs(out_dir, exist_ok=True)
            np.savez(ck, done=done, chunk_nu=np.array(chunk_nu), chunk_nu2=np.array(chunk_nu2), **acc)
    n_done = int(acc["cnt"][0])
    res = dict(acc)
    res["chunk_nu"], res["chunk_nu2"] = np.array(chunk_nu), np.array(chunk_nu2)
    res["chunk_m"] = m
    res["N"] = n_done
    res["elapsed"] = time.time() - t0
    return res


def k_and_sigma(res):
    n = res["N"]
    s1, s2 = res["sc"][0], res["sc"][1]
    k = s1 / n
    return k, np.sqrt(max(s2 / n - k * k, 0.0) / n)


def save_result(path, case, res, lib):
    meta = dict(spec=case.spec, nuc=case.nuc, N_dens=case.N, info=case.info, lib_mode=lib.mode,
                th_q=case.th_q, th_tau=case.th_tau, th_vbar=case.th_vbar, N=int(res["N"]), elapsed=res["elapsed"],
                chunk_m=int(res["chunk_m"]))
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta=json.dumps(meta, default=float), th_shape=case.th_shape, **arrays)


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, default=1e6, help="neutron histories per case (default 1e6)")
    ap.add_argument("--out", default="results_1e6", help="output folder")
    ap.add_argument("--cases", default="1,2,3,4", help="comma separated case ids")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints in --out")
    ap.add_argument("--coarse", action="store_true", help="use ONLY the 50-point assignment table (sensitivity test)")
    ap.add_argument("--xs-dir", default=None, help="folder with U235.csv U238.csv H1.csv H2.csv O16.csv to override the built-in data")
    ap.add_argument("--ratio3", type=float, default=None, help="H2O molecules per U atom for case 3")
    ap.add_argument("--ratio4", type=float, default=None, help="D2O molecules per U atom for case 4")
    ap.add_argument("--scan", action="store_true", help="k_inf versus moderator amount instead of the four main cases")
    ap.add_argument("--scan-n", type=float, default=2e5, help="histories per scan point")
    ap.add_argument("--chunks", type=int, default=64, help="chunks per parallel batch")
    ap.add_argument("--chunk-size", type=int, default=None, help="histories per chunk")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    lib = xs.Library("coarse" if a.coarse else "full", xs_dir=a.xs_dir)
    if a.ratio3:
        CASE_SPECS[3]["ratio"] = a.ratio3
    if a.ratio4:
        CASE_SPECS[4]["ratio"] = a.ratio4
    if a.scan:
        return scan(a, lib)
    for cid in [int(c) for c in a.cases.split(",")]:
        spec = CASE_SPECS[cid]
        case = prepare_case(spec, lib)
        print(f"== case {cid}: {spec['title']}   (N = {int(a.n):,d}, library = {lib.mode}) ==", flush=True)
        res = simulate(case, lib, a.n, seed=a.seed + 1000003 * cid, nchunk=a.chunks, m=a.chunk_size, out_dir=a.out,
                       tag=f"case{cid}", resume=a.resume)
        k, s = k_and_sigma(res)
        print(f"   RESULT case {cid}: k_inf = {k:.5f} +- {s:.5f}   ({res['elapsed']:.0f} s)", flush=True)
        save_result(os.path.join(a.out, f"case{cid}.npz"), case, res, lib)
        ck = os.path.join(a.out, f"case{cid}_ckpt.npz")
        if os.path.exists(ck):
            os.remove(ck)
    print("done ->", a.out)


def scan(a, lib):
    """k_inf as a function of moderator molecules per uranium atom (shows an optimum and the critical window)."""
    grids = {
        "2pct_U_H2O": (dict(enrichment=2.0, moderator="H2O"), np.geomspace(0.4, 40, 16)),
        "natU_D2O": (dict(natural=True, moderator="D2O"), np.geomspace(20, 1500, 16)),
        "natU_H2O": (dict(natural=True, moderator="H2O"), np.geomspace(0.4, 40, 16)),
        "2pct_U_D2O": (dict(enrichment=2.0, moderator="D2O"), np.geomspace(20, 1500, 16)),
    }
    out = {}
    for name, (spec, rs) in grids.items():
        out[name] = dict(ratio=[], k=[], sigma=[], vm_over_vu=[])
        for r in rs:
            case = prepare_case(dict(spec, ratio=float(r)), lib)
            res = simulate(case, lib, a.scan_n, seed=a.seed + 77, nchunk=64, m=None, verbose=False, rec_n=0)
            k, s = k_and_sigma(res)
            out[name]["ratio"].append(float(r))
            out[name]["k"].append(float(k))
            out[name]["sigma"].append(float(s))
            out[name]["vm_over_vu"].append(float(case.info["V_mod_over_V_U"]))
            print(f"   {name:12s} ratio = {r:9.3f}   k_inf = {k:.4f} +- {s:.4f}", flush=True)
    # unmoderated uranium versus enrichment
    out["enrichment"] = dict(w=[], k=[], sigma=[])
    for w in [0.0, 0.711, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 90]:
        spec = dict(pure238=True) if w == 0.0 else dict(enrichment=w)
        case = prepare_case(spec, lib)
        res = simulate(case, lib, a.scan_n, seed=a.seed + 99, nchunk=64, m=None, verbose=False, rec_n=0)
        k, s = k_and_sigma(res)
        out["enrichment"]["w"].append(w)
        out["enrichment"]["k"].append(float(k))
        out["enrichment"]["sigma"].append(float(s))
        print(f"   unmoderated U, {w:6.3f} wt% U-235: k_inf = {k:.4f} +- {s:.4f}", flush=True)
    with open(os.path.join(a.out, "scan.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("scan ->", os.path.join(a.out, "scan.json"))


if __name__ == "__main__":
    main()
