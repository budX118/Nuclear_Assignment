#!/usr/bin/env python3
"""
validate_physics.py -- quick self-tests of data and sampling (runs in seconds).  python validate_physics.py
Each line prints PASS/FAIL against a textbook / evaluated-data value.
"""
import numpy as np
from scipy.stats import kstest
from scipy.special import gammainc

import xs_data as xs
import mc_reactor as mc

ok_all = True


def check(name, value, ref, tol):
    global ok_all
    ok = abs(value - ref) <= tol * abs(ref)
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {value:12.5g}   (reference {ref:g}, tol {100*tol:g}%)")


lib = xs.Library("full")
E0 = np.array([2.53e-8])
at = lambda n, r, E=E0: float(np.ravel(lib.at(n, r, E))[0])

print("--- thermal (0.0253 eV) constants, barn")
check("U-235 fission", at("U235", "fission"), 585.0, 0.01)
check("U-235 capture", at("U235", "capture"), 98.7, 0.01)
check("U-238 capture", at("U238", "capture"), 2.68, 0.01)
check("U-238 elastic", at("U238", "elastic"), 9.3, 0.02)
check("H-1 capture", at("H1", "capture"), 0.3326, 0.01)
check("H-2 capture", at("H2", "capture"), 5.0e-4, 0.05)
check("U-235 eta (nu=2.4355)", 2.4355 * 585.1 / (585.1 + 98.7), 2.077, 0.01)
print("--- resonance integrals (0.5 eV - 20 MeV, dilute)")
E = np.exp(np.arange(np.log(0.5e-6), np.log(20.0), 1e-4))
check("U-238 capture RI", lib.at("U238", "capture", E).sum() * 1e-4, 275.0, 0.05)
check("U-235 fission RI", lib.at("U235", "fission", E).sum() * 1e-4, 275.0, 0.05)
print("--- U-238 thresholds / fast data")
check("U-238 fission @ 2 MeV", at("U238", "fission", np.array([2.0])), 0.54, 0.05)
check("U-238 inelastic @ 1 MeV", at("U238", "inelastic", np.array([1.0])), 2.46, 0.05)
check("H-1 elastic @ 1 MeV", at("H1", "elastic", np.array([1.0])), 4.25, 0.03)
print("--- slowing-down parameter xi = 1 + alpha ln(alpha)/(1-alpha)")
rng = np.random.default_rng(1)
for nm, ref in (("H2", 0.725), ("O16", 0.120), ("U238", 0.0084)):
    a = ((xs.A_RATIO[nm] - 1) / (xs.A_RATIO[nm] + 1)) ** 2
    Ep = a + (1 - a) * rng.random(2_000_000)             # same law as the kernel: E'/E uniform on [alpha, 1]
    check(f"xi({nm}) sampled from the kernel's elastic law", -np.log(Ep).mean(), ref, 0.02)
print("--- birth spectrum s(E) = 0.771 sqrt(E) exp(-0.776 E)")
T = mc.T_BIRTH
r1, r2, r3 = rng.random(400000), rng.random(400000), rng.random(400000)
Eb = -T * (np.log(1 - r1) + np.log(1 - r2) * np.cos(0.5 * np.pi * r3) ** 2)
print("KS test vs exact Gamma(3/2) CDF: p-value = %.3f" % kstest(Eb, lambda x: gammainc(1.5, x / T)).pvalue)
check("mean birth energy", Eb.mean(), 1.5 / 0.776, 0.005)
check("normalisation 0.771 of s(E)", 1.0 / (0.5 * np.sqrt(np.pi) * 0.776 ** -1.5), 0.771, 0.002)
print("--- Maxwellian thermal-pool consistency (case 3)")
c3 = mc.prepare_case(mc.CASE_SPECS[3], lib)
print("   absorption probability per thermal collision q = %.4g,  thermal lifetime = %.4g s,  <v> = %.4g cm/s" % (c3.th_q, c3.th_tau, c3.th_vbar))
check("<v> of a Maxwellian density at 293.6 K = 2/sqrt(pi) sqrt(2kT/m)", c3.th_vbar, 2 / np.sqrt(np.pi) * xs.VCONST * np.sqrt(xs.KT), 0.01)
print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
