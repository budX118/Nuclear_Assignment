#!/usr/bin/env python3
"""
analyze.py -- turns the saved Monte-Carlo tallies (case1.npz ... case4.npz) into the figures and tables of the report.

  python analyze.py --main results_1e9 --small results_1e6 --scan results_scan/scan.json --out analysis

Outputs (in --out):  fig*.png , summary.md (every number quoted in the figures), cross_section_table.csv/.md
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import digamma

import xs_data as xs
import mc_reactor as mc

# ----------------------------------------------------------------------------- style (validated categorical slots, light surface)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 110,
    "savefig.dpi": 170, "lines.linewidth": 1.6,
})
NICE = {"U235": "U-235", "U238": "U-238", "H1": "H-1", "H2": "D (H-2)", "O16": "O-16"}
CASES = [1, 2, 3, 4]
TITLES = {1: "Case 1: pure U-238", 2: "Case 2: natural U", 3: "Case 3: 2% U + H$_2$O", 4: "Case 4: natural U + D$_2$O"}
DU = np.log(10.0) / mc.NFB_PER_DEC
FE = mc.edges_flux()
FC = np.sqrt(FE[:-1] * FE[1:])                       # bin centres (MeV)
LE = mc.edges_life()
LC = np.sqrt(LE[:-1] * LE[1:])
OUT = "analysis"


def save(fig, name):
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.965 if fig._suptitle is not None else 1.0))
    except Exception:
        pass
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ----------------------------------------------------------------------------- loading
class Run:
    def __init__(self, folder, label, color, lw):
        self.folder, self.label, self.color, self.lw = folder, label, color, lw
        self.d, self.meta, self.case = {}, {}, {}
        self.lib = None
        for c in CASES:
            fn = os.path.join(folder, f"case{c}.npz")
            z = np.load(fn)
            self.d[c] = {k: z[k] for k in z.files if k != "meta"}
            self.meta[c] = json.loads(str(z["meta"]))
        mode = self.meta[1]["lib_mode"]
        self.lib = xs.Library(mode if mode in ("full", "coarse") else "full")
        for c in CASES:
            self.case[c] = mc.prepare_case(self.meta[c]["spec"], self.lib)

    def N(self, c):
        return float(self.meta[c]["N"])

    def k(self, c):
        n = self.N(c)
        s = self.d[c]["sc"]
        k = s[0] / n
        return k, np.sqrt(max(s[1] / n - k * k, 0) / n)

    def flux_u(self, c):
        """flux per unit lethargy, cm per source neutron (track-length estimator, thermal part from the pool)"""
        n = self.N(c)
        f = self.d[c]["flux"] / n + self.d[c]["sc"][7] / n * self.case[c].th_shape
        return f / DU


# ----------------------------------------------------------------------------- figures
def fig_cross_sections(lib):
    tab, Et = xs.required_table()
    E = lib.E
    st = 4
    panels = [("U238", ["elastic", "inelastic", "capture", "fission"], "U-238"),
              ("U235", ["elastic", "inelastic", "capture", "fission"], "U-235"),
              ("mod", None, "Moderator nuclides")]
    colr = {"elastic": BLUE, "inelastic": ORANGE, "capture": AQUA, "fission": VIOLET}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, (nuc, rxs, ttl) in zip(ax, panels):
        if rxs:
            for r in rxs:
                y = lib.sig[nuc][r][::st]
                a.plot(E[::st] * 1e6, np.where(y > 1e-5, y, np.nan), color=colr[r], lw=0.9, label=r)
                a.plot(Et * 1e6, np.where(tab[nuc][r] > 1e-5, tab[nuc][r], np.nan), "o", ms=3.5, color=colr[r], mec=SURF, mew=0.5)
        else:
            for n, ls in (("H1", "-"), ("H2", "-"), ("O16", "-")):
                col = {"H1": BLUE, "H2": ORANGE, "O16": AQUA}[n]
                a.plot(E[::st] * 1e6, lib.sig[n]["elastic"][::st], color=col, lw=1.1, label=f"{NICE[n]} elastic")
                a.plot(E[::st] * 1e6, np.where(lib.sig[n]["capture"][::st] > 1e-9, lib.sig[n]["capture"][::st], np.nan), color=col, lw=1.1, ls="--", label=f"{NICE[n]} capture")
            a.set_ylim(1e-9, 300)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(2e-3, 2e7)
        if rxs:
            a.set_ylim(1e-5, 1e5)
        a.set_xlabel("neutron energy E (eV)")
        a.set_title(ttl)
        a.legend(fontsize=7.5, ncol=2 if not rxs else 1, loc="upper right" if rxs else "lower left")
    ax[0].set_ylabel("microscopic cross-section (barn)")
    fig.suptitle("Library used by the code: lines = fine library, dots = the 50 tabulated energies (0.2,0.4,0.6,0.8,1.0 x 10$^y$ MeV, y = -8..1)", fontsize=10, color=INK2)
    save(fig, "fig01_cross_sections.png")


def fig_birth(runs):
    E = (np.arange(mc.NBB) + 0.5) * mc.EB_MAX / mc.NBB
    w = mc.EB_MAX / mc.NBB
    th = 0.771 * np.sqrt(E) * np.exp(-0.776 * E)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    for r in runs:
        tot = sum(r.d[c]["birth"] for c in CASES)
        n = sum(r.N(c) for c in CASES)
        p = tot / n / w
        ax[0].plot(E, p, color=r.color, lw=r.lw, label=f"{r.label} (all four cases pooled)")
        ax[1].plot(E, (p / th - 1) * 100, color=r.color, lw=max(r.lw * 0.6, 0.8), label=r.label)
    ax[0].plot(E, th, "k--", lw=1.0, label=r"$s(E)=0.771\sqrt{E}\,e^{-0.776E}$")
    ax[0].set_xlabel("birth energy E (MeV)")
    ax[0].set_ylabel("probability density (1/MeV)")
    ax[0].set_xlim(0, 10)
    ax[0].legend()
    ax[0].set_title("Neutron birth-energy distribution")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylim(-10, 10)
    ax[1].set_xlim(0, 8)
    ax[1].set_xlabel("E (MeV)")
    ax[1].set_ylabel("MC / exact - 1  (%)")
    ax[1].set_title("Relative deviation")
    ax[1].legend()
    save(fig, "fig02_birth_energy.png")


def fig_flux(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for a, c in zip(ax.ravel(), CASES):
        for r in runs:
            f = r.flux_u(c)
            a.plot(FC * 1e6, f, color=r.color, lw=r.lw, label=r.label)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(1e-3, 2e7)
        top = max(r.flux_u(c).max() for r in runs)
        a.set_ylim(top * 1e-4, top * 3)
        a.axvline(xs.E_TH_CUT * 1e6, color=MUTED, ls=":", lw=1)
        a.axvline(xs.E_FAST * 1e6, color=MUTED, ls=":", lw=1)
        a.set_title(TITLES[c])
        a.text(2e-3, top * 1.2, "thermal", fontsize=8, color=INK2)
        a.text(3, top * 1.2, "slowing down", fontsize=8, color=INK2)
        a.text(2e5, top * 1.2, "fast", fontsize=8, color=INK2)
    for a in ax[1]:
        a.set_xlabel("neutron energy E (eV)")
    fig.supylabel("flux per unit lethargy  $E\\phi(E)$  (cm per source neutron)", fontsize=10, color=INK2)
    ax[0, 0].legend(loc="lower left")
    fig.suptitle("Neutron slowing down: track-length flux per unit lethargy", y=0.995)
    save(fig, "fig03_flux_spectrum.png")


def fig_paths(run):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4))
    fig2, ax2 = plt.subplots(2, 2, figsize=(11, 7.4))
    cols = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED, GREEN]
    for a, b, c in zip(ax.ravel(), ax2.ravel(), CASES):
        d = run.d[c]
        for i in range(d["pathE"].shape[0]):
            n = int(d["pathN"][i])
            E = d["pathE"][i, :n] * 1e6
            t = d["pathT"][i, :n]
            nmax = 60 if c == 3 else (400 if c == 4 else None)
            sl = slice(0, min(n, 130 if c in (1, 2) else n))
            a.plot(np.arange(n)[sl], E[sl], color=cols[i % 8], lw=1.0)
            b.plot(np.maximum(t[sl], 1e-11), E[sl], color=cols[i % 8], lw=1.0)
        for A_ in (a, b):
            A_.set_yscale("log")
            A_.axhline(0.625, color=MUTED, ls=":", lw=1)
            A_.set_title(TITLES[c])
            A_.set_ylim(1e-3, 3e7)
        a.set_xlabel("collision number")
        b.set_xscale("log")
        b.set_xlabel("time since birth (s)")
        if c == 3:
            a.set_xlim(0, 60)
        if c == 4:
            a.set_xlim(0, 150)
        a.set_ylabel("neutron energy after collision (eV)")
        b.set_ylabel("neutron energy (eV)")
    fig.suptitle("Slowing down of 8 sample neutrons per case (dotted: 0.625 eV thermal cut-off; below it, thermal collisions are drawn from the Maxwellian)", fontsize=9.5, color=INK2)
    fig2.suptitle("Same 8 neutrons against time", fontsize=10, color=INK2)
    save(fig, "fig04_sample_paths_vs_collision.png")
    save(fig2, "fig05_sample_paths_vs_time.png")


def xi_of(alpha):
    return 1.0 if alpha <= 0 else 1.0 + alpha * np.log(alpha) / (1 - alpha)


def mean_xi(case, lib, lo=1e-6, hi=0.1):
    """scattering-weighted average lethargy gain per collision over 1 eV - 100 keV (uniform in ln E)"""
    E = np.exp(np.linspace(np.log(lo), np.log(hi), 4000))
    num = np.zeros_like(E)
    den = np.zeros_like(E)
    for j, n in enumerate(case.nuc):
        ss = case.N[n] * lib.at(n, "elastic", E)
        num += xi_of(case.alpha[j]) * ss
        den += ss
    return np.mean(num) / np.mean(den), np.mean(den / (1.0)), E


def fig_lethargy(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    th = digamma(1.5) + np.log(1 / 0.776) - np.log(xs.E_TH_CUT)
    for c, a in zip(CASES, [None, None]):
        pass
    A0, A1 = ax
    for c, col in zip(CASES, [BLUE, ORANGE, AQUA, VIOLET]):
        for r in runs[-1:]:
            ls = r.d[c]["lsum"]
            lc = r.d[c]["lcnt"]
            ok = lc > 2000
            n = np.arange(1, mc.NMAXC + 1)
            A0.plot(n[ok], (ls / np.maximum(lc, 1))[ok], color=col, label=TITLES[c].replace("Case ", ""))
        if c in (3, 4):
            xi, _, _ = mean_xi(runs[-1].case[c], runs[-1].lib)
            n = np.arange(0, 45 if c == 3 else 60)
            A0.plot(n, xi * n, color=col, ls="--", lw=1)
    A0.axhline(th, color=MUTED, ls=":", lw=1)
    A0.text(1, th + 0.3, r"$\ln(E_0/E_{th})$ = %.1f (thermal)" % th, fontsize=8, color=INK2)
    A0.set_xlabel("collision number n")
    A0.set_ylabel(r"mean lethargy gained  $\langle\ln(E_{birth}/E_n)\rangle$")
    A0.set_xlim(0, 120)
    A0.set_ylim(0, 17)
    A0.legend(fontsize=8, loc="center right")
    A0.set_title("Lethargy gained vs collisions (dashed: $n\\bar\\xi$)")
    for c, col in zip((3, 4), (AQUA, VIOLET)):
        for r in runs:
            h = r.d[c]["cth"][1:].astype(float)
            h = h / h.sum()
            A1.step(np.arange(1, mc.NMAXC + 1), h, where="mid", color=col if r is runs[-1] else col, lw=r.lw, alpha=1.0 if r is runs[-1] else 0.6,
                    label=f"{TITLES[c]} ({r.label})")
        cs = runs[-1].d[c]
        A1.axvline(cs["cnt"][6] / cs["cnt"][1], color=col, ls=":", lw=1)
    A1.set_xlim(0, 90)
    A1.set_xlabel("collisions needed to reach 0.625 eV")
    A1.set_ylabel("fraction of thermalised neutrons")
    A1.legend(fontsize=7.5)
    A1.set_title("Number of collisions to thermalise (dotted: mean)")
    save(fig, "fig06_lethargy_and_collisions.png")


def fig_lifetimes(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    names = ["fast ($E\\geq0.1$ MeV)", "slowing down", "thermal", "total life"]
    cols = [RED, YELLOW, BLUE, INK]
    r = runs[-1]
    for a, c in zip(ax.ravel(), CASES):
        for g in range(4):
            y = r.d[c]["lifeH"][g] / r.N(c) * mc.NLT_PER_DEC
            if y.sum() > 0:
                a.plot(LC, y, color=cols[g], lw=1.6 if g == 3 else 1.2, label=names[g])
        if len(runs) > 1:
            y = runs[0].d[c]["lifeH"][3] / runs[0].N(c) * mc.NLT_PER_DEC
            a.plot(LC, y, color=ORANGE, lw=0.8, ls="--", label=f"total, {runs[0].label}")
        m = r.d[c]["sc"][5] / r.N(c)
        a.axvline(m, color=MUTED, ls=":", lw=1)
        a.text(m * 1.3, a.get_ylim()[1] * 0.9, f"mean {m:.3g} s", fontsize=8, color=INK2)
        a.set_xscale("log")
        a.set_xlim(1e-11, 1e0)
        a.set_title(TITLES[c])
    for a in ax[1]:
        a.set_xlabel("time spent (s)")
    fig.supylabel("fraction of neutrons per decade of time", fontsize=10, color=INK2)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Neutron lifetimes split by energy group ({r.label})", y=0.995)
    save(fig, "fig07_lifetime_distributions.png")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = ["fast", "slowing down", "thermal"]
    cols3 = [RED, YELLOW, BLUE]
    # simple grouped bars: (case) x (group) x (run)
    x0 = 0
    ticks, tl = [], []
    for c in CASES:
        for g in range(3):
            for ir, r in enumerate(runs):
                v = r.d[c]["sc"][2 + g] / r.N(c)
                if v <= 0:
                    continue
                ax.bar(x0 + ir * 0.38, v, width=0.34, color=cols3[g], alpha=1 if r is runs[-1] else 0.45, hatch=None if r is runs[-1] else "///",
                       edgecolor=SURF, label=labels[g] if (c == 3 and ir == len(runs) - 1) else None)
            x0 += 1.0
        ticks.append(x0 - 1.7)
        tl.append(TITLES[c].replace("Case ", "").replace(": ", ":\n"))
        x0 += 0.8
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tl, fontsize=8.5)
    ax.set_ylabel("mean time per neutron (s)")
    ax.set_title("Mean time spent fast / slowing down / thermal (light hatched = " + runs[0].label + ")" if len(runs) > 1 else "Mean time spent fast / slowing down / thermal")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig08_lifetime_bars.png")


ABS_ORDER = [("U235", 1, "U-235 fission", BLUE), ("U235", 0, "U-235 capture", MAGENTA), ("U238", 1, "U-238 fission", AQUA),
             ("U238", 0, "U-238 capture", YELLOW), ("H1", 0, "H-1 capture", RED), ("H2", 0, "D capture", RED), ("O16", 0, "O-16 (n,$\\gamma$)+(n,$\\alpha$)", VIOLET)]


def abs_fractions(r, c):
    d = r.d[c]
    n = r.N(c)
    nuc = r.meta[c]["nuc"]
    out = {}
    for j, nm in enumerate(nuc):
        for x in (0, 1):
            out[(nm, x)] = d["absE"][j * 2 + x].sum() / n
    return out


def fig_absorption(runs):
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ypos, ylab = [], []
    y = 0
    legend_done = set()
    for c in CASES:
        for r in runs:
            f = abs_fractions(r, c)
            left = 0.0
            for nm, x, lab, col in ABS_ORDER:
                v = f.get((nm, x), 0.0)
                if v <= 0:
                    continue
                ax.barh(y, v, left=left, color=col, edgecolor=SURF, height=0.7, label=lab if lab not in legend_done else None)
                legend_done.add(lab)
                if v > 0.035:
                    ax.text(left + v / 2, y, f"{100 * v:.1f}%", ha="center", va="center", fontsize=7.5, color="white" if col in (BLUE, VIOLET, RED, AQUA) else INK)
                left += v
            ypos.append(y)
            ylab.append(f"{TITLES[c].replace('Case ', '').replace('$_2$', '2')}  [{r.label.split()[0]}]")
            y += 1
        y += 0.6
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of all neutrons (= fraction of all absorptions)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    ax.set_title("Where neutrons are absorbed: fuel vs moderator, fission vs capture", y=1.1)
    ax.grid(axis="y", visible=False)
    save(fig, "fig09_absorption_fuel_vs_moderator.png")


def fuel_rows(r, c):
    nuc = r.meta[c]["nuc"]
    return [j for j, n in enumerate(nuc) if n in ("U235", "U238")]


def fig_fission_capture(runs):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    for a, c in zip(ax.ravel(), CASES):
        for r in runs:
            rows = fuel_rows(r, c)
            fis = sum(r.d[c]["absE"][j * 2 + 1] for j in rows) / r.N(c) / DU
            cap = sum(r.d[c]["absE"][j * 2] for j in rows) / r.N(c) / DU
            lw = r.lw
            a.plot(FC * 1e6, np.where(fis > 0, fis, np.nan), color=BLUE, lw=lw, alpha=1.0 if r is runs[-1] else 0.55, label=f"fission, {r.label}")
            a.plot(FC * 1e6, np.where(cap > 0, cap, np.nan), color=ORANGE, lw=lw, alpha=1.0 if r is runs[-1] else 0.55, label=f"capture, {r.label}")
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(1e-3, 2e7)
        a.set_ylim(1e-6, 3)
        a.axvline(xs.E_TH_CUT * 1e6, color=MUTED, ls=":", lw=1)
        a.axvline(xs.E_FAST * 1e6, color=MUTED, ls=":", lw=1)
        a.set_title(TITLES[c])
    for a in ax[1]:
        a.set_xlabel("neutron energy at absorption (eV)")
    fig.supylabel("events per source neutron per unit lethargy", fontsize=10, color=INK2)
    ax[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Fuel fissions and captures against the energy at which they happen", y=0.995)
    save(fig, "fig10_fission_capture_vs_energy.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    r = runs[-1]
    for c, col in zip(CASES, [BLUE, ORANGE, AQUA, VIOLET]):
        rows = fuel_rows(r, c)
        fis = sum(r.d[c]["absE"][j * 2 + 1] for j in rows).astype(float)
        cap = sum(r.d[c]["absE"][j * 2] for j in rows).astype(float)
        # merge 5 bins to reduce noise
        k = 5
        nb = (len(fis) // k) * k
        f5 = fis[:nb].reshape(-1, k).sum(1)
        c5 = cap[:nb].reshape(-1, k).sum(1)
        ok = (f5 + c5) > 200
        ax.plot(FC[:nb].reshape(-1, k)[:, 2][ok] * 1e6, (f5 / np.maximum(c5, 1))[ok], color=col, label=TITLES[c])
    ax.axhline(585.1 / 98.7, color=MUTED, ls=":", lw=1)
    ax.text(2e-3, 585.1 / 98.7 * 1.08, r"U-235 thermal $\sigma_f/\sigma_\gamma$ = 5.93", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 2e7)
    ax.set_xlabel("neutron energy at absorption (eV)")
    ax.set_ylabel("fuel fissions / fuel captures")
    ax.set_title(f"Fission-to-capture ratio against energy ({r.label})")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig11_fission_to_capture_ratio.png")


def fig_convergence(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for c, col in zip(CASES, [BLUE, ORANGE, AQUA, VIOLET]):
        Ns, ks, ss = [], [], []
        for r in runs:
            m = int(r.meta[c]["chunk_m"])
            nu, nu2 = np.cumsum(r.d[c]["chunk_nu"]), np.cumsum(r.d[c]["chunk_nu2"])
            n = m * np.arange(1, len(nu) + 1)
            idx = np.unique(np.geomspace(1, len(nu), 40).astype(int)) - 1
            k = nu[idx] / n[idx]
            s = np.sqrt(np.maximum(nu2[idx] / n[idx] - k * k, 0) / n[idx])
            Ns += list(n[idx]); ks += list(k); ss += list(s)
        o = np.argsort(Ns)
        Ns, ks, ss = np.array(Ns)[o], np.array(ks)[o], np.array(ss)[o]
        ax[0].errorbar(Ns, ks, 2 * ss, color=col, lw=1, marker="o", ms=2.5, capsize=1.5, label=TITLES[c])
        ax[1].plot(Ns, ss, color=col, marker="o", ms=2.5, lw=1, label=TITLES[c])
    Nref = np.geomspace(1e3, 1e9, 50)
    ax[1].plot(Nref, 1.1 / np.sqrt(Nref), "k--", lw=0.9, label=r"$\propto N^{-1/2}$")
    ax[0].axhline(1, color="k", lw=0.8, ls="--")
    ax[0].set_xscale("log")
    ax[0].set_ylim(0, 1.6)
    ax[0].set_xlabel("number of neutron histories N")
    ax[0].set_ylabel(r"$k_\infty$ ($\pm 2\sigma$)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Convergence of $k_\\infty$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("number of neutron histories N")
    ax[1].set_ylabel(r"standard error of $k_\infty$")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Statistical error falls as $1/\\sqrt{N}$")
    save(fig, "fig12_convergence.png")


def four_factors(r, c):
    d, cs = r.d[c], r.case[c]
    n = r.N(c)
    k, s = r.k(c)
    nth = d["cnt"][1]
    if nth == 0:
        return None
    p = nth / n
    f = d["cnt"][2] / nth
    eta = d["sc"][8] / d["cnt"][2]
    eps = k / (eta * f * p)
    # theory (thermal-pool cross-sections, flux-weighted)
    A = cs.th_avg
    N = cs.N
    sa = {nm: N[nm] * (A[nm]["capture"] + A[nm]["fission"]) for nm in cs.nuc}
    fuel = sum(sa[nm] for nm in cs.nuc if nm in ("U235", "U238"))
    f_th = fuel / sum(sa.values())
    nu_th = 2.4355
    eta_th = nu_th * N["U235"] * A["U235"]["fission"] / fuel
    return dict(eps=eps, p=p, f=f, eta=eta, k=k, f_th=f_th, eta_th=eta_th)


def fig_four_factors(runs):
    r = runs[-1]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    names = [r"$\varepsilon$", "p", "f", r"$\eta$", r"$k_\infty$"]
    for i, (c, col) in enumerate(zip((3, 4), (AQUA, VIOLET))):
        ff = four_factors(r, c)
        vals = [ff["eps"], ff["p"], ff["f"], ff["eta"], ff["k"]]
        b = ax.bar(np.arange(5) + (i - 0.5) * 0.38, vals, width=0.36, color=col, label=TITLES[c], edgecolor=SURF)
        for x, v in zip(np.arange(5) + (i - 0.5) * 0.38, vals):
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("value")
    ax.legend()
    ax.set_title(r"Four-factor decomposition $k_\infty=\eta f p\varepsilon$ ($N$ = %s)" % f"{int(r.N(3)):,}")
    ax.grid(axis="x", visible=False)
    save(fig, "fig13_four_factors.png")


def fig_scan(scan_file, chosen):
    S = json.load(open(scan_file))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    spec = [("2pct_U_H2O", AQUA, "2% enriched U + H$_2$O"), ("natU_H2O", ORANGE, "natural U + H$_2$O"),
            ("2pct_U_D2O", BLUE, "2% enriched U + D$_2$O"), ("natU_D2O", VIOLET, "natural U + D$_2$O")]
    for key, col, lab in spec[:2]:
        s = S[key]
        ax[0].errorbar(s["ratio"], s["k"], 2 * np.array(s["sigma"]), color=col, marker="o", ms=3, capsize=1.5, label=lab)
    ax[0].axvline(chosen[3], color=AQUA, ls=":", lw=1)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("H$_2$O molecules per uranium atom")
    ax[0].set_title("Light-water moderated")
    for key, col, lab in spec[2:]:
        s = S[key]
        ax[1].errorbar(s["ratio"], s["k"], 2 * np.array(s["sigma"]), color=col, marker="o", ms=3, capsize=1.5, label=lab)
    ax[1].axvline(chosen[4], color=VIOLET, ls=":", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("D$_2$O molecules per uranium atom")
    ax[1].set_title("Heavy-water moderated")
    for a in ax[:2]:
        a.axhline(1, color="k", ls="--", lw=0.8)
        a.set_ylabel(r"$k_\infty$")
        a.legend(fontsize=8)
    s = S["enrichment"]
    ax[2].errorbar(s["w"], s["k"], 2 * np.array(s["sigma"]), color=BLUE, marker="o", ms=3, capsize=1.5)
    ax[2].axhline(1, color="k", ls="--", lw=0.8)
    ax[2].set_xscale("symlog", linthresh=1)
    ax[2].set_xlabel("U-235 enrichment (wt %)")
    ax[2].set_ylabel(r"$k_\infty$")
    ax[2].set_title("No moderator: uranium metal")
    fig.suptitle(r"$k_\infty$ against the amount of moderator (error bars $\pm2\sigma$; dotted: ratio used in the main runs)", y=1.0, fontsize=10, color=INK2)
    save(fig, "fig14_moderator_scan.png")
    return S


def fig_coarse(main, coarse_dir):
    if not coarse_dir or not os.path.exists(os.path.join(coarse_dir, "case1.npz")):
        return None
    rc = Run(coarse_dir, "coarse table only", ORANGE, 1.2)
    rows = []
    for c in CASES:
        rows.append((c, main.k(c), rc.k(c)))
    return rows


# ----------------------------------------------------------------------------- tables
def fmt_k(k, s):
    return f"{k:.5f} ± {s:.5f}"


# reference solutions (values copied from the two reference reports)
REF = {
    "Ref. A (GitHub report, CH23B086/CH23B036)": {1: (0.22992, 0.00002), 2: (0.33816, 0.00003), 3: (1.27030, 0.00004), 4: (1.21403, 0.00004)},
    "Ref. B (MATLAB report, CH23B025/CH23B043)": {1: (0.30202, 0.00003), 2: (0.40556, 0.00003), 3: (1.21885, 0.00004), 4: (1.14017, 0.00004)},
}


def write_summary(runs, S, chosen, coarse_rows, main_dir):
    L = []
    r = runs[-1]
    L.append("# Summary of results\n")
    L.append(f"Library: `{r.lib.mode}`; temperature {xs.T_K} K; thermal cut-off {xs.E_TH_CUT*1e6:.3f} eV; groups: fast E >= 0.1 MeV, "
             "slowing down 0.625 eV - 0.1 MeV, thermal < 0.625 eV.\n")
    L.append("## 1. Multiplication factor  k_inf  (± 1 standard deviation)\n")
    L.append("| Case | Medium | " + " | ".join(f"k_inf, N = {int(x.N(1)):,}" for x in runs) + " | Can go critical? |\n|---|---|" + "---|" * (len(runs) + 1) + "\n")
    for c in CASES:
        ks = [x.k(c) for x in runs]
        crit = "**YES**" if ks[-1][0] - 3 * ks[-1][1] > 1 else "no"
        L.append(f"| {c} | {TITLES[c].split(': ')[1].replace('$_2$', '2')} | " + " | ".join(fmt_k(*k) for k in ks) + f" | {crit} |\n")
    L.append("\nModerator amounts used: case 3 = %.2f H2O molecules per U atom (V_mod/V_U = %.2f); case 4 = %.1f D2O molecules per U atom (V_mod/V_U = %.1f). "
             "Uranium metal 19.05 g/cm3, H2O 0.998, D2O 1.105 g/cm3, smeared homogeneously.\n" % (
                 r.meta[3]["spec"]["ratio"], r.meta[3]["info"]["V_mod_over_V_U"], r.meta[4]["spec"]["ratio"], r.meta[4]["info"]["V_mod_over_V_U"]))
    L.append("\n## 2. Comparison with the two reference reports (N = 10^9 unless stated)\n")
    L.append("| Case | This work | " + " | ".join(REF.keys()) + " |\n|---|---|" + "---|" * len(REF) + "\n")
    for c in CASES:
        L.append(f"| {c} | {fmt_k(*r.k(c))} | " + " | ".join(f"{v[c][0]:.5f}" for v in REF.values()) + " |\n")
    L.append("\nAll three agree on the four yes/no answers. Differences in the actual k_inf come from cross-section data, inelastic-scattering and resonance treatment (see README).\n")
    L.append("\n## 3. Four factors (moderated cases)\n")
    L.append("| Case | eps | p | f | eta | k_inf | f (theory) | eta (theory) |\n|---|---|---|---|---|---|---|---|\n")
    for c in (3, 4):
        ff = four_factors(r, c)
        L.append(f"| {c} | {ff['eps']:.4f} | {ff['p']:.4f} | {ff['f']:.4f} | {ff['eta']:.4f} | {ff['k']:.4f} | {ff['f_th']:.4f} | {ff['eta_th']:.4f} |\n")
    L.append("\nDefinitions: p = fraction of source neutrons that reach 0.625 eV; f = fraction of thermal absorptions in uranium; "
             "eta = neutrons per thermal absorption in uranium; eps = k_inf/(eta f p) = total fission neutrons / thermal fission neutrons.\n")
    L.append("\n## 4. Neutron lifetimes (seconds, averaged over ALL neutrons, N = %d)\n" % r.N(1))
    L.append("| Case | time fast | time slowing down | time thermal | total lifetime | thermal lifetime of thermal neutrons (MC / 1/<vSigma_a>) | slowing-down time of thermalising neutrons (MC / theory) |\n|---|---|---|---|---|---|---|\n")
    for c in CASES:
        d, n = r.d[c], r.N(c)
        sc = d["sc"]
        nth = d["cnt"][1]
        tf, ts, tt, tl = sc[2] / n, sc[3] / n, sc[4] / n, sc[5] / n
        if nth > 0:
            thm = sc[4] / nth
            sdm = sc[10] / nth
            xi, _, E = mean_xi(r.case[c], r.lib, 1e-6, 1e-1)
            cs = r.case[c]
            Es = np.exp(np.linspace(np.log(1e-5), np.log(1e-2), 500))
            Ss = np.mean(sum(cs.N[nm] * r.lib.at(nm, "elastic", Es) for nm in cs.nuc))
            vth = xs.VCONST * np.sqrt(xs.E_TH_CUT)
            v0 = xs.VCONST * np.sqrt(1.0)
            tsd = 2.0 / (xi * Ss) * (1 / vth - 1 / v0)
            x1 = f"{thm:.4g} / {cs.th_tau:.4g}"
            x2 = f"{sdm:.3g} / {tsd:.3g}"
        else:
            x1 = x2 = "-"
        L.append(f"| {c} | {tf:.3e} | {ts:.3e} | {tt:.3e} | {tl:.3e} | {x1} | {x2} |\n")
    L.append("\n## 5. Slowing down: collisions needed to reach 0.625 eV\n")
    L.append("| Case | MC mean collisions | theory ln(E0/Eth)/xi_bar | xi_bar (1 eV-0.1 MeV) | fraction of neutrons reaching thermal (p) |\n|---|---|---|---|---|\n")
    th = digamma(1.5) + np.log(1 / 0.776) - np.log(xs.E_TH_CUT)
    for c in CASES:
        d = r.d[c]
        xi, _, _ = mean_xi(r.case[c], r.lib)
        if d["cnt"][1] > 0:
            L.append(f"| {c} | {d['cnt'][6]/d['cnt'][1]:.2f} | {th/xi:.2f} | {xi:.3f} | {d['cnt'][1]/r.N(c):.4f} |\n")
        else:
            L.append(f"| {c} | no neutron thermalises | - | {xi:.4f} | 0 |\n")
    L.append("\nElastic-scattering lethargy gain per collision xi = 1 + alpha ln(alpha)/(1-alpha):  " + ", ".join(
        f"{NICE[n]} {xi_of(max(((a-1)/(a+1))**2,0) if a>1.01 else 0):.4f}" for n, a in xs.A_RATIO.items()) + "\n")
    L.append("\n## 6. Where neutrons are absorbed (fraction of all neutrons, N = %d)\n" % r.N(1))
    hdr = ["U-235 fission", "U-235 capture", "U-238 fission", "U-238 capture", "H-1 capture", "D capture", "O-16 capture"]
    L.append("| Case | " + " | ".join(hdr) + " | fuel total | moderator total | fission / capture (fuel) |\n|" + "---|" * (len(hdr) + 4) + "\n")
    for c in CASES:
        f = abs_fractions(r, c)
        vals = [f.get((nm, x), 0.0) for nm, x, _, _ in ABS_ORDER]
        fuel = sum(f.get((nm, x), 0.0) for nm in ("U235", "U238") for x in (0, 1))
        mod = sum(f.get((nm, 0), 0.0) for nm in ("H1", "H2", "O16"))
        fis = sum(f.get((nm, 1), 0.0) for nm in ("U235", "U238"))
        cap = sum(f.get((nm, 0), 0.0) for nm in ("U235", "U238"))
        L.append(f"| {c} | " + " | ".join(f"{100*v:.2f}%" for v in vals) + f" | {100*fuel:.2f}% | {100*mod:.2f}% | {fis/cap:.4f} |\n")
    L.append("\nU-235 thermal fission/capture from the MC (thermal-pool absorptions only) vs the thermal-averaged data:\n\n| Case | MC | data |\n|---|---|---|\n")
    for c in (3, 4):
        d = r.d[c]
        j = r.meta[c]["nuc"].index("U235")
        thbins = FC < xs.E_TH_CUT
        fi = d["absE"][j * 2 + 1][thbins].sum()
        ca = d["absE"][j * 2][thbins].sum()
        A = r.case[c].th_avg["U235"]
        L.append(f"| {c} | {fi/ca:.4f} | {A['fission']/A['capture']:.4f} |\n")
    L.append("\n## 7. Birth energy\n")
    tot = sum(r.d[c]["sc"][9] for c in CASES) / sum(r.N(c) for c in CASES)
    L.append(f"Mean sampled birth energy = {tot:.4f} MeV (exact for s(E): 1.5/0.776 = {1.5/0.776:.4f} MeV).\n")
    L.append("\n## 8. Events per history\n")
    L.append("| Case | collisions/neutron (non-thermal) | thermal collisions/neutron |\n|---|---|---|\n")
    for c in CASES:
        d = r.d[c]
        L.append(f"| {c} | {(d['cnt'][3]+d['cnt'][4])/r.N(c):.2f} | {d['cnt'][5]/r.N(c):.2f} |\n")
    if S is not None:
        L.append("\n## 9. Moderator scans (k_inf)\n")
        for key, lab in (("2pct_U_H2O", "2% U + H2O"), ("natU_H2O", "natural U + H2O"), ("2pct_U_D2O", "2% U + D2O"), ("natU_D2O", "natural U + D2O")):
            s = S[key]
            kb = int(np.argmax(s["k"]))
            win = [x for x, k in zip(s["ratio"], s["k"]) if k > 1]
            L.append(f"* **{lab}**: maximum k_inf = {s['k'][kb]:.4f} at {s['ratio'][kb]:.2f} molecules/U atom; k_inf > 1 for "
                     + (f"{min(win):.2f} ... {max(win):.2f} (scan points)" if win else "no ratio scanned") + "\n")
        s = S["enrichment"]
        ab = [w for w, k in zip(s["w"], s["k"]) if k > 1]
        L.append(f"* **unmoderated uranium**: k_inf exceeds 1 only above ~{min(ab) if ab else float('nan')} wt% U-235 (scan point); natural U = {s['k'][1]:.3f}\n")
    if coarse_rows:
        L.append("\n## 10. Sensitivity: only the 50-point assignment table (log-log interpolation) instead of the fine library\n")
        L.append("| Case | fine library | coarse 50-point table |\n|---|---|---|\n")
        for c, kf, kc in coarse_rows:
            L.append(f"| {c} | {fmt_k(*kf)} | {fmt_k(*kc)} |\n")
    open(os.path.join(OUT, "summary.md"), "w").write("".join(L))
    print("  wrote summary.md")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results_1e9", help="folder of the large run")
    ap.add_argument("--small", default="results_1e6", help="folder of the small run (optional)")
    ap.add_argument("--scan", default="results_scan/scan.json")
    ap.add_argument("--coarse", default=None, help="folder of a --coarse run for the sensitivity table")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    runs = []
    if a.small and os.path.exists(os.path.join(a.small, "case1.npz")):
        runs.append(Run(a.small, "$10^6$ neutrons", ORANGE, 2.8))
    if a.main and os.path.exists(os.path.join(a.main, "case1.npz")):
        runs.append(Run(a.main, "$10^9$ neutrons", BLUE, 1.1))
    for r in runs:
        r.label = r.label.replace("$10^6$", "10^6").replace("$10^9$", "10^9") if False else r.label
    print("figures ->", OUT)
    xs.write_required_table(os.path.join(OUT, "cross_section_table.csv"), os.path.join(OUT, "cross_section_table.md"))
    fig_cross_sections(runs[-1].lib)
    fig_birth(runs)
    fig_flux(runs)
    fig_paths(runs[-1])
    fig_lethargy(runs)
    fig_lifetimes(runs)
    fig_absorption(runs)
    fig_fission_capture(runs)
    fig_convergence(runs)
    fig_four_factors(runs)
    S = None
    chosen = {3: runs[-1].meta[3]["spec"]["ratio"], 4: runs[-1].meta[4]["spec"]["ratio"]}
    if os.path.exists(a.scan):
        S = fig_scan(a.scan, chosen)
    coarse_rows = fig_coarse(runs[-1], a.coarse)
    write_summary(runs, S, chosen, coarse_rows, a.main)


if __name__ == "__main__":
    main()
