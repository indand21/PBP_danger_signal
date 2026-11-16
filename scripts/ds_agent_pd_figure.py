"""Supplementary Figure S1: where the endotoxin ranking survives PD differences.

Reads results/danger_signal/agent_pd_map.json and draws, through the phase 4
figure helpers (the same column width, fonts and saved-width check),
docs/danger_signal_study/figures/figS1_agent_pd_map.{png,pdf}. Both are
copied into the submission package's figures folder, as for the main
figures; the supplement embeds the PNG.

Panels A and B are the two 9 x 9 maps at the 40% fT>MIC target, the target
at which failing therapy appears: A varies ceftazidime's kill rate and C50
against the adopted meropenem, B varies meropenem's against the adopted
ceftazidime. Colour is rho_critical (S2.3), the ceftazidime:meropenem lysis-yield ratio at
which 48-h release is equal, on a diverging scale centred on equal yields.
Panel C places every scenario of the analysis, at every target, on the rho_critical
axis, grouped by which agent controls the infection, against the assumed
3.5-7 range.
"""
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.ds_phase4_figures as f4  # noqa: E402  (sets the house rcParams)

IN = os.path.join("results", "danger_signal", "agent_pd_map.json")
NAME = "figS1_agent_pd_map"
PACKAGE_FIGURES = os.path.join("docs", "danger_signal_submission_package", "figures")
TARGET = "40"
INK = "#333333"
RHO = r"$\rho_{\mathrm{critical}}$"   # the tipping ratio of S2.3


def grid(cells, target):
    """rho_critical and the controlled flags on the 9 x 9 grid, rows = kill factor."""
    n = int(round(np.sqrt(len(cells))))
    r = np.array([c["targets"][target]["r_star"] for c in cells]).reshape(n, n)
    caz_ok = np.array([c["targets"][target]["ceftazidime"]["controlled"] for c in cells]).reshape(n, n)
    mero_ok = np.array([c["targets"][target]["meropenem"]["controlled"] for c in cells]).reshape(n, n)
    return r, caz_ok, mero_ok


def draw_map(ax, cells, factors, target, varied, norm, tsuji_closer=None):
    r, caz_ok, mero_ok = grid(cells, target)
    n = len(factors)
    im = ax.imshow(np.log10(r), origin="lower", cmap="RdBu", norm=norm,
                   extent=(-0.5, n - 0.5, -0.5, n - 0.5), aspect="equal")
    failing = ~(caz_ok if varied == "ceftazidime" else mero_ok)
    ys, xs = np.nonzero(failing)
    ax.scatter(xs, ys, marker="x", s=14, linewidths=0.8, color=INK, zorder=3)
    # Where the other agent fails too, say so with a second mark.
    other = ~(mero_ok if varied == "ceftazidime" else caz_ok)
    ys, xs = np.nonzero(other)
    ax.scatter(xs, ys, marker="+", s=22, linewidths=0.8, color=INK, zorder=3)
    contours = []
    lr = np.log10(r)
    for level, style in ((np.log10(3.5), "--"), (np.log10(7.0), "-")):
        if lr.min() < level < lr.max():
            ax.contour(np.arange(n), np.arange(n), lr, levels=[level], colors=INK,
                       linewidths=1.0, linestyles=style)
            contours.append(level)
    if tsuji_closer is not None:
        ys, xs = np.nonzero(tsuji_closer.reshape(n, n))
        ax.scatter(xs, ys, marker="o", s=34, facecolors="none", edgecolors=INK,
                   linewidths=0.8, zorder=4)
    mid = n // 2
    ax.add_patch(plt.Rectangle((mid - 0.5, mid - 0.5), 1, 1, fill=False, lw=1.2,
                               edgecolor="black", zorder=5))
    ticks = [0, n // 4, mid, 3 * n // 4, n - 1]
    labels = [f"{factors[i]:g}" if factors[i] >= 1 else f"1/{1 / factors[i]:g}" for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    short = "ceftazidime" if varied == "ceftazidime" else "meropenem"
    ax.set_xlabel(f"{short} C50, x adopted")
    ax.set_ylabel(f"{short} kill rate, x adopted")
    return im, bool(other.any()), bool(contours)


def outcome_class(t):
    c, m = t["ceftazidime"]["controlled"], t["meropenem"]["controlled"]
    return {(True, True): 0, (False, True): 1, (True, False): 2, (False, False): 3}[(c, m)]


def main():
    with open(IN, encoding="utf-8") as fh:
        d = json.load(fh)
    factors = d["design"]["ceftazidime_factor_grid"]
    main_cells, mirror_cells = d["main_map"]["cells"], d["mirror_map"]["cells"]
    shared_misfit = d["tsuji_check"]["shared_pd_misfit_log10"]
    closer = np.array([c["tsuji_difference_misfit_log10"] < shared_misfit for c in main_cells])
    closer_b = np.array([c["tsuji_difference_misfit_log10"] < shared_misfit for c in mirror_cells])

    every = (main_cells + mirror_cells + d["hill_sweep"] + d["k_pers_sweep"]
             + [c for cells in d["meropenem_envelope"]["cells"].values() for c in cells])
    points = [(t["r_star"], outcome_class(t)) for c in every for t in c["targets"].values()]
    lo = np.floor(np.log10(min(p[0] for p in points)))
    hi = np.ceil(np.log10(max(p[0] for p in points)))
    span = max(abs(lo), abs(hi), 1.0)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)

    fig = plt.figure(figsize=(f4.COL_W, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.07, 1.0], width_ratios=[1, 1])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cax, ax_c = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])

    im, plus_a, lines_a = draw_map(ax_a, main_cells, factors, TARGET, "ceftazidime", norm, closer)
    _, plus_b, lines_b = draw_map(ax_b, mirror_cells, factors, TARGET, "meropenem", norm, closer_b)
    ax_a.set_title("A  Ceftazidime's PD varied", fontsize=9, loc="left")
    ax_b.set_title("B  Meropenem's PD varied", fontsize=9, loc="left")

    ticks = [v for v in np.arange(-span, span + 1) if v == int(v)]
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.set_xticklabels([f"{10 ** v:g}" for v in ticks])
    cb.set_label(RHO + ", CAZ:MERO yield ratio at equal 48-h release (40% fT>MIC)", fontsize=8)

    names = ["both control", "ceftazidime fails", "meropenem fails", "both fail"]
    rng = np.random.default_rng(0)        # vertical jitter only; fixed for reproducibility
    for k in range(4):
        xs = np.array([p[0] for p in points if p[1] == k])
        if len(xs):
            ax_c.scatter(xs, k + rng.uniform(-0.18, 0.18, len(xs)), s=6, color=INK,
                         alpha=0.45, linewidths=0)
        ax_c.text(1.02, k, f"n={len(xs)}", transform=ax_c.get_yaxis_transform(),
                  va="center", fontsize=7, color="#555555")
    ax_c.axvspan(3.5, 7.0, color="#7f8c8d", alpha=0.18, zorder=0)
    ax_c.axvline(1.0, color="#999999", lw=0.8, zorder=0)
    ax_c.set_xscale("log")
    ax_c.set_xlim(10 ** lo, 10 ** hi)
    decades = [10.0 ** v for v in range(int(lo), int(hi) + 1)]
    ax_c.set_xticks(decades)
    ax_c.set_xticklabels([f"{v:g}" for v in decades])
    ax_c.set_yticks(range(4))
    ax_c.set_yticklabels(names)
    ax_c.set_ylim(3.6, -0.6)
    ax_c.set_xlabel(RHO + ", every scenario and target")
    # The title states a conclusion only if the plotted points support it.
    beyond = {p[1] for p in points if p[0] > 3.5}
    title = ("C  Ranking reverses only where meropenem fails" if beyond == {2}
             else "C  Ranking holds across every scenario" if not beyond
             else "C  " + RHO + " across every scenario and target")
    ax_c.set_title(title, fontsize=9, loc="left")
    ax_c.text(np.sqrt(3.5 * 7.0), -0.45, "assumed 3.5-7x", ha="center", va="top",
              fontsize=7, color="#5d6d7e")

    note = ("Shaded band: the assumed 3.5-7x yield range. Left of the band the ranking holds "
            "anywhere in that range, right of it the ranking reverses, and inside it the "
            "outcome depends on the ratio. " + ("Dashed and solid lines: " + RHO + " = 3.5 and 7. " if lines_a or lines_b else "")
            + "x: the varied agent fails (48-h net growth)" + ("; +: the other agent fails. "
                                                               if plus_a or plus_b else ". ")
            + "Circles: cells closer than the shared kill function to Tsuji's paired 6-h "
            "counts. Box: equal PD. MIC reference fixed at 1 mg/L for both agents.")
    f4._footnote(fig, note, 7)
    f4._save(fig, NAME)
    os.makedirs(PACKAGE_FIGURES, exist_ok=True)
    # PNG for the supplement to embed, PDF for submission, as for the main figures.
    for ext in (".png", ".pdf"):
        shutil.copy2(os.path.join(f4.OUT_DIR, NAME + ext),
                     os.path.join(PACKAGE_FIGURES, NAME + ext))
    print(f"wrote {NAME}")


if __name__ == "__main__":
    main()
