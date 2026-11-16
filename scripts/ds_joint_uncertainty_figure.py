"""Supplementary Figure S3: the joint uncertainty analysis.

Reads results/danger_signal/joint_uncertainty.json and draws, through the
phase 4 figure helpers (column width, fonts, saved-width check),
docs/danger_signal_study/figures/figS3_joint_uncertainty.{png,pdf}, copied
into the submission package's figures folder as for Figures S1 and S2.

Panel A is the Morris screen at the 40% fT>MIC target: mu*, the mean absolute
elementary effect over the factor's full range, for the ranking (log10
rho_critical) and for each agent's 48-h net change, with sigma marked. Panel B
places every screening comparison by how much worse meropenem did than
ceftazidime (x) and by rho_critical (y), against the assumed 3.5-7 yield
range. Every number in the titles and note is read from the results file,
and each statement is made only if the data support it.
"""
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.ds_phase4_figures as f4  # noqa: E402  (sets the house rcParams)

IN = os.path.join("results", "danger_signal", "joint_uncertainty.json")
NAME = "figS3_joint_uncertainty"
PACKAGE_FIGURES = os.path.join("docs", "danger_signal_submission_package", "figures")
INK = "#333333"
NEUTRAL = "#5d6d7e"          # both agents control: the reference class, not a series hue
RHO = r"$\rho_{\mathrm{critical}}$"
SHORT = {"log10_burden": "starting burden", "immune_pool_fraction": "effector pool",
         "immune_recruit_fold": "immune recruitment", "interval_h": "dosing interval",
         "infusion_h": "infusion duration", "clearance_factor": "clearance",
         "mero_kill_factor": "MERO kill rate", "mero_c50_factor": "MERO C50",
         "release_mtt_h": "release delay", "mero_filament_fraction": "MERO sub-MIC term"}


def panel_a(fig, spec, d):
    eff = d["morris"]["effects"]
    order = [n for n, _ in d["headline"]["mu_star_rank_log10_r_star_at_40"]]
    outcomes = (("log10_r_star", "ranking\n(log10 " + RHO + ")", NEUTRAL),
                ("net_ceftazidime", "ceftazidime\nnet change", f4.CAZ_C),
                ("net_meropenem", "meropenem\nnet change", f4.MERO_C))
    # The factor names get a column of their own. As tick labels on the first
    # panel they would make the layout give every panel that left margin.
    sub = spec.subgridspec(1, 4, width_ratios=[1.0, 1, 1, 1], wspace=0.06)
    y = np.arange(len(order))[::-1]
    names = fig.add_subplot(sub[0, 0])
    names.set_ylim(-0.6, len(order) - 0.4)
    names.axis("off")
    for yi, n in zip(y, order):
        names.text(1.0, yi, SHORT[n], ha="right", va="center", fontsize=7)
    names.text(0.0, 1.2, "A  Morris screen, 40% fT>MIC", transform=names.transAxes,
               fontsize=9, ha="left", va="bottom")
    for i, (key, title, color) in enumerate(outcomes):
        ax = fig.add_subplot(sub[0, i + 1], sharey=names)
        mu = [eff[key]["40"][n]["mu_star"] for n in order]
        sd = [eff[key]["40"][n]["sigma"] for n in order]
        ax.barh(y, mu, height=0.62, color=color, edgecolor="white", linewidth=0.6)
        ax.plot(sd, y, "o", ms=3.2, mfc="white", mec=INK, mew=0.7, zorder=3)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(0, 1.12 * max(max(mu), max(sd)))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlabel("μ*", fontsize=8)


def panel_b(ax, d):
    h = d["headline"]
    cap = 10 ** d["design"]["log10_r_star_cap"]
    pts = [(t, p["targets"][t], p["factors"]) for tr in d["morris"]["points"] for p in tr
           for t in p["targets"] if p["targets"][t]["attainable"]]
    groups = {"both control": (NEUTRAL, lambda r: r["ceftazidime"]["controlled"] and r["meropenem"]["controlled"]),
              "meropenem fails": (f4.MERO_C, lambda r: r["ceftazidime"]["controlled"] and not r["meropenem"]["controlled"]),
              "ceftazidime fails": (f4.CAZ_C, lambda r: not r["ceftazidime"]["controlled"])}
    for label, (color, test) in groups.items():
        for sub_mic, filled in ((False, True), (True, False)):
            sel = [(r, f) for _, r, f in pts if test(r) and (f["mero_filament_fraction"] > 0) == sub_mic]
            if not sel:
                continue
            x = [r["net_difference_mero_minus_caz"] for r, _ in sel]
            yv = [min(r["r_star"], cap) if r["r_star"] is not None else cap for r, _ in sel]
            ax.scatter(x, yv, s=9, marker="o", linewidths=0.7, alpha=0.75,
                       facecolors=color if filled else "none", edgecolors=color, zorder=3)
    ax.axhspan(3.5, 7.0, color="#7f8c8d", alpha=0.18, zorder=0)
    ax.axhline(1.0, color="#999999", lw=0.8, zorder=0)
    thr = h["comparable_outcomes"]["threshold_log10"]
    ax.axvline(thr, color=INK, lw=0.9, ls="--", zorder=1)
    ax.set_yscale("log")
    xs = [r["net_difference_mero_minus_caz"] for _, r, _ in pts]
    finite = [r["r_star"] for _, r, _ in pts if r["r_star"] is not None]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(0.8 * min(finite), cap * 1.6)
    low = [10.0 ** e for e in range(int(np.floor(np.log10(0.8 * min(finite)))) + 1, 0)]
    ticks = low + [1, 3.5, 7, 100, cap]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks[:-1]] + [f"{cap:g}+"])
    ax.set_xlabel("meropenem minus ceftazidime 48-h net change, log10 CFU/mL")
    ax.set_ylabel(RHO)
    c = h["comparable_outcomes"]
    title = ("B  Ranking lost only where meropenem does worse" if c["all_hold"]
             else "B  " + RHO + " against the outcome gap")
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(thr, 0.85 * min(finite), f" {thr:.2f}", fontsize=6.5, color=INK, ha="left", va="bottom")
    handles = [Line2D([], [], marker="o", ls="", mfc=col, mec=col, ms=4, label=lab)
               for lab, (col, _) in groups.items()]
    handles += [Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=4, label="no MERO sub-MIC term"),
                Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=4, label="MERO sub-MIC term > 0")]
    ax.legend(handles=handles, fontsize=6.3, loc="upper left", frameon=False, ncol=1,
              handletextpad=0.3, labelspacing=0.25)


def note_text(d):
    h = d["headline"]
    c, ns, bo = h["comparable_outcomes"], h["not_holding_without_mero_subMIC"], h["not_holding_both_controlled"]
    cap = 10 ** d["design"]["log10_r_star_cap"]
    parts = [f"A: bars, mu* (mean absolute elementary effect over each factor's range); open "
             f"circles, sigma (their spread: interaction or curvature). B: {h['n_points']} screening "
             f"points x 3 targets, {h['n_attainable']} comparisons attainable within 60 g per dose; "
             f"shaded band, the assumed 3.5-7 yield range."]
    if c["all_hold"]:
        parts.append(f"Left of the dashed line (meropenem less than {c['threshold_log10']:.2f} log10 "
                     f"worse, {c['n']} comparisons) the ranking held everywhere, the critical yield "
                     f"ratio at most {c['max_r_star']:.2f}.")
    if ns["n"]:
        parts.append(f"Without a meropenem sub-MIC term, every loss of the ranking had meropenem at "
                     f"least {ns['min_mero_net_disadvantage']:.2f} log10 worse.")
    parts.append(f"Critical ratios above {cap:g}, and cases where meropenem's own sub-MIC release "
                 "exceeds ceftazidime's at every yield ratio, are drawn at the top.")
    return " ".join(parts)


def main():
    with open(IN, encoding="utf-8") as fh:
        d = json.load(fh)
    fig = plt.figure(figsize=(f4.COL_W, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.1])
    panel_a(fig, gs[0], d)
    panel_b(fig.add_subplot(gs[1]), d)
    f4._footnote(fig, note_text(d), 7)
    f4._save(fig, NAME)
    os.makedirs(PACKAGE_FIGURES, exist_ok=True)
    for ext in (".png", ".pdf"):
        shutil.copy2(os.path.join(f4.OUT_DIR, NAME + ext),
                     os.path.join(PACKAGE_FIGURES, NAME + ext))
    print(f"wrote {NAME}")


if __name__ == "__main__":
    main()
