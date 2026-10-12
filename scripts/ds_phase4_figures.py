"""Phase 4 figures for the danger-signal study.

Reads results/danger_signal/phase4_analyses.json (so figures and numbers cannot
drift) and writes docs/danger_signal_study/figures/.

Every agent comparison is drawn with the assumed 3.5-7x yield range as a band
or a range, per the Phase 3 reporting rule: the claim is a ranking plus a range,
never a single fold-change.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({"font.size": 9, "font.family": "sans-serif",
                     "axes.linewidth": 0.8, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.05})

IN = os.path.join("results", "danger_signal", "phase4_analyses.json")
OUT_DIR = os.path.join("docs", "danger_signal_study", "figures")

CAZ_C = "#c0392b"   # PBP-3, high yield
MERO_C = "#2471a3"  # PBP-2, low yield


def _load():
    with open(IN, encoding="utf-8") as f:
        return json.load(f)


def fig1_yield_table(d):
    """Yield by agent, grouped by PBP target -- the resource deliverable."""
    rows = d["analysis1_yield_table"]["rows"]
    labels = [r["agent"] for r in rows]
    vals = [r["Y_lysis_fg_per_CFU"] for r in rows]
    colors = ["#c0392b" if r["pbp"] == "PBP-3" else
              ("#2471a3" if r["pbp"] == "PBP-2" else "#7f8c8d") for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.1), constrained_layout=True)
    y = np.arange(len(rows))
    # Points with intervals, NOT bars: the axis is logarithmic, so a bar drawn
    # from zero has no meaningful length and would misrepresent the ratios
    # that are the whole point of this table.
    for i, (r, c) in enumerate(zip(rows, colors)):
        if r["interval"]:
            lo, hi = r["interval"]
            ax.plot([lo, hi], [i, i], color=c, lw=2.0, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([e, e], [i - 0.14, i + 0.14], color=c, lw=1.4)
        ax.plot([r["Y_lysis_fg_per_CFU"]], [i], "o", color=c, ms=7,
                zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 12)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="x", color="#dddddd", lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.set_xlabel("danger-signal yield per CFU killed (fg/CFU)")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", ms=6) for c in
               ("#c0392b", "#2471a3", "#7f8c8d")]
    # "PBP-3" rather than "cephalosporins": the red group includes
    # piperacillin, which is a penicillin, so the class label was wrong.
    ax.legend(handles, ["PBP-3-specific (cephalosporin, penicillin)",
                        "PBP-2-specific (carbapenems)",
                        "ribosomal (aminoglycoside)"], frameon=False,
              fontsize=7.5, loc="lower right")
    ax.set_title("Every agent here is bactericidal, yet yields differ "
                 "several-fold:\nthe yield tracks the PBP target, not killing "
                 "class or morphology", fontsize=8.5)
    _save(fig, "fig1_yield_table")


def fig2_matched_exposure(d):
    """Matched %fT>MIC: near-equal killing, divergent inflammatory cost."""
    rows = d["analysis3_matched_exposure"]["rows"]
    t = [r["target_fT_pct"] for r in rows]
    caz_kill = [r["arms"]["ceftazidime"]["net_log10_change"] for r in rows]
    mero_kill = [r["arms"]["meropenem"]["net_log10_change"] for r in rows]
    lo = [min(x["ratio_AUC_DS"] for x in r["cost_ratio_range"]) for r in rows]
    hi = [max(x["ratio_AUC_DS"] for x in r["cost_ratio_range"]) for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.2),
                                 constrained_layout=True)
    a1.plot(t, caz_kill, "o-", color=CAZ_C, label="ceftazidime (PBP-3)", ms=5)
    a1.plot(t, mero_kill, "s--", color=MERO_C, label="meropenem (PBP-2)", ms=5)
    a1.axhline(0.0, color="k", lw=0.6, ls=":")
    a1.set_xlabel("matched exposure target (% fT>MIC)")
    a1.set_ylabel("48 h net change (log$_{10}$ CFU)")
    a1.set_title("Killing is nearly equal at matched exposure", fontsize=8.5)
    a1.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=1.0,
              edgecolor="none")

    a2.fill_between(t, lo, hi, color=CAZ_C, alpha=0.22)
    a2.plot(t, lo, color=CAZ_C, lw=1.0)
    a2.plot(t, hi, color=CAZ_C, lw=1.0)
    a2.axhline(1.0, color="k", lw=0.8, ls=":")
    a2.set_yscale("log")
    a2.set_xlabel("matched exposure target (% fT>MIC)")
    a2.set_ylabel("danger-signal cost ratio\n(ceftazidime / meropenem)")
    a2.set_title("Inflammatory cost is not", fontsize=8.5)
    a2.set_ylim(0.6, 30)
    a2.text(0.03, 0.97,
            "shaded band = assumed 3.5-7x yield range;\n"
            "the ratio is reported as a RANGE, never a single value",
            transform=a2.transAxes, fontsize=7, va="top", color="#555555")
    _save(fig, "fig2_matched_exposure")


def fig3_cost_index(d):
    """Existing PK/PD indices do not rank inflammatory cost across agents."""
    a4 = d["analysis4_cost_index"]
    lib = a4["library"]
    order = ["yield_weighted_kill", "cum_killed_cfu_per_mL",
             "net_log10_change", "fCmax_MIC", "fT_pct", "fAUC_MIC"]
    nice = {"yield_weighted_kill": "yield x killed biomass (proposed)",
            "cum_killed_cfu_per_mL": "killed biomass (agent-blind)",
            "net_log10_change": "net log$_{10}$ kill",
            "fCmax_MIC": "fC$_{max}$/MIC", "fT_pct": "% fT>MIC",
            "fAUC_MIC": "fAUC/MIC"}
    rho = [a4["candidates"][k]["spearman_rho"] for k in order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.3),
                                 constrained_layout=True)
    y = np.arange(len(order))
    cols = ["#1e8449" if r > 0.9 else "#b03a2e" if abs(r) < 0.4 else "#b9770e"
            for r in rho]
    a1.barh(y, rho, color=cols, height=0.62)
    a1.set_yticks(y)
    a1.set_yticklabels([nice[k] for k in order], fontsize=7.5)
    a1.invert_yaxis()
    a1.axvline(0.0, color="k", lw=0.7)
    a1.set_xlim(-1.05, 1.05)
    a1.set_xlabel("Spearman $\\rho$ with danger-signal cost (across agents)")
    a1.set_title(f"Ranking performance, {a4['n_regimens']} regimens",
                 fontsize=8.5)

    for agent, c, m in (("ceftazidime", CAZ_C, "o"), ("meropenem", MERO_C, "s")):
        sub = [r for r in lib if r["agent"] == agent]
        a2.scatter([r["fT_pct"] for r in sub],
                   [r["cost_AUC_DS"] for r in sub],
                   s=18, c=c, marker=m, alpha=0.8, label=f"{agent}")
    a2.set_yscale("log")
    a2.set_xlabel("% fT>MIC")
    a2.set_ylabel("inflammatory cost\n(attributable AUC$_{DS}$, pg·h/mL)")
    a2.set_title("Same exposure, two different costs", fontsize=8.5)
    # Limits derived from the data, with headroom: a hard-coded upper bound
    # silently clipped the highest-cost regimen in an earlier version.
    # The extra room below the lowest point is for the annotation, which would
    # otherwise be drawn over the low-cost meropenem cluster.
    _c = [r["cost_AUC_DS"] for r in lib if r["cost_AUC_DS"] > 0]
    a2.set_ylim(min(_c) * 0.22, max(_c) * 1.6)
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    # The quoted range is read from analysis 3 rather than hard-coded, so the
    # annotation cannot drift from the data it describes.
    _r = [x["ratio_AUC_DS"] for e in d["analysis3_matched_exposure"]["rows"]
          for x in e["cost_ratio_range"]]
    a2.text(0.03, 0.04,
            "at matched %fT>MIC the two agents differ by\n"
            f"several-fold in cost (range {min(_r):.1f}-{max(_r):.1f}x)",
            transform=a2.transAxes, fontsize=7, ha="left", va="bottom",
            color="#555555")
    _save(fig, "fig3_cost_index")


def fig4_exposure_response(d):
    """Cost versus exposure: flat above stasis, explosive below."""
    a5 = d["analysis5_divergence_map"]
    cells = a5["cells"]
    facs = sorted({c["mic_factor"] for c in cells})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3),
                                 constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, fac in enumerate(facs):
        sub = sorted([c for c in cells if c["mic_factor"] == fac
                      and c.get("ceftazidime")],
                     key=lambda c: c["target_fT_pct"])
        if not sub:
            continue
        x = [c["target_fT_pct"] for c in sub]
        cost = [c["ceftazidime"]["cost_AUC_DS"] for c in sub]
        kill = [c["ceftazidime"]["net_log10_change"] for c in sub]
        col = cmap(i / max(len(facs) - 1, 1))
        a1.plot(x, cost, "o-", color=col, ms=4, label=f"MIC x{fac:g}")
        a2.plot(x, kill, "o-", color=col, ms=4)
    a1.set_yscale("log")
    a1.set_xlabel("exposure target (% fT>MIC)")
    a1.set_ylabel("attributable AUC$_{DS}$ (pg·h/mL)")
    a1.set_title("Cost falls with exposure, then flattens", fontsize=8.5)
    a1.set_ylim(1, 1e5)
    a1.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    # Worded to match the manuscript: rescaling potency and re-solving the
    # dose preserves C/C50 in a linear PK system, so the superposition is
    # largely a consequence of how susceptibility is represented.
    a1.text(0.03, 0.04,
            "susceptibility curves superimpose, largely by\n"
            "construction (potency and dose rescale together)",
            transform=a1.transAxes, fontsize=7, va="bottom", color="#555555")
    a2.axhline(0.0, color="k", lw=0.7, ls=":")
    a2.set_xlabel("exposure target (% fT>MIC)")
    a2.set_ylabel("48 h net change (log$_{10}$ CFU)")
    a2.set_title("Stasis boundary (dotted)", fontsize=8.5)
    cov = a5.get("coverage", {}).get("populated_arms", {})
    fig.text(0.5, -0.04,
             "No evaluated exposure increase raised inflammatory cost "
             f"(ceftazidime arms shown; {cov.get('ceftazidime', '?')} of "
             f"{a5.get('coverage', {}).get('n_possible_arms_per_agent', '?')} "
             "populated - missing points lie outside the dose bracket).",
             ha="center", fontsize=7.5, color="#555555")
    _save(fig, "fig4_exposure_response")


def fig5_dual_pathway(d):
    """The kill-independent pathway is required by the P. aeruginosa data."""
    a2d = d["analysis2_dual_pathway"]
    levels = ["0.5x MIC", "1x MIC", "50x MIC"]
    dual = [a2d["levels"][k]["dual"]["AUC_DS_pg_h_mL"] for k in levels]
    lysis = [a2d["levels"][k]["lysis_only"]["AUC_DS_pg_h_mL"] for k in levels]
    fig, ax = plt.subplots(figsize=(5.2, 3.3), constrained_layout=True)
    x = np.arange(len(levels))
    w = 0.36
    ax.bar(x - w / 2, dual, w, label="dual pathway", color=CAZ_C)
    ax.bar(x + w / 2, lysis, w, label="lysis-coupled only", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("danger-signal exposure (AUC$_{DS}$, pg·h/mL)")
    ax.set_title("Jackson & Kropp observed sub-MIC release matching 1-50x MIC;\n"
                 "a lysis-only model cannot produce it", fontsize=8.5)
    ax.set_ylim(0, max(dual) * 1.18)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.text(0.5, -0.05,
             "N.B. Prins (E. coli) reports the opposite at sub-MIC "
             "-- an unresolved conflict between held-out sources",
             ha="center", fontsize=7, color="#555555")
    _save(fig, "fig5_dual_pathway")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    d = _load()
    print("Phase 4 figures:")
    fig1_yield_table(d)
    fig2_matched_exposure(d)
    fig3_cost_index(d)
    fig4_exposure_response(d)
    fig5_dual_pathway(d)


if __name__ == "__main__":
    main()
