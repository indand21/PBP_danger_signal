"""Supplementary Figure S2: matched on cumulative killing, and the ratio decomposed.

Reads results/danger_signal/kill_matched.json and draws, through the phase 4
figure helpers (column width, fonts, saved-width check),
docs/danger_signal_study/figures/figS2_kill_matched.{png,pdf}, copied into
the submission package's figures folder as for Figure S1.

Panel A is cumulative drug-induced killing over 48 h against the q8h dose for
each agent, solid where the regimen controls the infection and dotted where it
fails, with the Table 1 doses and every equal-killing dose marked. Panel B is
what multiplies the assumed yield ratio in each controlled comparison: the
killing factor and the sub-MIC factor of the exact decomposition, stacked on a
log axis so that the segments add. Every number the note quotes is read from
the results file, and each statement in it is made only if the data support it.
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

IN = os.path.join("results", "danger_signal", "kill_matched.json")
NAME = "figS2_kill_matched"
PACKAGE_FIGURES = os.path.join("docs", "danger_signal_submission_package", "figures")
INK = "#333333"
COLOR = {"ceftazidime": f4.CAZ_C, "meropenem": f4.MERO_C}
SEGMENT = {"killing": ("#b9770e", "killing"),
           "sub_mic": ("#7d3c98", "sub-MIC term")}   # palette checked with the dataviz validator


def curve(ax, sc, agent, b0):
    d = np.array(sc["dose_mg_q8h"])
    k = np.array(sc["cum_killed_cfu_per_mL"]) / b0
    ok = np.array(sc["net_log10_change"]) < 0.0
    # Split where control changes, overlapping one point so the line is unbroken.
    edges = [0] + [i for i in range(1, len(d)) if ok[i] != ok[i - 1]] + [len(d)]
    for a, b in zip(edges, edges[1:]):
        lo = max(a - 1, 0)
        ax.plot(d[lo:b], k[lo:b], color=COLOR[agent], lw=1.4,
                ls="-" if ok[a] else ":", zorder=2)
    # The two peaks sit close together; label each on its outer side.
    i = int(np.argmax(k))
    left = agent == "ceftazidime"
    ax.annotate(agent, (d[i], k[i]), xytext=(-5 if left else 5, 0), textcoords="offset points",
                ha="right" if left else "left", va="center", fontsize=7, color=INK)


def panel_a(ax, d):
    b0 = d["design"]["initial_burden_cfu_per_mL"]
    for agent, sc in d["killing_curve"].items():
        curve(ax, sc, agent, b0)
    for t, pair in d["table1"].items():
        for agent in ("ceftazidime", "meropenem"):
            a = pair[agent]
            ax.plot(a["dose_mg_q8h"], a["killed_over_initial_burden"], "o", ms=4.5,
                    color=COLOR[agent], mec="white", mew=0.6, zorder=4)
    for block in d["kill_matched"].values():
        for direction, m in block.items():
            agent = direction.replace("_solved", "")
            for s in m["solutions"]:
                ax.plot(s["arm"]["dose_mg_q8h"], s["arm"]["killed_over_initial_burden"],
                        "D" if s["branch"] == "above_peak" else "s", ms=4.5, mfc="white",
                        mec=COLOR[agent], mew=1.0, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3, 6e4)
    ax.set_ylim(0.4, 200)
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10", "100", "1000", "10000"])
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_xlabel("dose every 8 h, mg")
    ax.set_ylabel("killed / starting burden")
    ax.set_title("A  Cumulative killing over 48 h", fontsize=9, loc="left")
    handles = [
        Line2D([], [], color=INK, lw=1.4, ls="-", label="controls"),
        Line2D([], [], color=INK, lw=1.4, ls=":", label="fails"),
        Line2D([], [], marker="o", ls="", color=INK, mec="white", ms=4.5, label="Table 1 dose"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=4.5, label="equal killing, controls"),
        Line2D([], [], marker="s", ls="", mfc="white", mec=INK, ms=4.5, label="equal killing, fails"),
    ]
    ax.legend(handles=handles, fontsize=6.5, loc="upper right", frameon=False,
              handlelength=1.8, labelspacing=0.3)


def panel_b(ax, d):
    groups = [("Matched %fT>MIC (Table 1)",
               [(t, v["decomposition"]["3.5"]) for t, v in d["table1"].items()]),
              ("Equal killing, meropenem dose solved",
               [(t, s["decomposition"]["3.5"]) for t, b in d["kill_matched"].items()
                for s in b["meropenem_solved"]["solutions"] if s["branch"] == "above_peak"]),
              ("Equal killing, ceftazidime dose solved",
               [(t, s["decomposition"]["3.5"]) for t, b in d["kill_matched"].items()
                for s in b["ceftazidime_solved"]["solutions"] if s["branch"] == "above_peak"])]
    y, ticks, labels, heads = 0.0, [], [], []
    top = 0.0
    for title, rows in groups:
        heads.append((y, title))
        y -= 0.85
        for t, dec in rows:
            left = 0.0
            for key in ("killing", "sub_mic"):
                w = np.log10(dec["factors"][key])
                ax.barh(y, w, left=left, height=0.62, color=SEGMENT[key][0],
                        edgecolor="white", linewidth=0.6)
                left += w
            total = dec["factors"]["killing"] * dec["factors"]["sub_mic"]
            ax.text(max(left, 0.0) + 0.003, y, f"{total:.2f}", va="center", fontsize=6.5, color=INK)
            top = max(top, left)
            ticks.append(y)
            labels.append(f"{t}%")
            y -= 1.0
        y -= 0.35
    for yh, title in heads:
        ax.text(0.0, yh - 0.25, title, fontsize=7, color=INK, va="center", ha="left",
                transform=ax.get_yaxis_transform(), fontweight="bold")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(y + 0.6, 0.6)
    span = [1.0, 1.1, 1.2, 1.3, 1.4]
    span = [v for v in span if np.log10(v) <= top * 1.25 + 1e-9]
    ax.set_xticks([np.log10(v) for v in span])
    ax.set_xticklabels([f"{v:.1f}" for v in span])
    ax.set_xlim(0.0, max(top * 1.25, np.log10(1.05)))
    ax.set_xlabel("factor multiplying the assumed yield ratio (log scale)")
    ax.axvline(0.0, color="#999999", lw=0.8)
    ax.set_title("B  What multiplies the yield ratio, where both agents control",
                 fontsize=9, loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in SEGMENT.values()],
              fontsize=6.5, loc="lower right", frameon=False)


def note_text(d):
    h = d["headline"]
    bd = h["by_direction"]
    t1_timing = h["table1_timing_factor_range"]
    timing_dev = max(abs(v - 1.0) for v in t1_timing + h["above_peak"]["timing_factor_range"])
    parts = [f"Table 1 setting (effective MIC {d['design']['mic_effect_site_mgL']:.2f} mg/L, "
             "q8h for 48 h). The CAZ:MERO ratio of 48-h release is the assumed 3.5-7 yield "
             "ratio times the factor drawn in B."]
    if timing_dev < 1e-4:
        exponent = int(np.ceil(np.log10(timing_dev)))
        parts.append(f"The timing factor is 1 to within 1 part in {10 ** -exponent:,} with "
                     "immediate release and is not drawn.")
    mf, cf = bd["meropenem_solved"]["below_peak"], bd["ceftazidime_solved"]["below_peak"]
    lo, hi = cf["sub_mic_factor_range"]
    text = ("Each killing level is also reached by a failing regimen (squares in A, "
            f"{h['below_peak']['dose_mg_q8h_range'][0]:.0f}-{h['below_peak']['dose_mg_q8h_range'][1]:.0f} "
            "mg q8h)")
    if h["meropenem_failing_vs_controlled_ratio_max_rel_diff"] < 1e-3 and not mf["all_controlled"]:
        text += ": with meropenem failing the ratio is unchanged"
        if not cf["all_controlled"]:
            text += (f"; with ceftazidime failing the sub-MIC term multiplies it by "
                     f"{lo:.0f}-{hi:.0f}")
    parts.append(text + ".")
    return " ".join(parts)


def main():
    with open(IN, encoding="utf-8") as fh:
        d = json.load(fh)
    fig = plt.figure(figsize=(f4.COL_W, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
    panel_a(fig.add_subplot(gs[0]), d)
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
