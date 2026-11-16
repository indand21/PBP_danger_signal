"""Supplementary Figure S4: the full broth schedules against a first-exposure fit.

Reads results/danger_signal/broth_replay.json and draws, through the phase 4
figure helpers, docs/danger_signal_study/figures/figS4_broth_replay.{png,pdf},
copied into the submission package's figures folder.

One row per panel of Nussbaumer-Proll et al. Figure 2 (A one exposure, B two,
C three; exposure windows shaded). For three arms (1, 2 and 8 x MIC) the
observed readings are drawn with the source figure's marker shapes, and the
lines are the bacterial model refitted to the FIRST exposure window only, so
everything after the first removal is a prediction. Readings below the 10
CFU/mL plot floor are drawn as open downward triangles on the floor; a
reading hidden in a cluster is drawn as its interval. The note's numbers are
read from the results file.
"""
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.ds_broth_replay as br  # noqa: E402
import scripts.ds_phase4_figures as f4  # noqa: E402  (sets the house rcParams)

IN = os.path.join("results", "danger_signal", "broth_replay.json")
NAME = "figS4_broth_replay"
PACKAGE_FIGURES = os.path.join("docs", "danger_signal_submission_package", "figures")
INK = "#333333"
#: The dataviz reference categorical slots 1-3, checked with its validator;
#: aqua is below 3:1 contrast, so every arm is also direct-labelled and keeps
#: the source figure's marker shape.
ARMS = ((1.0, "1x MIC", "#2a78d6", "*"), (2.0, "2x MIC", "#eb6834", "D"), (8.0, "8x MIC", "#1baf7a", "^"))
MODEL = ("first_window", "refit")


def main():
    with open(IN, encoding="utf-8") as fh:
        d = json.load(fh)
    h = d["headline"]
    growths = {p: (v["rate_per_h"], v["log10_initial"], v["log10_capacity"]) for p, v in h["growth"].items()}
    kill = d["prediction_tests"][MODEL[0]][MODEL[1]]["kill"]
    kill = tuple(kill[k] for k in br.KILL_KEYS)
    obs = br.load()

    fig, axes = plt.subplots(3, 1, figsize=(f4.COL_W, 6.6), sharex=True, constrained_layout=True)
    grid = np.linspace(0.0, 24.0, 241)
    for ax, panel in zip(axes, "ABC"):
        for a, b in br.SCHEDULES[panel]:
            ax.axvspan(a, b, color="#d5d8dc", alpha=0.6, lw=0, zorder=0)
        sims = br.simulate_panel(panel, growths[panel], kill, [c for c, *_ in ARMS], grid)
        for j, (conc, label, color, marker) in enumerate(ARMS):
            ax.plot(grid, [sims[t][j] for t in grid], color=color, lw=1.4, zorder=2)
            pts = [o for o in obs if o["panel"] == panel and o["conc"] == conc]
            for o in pts:
                if o["kind"] == "point":
                    ax.plot(o["t"], o["value"], marker, ms=5 if marker != "*" else 7, color=color,
                            mec="white", mew=0.5, zorder=4)
                elif o["kind"] == "below_floor":
                    ax.plot(o["t"], br.FLOOR, "v", ms=5, mfc="white", mec=color, mew=1.0, zorder=4)
                else:
                    # A cluster interval is shared by every arm in it: neutral, not an arm color.
                    ax.plot([o["t"], o["t"]], [o["lower"], o["upper"]], color="#7f8c8d", lw=2.2,
                            alpha=0.7, zorder=3)
        # Direct labels at the line ends, pushed apart where the lines converge.
        ends = sorted(((sims[24.0][j], label) for j, (_, label, _, _) in enumerate(ARMS)))
        placed = []
        for y, label in ends:
            y_lab = max(y, placed[-1] + 0.55) if placed else y
            placed.append(y_lab)
            ax.annotate(label, (24.0, y), xytext=(24.4, y_lab), textcoords="data",
                        fontsize=6.5, color=INK, va="center")
        ax.axhline(br.FLOOR, color="#999999", lw=0.7, ls=":", zorder=1)
        ax.set_ylim(0.6, 9.6)
        ax.set_yticks([1, 3, 5, 7, 9])
        ax.set_ylabel("log10 CFU/mL")
        n_exp = len(br.SCHEDULES[panel])
        ax.set_title(f"{panel}  {n_exp} exposure{'s' if n_exp > 1 else ''}", fontsize=9, loc="left")
    axes[-1].set_xlim(0, 27.5)
    axes[-1].set_xticks(range(0, 25, 4))
    axes[-1].set_xlabel("time, h")

    r = h["first_window_reexposure"]["refit"]
    note = (f"Shaded: drug present. Lines: the bacterial model refitted to the first exposure window "
            f"only, with each panel's growth fitted to its untreated controls; after the first removal "
            f"they are predictions. Open triangles: readings below the 10 CFU/mL plot floor. On "
            f"re-exposure the fit predicts more killing than observed (mean residual "
            f"{r['bias']:+.2f} log10 over {r['n']} readings, {r['bias_without_C_2x']:+.2f} without "
            f"panel C's 2x MIC arm). Source: Nussbaumer-Proll et al. 2022, Figure 2 (CC BY 4.0).")
    f4._footnote(fig, note, 7)
    f4._save(fig, NAME)
    os.makedirs(PACKAGE_FIGURES, exist_ok=True)
    for ext in (".png", ".pdf"):
        shutil.copy2(os.path.join(f4.OUT_DIR, NAME + ext), os.path.join(PACKAGE_FIGURES, NAME + ext))
    print(f"wrote {NAME}")


if __name__ == "__main__":
    main()
