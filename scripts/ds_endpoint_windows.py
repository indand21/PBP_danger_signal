"""Does the exposure-release result depend on the endpoint and its window?

The paper's endpoint is the 48-hour AUC of plasma endotoxin above the
pre-treatment level. This script re-reads the same regimens on shorter windows
(0-1, 0-2, 0-4, 0-8 and 0-24 h), and on the peak concentration, its timing and
the maximum release rate, to find out which statements survive a change of
endpoint.

It also measures how far the susceptibility sweep in phase 4's
analysis5_divergence_map is a replicate of itself. That sweep scales kill_C50,
re-derives the MIC from the model's own stasis point and re-solves the dose, so
C/C50 and C/MIC follow the same trajectory at every MIC level: the seven levels
are one simulation per exposure target, reproduced to the dose solver's
tolerance. The summaries the paper quotes are therefore recomputed here over
the distinct exposure levels (the MIC x1 stratum, which is fully populated).

Nothing is re-solved. The doses are phase 4's stored ones, and each arm's 48-h
AUC must reproduce the stored value before anything else is reported, so these
are the same simulations the paper already rests on, read differently.

Window AUCs are exact: the linear endotoxin compartment D' = src(t) - k D is
integrated in closed form over the release rate, which is evaluated exactly
at eight sub-steps per solver step (src/analysis/exact_release.py). The
trapezoid over the sampled trajectory is kept as a check, and its
disagreement is reported as that rule's error.

Writes results/danger_signal/endpoint_windows.json.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"
OUT = ROOT / "results/danger_signal/endpoint_windows.json"
AGENTS = ("ceftazidime", "meropenem")
WINDOWS_H = (1.0, 2.0, 4.0, 8.0, 24.0, 48.0)
INTERVAL_H, N_DOSES = 8.0, 6
REFERENCE_TARGET = 40.0   # lowest exposure that controls the infection
# The endotoxin AUC is a closed-form function of a deterministic bacterial
# solve, so a recomputation at the stored dose reproduces the stored value to
# rounding; any real change of model or dose fails this.
REPRODUCTION_RTOL = 1e-9


def load_phase4_module():
    spec = importlib.util.spec_from_file_location(
        "endpoint_windows_p4", ROOT / "scripts/ds_phase4_analyses.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exact_window_aucs(t, src, k, windows):
    """AUC of D over [0, T] for D' = src - k D, D(0) = 0, src piecewise linear.

    Over a step of length h with src = a + b s, the exact update is
        D1 = D0 e^{-kh} + a (1 - e^{-kh})/k + b (h/k - (1 - e^{-kh})/k^2)
    and the exact integral of D over the step is
        D0 (1 - e^{-kh})/k + a (h/k - (1 - e^{-kh})/k^2)
          + b (h^2/(2k) - h/k^2 + (1 - e^{-kh})/k^3).
    Window ends are inserted into the grid, so no step straddles one. The
    update itself is src/analysis/exact_release.linear_source_ds.
    """
    from src.analysis.exact_release import linear_source_ds

    t = np.asarray(t, float)
    src = np.asarray(src, float)
    grid = np.unique(np.concatenate([t, [w for w in windows if t[0] < w <= t[-1]]]))
    _, area, _ = linear_source_ds(grid, np.interp(grid, t, src), k)
    return {w: float(area[int(np.argmin(np.abs(grid - w)))]) for w in windows
            if t[0] < w <= t[-1] + 1e-12}


def window_auc(t, y, T):
    """Trapezoid AUC of y over [t0, T], interpolating y at T."""
    m = t <= T
    tt = np.append(t[m], T) if t[m][-1] < T else t[m]
    yy = np.append(y[m], np.interp(T, t, y)) if t[m][-1] < T else y[m]
    return float(trapezoid(yy, tt))


def endpoints(p4, agent, cell):
    """Every endpoint of one stored arm, plus its reproduction check."""
    sim = p4.simulate(agent, cell[agent]["dose_mg"], INTERVAL_H, N_DOSES,
                      mic=cell["mic_mgL"])
    out, summ = p4.ds_cost(sim, agent=agent, cytokines=False)
    t = np.asarray(out["t"], float)
    D = np.asarray(out["DS"], float) - out["DS"][0]
    q = p4.P_REF
    to_conc = 1e-6 / q.V_ds_L * q.translocation_fraction
    rate_ng_h = np.asarray(out["release_lysis"]) + np.asarray(out["release_filament"])
    src = rate_ng_h * 1e6 * to_conc
    stored = cell[agent]["AUC_DS_pg_h_mL"]
    reproduced = summ["AUC_DS_pg_h_mL"]
    rel = abs(reproduced - stored) / stored
    if rel > REPRODUCTION_RTOL:
        raise RuntimeError(f"{agent} at {cell['target_fT_pct']}% reproduces the "
                           f"stored 48-h AUC only to {rel:.2e}; not the same simulation")
    trap = {w: window_auc(t, D, w) for w in WINDOWS_H}
    exact = exact_window_aucs(t, src, q.k_ds_clear, WINDOWS_H)
    k_pk, k_rt = int(np.argmax(D)), int(np.argmax(rate_ng_h))
    return {
        "agent": agent,
        "target_fT_pct": cell["target_fT_pct"],
        "dose_mg_q8h": cell[agent]["dose_mg"],
        "achieved_fT_pct": sim["fT"],
        "net_log10_change": sim["net_log10"],
        "controlled": bool(sim["net_log10"] < 0.0),
        "window_AUC": {f"0-{w:g}h": exact[w] for w in WINDOWS_H},
        "window_AUC_trapezoid": {f"0-{w:g}h": trap[w] for w in WINDOWS_H},
        "peak_excess": float(D[k_pk]),
        "t_peak_h": float(t[k_pk]),
        "max_release_rate_ng_per_h": float(rate_ng_h[k_rt]),
        "t_max_release_h": float(t[k_rt]),
        "fraction_of_48h_AUC_in_first_hour": exact[1.0] / exact[48.0],
        "stored_48h_AUC_reproduced_rel_diff": rel,
        "n_grid_points_first_hour": int(np.sum(t <= 1.0)),
    }


ENDPOINT_KEYS = [f"0-{w:g}h" for w in WINDOWS_H] + ["peak_excess",
                                                     "max_release_rate_ng_per_h"]


def value(arm, key):
    return arm["window_AUC"][key] if key in arm["window_AUC"] else arm[key]


def susceptibility_invariance(cells):
    """How far each MIC level reproduces the MIC x1 arm at the same target."""
    rows, worst = [], {"dose_per_mic": 0.0, "AUC_DS": 0.0, "net_log10": 0.0}
    for agent in AGENTS:
        for target in sorted({c["target_fT_pct"] for c in cells}):
            arms = [(c["mic_factor"], c[agent]) for c in cells
                    if c["target_fT_pct"] == target and c.get(agent)]
            dpm = [a["dose_mg"] / f for f, a in arms]
            auc = [a["AUC_DS_pg_h_mL"] for _, a in arms]
            net = [a["net_log10_change"] for _, a in arms]
            row = {"agent": agent, "target_fT_pct": target,
                   "mic_factors_populated": [f for f, _ in arms],
                   "dose_per_mic_spread_rel": max(dpm) / min(dpm) - 1,
                   "AUC_DS_spread_rel": max(auc) / min(auc) - 1,
                   "net_log10_spread_abs": max(net) - min(net)}
            rows.append(row)
            worst["dose_per_mic"] = max(worst["dose_per_mic"], row["dose_per_mic_spread_rel"])
            worst["AUC_DS"] = max(worst["AUC_DS"], row["AUC_DS_spread_rel"])
            worst["net_log10"] = max(worst["net_log10"], row["net_log10_spread_abs"])
    return rows, worst


def main():
    p4 = load_phase4_module()
    family = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(family, key=lambda e: e["cost"]))["params"]
    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    a5 = phase4["analysis5_divergence_map"]
    base = sorted((c for c in a5["cells"] if c["mic_factor"] == 1.0),
                  key=lambda c: c["target_fT_pct"])
    targets = [c["target_fT_pct"] for c in base]

    arms = {}
    for agent in AGENTS:
        for cell in base:
            if not cell.get(agent):
                raise RuntimeError(f"MIC x1 stratum lacks {agent} at "
                                   f"{cell['target_fT_pct']}%")
            arms[(agent, cell["target_fT_pct"])] = endpoints(p4, agent, cell)
            print(f"  {agent:12s} {cell['target_fT_pct']:4.0f}%  done", flush=True)

    # Each exposure step within an agent, read on every endpoint. A step is
    # "divergent" on an endpoint when net killing improved and the endpoint
    # rose: the trade-off the study set out to look for.
    # Every direction is re-read on the trapezoid window integrals; a step
    # whose sign depends on the quadrature would not be a result.
    steps, divergent = [], {k: 0 for k in ENDPOINT_KEYS}
    worst_fold_gap, sign_disagreements = 0.0, []
    for agent in AGENTS:
        for lo, hi in zip(targets, targets[1:]):
            a, b = arms[(agent, lo)], arms[(agent, hi)]
            better = b["net_log10_change"] < a["net_log10_change"]
            fold = {k: value(b, k) / value(a, k) for k in ENDPOINT_KEYS}
            for k in a["window_AUC_trapezoid"]:
                ft = b["window_AUC_trapezoid"][k] / a["window_AUC_trapezoid"][k]
                worst_fold_gap = max(worst_fold_gap, abs(ft - fold[k]))
                if (ft > 1.0) != (fold[k] > 1.0):
                    sign_disagreements.append(f"{agent} {lo:g}->{hi:g}% {k}")
            for k in ENDPOINT_KEYS:
                divergent[k] += int(better and fold[k] > 1.0)
            steps.append({"agent": agent, "from_fT": lo, "to_fT": hi,
                          "kill_improved": better, "fold_change": fold})
    if sign_disagreements:
        raise RuntimeError("step direction depends on quadrature: "
                           + "; ".join(sign_disagreements))

    relative_to_reference = {
        agent: {f"{t:g}": {k: value(arms[(agent, t)], k)
                           / value(arms[(agent, REFERENCE_TARGET)], k)
                           for k in ENDPOINT_KEYS}
                for t in targets}
        for agent in AGENTS}

    agent_ratio = {f"{t:g}": {k: value(arms[("ceftazidime", t)], k)
                              / value(arms[("meropenem", t)], k)
                              for k in ENDPOINT_KEYS}
                   for t in targets}
    controlled_targets = [t for t in targets
                          if all(arms[(a, t)]["controlled"] for a in AGENTS)]
    ratio_values = [v for t in controlled_targets for v in agent_ratio[f"{t:g}"].values()]

    # The summaries phase 4 quotes, over the distinct exposure levels.
    distinct = {}
    for agent in AGENTS:
        a48 = {t: arms[(agent, t)]["window_AUC"]["0-48h"] for t in targets}
        ctl = [t for t in targets if arms[(agent, t)]["controlled"]]
        fail = [t for t in targets if not arms[(agent, t)]["controlled"]]
        med_ctl = float(np.median([a48[t] for t in ctl]))
        distinct[agent] = {
            "controlled_targets": ctl, "failing_targets": fail,
            "cost_span_above_stasis_fold": max(a48[t] for t in ctl) / min(a48[t] for t in ctl),
            "median_cost_controlled": med_ctl,
            "failing_over_controlled_fold": (float(np.median([a48[t] for t in fail])) / med_ctl
                                             if fail else None),
        }

    inv_rows, inv_worst = susceptibility_invariance(a5["cells"])
    numerical = max(abs(arm["window_AUC_trapezoid"][k] - arm["window_AUC"][k])
                    / arm["window_AUC"][k]
                    for arm in arms.values() for k in arm["window_AUC"])

    fl = [arms[(a, t)]["fraction_of_48h_AUC_in_first_hour"]
          for a in AGENTS for t in controlled_targets]

    # The values the documents quote, in the form they quote them (percent
    # changes, not folds), so the consistency audit can bind each one here.
    top, lowest = max(targets), min(targets)

    def pct(agent, t, k):
        return 100.0 * (relative_to_reference[agent][f"{t:g}"][k] - 1.0)

    headline = {
        agent: {
            "first_hour_change_pct_failing_to_reference":
                100.0 * (value(arms[(agent, REFERENCE_TARGET)], "0-1h")
                         / value(arms[(agent, lowest)], "0-1h") - 1.0),
            "first_hour_change_pct_reference_to_top": pct(agent, top, "0-1h"),
            "max_release_rate_change_pct_reference_to_top":
                pct(agent, top, "max_release_rate_ng_per_h"),
            "auc48_change_pct_reference_to_top": pct(agent, top, "0-48h"),
            "peak_change_pct_reference_to_top": pct(agent, top, "peak_excess"),
        } for agent in AGENTS}
    headline["max_abs_peak_change_pct_reference_to_top"] = max(
        abs(headline[a]["peak_change_pct_reference_to_top"]) for a in AGENTS)
    headline["first_hour_share_pct_controlled_range"] = [100 * min(fl), 100 * max(fl)]
    headline["invariance_max_spread_pct"] = {
        "dose_per_mic": 100 * inv_worst["dose_per_mic"],
        "AUC_DS": 100 * inv_worst["AUC_DS"]}
    headline["invariance_max_net_log10_spread"] = inv_worst["net_log10"]
    headline["numerical_max_window_AUC_error_pct"] = 100 * numerical
    payload = {
        "status": "re-reading of phase 4's stored regimens; no dose re-solved, "
                  "no parameter changed",
        "endpoint": "plasma endotoxin above the pre-treatment level, uncalibrated "
                    "units; windows are AUCs from the first dose",
        "windows_h": list(WINDOWS_H),
        "reference_target_fT_pct": REFERENCE_TARGET,
        "regimen": f"q{INTERVAL_H:g}h x {N_DOSES}, 30-min infusion, MIC x1 "
                   "stratum of analysis5_divergence_map",
        "headline": headline,
        "arms": [arms[(a, t)] for a in AGENTS for t in targets],
        "exposure_steps": steps,
        "divergent_steps_by_endpoint": divergent,
        "n_steps": len(steps),
        "relative_to_reference_target": relative_to_reference,
        "agent_ratio_caz_over_mero": agent_ratio,
        "agent_ratio_range_controlled_targets": [min(ratio_values), max(ratio_values)],
        "fraction_of_48h_AUC_in_first_hour_controlled_range": [min(fl), max(fl)],
        "distinct_exposure_summary": distinct,
        "susceptibility_invariance": {
            "rows": inv_rows,
            "max_spread": inv_worst,
            "n_distinct_exposure_levels_per_agent": len(targets),
            "n_cells_in_sweep": len(a5["cells"]),
            "explanation": "kill depends on C only through C/kill_C50 and the "
                           "effective MIC is proportional to kill_C50; with "
                           "linear PK the re-solved dose scales by the same "
                           "factor, so every MIC level reproduces one "
                           "trajectory. Residual spread is the tolerance of "
                           "the effective-MIC root (the dose is solved to "
                           "1e-6 mg from the PK alone).",
        },
        "numerical_check": {
            "max_rel_diff_trapezoid_vs_exact_window_AUC": numerical,
            "max_rel_diff_reproduced_vs_stored_48h_AUC":
                max(a["stored_48h_AUC_reproduced_rel_diff"] for a in arms.values()),
            "reproduction_tolerance": REPRODUCTION_RTOL,
            "max_abs_fold_change_gap_trapezoid_vs_exact": worst_fold_gap,
            "step_directions_agree_with_exact": True,
            "min_grid_points_first_hour": min(a["n_grid_points_first_hour"]
                                              for a in arms.values()),
            "method": "reported windows integrate D' = src - k D exactly over "
                      "the release rate evaluated at 8 sub-steps per solver "
                      "step; the trapezoid over the sampled trajectory is the "
                      "check",
        },
        "caveat": "Release is lysis-coupled and immediate in this model, so "
                  "short-window results depend on that assumption. A delay "
                  "between killing and release, as fitted for cefuroxime by "
                  "Thorsted et al. 2020, would move release later; it is not "
                  "modeled here.",
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {PHASE4.name: hashlib.sha256(PHASE4.read_bytes()).hexdigest()},
        "provenance": provenance(note="Endpoint and observation-window re-reading "
                                      "of phase 4 regimens"),
    }
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: payload[k] for k in
                      ["divergent_steps_by_endpoint", "n_steps",
                       "agent_ratio_range_controlled_targets",
                       "fraction_of_48h_AUC_in_first_hour_controlled_range",
                       "distinct_exposure_summary", "numerical_check"]},
                     indent=2), flush=True)
    print(json.dumps(inv_worst, indent=2))


if __name__ == "__main__":
    main()
