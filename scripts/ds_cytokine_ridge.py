"""What the IL-6 calibration's flat ridge moves, and what the data do not decide.

``calibrate_family`` fits the IL-6 layer to Hijma 2020 at each DS50 on a grid
and reads the IL-6 time-to-peak off the 0.005-h simulation grid with argmax.
That residual is piecewise constant in the parameters, so the optimizer sees
no gradient in it and stops wherever it is along a ridge of equal cost. The
TNF fit had the same flaw and was corrected on 2026-11-16
(``interpolated_peak``); the IL-6 fit was deliberately left as it is
(decision of 2026-11-17), because nothing the paper reports depends on it:
the reported endpoint, the 48-h endotoxin AUC, is upstream of the cytokine
layer. The reference parameterization, DS50 = 300 pg/mL, is therefore kept as
a convention, and this script records what that convention leaves open.

1. Reproducibility, and how far the ridge reaches. The family is refitted
   and its reference compared with the one stored in phase4_analyses.json;
   with the same code the fit is deterministic and they agree exactly. The
   reference stored there until plan item 8 came from an earlier,
   uncommitted state of the code and sits elsewhere on the ridge at the same
   cost (EARLIER_REFERENCE). All three are run through the Table 1 arms, to
   measure how far a move along the ridge shifts the cytokine-layer outputs
   and, as a check, the endotoxin AUC, which is upstream of them.

2. What the grid-read peak decides. The family is refitted with the peak
   located between grid points instead, in this process only. Every DS50 then
   fits, including the two the grid-read screen rejects, so neither the
   choice of reference nor the lower end of the accepted DS50 range is
   decided by the volunteer data.

NO DEFAULT IS CHANGED and nothing downstream reads this file.
Writes results/danger_signal/cytokine_ridge.json.
"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.analysis.cytokine_calibration as cc  # noqa: E402
from scripts.ds_endpoint_windows import load_phase4_module  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"
OUT = ROOT / "results/danger_signal/cytokine_ridge.json"
FITTED = ("hill", "k_IL6_max", "k_signal_off")
FIELD = {"hill": "hill_ds", "k_IL6_max": "k_IL6_max", "k_signal_off": "k_signal_off"}
OUTPUTS = ("AUC_DS_pg_h_mL", "AUC_IL6_excess_pg_h_mL", "IL6_peak_pg_mL", "saturation_peak")


def predict_peaks_between_grid_points(p, doses=None):
    """``predict_peaks`` with each peak located by ``interpolated_peak``."""
    doses = doses or cc.HIJMA_2020["doses_ng_per_kg"]
    out = {k: [] for k in ("IL6_peak", "TNF_peak", "t_peak_IL6", "t_peak_TNF",
                           "saturation_peak")}
    for d in doses:
        r = cc.simulate_endotoxemia(d, p)
        t6, v6 = cc.interpolated_peak(r["t"], r["IL6"])
        tt, vt = cc.interpolated_peak(r["t"], r["TNF"])
        out["IL6_peak"].append(v6)
        out["t_peak_IL6"].append(t6)
        out["TNF_peak"].append(vt)
        out["t_peak_TNF"].append(tt)
        out["saturation_peak"].append(float(np.max(r["saturation"])))
    return out


def family_between_grid_points():
    original = cc.predict_peaks
    cc.predict_peaks = predict_peaks_between_grid_points   # this process only
    try:
        return cc.calibrate_family()
    finally:
        cc.predict_peaks = original


def summary(e):
    return {"DS50": e["DS50"], "feasible": e["feasible"], "cost": e["cost"],
            "hill": e["hill"], "k_IL6_max": e["k_IL6_max"],
            "k_signal_off": e["k_signal_off"],
            "pred_t_peak_IL6_h": e["pred_t_peak_IL6_h"],
            "max_rel_peak_error": e["max_rel_peak_error"],
            "saturation_peak_at_2ng_kg": e["saturation_peak_at_2ng_kg"]}


#: The Layer-2 reference phase4_analyses.json carried until plan item 8
#: (commit d429b32), written from an uncommitted state of the code: the same
#: DS50 and cost as the current reference, another point on the ridge.
EARLIER_REFERENCE = {"hill": 1.713984131094413, "k_IL6_max": 176080.53594356967,
                     "k_signal_off": 0.931066454178522, "cost": 0.0012247330282485073}


def main():
    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    stored = phase4["meta"]["layer2_reference"]

    # ---- 1. reproducibility ------------------------------------------------
    grid_read = cc.calibrate_family()
    today = min((e for e in grid_read if e["feasible"]), key=lambda e: e["cost"])
    if today["DS50"] != stored["DS50"]:
        raise RuntimeError("the refit chose a different reference DS50")
    stored_params = replace(today["params"], **{FIELD[k]: float(stored[k]) for k in FITTED})
    earlier_params = replace(today["params"], **{FIELD[k]: float(EARLIER_REFERENCE[k]) for k in FITTED})
    refs = {"stored": cc.calibrate_tnf({"params": stored_params})["params"],
            "refit": cc.calibrate_tnf(today)["params"],
            "earlier": cc.calibrate_tnf({"params": earlier_params})["params"]}

    p4 = load_phase4_module()
    p4.MIC_EFFECT = phase4["meta"]["mic_effect_site_mgL"]
    arms = {}
    for row in phase4["analysis3_matched_exposure"]["rows"]:
        if not row.get("evaluated"):
            continue
        for agent in ("ceftazidime", "meropenem"):
            sim = p4.simulate(agent, row["arms"][agent]["dose_mg"], 8.0, 6)
            per = {}
            for name, q in refs.items():
                p4.P_REF = q
                _, s = p4.ds_cost(sim, agent=agent)
                per[name] = {k: float(s[k]) for k in OUTPUTS}
            arms[f"{agent} {row['target_fT_pct']:g}%"] = per
    table1_rel = {k: max(abs(v["refit"][k] / v["stored"][k] - 1.0) for v in arms.values())
                  for k in OUTPUTS}
    table1_rel_earlier = {k: max(abs(v["earlier"][k] / v["stored"][k] - 1.0) for v in arms.values())
                          for k in OUTPUTS}

    # ---- 2. the peak located between grid points ---------------------------
    between = family_between_grid_points()
    by_ds50 = {f"{e['DS50']:g}": e for e in between}
    rejected = [e["DS50"] for e in grid_read if not e["feasible"]]

    headline = {
        "reference_DS50": stored["DS50"],
        "cost_stored": stored["cost"], "cost_refit": today["cost"],
        "cost_rel_diff": abs(today["cost"] / stored["cost"] - 1.0),
        "param_rel_diff_refit_vs_stored": {k: today[k] / stored[k] - 1.0 for k in FITTED},
        "table1_max_rel_diff_refit_vs_stored": table1_rel,
        "cost_rel_diff_earlier": abs(EARLIER_REFERENCE["cost"] / stored["cost"] - 1.0),
        "param_rel_diff_earlier_vs_stored": {k: EARLIER_REFERENCE[k] / stored[k] - 1.0
                                             for k in FITTED},
        "table1_max_rel_diff_earlier_vs_stored": table1_rel_earlier,
        "grid_read": {
            "n_feasible": sum(e["feasible"] for e in grid_read),
            "rejected_DS50": rejected,
            "rejected_t_peak_IL6_h": [e["pred_t_peak_IL6_h"][0] for e in grid_read if not e["feasible"]],
            "reference_t_peak_IL6_h": today["pred_t_peak_IL6_h"][0],
        },
        "between_grid_points": {
            "n_feasible": sum(e["feasible"] for e in between),
            "n_grid": len(between),
            "max_cost": max(e["cost"] for e in between),
            "min_cost_DS50": min(between, key=lambda e: e["cost"])["DS50"],
            "t_peak_IL6_h_range": [min(e["pred_t_peak_IL6_h"][0] for e in between),
                                   max(e["pred_t_peak_IL6_h"][0] for e in between)],
            "k_signal_off_range": [min(e["k_signal_off"] for e in between),
                                   max(e["k_signal_off"] for e in between)],
            "saturation_at_2ng_kg_of_rejected": {
                f"{d:g}": by_ds50[f"{d:g}"]["saturation_peak_at_2ng_kg"] for d in rejected},
            "saturation_at_2ng_kg_range": [min(e["saturation_peak_at_2ng_kg"] for e in between),
                                           max(e["saturation_peak_at_2ng_kg"] for e in between)],
        },
    }
    payload = {
        "headline": headline,
        "stored_reference": {k: stored[k] for k in ("DS50",) + FITTED + ("cost", "pred_t_peak_IL6_h",
                                                                         "saturation_peak_at_2ng_kg")},
        "family_grid_read": [summary(e) for e in grid_read],
        "family_between_grid_points": [summary(e) for e in between],
        "table1_arms": arms,
        "input_sha256": {PHASE4.name: hashlib.sha256(PHASE4.read_bytes()).hexdigest()},
        "provenance": provenance(note="IL-6 calibration ridge: reproducibility and "
                                      "what the grid-read peak decides; no default changed"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
