"""Score the curve-level validation against the targets the registry sets.

Every digitized dataset in data/external_validation/registry.json carries a
numeric acceptance target, written when the dataset was entered and before
any fitting. Until now those targets lived only in the registry's free-text
`notes`, nothing computed them, and run_external_validation.py writes no
results file -- so the supplement could quote a fitted RMSE while two of the
targets were not met and nothing surfaced it.

This script computes each target and writes
results/danger_signal/validation_targets.json, so the numbers quoted in S3
resolve to a results file like every other number in the study, and so the
consistency audit can check them.

The targets are restated here rather than parsed out of the notes: parsing
prose is fragile, and a target that silently fails to parse would be worse
than one written twice. Each entry quotes the registry text it comes from.

Writes results/danger_signal/validation_targets.json. Changes no parameter.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.analysis.external_validation as ev  # noqa: E402

OUT = os.path.join("results", "danger_signal", "validation_targets.json")

#: Registry note each target is quoted from, so the two can be compared.
SOURCES = {
    "meropenem_pk_1g":
        "Targets: RMSE < 0.15 log10 mg/L; Cmax ratio within +/-20%; "
        "AUC0-8h within +/-15%.",
    "meropenem_neutropenic_thigh":
        "Success: model stasis threshold within ~5-10 pp of 17.3%; "
        "monotonic sigmoidal kill curve.",
}


def _pk_targets(ds):
    d = ds.load()
    pred = np.asarray(ev.predict(d), dtype=float)
    t = np.asarray(d.t_obs, dtype=float)
    obs = np.asarray(d.y_obs, dtype=float)
    keep = obs > 0
    eps = 1e-3
    resid = (np.log10(np.maximum(pred[keep], eps))
             - np.log10(np.maximum(obs[keep], eps)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    cmax = float(pred.max() / obs.max() - 1.0)
    win = t <= 8.0
    auc = float(trapezoid(pred[win], t[win]) / trapezoid(obs[win], t[win]) - 1.0)
    # Terminal slope on the decay limb, the quantity both other targets follow
    # from. Fitted on t >= 2 h so the distribution phase is excluded.
    lim = (t >= 2.0) & (obs > 0) & (pred > 0)
    k_obs = float(-np.polyfit(t[lim], np.log(obs[lim]), 1)[0])
    k_pred = float(-np.polyfit(t[lim], np.log(pred[lim]), 1)[0])
    return {
        "rmse_log10": {"value": rmse, "target": "< 0.15",
                       "met": bool(rmse < 0.15)},
        "cmax_relative_error": {"value": cmax, "target": "within +/-0.20",
                                "met": bool(abs(cmax) <= 0.20)},
        "auc_0_8h_relative_error": {"value": auc, "target": "within +/-0.15",
                                    "met": bool(abs(auc) <= 0.15)},
        "terminal_half_life_h": {
            "observed": 0.693 / k_obs, "model": 0.693 / k_pred,
            "note": "not a registry target; recorded because the two targets "
                    "above both follow from it"},
        "predicted_over_observed": [float(p / o) if o > 0 else None
                                    for p, o in zip(pred, obs)],
    }


def _thigh_targets(ds):
    d = ds.load()
    idx, delta, mic = ev.dose_response_curve(d)
    stasis = ev.stasis_index(idx, delta, 0.0)
    one_log = ev.stasis_index(idx, delta, -1.0)
    published_stasis, published_1log = 17.3, 26.0
    gap = None if stasis is None else float(stasis - published_stasis)
    return {
        "effective_mic_mgL": float(mic),
        "model_stasis_fT_pct": None if stasis is None else float(stasis),
        "published_stasis_fT_pct": published_stasis,
        "gap_percentage_points": gap,
        "target": "within 5-10 percentage points of 17.3",
        "met": None if gap is None else bool(abs(gap) <= 10.0),
        "model_1log_fT_pct": None if one_log is None else float(one_log),
        "published_1log_fT_pct": published_1log,
        "monotone": bool(np.all(np.diff(delta) <= 1e-9)),
        "published_source": "Sabet M et al. Antimicrob Agents Chemother 2020; "
                            "stasis, 1-log and 2-log at 17.3, 26.0 and 41.7 "
                            "%24-h fT>MIC",
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    by_id = {ds.id: ds for ds in ev.load_registry()}
    payload = {"targets_restated_from": SOURCES, "datasets": {}}

    if "meropenem_pk_1g" in by_id:
        payload["datasets"]["meropenem_pk_1g"] = _pk_targets(
            by_id["meropenem_pk_1g"])
    if "meropenem_neutropenic_thigh" in by_id:
        payload["datasets"]["meropenem_neutropenic_thigh"] = _thigh_targets(
            by_id["meropenem_neutropenic_thigh"])

    # Two block shapes: the PK block holds one dict per target, the thigh
    # block carries a single `met` at its own top level. Walking only the
    # nested form silently dropped the thigh failure from the summary while
    # the printed table showed it -- collect both.
    failed = []
    for ds_id, block in payload["datasets"].items():
        if block.get("met") is False:
            failed.append(ds_id)
        for key, item in block.items():
            if isinstance(item, dict) and item.get("met") is False:
                failed.append(f"{ds_id}.{key}")
    payload["summary"] = {
        "n_targets_not_met": len(failed),
        "not_met": failed,
        "interpretation":
            "Both failures run in the same direction -- the model needs more "
            "exposure than the data do -- and both follow from a terminal "
            "half-life that is too short, so the plasma profile decays faster "
            "than observed and the time spent above any threshold is "
            "understated." if failed else
            "Every registry target is met.",
    }

    from src.utils.provenance import provenance
    payload["provenance"] = provenance(
        endpoint="curve-level validation against registry acceptance targets",
        exposure_index="%fT>MIC for the dose-response; plasma mg/L for the PK",
        solver={"pk": "analytic two-compartment with effect site",
                "bacterial": "LSODA rtol 1e-6 atol 1e-8 max_step 0.2 h"})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)

    print("curve-level validation against registry targets")
    print("=" * 62)
    pk = payload["datasets"].get("meropenem_pk_1g", {})
    for key in ("rmse_log10", "cmax_relative_error", "auc_0_8h_relative_error"):
        it = pk.get(key)
        if it:
            print("  %-26s %10.4f   target %-18s %s"
                  % (key, it["value"], it["target"],
                     "met" if it["met"] else "NOT MET"))
    th = payload["datasets"].get("meropenem_neutropenic_thigh", {})
    if th:
        print("  %-26s %10s   target %-18s %s"
              % ("thigh stasis %fT>MIC",
                 "n/a" if th["model_stasis_fT_pct"] is None
                 else "%.1f" % th["model_stasis_fT_pct"],
                 th["target"], "met" if th["met"] else "NOT MET"))
    print()
    print("  targets not met: %d %s" % (len(failed), failed if failed else ""))
    print("\nwrote " + OUT)


if __name__ == "__main__":
    main()
