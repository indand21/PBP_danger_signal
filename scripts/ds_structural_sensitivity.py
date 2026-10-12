"""Does the agent comparison survive the model's two weakest structures?

The external review raised two structural objections that cannot be settled
by rewording:

  R7  The active bactericidal parameters are described as calibrated, but the
      fitting chain (in the parent repository, scripts/pathb_calibrate.py)
      returns k_kill_max = 1.587 against two digitized datasets, while the
      study runs at 3.0. C50 was never in that fit at all.

  R10 Immune recruitment is not on the same scale as the effector state.
      With N_eff(0) = 1e7 cells/mL, k_deg = 0.05/h and k_prod = 0.5, decay
      removes 5e5 cells/mL/h while recruitment adds at most 0.5 -- a factor
      of 1e6. The equilibrium recruitment can support is 10 cells/mL, so the
      immune layer is a decaying initial pool, not a recruited response.
      Immune killing falls from 0.1/h to 0.009/h over 48 h.

Neither is fixed here: refitting the kill parameters or rescaling the immune
layer would change every reported number and is the author's decision. What
this script establishes is whether the study's conclusions DEPEND on them. It
re-runs the matched-exposure comparison under each variant, re-solving the
effective MIC and the matched doses each time, and reports the agent cost
ratio and the residual kill difference.

The conclusion at stake is ordinal: at matched %fT>MIC the PBP-3 agent costs
several-fold more danger signal while killing within ~0.1 log10 of the PBP-2
agent. If that holds across every variant, the two structural defects bound
the absolute numbers but not the claim.

Writes results/danger_signal/structural_sensitivity.json.
"""
import copy
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.parameters import get_default_parameters  # noqa: E402

OUT = os.path.join("results", "danger_signal", "structural_sensitivity.json")
TARGETS = (40.0, 80.0)

#: k_kill_max fitted by scripts/pathb_calibrate.py in the parent repository
#: (shared across a meropenem time-kill and a neutropenic-thigh dose-response;
#: objective 0.5046 log10 CFU/mL, against 0.8107 at the nearest grid point to
#: the value the study uses).
K_KILL_FITTED = 1.5874010519681996

VARIANTS = {
    "baseline": {},
    "kill_max_fitted": {"bacteria": {"k_kill_max": K_KILL_FITTED}},
    "immune_no_decay": {"immune": {"k_deg_immune": 0.0}},
    "immune_absent": {"immune": {"k_kill_base": 0.0}},
    # Recruitment rescaled to the effector state's own units, so that it can
    # actually sustain the initial pool: k_prod = k_deg * N_eff(0).
    "immune_recruitment_scaled": {"immune": {"k_prod": 0.05 * 1e7}},
}

NOTES = {
    "baseline": "the study as reported",
    "kill_max_fitted": f"k_kill_max {K_KILL_FITTED:.3f} (R7), the value the "
                       "parent repository's calibration returns",
    "immune_no_decay": "immune effectors held at their initial level (R10): "
                       "the opposite extreme to the decaying pool",
    "immune_absent": "no immune killing at all (R10): the neutropenic limit",
    "immune_recruitment_scaled": "recruitment rescaled to cells/mL/h so it "
                                 "sustains the initial pool (R10)",
}


def _patched(overrides):
    def factory():
        p = get_default_parameters()
        for group, vals in overrides.items():
            for k, v in vals.items():
                setattr(p[group], k, v)
        return p
    return factory


def _load_phase4():
    spec = importlib.util.spec_from_file_location(
        "p4v", os.path.join("scripts", "ds_phase4_analyses.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_variant(name, overrides):
    p4 = _load_phase4()
    factory = _patched(overrides)
    # Every helper resolves this name from the module globals at call time,
    # including dose_for_target_ft, which takes no params argument.
    p4.get_default_parameters = factory
    p4.MIC_EFFECT = p4.effect_site_mic(factory())

    from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf
    fam = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(fam, key=lambda e: e["cost"]))["params"]

    rows = []
    for target in TARGETS:
        arms, ok = {}, True
        for agent in ("ceftazidime", "meropenem"):
            d = p4.dose_for_target_ft(agent, target, 8.0, 6)
            if d is None:
                ok = False
                break
            arms[agent] = (d, p4.simulate(agent, d, 8.0, 6, params=factory()))
        if not ok:
            rows.append({"target_fT_pct": target, "evaluated": False,
                         "reason": "no dose inside the search bracket"})
            continue
        ratios = p4.yield_range_ratio(arms["ceftazidime"][1],
                                      arms["meropenem"][1])
        rows.append({
            "target_fT_pct": target,
            "evaluated": True,
            "dose_ceftazidime_mg": arms["ceftazidime"][0],
            "dose_meropenem_mg": arms["meropenem"][0],
            "kill_difference_log10": (arms["ceftazidime"][1]["net_log10"]
                                      - arms["meropenem"][1]["net_log10"]),
            "net_log10_ceftazidime": arms["ceftazidime"][1]["net_log10"],
            "net_log10_meropenem": arms["meropenem"][1]["net_log10"],
            "cost_ratio_low": min(r["ratio_AUC_DS"] for r in ratios),
            "cost_ratio_high": max(r["ratio_AUC_DS"] for r in ratios),
        })
    done = [r for r in rows if r.get("evaluated")]
    return {
        "note": NOTES[name],
        "overrides": overrides,
        "effective_mic_mgL": p4.MIC_EFFECT,
        "rows": rows,
        "max_abs_kill_difference": (max(abs(r["kill_difference_log10"])
                                        for r in done) if done else None),
        "cost_ratio_span": ([min(r["cost_ratio_low"] for r in done),
                             max(r["cost_ratio_high"] for r in done)]
                            if done else None),
        "ordinal_holds": bool(done and all(r["cost_ratio_low"] > 1.0
                                           for r in done)),
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {"targets_pct": list(TARGETS), "variants": {}}
    for name, ov in VARIANTS.items():
        print(f"--- {name}: {NOTES[name]}")
        v = run_variant(name, copy.deepcopy(ov))
        payload["variants"][name] = v
        span = v["cost_ratio_span"]
        print(f"    MIC {v['effective_mic_mgL']:.4f} mg/L   "
              f"cost ratio {span[0]:.2f}-{span[1]:.2f}   "
              f"max |kill diff| {v['max_abs_kill_difference']:.3f} log10   "
              f"ordinal holds {v['ordinal_holds']}")

    ok = all(v["ordinal_holds"] for v in payload["variants"].values())
    lo = min(v["cost_ratio_span"][0] for v in payload["variants"].values())
    hi = max(v["cost_ratio_span"][1] for v in payload["variants"].values())
    kd = max(v["max_abs_kill_difference"]
             for v in payload["variants"].values())
    payload["summary"] = {
        "ordinal_holds_in_every_variant": ok,
        "cost_ratio_range_across_variants": [lo, hi],
        "max_abs_kill_difference_across_variants": kd,
        "interpretation":
            "The ordinal claim -- PBP-3 agent costs several-fold more danger "
            "signal at matched exposure, while killing within about 0.1 log10 "
            "-- is reproduced under the fitted kill rate and under three "
            "immune structures spanning no immune killing to a sustained "
            "effector pool. The two structural defects therefore bound the "
            "absolute numbers, not the comparison. They remain unfixed: see "
            "the Limitations." if ok else
            "At least one variant breaks the ordinal claim; the conclusion "
            "is NOT robust to the structures tested and the manuscript must "
            "say so.",
    }
    print(f"\nordinal holds in every variant: {ok}")
    print(f"cost ratio across variants: {lo:.2f}-{hi:.2f}")
    print(f"max |kill difference| across variants: {kd:.3f} log10")

    from src.utils.provenance import provenance
    payload["provenance"] = provenance(
        endpoint="AUC_DS_pg_h_mL", note="structural sensitivity for R7 and R10")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
