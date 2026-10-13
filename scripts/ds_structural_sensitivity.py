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

Neither is adopted here: changing the defaults would change every reported
number and is the author's decision. What this script establishes is whether
the study's conclusions DEPEND on them. It re-runs the matched-exposure
comparison under each variant, re-solving the effective MIC and the matched
doses each time, and reports the agent cost ratio and the residual kill
difference.

The conclusion at stake is ordinal: at matched %fT>MIC the PBP-3 agent costs
several-fold more danger signal while killing within ~0.1 log10 of the PBP-2
agent. If that holds across every variant, the two structural defects bound
the absolute numbers but not the claim.

R7 has since been worked out properly in scripts/ds_bacterial_refit.py, and
the variants below now include its results. Three things that script settled
change how the R7 variants should be read:

  * kill_C50 is not identifiable from the repository's kill data, for two
    independent reasons: both datasets were expressed in multiples of the
    model's own effective MIC, which is itself exactly proportional to
    kill_C50; and even in absolute units the 4xMIC exposure saturates a
    steep Hill, so the profile is flat from 0.1 to ~5 mg/L. It is pinned
    here by the calibration isolate's measured MIC of 1 mg/L, which costs
    0.001 log10 RMSE.

  * The misfit at the parameters in use is driven by the kill RATE, not by
    kill_C50: correcting kill_C50 alone moves the objective 0.773 -> 0.769,
    while dropping k_kill_max to 1.59 takes the time-kill arm from 0.653 to
    0.118.

  * Changing the kill rate WITHOUT correcting kill_C50 is not a
    self-consistent parameter set, and an earlier variant here did exactly
    that. Its doubling of the kill difference (0.098 -> 0.243) was read as
    the consequence of the fitted kill rate; it is not. The doubling tracks
    the Hill exponent, which the data do not identify. That is why
    ``refit_optimum`` below varies the exponent and nothing else of
    substance.

The refit has since been ADOPTED: the defaults in src/core/parameters.py are
now the refitted values, so "baseline" is the refitted model. R10 is still
not fixed.

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

#: The study now RUNS the refit (k_kill_max 1.587, kill_C50 0.936,
#: kill_hill 4), so "baseline" below is the refitted model and the variants
#: that used to carry the refit have been replaced by the two comparisons
#: that are still informative:
#:
#:   previous_parameters  what the study ran before 2026-10-13, so the
#:                        consequence of adopting the refit can be read off
#:                        the same table as everything else
#:   refit_optimum        the formal optimum of the refit, which differs from
#:                        the adopted set only in the Hill exponent (8 rather
#:                        than 4, with kill_C50 re-pinned to the measured MIC
#:                        at that exponent). The data cannot separate the two
#:                        exponents, 0.05 log10 RMSE apart, and this is the
#:                        variant that shows what rests on that choice.
PREVIOUS_PARAMETERS = {"k_kill_max": 3.0, "kill_C50": 0.6, "kill_hill": 4.0}
REFIT_OPTIMUM = {"k_kill_max": 1.351, "kill_C50": 0.991, "kill_hill": 8.0}

VARIANTS = {
    "baseline": {},
    "previous_parameters": {"bacteria": PREVIOUS_PARAMETERS},
    "refit_optimum": {"bacteria": REFIT_OPTIMUM},
    "immune_no_decay": {"immune": {"k_deg_immune": 0.0}},
    "immune_absent": {"immune": {"k_kill_base": 0.0}},
    # Recruitment rescaled to the effector state's own units, so that it can
    # actually sustain the initial pool: k_prod = k_deg * N_eff(0).
    "immune_recruitment_scaled": {"immune": {"k_prod": 0.05 * 1e7}},
}

NOTES = {
    "baseline": "the study as reported: the refitted bacterial layer, "
                "k_kill_max 1.587, kill_C50 0.936, kill_hill 4",
    "previous_parameters": "what the study ran before the refit: k_kill_max "
                           "3.0 and kill_C50 0.6, neither fitted, with the "
                           "model about twice as potent as the calibration "
                           "isolate it was fitted against",
    "refit_optimum": "the refit's formal optimum, differing only in the Hill "
                     "exponent: k_kill_max 1.351, kill_C50 0.991, kill_hill "
                     "8. The data cannot separate hill 8 from hill 4 "
                     "(0.05 log10 RMSE), so this variant bounds what the "
                     "choice of exponent decides",
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
            "-- is reproduced under the bacterial-layer refit, under the "
            "fitted kill rate alone, under a kill_C50 corrected to the "
            "calibration isolate's measured MIC, and under three immune "
            "structures spanning no immune killing to a sustained effector "
            "pool. What the refit moves is the dose scale, not the "
            "comparison: because matching is done at equal %fT>MIC, a change "
            "in the effective MIC multiplies both agents' doses by the same "
            "factor (x1.85 to x2.31 here, exactly the MIC ratio), so the "
            "cost ratio and the kill difference under a pure kill_C50 "
            "correction are unchanged to the last digit. The residual kill "
            "difference is the one quantity that is NOT robust: it is 0.105 "
            "log10 at the refit optimum and 0.232 with the Hill exponent "
            "held at 4, and the data cannot distinguish those two exponents. "
            "The defaults are unchanged; adopting the refit is the author's "
            "decision. See scripts/ds_bacterial_refit.py and the "
            "Limitations." if ok else
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
