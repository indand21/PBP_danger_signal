"""What it would cost to drop the PBP-3 exclusivity of the filament term.

The model gives the kill-independent release term to ceftazidime only, on the
grounds that PBP-3 specificity produces the filamentous lesion. Tsuji 2003
reports that carbapenems also filament at sub-inhibitory concentrations and
also release heavily there: moving from 10x to 0.1x MIC raised free LPS
88-fold for doripenem and 48-fold for meropenem, against 5.7-fold for
ceftazidime. On that reading the kill-independent pathway is a sub-inhibitory
REGIME effect that both classes enter, not an agent property.

Nothing here estimates meropenem's filament coefficient, which no experiment
in this study's evidence base identifies. The question asked is narrower and
answerable: if meropenem carried a filament yield that is some fraction f of
ceftazidime's, how much of this study's output would move? The fraction is
swept from 0 (the adopted structure) to 1 (full equality, which no source
supports and which the matched-killing pairs argue against).

NO DEFAULT IS CHANGED. Writes results/danger_signal/filament_structure.json.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.ds_phase4_analyses as p4
from src.utils.provenance import provenance

#: Fractions of the ceftazidime filament coefficient given to meropenem.
FRACTIONS = (0.0, 0.25, 0.5, 1.0)
#: Matched exposure targets, as in Table 1.
TARGETS = (40.0, 60.0, 80.0, 95.0)
#: Sub-MIC multiples at which the dual-pathway comparison is made.
SUB_MIC = (0.5, 1.0)


def matched_exposure(fraction):
    """Table 1's doses and cost ratio with meropenem given a filament yield."""
    yf_mero = fraction * p4.Y_FILAMENT_CAZ
    rows = []
    for target in TARGETS:
        doses, costs, kills = {}, {}, {}
        for agent in ("ceftazidime", "meropenem"):
            dose = p4.dose_for_target_ft(agent, target, 8.0, 6)
            if dose is None:
                break
            sim = p4.simulate(agent, dose, 8.0, 6)
            yf = yf_mero if agent == "meropenem" else p4.AGENT_FILAMENT[agent]
            _, summary = p4.ds_cost(sim, agent=agent, Yf=yf)
            doses[agent] = dose
            costs[agent] = summary["AUC_DS_pg_h_mL"]
            kills[agent] = sim["net_log10"]
        if len(doses) != 2:
            rows.append({"target_fT_pct": target, "evaluated": False})
            continue
        rows.append({
            "target_fT_pct": target,
            "evaluated": True,
            "kill_difference_log10": kills["ceftazidime"] - kills["meropenem"],
            "cost_ratio_at_nominal_yield": costs["ceftazidime"] / costs["meropenem"],
        })
    return rows


def sub_mic_release(fraction):
    """Where the relaxed structure actually bites: below the MIC."""
    yf_mero = fraction * p4.Y_FILAMENT_CAZ
    out = {}
    for mult in SUB_MIC:
        caz = p4.const_exposure(mult, p4.AGENT_YIELD["ceftazidime"], p4.Y_FILAMENT_CAZ)
        mero = p4.const_exposure(mult, p4.AGENT_YIELD["meropenem"], yf_mero)
        out[f"{mult}xMIC"] = {
            "ceftazidime_AUC_DS": caz["AUC_DS_pg_h_mL"],
            "meropenem_AUC_DS": mero["AUC_DS_pg_h_mL"],
            "cost_ratio": caz["AUC_DS_pg_h_mL"] / mero["AUC_DS_pg_h_mL"],
        }
    return out


def main():
    # Same initialisation as ds_phase4_analyses.main, so the cost model and the
    # exposure denominator are identical to the ones Table 1 is computed with.
    from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf
    p4.MIC_EFFECT = p4.effect_site_mic()
    feasible = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(feasible, key=lambda e: e["cost"]))["params"]

    scenarios = {}
    for f in FRACTIONS:
        scenarios[f"mero_filament_{f:g}x_caz"] = {
            "meropenem_Y_filament_fg_per_CFU_per_h": f * p4.Y_FILAMENT_CAZ,
            "matched_exposure": matched_exposure(f),
            "sub_mic": sub_mic_release(f),
        }

    base = scenarios["mero_filament_0x_caz"]
    full = scenarios["mero_filament_1x_caz"]

    def ratios(block):
        return [r["cost_ratio_at_nominal_yield"] for r in block["matched_exposure"]
                if r.get("evaluated")]

    out = {
        "status": "structural sensitivity; no default changed",
        "question": ("The filament term is assigned to PBP-3 agents alone. Tsuji 2003 "
                     "shows carbapenems filament and release heavily below the MIC too. "
                     "What would relaxing the exclusivity cost?"),
        "fraction_meaning": ("Fraction of the ceftazidime filament coefficient "
                             "(1.5 fg/CFU/h) given to meropenem. Not an estimate: no "
                             "source in this study's evidence base identifies it."),
        "scenarios": scenarios,
        "summary": {
            "matched_exposure_cost_ratio_range_at_f0": [min(ratios(base)), max(ratios(base))],
            "matched_exposure_cost_ratio_range_at_f1": [min(ratios(full)), max(ratios(full))],
            "worst_case_relative_change_matched": max(
                abs(a - b) / b for a, b in zip(ratios(full), ratios(base))),
            "sub_mic_cost_ratio_0.5xMIC": {
                "f0": base["sub_mic"]["0.5xMIC"]["cost_ratio"],
                "f1": full["sub_mic"]["0.5xMIC"]["cost_ratio"]},
        },
        "interpretation": (
            "This assumption is load-bearing, more so than expected. A matched "
            "%fT>MIC target of 40% leaves 60% of each interval BELOW the MIC, so the "
            "filament window is open for much of the matched-exposure comparison too, "
            "not only in failing arms. Giving meropenem the full ceftazidime "
            "coefficient narrows the matched-exposure cost ratio by up to 35% at the "
            "nominal yield, and collapses the 0.5x MIC ratio from about 35 to about "
            "1.1. The ORDERING survives every fraction tested and the matched-exposure "
            "ratio stays several-fold, so no conclusion reverses; but the claim that "
            "the kill-independent pathway is PBP-3-exclusive is doing real work in the "
            "reported magnitudes, and the evidence for exclusivity is weak. The "
            "sub-MIC regime is where this is decided."),
        "not_done": ("Meropenem's filament coefficient is not estimated and no default "
                     "is changed. Doing so needs an experiment measuring release and "
                     "viable counts together across sub-inhibitory exposures for both "
                     "agents, which is prediction 1 of the manuscript extended below "
                     "the MIC."),
        "provenance": provenance(),
    }
    path = ROOT / "results" / "danger_signal" / "filament_structure.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("matched-exposure cost ratio, f=0:", out["summary"]["matched_exposure_cost_ratio_range_at_f0"])
    print("matched-exposure cost ratio, f=1:", out["summary"]["matched_exposure_cost_ratio_range_at_f1"])
    print("worst relative change (matched):", out["summary"]["worst_case_relative_change_matched"])
    print("0.5xMIC cost ratio f=0 -> f=1:", out["summary"]["sub_mic_cost_ratio_0.5xMIC"])


if __name__ == "__main__":
    main()
