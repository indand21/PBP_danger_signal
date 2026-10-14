"""Conditional Table 1 sensitivity to the exploratory multiarm PD fits.

Run after ds_multiarm_refit.py. Keep in-vivo growth and immune parameters at
the adopted defaults. Linear PK makes every matched-fT dose scale with the
effective MIC; rerun the bacterial/danger-signal dynamics at those doses.
The reading-perturbation ranges are conditional sensitivity ranges, not CIs.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.parameters import get_default_parameters
from src.utils.provenance import provenance
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf

INPUT = ROOT / "results/danger_signal/multiarm_refit.json"
REFERENCE = ROOT / "results/danger_signal/phase4_analyses.json"
OUT = ROOT / "results/danger_signal/multiarm_downstream.json"


def load_phase4():
    spec = importlib.util.spec_from_file_location("multiarm_p4", ROOT / "scripts/ds_phase4_analyses.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def factory(params):
    def patched():
        p = get_default_parameters()
        for name in ["k_kill_max", "kill_C50", "kill_hill"]:
            setattr(p["bacteria"], name, params[name])
        return p
    return patched


def main():
    fits = json.loads(INPUT.read_text(encoding="utf-8"))
    ref = json.loads(REFERENCE.read_text(encoding="utf-8"))
    old_mic = ref["meta"]["mic_effect_site_mgL"]
    reference_rows = ref["analysis3_matched_exposure"]["rows"]
    fam = [e for e in calibrate_family() if e["feasible"]]
    signal = calibrate_tnf(min(fam, key=lambda e:e["cost"]))["params"]
    variants = {}
    labels = ["primary", "0.5x_assumed_at_native_control", "0.5x_assumed_at_centrifuged_control"]
    for label in labels:
        p4 = load_phase4()
        pd_fit = fits["scenarios"][label]["parameters"]
        patched = factory(pd_fit)
        p4.get_default_parameters = patched
        p4.MIC_EFFECT = p4.effect_site_mic(patched())
        p4.P_REF = signal
        factor = p4.MIC_EFFECT / old_mic
        rows = []
        for row in reference_rows:
            if not row.get("evaluated"): continue
            sims, doses = {}, {}
            for agent in ["ceftazidime", "meropenem"]:
                dose = row["arms"][agent]["dose_mg"] * factor
                doses[agent] = dose
                sims[agent] = p4.simulate(agent,dose,8.,6,params=patched())
                assert abs(sims[agent]["fT"]-row["arms"][agent]["achieved_fT_pct"]) < .15
            ratios = p4.yield_range_ratio(sims["ceftazidime"],sims["meropenem"])
            rows.append({"target_fT_pct":row["target_fT_pct"],"dose_mg_q8h":doses,
                         "achieved_fT_pct":{a:s["fT"] for a,s in sims.items()},
                         "net_log10_change":{a:s["net_log10"] for a,s in sims.items()},
                         "kill_difference_log10":sims["ceftazidime"]["net_log10"]-sims["meropenem"]["net_log10"],
                         "cost_ratio_range":ratios})
        variants[label] = {"parameters":pd_fit,"effective_mic_mgL":p4.MIC_EFFECT,
                           "dose_multiplier_vs_Table1":factor,"rows":rows}
        print(label,json.dumps(variants[label]),flush=True)

    # This range describes repeated reading perturbations of the primary
    # readable-arm subset only. The separate ambiguous-arm variants above
    # represent a different, larger source of uncertainty.
    p4 = load_phase4()
    reading_mics = []
    for fit in fits["reading_perturbations"]["fits"]:
        reading_mics.append(p4.effect_site_mic(factory(fit["parameters"])()))
    scale = np.asarray(reading_mics) / old_mic
    payload = {"status":"conditional sensitivity; no defaults or Table 1 replaced",
               "analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "input_fit_sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest(),
               "reference_results_sha256":hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               "reference_effective_mic_mgL":old_mic,
               "fixed": "Adopted in-vivo growth, immune recruitment, PK, signal model and yield range",
               "interpretation": "No statistical confidence level. Reading perturbations omit "
                                 "replicate noise, shared digitisation error and structural uncertainty. "
                                 "Do not interpret a best fit to noisy raster summaries as a measured constant.",
               "variants":variants,
               "primary_reading_perturbation_sensitivity":{
                   "n":len(reading_mics),
                   "effective_mic_range_mgL":[min(reading_mics),max(reading_mics)],
                   "dose_multiplier_range":[float(scale.min()),float(scale.max())]},
               "provenance":provenance(note="Conditional downstream propagation of multiarm fits")}
    OUT.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("SAVED",OUT,flush=True)


if __name__=="__main__":
    main()
