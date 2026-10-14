"""Reassess the thigh PD benchmark with source timing and fixed strain MICs.

No global default is fitted to a desired stasis threshold. The source's
unpaired control-growth range constrains growth separately; midpoints and
endpoints are scenarios, not confidence limits. Treatment marker clusters
retain strain identity and are not presented as recovered animal replicates.
"""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src.analysis.external_validation as ev
from src.analysis.pd_context import (load_thigh_strains, growth_from_control,
    strain_scenario, CONTROL_ELAPSED_H, CONTROL_DELTA_RANGE, THIGH_CAPACITY_CFU)
from src.utils.provenance import provenance


def run(strain,g,h=4.,interval=6.):
    sc=strain_scenario(strain,g,interval_h=interval,hill=h)
    ds=ev.DigitizedDataset(id=strain["strain"],kind="dose_response",drug="meropenem",
        drug_class="cidal",role="calibration",csv="unused",y_unit="log10 CFU/thigh",
        y_space="log10",source="Sabet 2020, Tables 1,3; Figures 1-6",scenario=sc)
    x,y,mic=ev.dose_response_curve(ds)
    return {"strain":strain["strain"],"growth_h_inv":g,"hill":h,"interval_h":interval,
            "mic_reference_mgL":mic,"predicted_stasis_ft_pct":ev.stasis_index(x,y,0),
            "predicted_one_log_ft_pct":ev.stasis_index(x,y,-1),
            "predicted_two_log_ft_pct":ev.stasis_index(x,y,-2),
            "published_stasis_ft_pct":strain["stasis_ft_pct"],
            "published_one_log_ft_pct":strain["one_log_ft_pct"],
            "published_two_log_ft_pct":strain["two_log_ft_pct"],
            "index_grid":x.tolist(),"delta_grid":y.tolist()}


def main():
    strains=load_thigh_strains()
    growth_records=[];scenarios=[]
    for s in strains:
        initial=10**s["baseline_log10_cfu_thigh"]
        rates=[growth_from_control(d,initial) for d in
               [CONTROL_DELTA_RANGE[0],np.mean(CONTROL_DELTA_RANGE),CONTROL_DELTA_RANGE[1]]]
        record={"strain":s["strain"],"growth_range_h_inv":[rates[0],rates[2]],
                "midpoint_scenario_h_inv":rates[1],"control_elapsed_h":CONTROL_ELAPSED_H,
                "not_a_strain_specific_estimate":"The source reports an unpaired growth range only",
                "12h_interval_sensitivity_h_inv":growth_from_control(np.mean(CONTROL_DELTA_RANGE),initial,12.)}
        growth_records.append(record)
        for label,g in zip(["control_low","control_mid","control_high"],rates):
            row=run(s,g);row["scenario"]=label;scenarios.append(row)
        for h in [.9,8.,29.49]:
            row=run(s,rates[1],h=h);row["scenario"]=f"hill_{h}";scenarios.append(row)
        for interval in [2.,4.]:
            row=run(s,rates[1],interval=interval);row["scenario"]=f"interval_{interval}";scenarios.append(row)
        print(s["strain"],"control growth range",rates[0],rates[2],flush=True)
    summaries={}
    for label in dict.fromkeys(r["scenario"] for r in scenarios):
        rows=[r for r in scenarios if r["scenario"]==label]
        def average(key):
            vals=[r[key] for r in rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None
        summaries[label]={"mean_predicted_stasis_ft_pct":average("predicted_stasis_ft_pct"),
                          "mean_published_stasis_ft_pct":average("published_stasis_ft_pct"),
                          "mean_predicted_one_log_ft_pct":average("predicted_one_log_ft_pct"),
                          "n_strains_with_stasis":sum(r["predicted_stasis_ft_pct"] is not None for r in rows)}
    payload={"status":"context and transport sensitivity, not validated murine reconstruction",
             "control_elapsed_h":CONTROL_ELAPSED_H,"control_delta_range_log10":CONTROL_DELTA_RANGE,
             "carrying_capacity_CFU_per_thigh":THIGH_CAPACITY_CFU,
             "capacity_basis":"Context assumption above the possible observed count range, "
                              "not the broth 1e9 CFU/mL parameter or an estimate from one endpoint",
             "growth_contexts":growth_records,"scenarios":scenarios,"summaries":summaries,
             "interpretation":"Growth and measured MIC are separated. No parameter is chosen "
                              "to hit 17.3%. Missing per-point murine regimens and unpaired controls "
                              "prevent a definitive mechanistic calibration. Historical 24h-control "
                              "interpretation and moving MIC denominator are not used here.",
             "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "input_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                ["data/external_validation/sabet2020_strains.csv","data/external_validation/sabet2020_strain_points.csv"]},
             "provenance":provenance(note="Fixed measured strain MIC; source-timed control constraints")}
    out=ROOT/"results/danger_signal/pd_context.json"
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":main()
