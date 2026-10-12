"""Phase 2 of the danger-signal study: build and calibrate the layers, then
re-run the stop-gates against the CALIBRATED model.

The Phase 1 pilot cleared the gates using an assumed cytokine dose-response.
Phase 2 replaces the assumption with a fit to human experimental endotoxemia
(Hijma 2020, two intravenous LPS doses, one study and assay), adds the
transduction state that the calibration showed to be structurally necessary,
and builds the ordinal injury layer. The gates are then re-run -- because a
calibrated layer is a different layer, and a gate passed by the prototype is
not a gate passed by the model.

Sections:
  1. Layer-2 calibration and its identifiability
  2. GATE 1 re-run: operating point, now at calibrated parameters
  3. GATE 2 re-run: smoothness in Y_lysis across the calibrated family
  4. Agent comparison at matched killing, with calibrated amplification
  5. Layer 3: ordinal injury and the rank-invariance test
  6. Extrapolation audit: how far outside the calibration range do we run?

Writes results/danger_signal/phase2_build.json.
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.cytokine_calibration import (  # noqa: E402
    DILLINGH_2014, HIJMA_2020, bolus_plasma_ds0, calibrate_family,
    check_lower_bound_consistency, observed_log_slope,
)
from src.analysis.danger_signal import (  # noqa: E402
    YIELD_ANCHORS_FG_PER_CFU, check_operating_point, check_smoothness,
    simulate_danger_signal, summarize,
)
from src.analysis.injury_layer import (  # noqa: E402
    InjuryParameters, check_rank_invariance, saturation_diagnostics,
)
from src.core.parameters import (  # noqa: E402
    get_default_parameters, get_drug_pd_parameters, get_drug_pk_parameters,
)
from src.core.pd_model import create_ode_system  # noqa: E402
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel  # noqa: E402
from src.core.simulation import run_simulation  # noqa: E402

OUT = os.path.join("results", "danger_signal", "phase2_build.json")

WEIGHT = 70.0
T_END = 48.0
IC = dict(B_rep=1e7, B_pers=1e4, B_SCV=0, N_eff=1e7,
          Damage=0, IL6=10, TNF=5, PAMP=0, D_host=0)
V_FOCUS = 10.0
Y_FILAMENT_CAZ = 1.5
Y_GRID = np.logspace(-1, 1, 31)

REGIMENS = {
    "ceftazidime": dict(drug="ceftazidime", dose_mg=2000.0, interval_h=8.0,
                        n_doses=6, inf_min=30),
    "meropenem": dict(drug="meropenem", dose_mg=1000.0, interval_h=8.0,
                      n_doses=6, inf_min=30),
}

MIC_EFFECT = None


def effect_site_mic():
    from scripts.model_qualification import effective_mic
    return float(effective_mic()[0])


def solve_bacteria(spec):
    p = get_drug_pk_parameters(spec["drug"])
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka,
                                Kp=p.Kp, effect_site_model=True)
    reg = DosingRegimen(dose_mg=spec["dose_mg"], interval_hours=spec["interval_h"],
                        start_time=0, n_doses=spec["n_doses"],
                        infusion_duration_min=spec["inf_min"])
    pd = create_ode_system(get_default_parameters())
    # Kp is a total-concentration partition; only unbound drug acts.
    fu = get_drug_pd_parameters(spec["drug"]).fraction_unbound
    r = run_simulation(pkm, reg, pd, dict(IC), t_span=(0, T_END),
                       drug_class="cidal", weight_kg=WEIGHT,
                       fraction_unbound=fu)
    t = r.t
    C_eff = r.get_free_effect_site_concentration(pk_model=pkm, weight_kg=WEIGHT)
    above = (C_eff > MIC_EFFECT).astype(float)
    dt = np.diff(t)
    ft = float(np.sum(0.5 * (above[:-1] + above[1:]) * dt) / (t[-1] - t[0]) * 100)
    _, B_total = r.get_bacterial_burden()
    return dict(t=t, B_rep=r.y[:, r.state_names.index("B_rep")], C_eff=C_eff,
                kill=np.asarray(pd.cidal_kill_rate(C_eff), dtype=float),
                net_log10=float(np.log10(max(B_total[-1], 1e-6) / B_total[0])),
                fT=ft)


def run_ds(driver, p, y_lysis, y_filament):
    q = replace(p, Y_lysis=y_lysis, Y_filament=y_filament, V_focus_mL=V_FOCUS)
    out = simulate_danger_signal(driver["t"], driver["B_rep"], driver["kill"],
                                 driver["C_eff"], MIC_EFFECT, q)
    return out, summarize(out)


def main():
    global MIC_EFFECT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    MIC_EFFECT = effect_site_mic()
    payload = {"meta": {"mic_effect_site_mgL": MIC_EFFECT, "t_end_h": T_END,
                        "V_focus_mL": V_FOCUS, "regimens": REGIMENS,
                        "initial_conditions": IC}}

    # ------------------------------------------------------- 1. calibration
    print("=" * 74)
    print("1. LAYER-2 CALIBRATION to human experimental endotoxemia")
    print("=" * 74)
    print(f"  data: {HIJMA_2020['source']}")
    print(f"        {HIJMA_2020['n_per_dose']} subjects per dose; "
          f"IL-6 peaks {HIJMA_2020['IL6_peak_pg_mL']} pg/mL at "
          f"{HIJMA_2020['t_peak_IL6_h']} h for {HIJMA_2020['doses_ng_per_kg']} ng/kg")
    slope = observed_log_slope()
    print(f"  observed local log-log slope of the IL-6 response: {slope:.2f}")
    print("  (the pilot assumed 1.0 -- i.e. it assumed a proportional response)")

    fam = calibrate_family()
    feasible = [e for e in fam if e["feasible"]]
    print("\n  DS50     hill   k_sig_off   pred IL-6 peaks    t_peak   occupancy@2ng/kg  fit")
    for e in fam:
        print(f"  {e['DS50']:>6.0f}  {e['hill']:5.3f}   {e['k_signal_off']:6.3f}/h   "
              f"{e['pred_IL6_peak'][0]:6.1f}/{e['pred_IL6_peak'][1]:6.1f}      "
              f"{e['pred_t_peak_IL6_h'][0]:4.2f} h   {e['saturation_peak_at_2ng_kg']:8.4f}      "
              f"{'OK' if e['feasible'] else 'rejected'}")
    print(f"\n  feasible DS50 range: {[e['DS50'] for e in feasible]}")
    print(f"  Hill coefficient across the feasible family: "
          f"{min(e['hill'] for e in feasible):.2f} to {max(e['hill'] for e in feasible):.2f}")

    print(f"\n  held-out check ({DILLINGH_2014['source']}, different laboratory,"
          f" not used in the fit):")
    foot_ok = True
    for e in feasible:
        c = check_lower_bound_consistency(e)
        ok = c["detectable_at_0.5"] and c["quiet_at_0.2"]
        foot_ok &= ok
        print(f"    DS50={e['DS50']:>6.0f}  IL-6 rise {c['IL6_rise_at_0.2_ng_kg']:5.2f} pg/mL "
              f"at 0.2 ng/kg (should be quiet), {c['IL6_rise_at_0.5_ng_kg']:5.2f} at "
              f"0.5 ng/kg (should be detectable)  -> {'pass' if ok else 'FAIL'}")

    ref = min(feasible, key=lambda e: e["cost"])
    p_ref = ref["params"]
    print(f"\n  REFERENCE parameterization (best fit): DS50={ref['DS50']:.0f} pg/mL, "
          f"hill={ref['hill']:.3f}, k_signal_off={ref['k_signal_off']:.3f}/h, "
          f"k_IL6_max={ref['k_IL6_max']:.0f} pg/mL/h")

    payload["calibration"] = {
        "sources": {"fit": HIJMA_2020["source"], "held_out": DILLINGH_2014["source"]},
        "observed_log_slope": slope,
        "family": [{k: v for k, v in e.items() if k != "params"} for e in fam],
        "feasible_DS50": [e["DS50"] for e in feasible],
        "hill_range": [min(e["hill"] for e in feasible),
                       max(e["hill"] for e in feasible)],
        "held_out_pass": bool(foot_ok),
        "reference": {k: v for k, v in ref.items() if k != "params"},
    }

    # --------------------------------------------------- bacterial drivers
    drivers = {k: solve_bacteria(v) for k, v in REGIMENS.items()}
    print("\n  bacterial drivers (unchanged from Phase 1):")
    for k, d in drivers.items():
        print(f"    {k:12s} %fT>MIC {d['fT']:5.1f}   48 h net {d['net_log10']:+.2f} log10")

    # ------------------------------------------------------------ 2. GATE 1
    print("\n" + "=" * 74)
    print("2. GATE 1 re-run -- operating point at CALIBRATED parameters")
    print("=" * 74)
    g1_cells, g1_pass = [], True
    for e in feasible:
        for agent in ("ceftazidime", "meropenem"):
            yl = YIELD_ANCHORS_FG_PER_CFU[agent]
            yf = Y_FILAMENT_CAZ if agent == "ceftazidime" else 0.0
            _, s = run_ds(drivers[agent], e["params"], yl, yf)
            chk = check_operating_point(s["saturation_peak"])
            g1_cells.append(dict(DS50=e["DS50"], agent=agent,
                                 DS_peak_pg_mL=s["DS_peak_pg_mL"],
                                 IL6_peak_pg_mL=s["IL6_peak_pg_mL"], **chk))
            if agent == "ceftazidime" and not chk["below_saturation"]:
                g1_pass = False
            print(f"  DS50={e['DS50']:>6.0f}  {agent:12s} DS peak {s['DS_peak_pg_mL']:8.2f} pg/mL  "
                  f"occupancy {s['saturation_peak']:.5f}  IL-6 peak {s['IL6_peak_pg_mL']:8.1f} pg/mL")
    n_sat = sum(1 for c in g1_cells if not c["below_saturation"])
    print(f"\n  saturated cells: {n_sat}/{len(g1_cells)}  ->  GATE 1 "
          f"{'PASS' if g1_pass else 'FAIL'}")
    payload["gate1"] = {"cells": g1_cells, "n_saturated": n_sat, "pass": bool(g1_pass)}

    # ------------------------------------------------------------ 3. GATE 2
    print("\n" + "=" * 74)
    print("3. GATE 2 re-run -- smoothness in Y_lysis across the calibrated family")
    print("=" * 74)
    g2_cells, g2_pass = [], True
    base = drivers["ceftazidime"]
    for e in feasible:
        auc = np.array([run_ds(base, e["params"], y, 0.0)[1]["AUC_IL6_excess_pg_h_mL"]
                        for y in Y_GRID])
        chk = check_smoothness(Y_GRID, auc)
        g2_cells.append(dict(DS50=e["DS50"], **chk))
        g2_pass &= bool(chk["monotone"] and chk["no_step_function"])
        print(f"  DS50={e['DS50']:>6.0f}  monotone={chk['monotone']}  "
              f"no_step={chk['no_step_function']}  "
              f"largest step {chk['max_step_fraction_of_span']:.3f} of span  "
              f"dynamic range {chk['dynamic_range']:.0f}x")
    print(f"\n  GATE 2 {'PASS' if g2_pass else 'FAIL'}")
    payload["gate2"] = {"cells": g2_cells, "pass": bool(g2_pass)}

    # ----------------------------------------------------- 4. agent contrast
    print("\n" + "=" * 74)
    print("4. AGENT COMPARISON at matched killing, calibrated layer")
    print("=" * 74)
    arms_by_ds50, comp = {}, {}
    for e in feasible:
        arms = {}
        row = {}
        for agent in ("ceftazidime", "meropenem"):
            yl = YIELD_ANCHORS_FG_PER_CFU[agent]
            yf = Y_FILAMENT_CAZ if agent == "ceftazidime" else 0.0
            out, s = run_ds(drivers[agent], e["params"], yl, yf)
            arms[agent] = out
            row[agent] = s
        arms_by_ds50[e["DS50"]] = arms
        r_ds = (row["ceftazidime"]["AUC_DS_pg_h_mL"]
                / row["meropenem"]["AUC_DS_pg_h_mL"])
        r_il6 = (row["ceftazidime"]["AUC_IL6_excess_pg_h_mL"]
                 / row["meropenem"]["AUC_IL6_excess_pg_h_mL"])
        comp[e["DS50"]] = {"agents": row, "ratio_AUC_DS": r_ds,
                           "ratio_AUC_IL6_excess": r_il6}
        print(f"  DS50={e['DS50']:>6.0f}  AUC_DS ratio {r_ds:6.2f}x   "
              f"AUC_IL6 ratio {r_il6:8.2f}x   "
              f"(CAZ IL-6 peak {row['ceftazidime']['IL6_peak_pg_mL']:7.1f}, "
              f"MERO {row['meropenem']['IL6_peak_pg_mL']:6.1f} pg/mL)")
    d_kill = drivers["ceftazidime"]["net_log10"] - drivers["meropenem"]["net_log10"]
    print(f"\n  difference in 48 h kill: {d_kill:+.2f} log10 "
          f"(the comparison is at matched killing by construction)")
    print(f"  input yield ratio: "
          f"{YIELD_ANCHORS_FG_PER_CFU['ceftazidime'] / YIELD_ANCHORS_FG_PER_CFU['meropenem']:.1f}x")
    print("  NOTE: the measured Hill slope (~1.7) AMPLIFIES the yield ratio rather")
    print("        than transmitting it, because the layer runs below saturation.")
    payload["agent_comparison"] = {
        str(k): {"ratio_AUC_DS": v["ratio_AUC_DS"],
                 "ratio_AUC_IL6_excess": v["ratio_AUC_IL6_excess"],
                 "agents": v["agents"]} for k, v in comp.items()}
    payload["agent_comparison"]["delta_net_log10_kill"] = d_kill

    # --------------------------------------------------------- 5. Layer 3
    print("\n" + "=" * 74)
    print("5. LAYER 3 -- ordinal injury and rank invariance")
    print("=" * 74)
    inj = {}
    for ds50, arms in arms_by_ds50.items():
        res = check_rank_invariance(arms, metric="peak")
        sat = saturation_diagnostics(arms)
        inj[ds50] = {"rank": res, "saturation": sat}
        print(f"  DS50={ds50:>6.0f}  invariant={res['invariant']}  "
              f"distinct rankings={res['n_distinct_rankings']}  "
              f"consensus={res['consensus_ranking']}")
        for label, d in sat.items():
            print(f"      {label:12s} peak IL-6/ref {d['max_x_over_ref']:7.3f}  "
                  f"hill1 occupancy {d['hill1_peak_occupancy']:.3f}  "
                  f"hill2 {d['hill2_peak_occupancy']:.3f}")
    all_inv = all(v["rank"]["invariant"] for v in inj.values())
    print(f"\n  ordinal claim licensed (2 arms): {all_inv}")

    # The two literature anchors sit in two widely separated clusters
    # (ceftazidime/piperacillin ~5 fg/CFU, imipenem/meropenem/isepamicin
    # ~0.25), which is an easy test: any monotone map separates them. The real
    # question is whether a SATURATING map can still rank an INTERMEDIATE
    # agent, since both high arms may already be pegged. A synthetic
    # intermediate yield (the geometric mean) stresses exactly that.
    y_lo = YIELD_ANCHORS_FG_PER_CFU["meropenem"]
    y_hi = YIELD_ANCHORS_FG_PER_CFU["ceftazidime"]
    y_mid = float(np.sqrt(y_lo * y_hi))
    arms3 = {}
    for label, y in (("low_yield", y_lo), ("mid_yield", y_mid),
                     ("high_yield", y_hi)):
        out, _ = run_ds(drivers["ceftazidime"], p_ref, y, 0.0)
        arms3[label] = out
    res3 = check_rank_invariance(arms3, metric="peak")
    sat3 = saturation_diagnostics(arms3)
    print(f"\n  3-arm stress test (yields {y_lo:.2f} / {y_mid:.2f} / {y_hi:.2f} fg/CFU,"
          f" same regimen):")
    print(f"    invariant={res3['invariant']}  distinct={res3['n_distinct_rankings']}  "
          f"consensus={res3['consensus_ranking']}")
    for label, d in sat3.items():
        print(f"      {label:11s} peak IL-6/ref {d['max_x_over_ref']:8.3f}  "
              f"hill1 occupancy {d['hill1_peak_occupancy']:.3f}  "
              f"hill2 {d['hill2_peak_occupancy']:.3f}")
    if not res3["invariant"]:
        print("    -> rankings differ by map:")
        for m, rk in res3["rankings"].items():
            print(f"       {m:16s} {rk}")
    all_inv = all_inv and bool(res3["invariant"])
    print(f"\n  ordinal claim licensed (including intermediate arm): {all_inv}")
    payload["layer3"] = {
        str(k): {"invariant": v["rank"]["invariant"],
                 "n_distinct_rankings": v["rank"]["n_distinct_rankings"],
                 "consensus_ranking": v["rank"]["consensus_ranking"],
                 "rankings": v["rank"]["rankings"],
                 "saturation": v["saturation"]} for k, v in inj.items()}
    payload["layer3"]["rank_invariant_everywhere"] = bool(all_inv)
    payload["layer3"]["three_arm_stress"] = {
        "yields_fg_per_CFU": [y_lo, y_mid, y_hi],
        "invariant": res3["invariant"],
        "n_distinct_rankings": res3["n_distinct_rankings"],
        "consensus_ranking": res3["consensus_ranking"],
        "rankings": res3["rankings"],
        "saturation": sat3,
    }

    # ------------------------------------------------ 6. extrapolation audit
    print("\n" + "=" * 74)
    print("6. EXTRAPOLATION AUDIT -- is the layer used where it was calibrated?")
    print("=" * 74)
    cal_lo = bolus_plasma_ds0(DILLINGH_2014["no_response_at_ng_per_kg"], p_ref)
    cal_hi = bolus_plasma_ds0(max(HIJMA_2020["doses_ng_per_kg"]), p_ref)
    print(f"  calibrated plasma-DS range: {cal_lo:.1f} to {cal_hi:.1f} pg/mL")
    audit = {"calibrated_range_pg_mL": [cal_lo, cal_hi], "arms": {}}
    for agent, arms in [(a, arms_by_ds50[ref["DS50"]][a])
                        for a in ("ceftazidime", "meropenem")]:
        ds = arms["DS"]
        peak = float(np.max(ds))
        frac_in = float(np.mean((ds >= cal_lo) & (ds <= cal_hi)))
        print(f"  {agent:12s} DS peak {peak:8.2f} pg/mL   "
              f"{100*frac_in:5.1f}% of the time within the calibrated range   "
              f"{'WITHIN' if peak <= cal_hi else 'ABOVE'} the upper anchor")
        audit["arms"][agent] = {"DS_peak_pg_mL": peak,
                                "fraction_of_time_in_calibrated_range": frac_in,
                                "peak_within_calibrated_range": bool(peak <= cal_hi)}
    print("\n  Meropenem's danger-signal concentrations sit BELOW the calibrated")
    print("  range, so its cytokine prediction extrapolates down the Hill foot.")
    print("  The amplification factor is therefore a lower-confidence quantity")
    print("  than the ordinal ranking, which is why Layer 3 is ordinal.")
    payload["extrapolation_audit"] = audit

    # ------------------------------------------------------------- summary
    gate = bool(g1_pass and g2_pass and foot_ok and all_inv)
    print("\n" + "=" * 74)
    print(f"PHASE 2 GATE 1 (operating point, calibrated):   {'PASS' if g1_pass else 'FAIL'}")
    print(f"PHASE 2 GATE 2 (smoothness, calibrated family): {'PASS' if g2_pass else 'FAIL'}")
    print(f"HELD-OUT Layer-2 check (Dillingh):              {'PASS' if foot_ok else 'FAIL'}")
    print(f"LAYER-3 rank invariance:                        {'PASS' if all_inv else 'FAIL'}")
    print(f"=> PHASE 2 {'COMPLETE -- proceed to Phase 3' if gate else 'BLOCKED'}")
    print("=" * 74)
    payload["phase2_pass"] = gate

    from src.utils.provenance import provenance
    payload["provenance"] = provenance(
        note="historical phase; retained for the record")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
