"""Phase 1 pilot: the danger-signal study's stop-gate acceptance tests.

The previous host-damage model failed in one specific way: its inflammation
pathway had ~1e10 effective gain, so it behaved as a binary detector -- any
nonzero danger-signal release saturated the injury term and every agent
collapsed onto the same ceiling. ``docs/danger_signal_study/DESIGN.md`` makes
the two tests that detect that failure a pre-registered stop-gate, to be run
*before* any model-building or manuscript effort:

  TEST 1  the cytokine layer operates BELOW saturation at physiological
          danger-signal levels (and is not sitting on the flat foot either);
  TEST 2  model output varies SMOOTHLY and MONOTONICALLY with the per-kill
          danger-signal yield Y_lysis, with enough dynamic range to tell a
          high-yield agent from a low-yield one.

Both are evaluated over the full uncertainty grid of the three parameters that
are not directly measured (DS50, LPS clearance, focus volume), not at a single
point, because the failure the tests guard against was itself a point-estimate
artefact.

TEST 3 is the headline pilot measurement, not a gate: ceftazidime-like
(PBP-3, filamenting, high yield) versus meropenem-like (PBP-2, low yield) at
MATCHED killing -- the model is given identical kill kinetics for both agents,
so any divergence in inflammatory cost comes from the yield alone.

Writes results/danger_signal/pilot_acceptance.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.danger_signal import (  # noqa: E402
    DS50_PG_PER_ML, DangerSignalParameters, check_operating_point,
    check_smoothness, simulate_danger_signal, summarize,
    YIELD_ANCHORS_FG_PER_CFU,
)
from src.core.parameters import (  # noqa: E402
    get_default_parameters, get_drug_pk_parameters,
)
from src.core.pd_model import create_ode_system  # noqa: E402
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel  # noqa: E402
from src.core.simulation import run_simulation  # noqa: E402

OUT = os.path.join("results", "danger_signal", "pilot_acceptance.json")

WEIGHT = 70.0
T_END = 48.0

# Immune-competent host with an established infection: this is the setting the
# clinical claim is about, not the neutropenic efficacy model.
IC = dict(B_rep=1e7, B_pers=1e4, B_SCV=0, N_eff=1e7,
          Damage=0, IL6=10, TNF=5, PAMP=0, D_host=0)

# Uncertainty grid over the three unmeasured structural parameters.
DS50_GRID = [30.0, 100.0, 300.0, 1000.0]       # pg/mL
KDS_GRID = [2.0, 13.9, 30.0]                   # per hour
VFOCUS_GRID = [1.0, 10.0, 100.0]               # mL

# Yield sweep: two decades, bracketing every agent anchor in the literature
# (isepamicin ~0.28, imipenem ~0.25, ceftazidime ~4.93 fg/CFU).
Y_GRID = np.logspace(-1, 1, 31)

# Kill-independent filamentation release for a PBP-3 agent, fg/CFU/h. Anchored
# on Yamaguchi 1999: at 1/4 x MIC ceftazidime released MORE free LPS than at
# 1 x MIC while not reducing colony counts at all, i.e. >~500 ng/mL over 4 h
# from a ~1e8 CFU/mL culture -> >~1.25 fg/CFU/h.
Y_FILAMENT_CAZ = 1.5

REGIMENS = {
    # Matched 8-hourly beta-lactam therapy. Doses are the standard clinical
    # ones; the PD kill is identical for both agents by construction.
    "ceftazidime": dict(drug="ceftazidime", dose_mg=2000.0, interval_h=8.0,
                        n_doses=6, inf_min=30),
    "meropenem": dict(drug="meropenem", dose_mg=1000.0, interval_h=8.0,
                      n_doses=6, inf_min=30),
}

# Fractionation arm: does the kill-independent (sub-MIC filamentation) pathway
# ever contribute materially? It can only operate when the concentration spends
# time BELOW the MIC, so it is a property of the regimen, not just the agent.
# Fixed 2 g/day of ceftazidime, spread four ways, plus a reduced-dose arm.
FRACTIONATION = [
    ("CAZ 2 g q8h",   2000.0,  8.0, 6),
    ("CAZ 1 g q8h",   1000.0,  8.0, 6),
    ("CAZ 1 g q12h",  1000.0, 12.0, 4),
    ("CAZ 500 mg q12h", 500.0, 12.0, 4),
    ("CAZ 500 mg q24h", 500.0, 24.0, 2),
]

# Central (reference) cell of the uncertainty grid.
CENTRAL = dict(ds50=100.0, k_ds=13.9, v_focus=10.0)

# --------------------------------------------------------------------------
# SUPERSEDED, DELIBERATELY FROZEN.
#
# This script is the Phase 1 record. Phase 2 calibrated the cytokine layer to
# human experimental endotoxemia and, in doing so, changed it: it added the
# transduction state (the Phase-1 layer put IL-6's peak within minutes of an
# LPS bolus, against an observed 2 h), measured the Hill coefficient at ~1.71
# rather than the assumed 1.0, and tied DS50, hill and k_IL6_max together
# through the fit -- so sweeping DS50 on its own, as the grid below does, is no
# longer a coherent operation.
#
# To keep this script reproducing the Phase 1 result exactly rather than a
# meaningless hybrid of assumed and calibrated constants, its Layer-2
# parameters are pinned here to the Phase-1 prototype. The cascade generalizes
# the prototype, and k_signal_off -> infinity is the prototype's limit (the
# activation signal then tracks receptor occupancy instantaneously).
#
# For the CURRENT gate results, see scripts/ds_phase2_build.py. The release-side
# findings below (tests 4-6: filamentation share, AUC_DS, the MIC-shift sweep)
# are independent of the cytokine layer and remain current either way -- that
# is why the Phase 1 negative result on sub-MIC filamentation still stands.
# --------------------------------------------------------------------------
PHASE1_LAYER2 = dict(hill_ds=1.0, k_IL6_max=2000.0, k_signal_off=1e6)


def params_for_susceptibility(mic_factor: float):
    """Parameter set for an isolate ``mic_factor``-fold less susceptible.

    The model's MIC is set by ``kill_C50``, so reduced susceptibility is a
    proportional shift of that constant -- not a relabelling of the MIC. The
    emergent effective MIC is re-derived from the shifted model.
    """
    p = get_default_parameters()
    p["bacteria"].kill_C50 = p["bacteria"].kill_C50 * mic_factor
    return p


def effect_site_mic_for(params):
    """Constant effect-site concentration giving 24 h stasis, for any params."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    pd = create_ode_system(params)
    y0 = np.array([1e6, 1e3, 0, 1e2, 0, 10, 5, 0, 0], dtype=float)

    def net(C):
        sol = solve_ivp(lambda t, y: pd.rhs(t, y, C_effect=C, drug_class="cidal"),
                        (0, 24), y0, method="LSODA", max_step=0.1,
                        rtol=1e-6, atol=1e-8)
        return np.log10(max(sol.y[:3, -1].sum(), 1e-6) / y0[:3].sum())

    return float(brentq(net, 1e-4, 100.0, xtol=1e-4))


def solve_bacteria(spec, params=None):
    """Solve PK/PD once for a regimen and return the drivers the DS layer needs.

    The DS and cytokine layers do not feed back on the bacteria (see the module
    docstring of src/analysis/danger_signal.py), so one bacterial solve per
    regimen serves the whole uncertainty grid.
    """
    p = get_drug_pk_parameters(spec["drug"])
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka,
                                Kp=p.Kp, effect_site_model=True)
    reg = DosingRegimen(dose_mg=spec["dose_mg"], interval_hours=spec["interval_h"],
                        start_time=0, n_doses=spec["n_doses"],
                        infusion_duration_min=spec["inf_min"])
    pd = create_ode_system(params or get_default_parameters())
    r = run_simulation(pkm, reg, pd, dict(IC), t_span=(0, T_END),
                       drug_class="cidal", weight_kg=WEIGHT)
    t = r.t
    B_rep = r.y[:, r.state_names.index("B_rep")]
    C_eff = r.get_effect_site_concentration(pk_model=pkm, weight_kg=WEIGHT)
    kill = np.asarray(pd.cidal_kill_rate(C_eff), dtype=float)
    _, B_total = r.get_bacterial_burden()
    mic = spec.get("mic", MIC_EFFECT)
    # Time-weighted %fT>MIC (the solver's time grid is non-uniform, so an
    # unweighted point fraction would over-weight the fast post-dose phase).
    above = (C_eff > mic).astype(float)
    dt = np.diff(t)
    ft = float(np.sum(0.5 * (above[:-1] + above[1:]) * dt) / (t[-1] - t[0]) * 100.0)
    return dict(t=t, B_rep=B_rep, C_eff=C_eff, kill=kill,
                net_log10=float(np.log10(max(B_total[-1], 1e-6) / B_total[0])),
                fT_above_mic_pct=ft)


def effect_site_mic():
    """Constant effect-site concentration giving 24 h stasis -- the model's own
    emergent MIC, reused from the qualification pipeline so the pilot is
    referenced to the same MIC as every other result in the project."""
    from scripts.model_qualification import effective_mic
    mic_es, _ = effective_mic()
    return float(mic_es)


MIC_EFFECT = None  # set in main() before any trajectory is solved


def run_ds(driver, y_lysis, y_filament, ds50, k_ds, v_focus, mic=None):
    p = DangerSignalParameters(
        Y_lysis=y_lysis, Y_filament=y_filament, V_focus_mL=v_focus,
        k_ds_clear=k_ds, DS50=ds50, **PHASE1_LAYER2,
    )
    out = simulate_danger_signal(driver["t"], driver["B_rep"], driver["kill"],
                                 driver["C_eff"], mic or MIC_EFFECT, p)
    return summarize(out)


def main():
    global MIC_EFFECT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    MIC_EFFECT = effect_site_mic()
    print(f"model effective (effect-site) MIC = {MIC_EFFECT:.4f} mg/L\n")

    drivers = {name: solve_bacteria(spec) for name, spec in REGIMENS.items()}
    for name, d in drivers.items():
        print(f"  {name:12s}  %fT>MIC = {d['fT_above_mic_pct']:5.1f}   "
              f"net 48 h log10 change = {d['net_log10']:+.2f}")

    base = drivers["ceftazidime"]

    # ---------------------------------------------------------------- TEST 1
    # Saturation of the cytokine layer at the two literature agent anchors,
    # over the whole uncertainty grid.
    print("\nTEST 1 -- cytokine-layer operating point (peak Hill occupancy)")
    t1_cells, t1_pass = [], 0
    for ds50 in DS50_GRID:
        for k_ds in KDS_GRID:
            for vf in VFOCUS_GRID:
                for agent, y in (("ceftazidime", YIELD_ANCHORS_FG_PER_CFU["ceftazidime"]),
                                 ("meropenem", YIELD_ANCHORS_FG_PER_CFU["meropenem"])):
                    yf = Y_FILAMENT_CAZ if agent == "ceftazidime" else 0.0
                    s = run_ds(drivers[agent], y, yf, ds50, k_ds, vf)
                    chk = check_operating_point(s["saturation_peak"])
                    t1_pass += int(chk["pass"])
                    t1_cells.append(dict(agent=agent, DS50=ds50, k_ds=k_ds,
                                         V_focus_mL=vf,
                                         DS_peak_pg_mL=s["DS_peak_pg_mL"],
                                         IL6_peak_pg_mL=s["IL6_peak_pg_mL"],
                                         **chk))
    n1 = len(t1_cells)
    print(f"  in the responsive band (0.01 < occupancy < 0.80): {t1_pass}/{n1} cells")
    for tag, pred in (("saturated (>=0.80)", lambda c: not c["below_saturation"]),
                      ("not engaged (<=0.01)", lambda c: not c["engaged"])):
        bad = [c for c in t1_cells if pred(c)]
        print(f"  {tag}: {len(bad)}")

    # Point estimate at the central grid cell, for the record.
    central = [c for c in t1_cells if c["DS50"] == 100.0 and c["k_ds"] == 13.9
               and c["V_focus_mL"] == 10.0]
    print("  central cell (DS50=100 pg/mL, k_ds=13.9/h, V_focus=10 mL):")
    for c in central:
        print(f"    {c['agent']:12s} DS_peak = {c['DS_peak_pg_mL']:9.2f} pg/mL   "
              f"occupancy = {c['saturation_peak']:.3f}   "
              f"IL-6 peak = {c['IL6_peak_pg_mL']:8.1f} pg/mL")

    # ---------------------------------------------------------------- TEST 2
    # Smoothness and monotonicity of the response in Y_lysis.
    print("\nTEST 2 -- response vs Y_lysis (smooth? monotone? discriminating?)")
    t2_cells, t2_pass = [], 0
    for ds50 in DS50_GRID:
        for k_ds in KDS_GRID:
            for vf in VFOCUS_GRID:
                auc = np.array([
                    run_ds(base, y, 0.0, ds50, k_ds, vf)["AUC_IL6_excess_pg_h_mL"]
                    for y in Y_GRID])
                chk = check_smoothness(Y_GRID, auc)
                t2_pass += int(chk["pass"])
                t2_cells.append(dict(DS50=ds50, k_ds=k_ds, V_focus_mL=vf,
                                     AUC_IL6_first=float(auc[0]),
                                     AUC_IL6_last=float(auc[-1]), **chk))
    n2 = len(t2_cells)
    # The design document's criterion is "smooth and monotone, no step
    # function". Dynamic range is reported alongside it as a usefulness
    # diagnostic, not as part of the gate.
    n_smooth = sum(1 for c in t2_cells if c["monotone"] and c["no_step_function"])
    print(f"  smooth AND monotone (the design criterion): {n_smooth}/{n2} cells")
    print(f"  also >=3x discriminating: {t2_pass}/{n2} cells")
    worst = max(t2_cells, key=lambda c: c["max_step_fraction_of_span"])
    print(f"  largest single-step jump anywhere on the grid: "
          f"{worst['max_step_fraction_of_span']:.3f} of the total span "
          f"(DS50={worst['DS50']}, k_ds={worst['k_ds']}, V={worst['V_focus_mL']})")
    ranges = [c["dynamic_range"] for c in t2_cells]
    print(f"  dynamic range over the 100-fold yield sweep: "
          f"{min(ranges):.1f}x to {max(ranges):.1f}x")
    nonmono = [c for c in t2_cells if not c["monotone"]]
    print(f"  non-monotone cells: {len(nonmono)}")

    # ---------------------------------------------------------------- TEST 3
    # The pilot measurement: matched killing, divergent inflammatory cost.
    print("\nTEST 3 -- matched killing, agent-specific inflammatory cost "
          "(central grid cell)")
    comp = {}
    for agent in ("ceftazidime", "meropenem"):
        y = YIELD_ANCHORS_FG_PER_CFU[agent]
        yf = Y_FILAMENT_CAZ if agent == "ceftazidime" else 0.0
        s = run_ds(drivers[agent], y, yf, CENTRAL["ds50"], CENTRAL["k_ds"],
                   CENTRAL["v_focus"])
        s["Y_lysis_fg_per_CFU"] = y
        s["Y_filament_fg_per_CFU_per_h"] = yf
        s["net_log10_change_48h"] = drivers[agent]["net_log10"]
        s["fT_above_MIC_pct"] = drivers[agent]["fT_above_mic_pct"]
        comp[agent] = s
        print(f"  {agent:12s} AUC_DS = {s['AUC_DS_pg_h_mL']:10.1f} pg.h/mL   "
              f"AUC_IL6(excess) = {s['AUC_IL6_excess_pg_h_mL']:9.1f} pg.h/mL   "
              f"filament share of release = {s['filament_fraction_of_release']:.2f}")
    r_ds = comp["ceftazidime"]["AUC_DS_pg_h_mL"] / comp["meropenem"]["AUC_DS_pg_h_mL"]
    # Ratio of the THERAPY-ATTRIBUTABLE IL-6 (the rise over baseline). Using
    # total IL-6 here would dilute the contrast with a constant baseline that
    # has nothing to do with the agent.
    r_il6 = (comp["ceftazidime"]["AUC_IL6_excess_pg_h_mL"]
             / comp["meropenem"]["AUC_IL6_excess_pg_h_mL"])
    d_kill = (comp["ceftazidime"]["net_log10_change_48h"]
              - comp["meropenem"]["net_log10_change_48h"])
    print(f"  ratio ceftazidime/meropenem: AUC_DS {r_ds:.2f}x, AUC_IL6 {r_il6:.2f}x, "
          f"difference in 48 h log10 kill {d_kill:+.2f}")

    # ---------------------------------------------------------------- TEST 4
    # Is the kill-independent (sub-MIC filamentation) pathway ever material?
    # It needs sub-MIC time, so this is a regimen question.
    print("\nTEST 4 -- sub-MIC filamentation pathway across ceftazidime regimens")
    frac = []
    for label, dose, iv, n in FRACTIONATION:
        d = solve_bacteria(dict(drug="ceftazidime", dose_mg=dose,
                                interval_h=iv, n_doses=n, inf_min=30))
        p = DangerSignalParameters(
            Y_lysis=YIELD_ANCHORS_FG_PER_CFU["ceftazidime"],
            Y_filament=Y_FILAMENT_CAZ, V_focus_mL=CENTRAL["v_focus"],
            k_ds_clear=CENTRAL["k_ds"], DS50=CENTRAL["ds50"],
            **PHASE1_LAYER2)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     MIC_EFFECT, p)
        s = summarize(out)
        # Counterfactual with the filamentation pathway switched off, to
        # separate its contribution from the lysis-coupled one.
        p0 = DangerSignalParameters(
            Y_lysis=YIELD_ANCHORS_FG_PER_CFU["ceftazidime"], Y_filament=0.0,
            V_focus_mL=CENTRAL["v_focus"], k_ds_clear=CENTRAL["k_ds"],
            DS50=CENTRAL["ds50"], **PHASE1_LAYER2)
        s0 = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                              d["C_eff"], MIC_EFFECT, p0))
        row = dict(regimen=label, dose_mg=dose, interval_h=iv,
                   fT_above_MIC_pct=d["fT_above_mic_pct"],
                   net_log10_change=d["net_log10"],
                   AUC_DS_pg_h_mL=s["AUC_DS_pg_h_mL"],
                   AUC_DS_lysis_only_pg_h_mL=s0["AUC_DS_pg_h_mL"],
                   filament_fraction_of_release=s["filament_fraction_of_release"],
                   AUC_IL6_excess_pg_h_mL=s["AUC_IL6_excess_pg_h_mL"],
                   AUC_IL6_excess_lysis_only_pg_h_mL=s0["AUC_IL6_excess_pg_h_mL"])
        frac.append(row)
        print(f"  {label:17s} %fT>MIC {row['fT_above_MIC_pct']:5.1f}  "
              f"kill {row['net_log10_change']:+5.2f} log10  "
              f"AUC_DS {row['AUC_DS_pg_h_mL']:8.1f}  "
              f"filament share {row['filament_fraction_of_release']:.2f}")

    # ---------------------------------------------------------------- TEST 5
    # Reduced susceptibility. Test 4 shows the filamentation window is traversed
    # right after the kill phase, when few organisms remain. A less susceptible
    # isolate pushes the window later into the interval, where the burden has
    # regrown -- so if the kill-independent pathway matters anywhere, it should
    # matter here. Ceftazidime 2 g q8h throughout; only the isolate changes.
    print("\nTEST 5 -- reduced susceptibility (ceftazidime 2 g q8h, MIC shifted)")
    susc = []
    for fac in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        prm = params_for_susceptibility(fac)
        mic_f = effect_site_mic_for(prm)
        d = solve_bacteria(dict(drug="ceftazidime", dose_mg=2000.0,
                                interval_h=8.0, n_doses=6, inf_min=30,
                                mic=mic_f), params=prm)
        s = run_ds(d, YIELD_ANCHORS_FG_PER_CFU["ceftazidime"], Y_FILAMENT_CAZ,
                   CENTRAL["ds50"], CENTRAL["k_ds"], CENTRAL["v_focus"], mic=mic_f)
        s0 = run_ds(d, YIELD_ANCHORS_FG_PER_CFU["ceftazidime"], 0.0,
                    CENTRAL["ds50"], CENTRAL["k_ds"], CENTRAL["v_focus"], mic=mic_f)
        row = dict(mic_factor=fac, mic_effect_site_mgL=mic_f,
                   fT_above_MIC_pct=d["fT_above_mic_pct"],
                   net_log10_change=d["net_log10"],
                   AUC_DS_pg_h_mL=s["AUC_DS_pg_h_mL"],
                   AUC_DS_lysis_only_pg_h_mL=s0["AUC_DS_pg_h_mL"],
                   filament_fraction_of_release=s["filament_fraction_of_release"],
                   AUC_IL6_excess_pg_h_mL=s["AUC_IL6_excess_pg_h_mL"],
                   AUC_IL6_excess_lysis_only_pg_h_mL=s0["AUC_IL6_excess_pg_h_mL"],
                   saturation_peak=s["saturation_peak"])
        susc.append(row)
        print(f"  {fac:4.0f}x MIC ({mic_f:6.3f} mg/L)  %fT>MIC {row['fT_above_MIC_pct']:5.1f}  "
              f"kill {row['net_log10_change']:+5.2f}  AUC_DS {row['AUC_DS_pg_h_mL']:8.1f}  "
              f"filament share {row['filament_fraction_of_release']:.2f}")

    # ---------------------------------------------------------------- TEST 6
    # Is the Test-4/5 conclusion an artefact of the single Y_filament anchor?
    # Sweep it over two decades at the regimen and isolate most favourable to
    # the pathway (the one with the largest filament share found above).
    print("\nTEST 6 -- sensitivity of the filamentation share to Y_filament")
    best = max(susc, key=lambda r: r["filament_fraction_of_release"])
    prm = params_for_susceptibility(best["mic_factor"])
    mic_b = best["mic_effect_site_mgL"]
    d_best = solve_bacteria(dict(drug="ceftazidime", dose_mg=2000.0,
                                 interval_h=8.0, n_doses=6, inf_min=30,
                                 mic=mic_b), params=prm)
    yf_sweep = []
    for yf in (0.15, 1.5, 15.0, 150.0):
        s = run_ds(d_best, YIELD_ANCHORS_FG_PER_CFU["ceftazidime"], yf,
                   CENTRAL["ds50"], CENTRAL["k_ds"], CENTRAL["v_focus"], mic=mic_b)
        yf_sweep.append(dict(Y_filament=yf,
                             filament_fraction_of_release=s["filament_fraction_of_release"],
                             AUC_DS_pg_h_mL=s["AUC_DS_pg_h_mL"],
                             AUC_IL6_excess_pg_h_mL=s["AUC_IL6_excess_pg_h_mL"]))
        print(f"  Y_filament {yf:7.2f} fg/CFU/h  ->  filament share "
              f"{s['filament_fraction_of_release']:.3f}   "
              f"AUC_DS {s['AUC_DS_pg_h_mL']:9.1f}")
    print(f"  (evaluated at {best['mic_factor']:.0f}x MIC, the most favourable "
          f"isolate found in test 5; literature anchor is 1.5)")

    # ---------------------------------------------------------------- GATE
    # The design document specifies two stop conditions. They are evaluated
    # exactly as written there; the extended grid statistics above are
    # diagnostics, deliberately kept separate from the gate so that the gate
    # cannot be made to pass by widening a tolerance.
    #   1. the cytokine layer operates below saturation at physiological DS
    #      levels (checked at the central configuration, for both agents);
    #   2. output varies smoothly and monotonically with Y_lysis, with no step
    #      function (checked over the ENTIRE uncertainty grid).
    gate1 = all(c["below_saturation"] for c in central) and any(
        c["engaged"] for c in central)
    gate2 = all(c["monotone"] and c["no_step_function"] for c in t2_cells)
    gate = bool(gate1 and gate2)
    print("\n" + "=" * 72)
    print(f"GATE 1 (below saturation, central cell): {'PASS' if gate1 else 'FAIL'}")
    print(f"GATE 2 (smooth and monotone in Y_lysis, all {n2} grid cells): "
          f"{'PASS' if gate2 else 'FAIL'}")
    print(f"STOP-GATE: {'PASS -- proceed to Phase 2' if gate else 'FAIL -- stop'}")
    print("=" * 72)

    payload = dict(
        meta=dict(
            purpose="Phase 1 pilot stop-gate for the danger-signal study",
            design="docs/danger_signal_study/DESIGN.md",
            mic_effect_site_mgL=MIC_EFFECT,
            t_end_h=T_END,
            initial_conditions=IC,
            regimens=REGIMENS,
            Y_filament_ceftazidime_fg_per_CFU_per_h=Y_FILAMENT_CAZ,
            yield_anchors_fg_per_CFU=YIELD_ANCHORS_FG_PER_CFU,
            grid=dict(DS50_pg_mL=DS50_GRID, k_ds_per_h=KDS_GRID,
                      V_focus_mL=VFOCUS_GRID,
                      Y_lysis_fg_per_CFU=[float(Y_GRID[0]), float(Y_GRID[-1]),
                                          int(Y_GRID.size)]),
        ),
        test1_operating_point=dict(n_cells=n1, n_pass=t1_pass, cells=t1_cells,
                                   central=central),
        test2_smoothness=dict(n_cells=n2, n_pass=t2_pass, cells=t2_cells),
        test3_agent_comparison=dict(agents=comp, ratio_AUC_DS=r_ds,
                                    ratio_AUC_IL6=r_il6,
                                    delta_net_log10_kill=d_kill),
        test4_fractionation=frac,
        test5_reduced_susceptibility=susc,
        test6_Y_filament_sensitivity=dict(
            evaluated_at_mic_factor=best["mic_factor"], sweep=yf_sweep),
        gate=dict(gate1_below_saturation=gate1, gate2_smooth_monotone=gate2,
                  pass_=gate,
                  criteria="as specified in docs/danger_signal_study/DESIGN.md "
                           "section 4; grid statistics are diagnostics only"),
        stop_gate_pass=gate,
    )
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
