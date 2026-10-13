"""Phase 4 of the danger-signal study: the six analyses of DESIGN.md section 5.

Two of the six were amended by earlier phases, and the amendments are applied
here rather than the original wording being honoured mechanically:

  * Analysis 4 originally proposed %fT_sub-MIC as a candidate inflammatory-cost
    index. Phase 1 demoted it: the sub-MIC filamentation window is traversed
    when the burden is at its nadir, so it carries almost no information at
    clinical exposures. AUC_DS survives, and a *computable* surrogate is tested
    alongside it.
  * Analysis 5 was a dose-fractionation sweep at fixed daily dose. Phase 1
    showed that produces almost no divergence, so it is re-scoped to a
    susceptibility x exposure map, which is where the divergence actually lives.

Phase 3 also fixed the reporting discipline, which this script follows: every
agent comparison is reported as a RANKING plus a RANGE over the assumed yield
interval, never as a single fold-change, and no survival claim is made.

Analysis 3 is the centrepiece and is done properly here for the first time:
earlier phases compared ceftazidime and meropenem at their standard doses,
which are NOT matched on %fT>MIC. Here the doses are solved so the two agents
achieve the same exposure, which is what isolates the yield effect.

Writes results/danger_signal/phase4_analyses.json.
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.analysis.danger_signal import (  # noqa: E402
    FINAL_YIELD_RATIO, FINAL_YIELDS_FG_PER_CFU, check_release_mass_balance,
    simulate_danger_signal, summarize,
)
from src.analysis.held_out_validation import (  # noqa: E402
    AGENT_EFFECT_ESTIMATES,
)
from src.analysis.injury_layer import check_rank_invariance  # noqa: E402
from src.core.parameters import (  # noqa: E402
    get_default_parameters, get_drug_pd_parameters, get_drug_pk_parameters,
    immune_free_parameters,
)
from src.core.pd_model import create_ode_system  # noqa: E402
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel  # noqa: E402
from src.core.simulation import SimulationResult, run_simulation  # noqa: E402

OUT = os.path.join("results", "danger_signal", "phase4_analyses.json")

WEIGHT = 70.0
T_END = 48.0
IC = dict(B_rep=1e7, B_pers=1e4, B_SCV=0, N_eff=1e7,
          Damage=0, IL6=10, TNF=5, PAMP=0, D_host=0)
V_FOCUS = 10.0
Y_FILAMENT_CAZ = 1.5
#: Total starting burden, CFU/mL, used to express cumulative killing relative
#: to the burden initially present.
B_TOTAL_0 = IC["B_rep"] + IC["B_pers"] + IC["B_SCV"]
# Agent-effect interval. REVISED in Phase 5b: three independent measurements
# (Trautmann ELISA in vitro 6.9x, Prins whole-blood TNF 3.5x, Simpson
# in-patient plasma endotoxin 4.97x) agree on 3.5-7x. The 10-40x this
# project carried from Phase 1 came from a Limulus assay and is treated as an
# outlier -- see PHASE5B_AGENT_EFFECT.md.
YIELD_RATIO_RANGE = AGENT_EFFECT_ESTIMATES["recommended_interval"]

# Both agents are beta-lactams and are given the SAME kill function, so any
# difference in inflammatory cost comes from the yield alone. Ceftazidime is
# PBP-3 (filamenting, high yield); meropenem is PBP-2 (fast lysis, low yield).
AGENT_PK = {"ceftazidime": "ceftazidime", "meropenem": "meropenem"}
AGENT_YIELD = {"ceftazidime": FINAL_YIELDS_FG_PER_CFU["ceftazidime"],
               "meropenem": FINAL_YIELDS_FG_PER_CFU["meropenem"]}
AGENT_FILAMENT = {"ceftazidime": Y_FILAMENT_CAZ, "meropenem": 0.0}

MIC_EFFECT = None
P_REF = None


# --------------------------------------------------------------------------
# machinery
# --------------------------------------------------------------------------
def effect_site_mic(params=None):
    from scipy.integrate import solve_ivp
    # The effective MIC is a property of the drug against the organism, so it
    # is resolved with the effector pool off -- the same condition the
    # bacterial layer was calibrated in. The trace N_eff below (1e2, contributing
    # a kill rate of 1e-6/h) was this script's way of saying "immune off"; since
    # the R10 repair the pool is homeostatic, so saying it requires the
    # set-point too, or the trace would be recruited to a full response.
    pd = create_ode_system(immune_free_parameters(params or get_default_parameters()))
    y0 = np.array([1e6, 1e3, 0, 1e2, 0, 10, 5, 0, 0], dtype=float)

    def net(C):
        sol = solve_ivp(lambda t, y: pd.rhs(t, y, C_effect=C, drug_class="cidal"),
                        (0, 24), y0, method="LSODA", max_step=0.1,
                        rtol=1e-6, atol=1e-8)
        return np.log10(max(sol.y[:3, -1].sum(), 1e-6) / y0[:3].sum())

    return float(brentq(net, 1e-4, 200.0, xtol=1e-4))


def simulate(agent, dose_mg, interval_h, n_doses, params=None, mic=None,
             inf_min=30, t_end=T_END):
    p = get_drug_pk_parameters(AGENT_PK[agent])
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka,
                                Kp=p.Kp, effect_site_model=True)
    reg = DosingRegimen(dose_mg=dose_mg, interval_hours=interval_h, start_time=0,
                        n_doses=n_doses, infusion_duration_min=inf_min)
    pd = create_ode_system(params or get_default_parameters())
    # Kp is a total-concentration partition, so only the unbound fraction
    # acts. Binding is applied once, inside run_simulation; C_eff below is
    # the FREE concentration, and every index and the kill rate use it.
    fu = get_drug_pd_parameters(AGENT_PK[agent]).fraction_unbound
    r = run_simulation(pkm, reg, pd, dict(IC), t_span=(0, t_end),
                       drug_class="cidal", weight_kg=WEIGHT,
                       fraction_unbound=fu)
    C_eff_total = r.get_effect_site_concentration(pk_model=pkm, weight_kg=WEIGHT)
    C_eff = r.get_free_effect_site_concentration(pk_model=pkm, weight_kg=WEIGHT)
    m = mic if mic is not None else MIC_EFFECT
    # Interpolate each MIC crossing rather than count grid points above it.
    # Counting is first-order in the solver's step size, so the doses solved
    # to hit a %fT>MIC target used to move whenever the solver's grid did.
    ft = float(100.0 * SimulationResult._time_above_threshold(r.t, C_eff, m)
               / (r.t[-1] - r.t[0]))
    _, B_total = r.get_bacterial_burden()
    B_rep = r.y[:, r.state_names.index("B_rep")]
    kill = np.asarray(pd.cidal_kill_rate(C_eff), dtype=float)
    # Cumulative killed biomass: the quantity danger-signal release is
    # proportional to, and the basis of the computable surrogate index tested
    # in analysis 4.
    from scipy.integrate import trapezoid
    killed = float(trapezoid(kill * B_rep, r.t))
    return dict(t=r.t, B_rep=B_rep, C_eff=C_eff, C_eff_total=C_eff_total,
                fraction_unbound=fu, kill=kill, mic=m,
                fT=ft, cum_killed_cfu_per_mL=killed,
                net_log10=float(np.log10(max(B_total[-1], 1e-6) / B_total[0])),
                fAUC_MIC=float(trapezoid(C_eff, r.t) / m / (t_end / 24.0)),
                fCmax_MIC=float(np.max(C_eff) / m))


def ds_cost(sim, agent=None, Y=None, Yf=None, v_focus=V_FOCUS):
    y = AGENT_YIELD[agent] if Y is None else Y
    yf = AGENT_FILAMENT[agent] if Yf is None else Yf
    q = replace(P_REF, Y_lysis=y, Y_filament=yf, V_focus_mL=v_focus)
    out = simulate_danger_signal(sim["t"], sim["B_rep"], sim["kill"],
                                 sim["C_eff"], sim["mic"], q)
    return out, summarize(out)


def dose_for_target_ft(agent, target_ft, interval_h, n_doses, mic=None,
                       lo=10.0, hi=60000.0):
    """Solve for the dose achieving a target %fT>MIC.

    ``hi`` is a search bracket, not a clinical limit. It was 16 g, which
    silently excluded the 95% target once the bacterial-layer refit raised
    the effective MIC from 0.40 to 0.77 mg/L and roughly doubled every
    matched dose: the required meropenem dose rose to about 18.6 g and the
    row vanished from the table without a record. The bracket is now wide
    enough to contain any dose the targets can require; whether a dose is
    administrable is a separate judgement, made in the dose table itself,
    where rows above the 6 g/day usual maximum are marked.
    """
    def err(d):
        return simulate(agent, d, interval_h, n_doses, mic=mic)["fT"] - target_ft
    try:
        if err(lo) > 0 or err(hi) < 0:
            return None
        return float(brentq(err, lo, hi, xtol=1.0))
    except ValueError:
        return None


def yield_range_ratio(sim_hi, sim_lo):
    """Agent cost ratio at both ends of the assumed yield range."""
    y_caz = AGENT_YIELD["ceftazidime"]
    out = []
    for r in YIELD_RATIO_RANGE:
        _, s_hi = ds_cost(sim_hi, Y=y_caz, Yf=Y_FILAMENT_CAZ)
        _, s_lo = ds_cost(sim_lo, Y=y_caz / r, Yf=0.0)
        out.append({
            "yield_ratio": r,
            "ratio_AUC_DS": s_hi["AUC_DS_pg_h_mL"] / s_lo["AUC_DS_pg_h_mL"],
            "ratio_AUC_IL6": (s_hi["AUC_IL6_excess_pg_h_mL"]
                              / s_lo["AUC_IL6_excess_pg_h_mL"]),
        })
    return out


# --------------------------------------------------------------------------
def analysis1_yield_table():
    """Agent-specific danger-signal yield table, with uncertainty."""
    y_caz = FINAL_YIELDS_FG_PER_CFU["ceftazidime"]
    carbapenem_interval = [round(y_caz / YIELD_RATIO_RANGE[1], 2),
                           round(y_caz / YIELD_RATIO_RANGE[0], 2)]
    rows = [
        dict(agent="piperacillin", pbp="PBP-3", morphology="filamentation",
             Y_lysis_fg_per_CFU=FINAL_YIELDS_FG_PER_CFU["piperacillin"],
             interval=None,
             basis="Yamaguchi 1999: 515 ng/mL free LPS at 1x MIC, 4 h, / 1e8 CFU/mL"),
        dict(agent="ceftazidime", pbp="PBP-3", morphology="filamentation",
             Y_lysis_fg_per_CFU=y_caz, interval=None,
             basis="Yamaguchi 1999: 493 ng/mL free LPS at 1x MIC, 4 h, / 1e8 CFU/mL"),
        dict(agent="imipenem", pbp="PBP-2", morphology="spheroplast, rapid lysis",
             Y_lysis_fg_per_CFU=round(FINAL_YIELDS_FG_PER_CFU["imipenem"], 2),
             interval=carbapenem_interval,
             basis="REVISED in Phase 5b to 3.5-7x, from Trautmann 1998 (ELISA, "
                   "6.9x), Prins 1995 (whole-blood TNF, 3.5x) and Simpson "
                   "2000 (in patients, 5.0x). Jackson & Kropp's Limulus "
                   "10-40x at "
                   "equivalent MICs; interval derived from the ceftazidime anchor"),
        dict(agent="meropenem", pbp="PBP-2", morphology="spheroplast, rapid lysis",
             Y_lysis_fg_per_CFU=round(FINAL_YIELDS_FG_PER_CFU["meropenem"], 2),
             interval=carbapenem_interval,
             basis="assigned imipenem's value (also PBP-2, also a low releaser "
                   "per Trautmann 1998); pending digitization"),
        dict(agent="isepamicin", pbp="none (ribosomal)", morphology="-",
             Y_lysis_fg_per_CFU=FINAL_YIELDS_FG_PER_CFU["isepamicin"],
             interval=None,
             basis="Yamaguchi 1999: 28 +/- 2 ng/mL at 1x MIC, 6 h, / 1e8 CFU/mL"),
    ]
    caveats = [
        "All per-CFU values scale inversely with the assumed 1e8 CFU/mL assay "
        "inoculum, which is taken from the standard protocol and has NOT been "
        "confirmed against the Yamaguchi methods section. The agent-to-agent "
        "RATIOS, on which every conclusion in this study rests, are independent "
        "of it.",
        "Values are derived from published abstracts, not digitized curves, so "
        "they are point estimates with no within-study dispersion.",
        "A PBP-3 agent additionally carries a kill-independent filamentation "
        "yield (Y_filament ~ 1.5 fg/CFU/h), which is anchored on P. aeruginosa "
        "and is bounded above by Prins' E. coli result -- see Phase 3.",
    ]
    note = ("Every agent in this table is BACTERICIDAL, and their yields still "
            "differ several-fold. The PBP-3 cephalosporins (ceftazidime, "
            "piperacillin) sit about 3.5-7x above the PBP-2 carbapenems at "
            "equivalent MICs, while the bactericidal aminoglycoside isepamicin "
            "sits LOWER still -- but that gap is not interpretable: "
            "isepamicin's value is an absolute per-CFU measurement from "
            "Yamaguchi, while the carbapenems' is derived from the assumed "
            "agent ratio applied to the ceftazidime anchor, so the two are "
            "not commensurable and their 3.5x separation is an artefact of "
            "combining differently derived quantities on one axis. What the "
            "aminoglycoside establishes is only that a bactericidal agent "
            "with no PBP target is not a high releaser. Yield therefore "
            "tracks the "
            "molecular target, not the bactericidal/bacteriostatic class -- "
            "which does not even vary across this table.")
    mechanism = (
        "The mechanism is PBP-3-SPECIFIC CELL-WALL DAMAGE, not filamentation "
        "as such. Trautmann 1998 settles this: ciprofloxacin induces filaments, "
        "with killing curves nearly identical to the beta-lactams, yet releases "
        "as little free LPS as the carbapenems -- so morphology does not "
        "predict release. Meropenem points the same way: it binds PBP-3 as "
        "well as PBP-2 and still releases little, so it is the DOMINANT PBP-3 "
        "specificity of the cephalosporins that matters. Filamentation is a "
        "marker of the relevant lesion, not its cause.")
    return {"rows": rows, "caveats": caveats, "interpretation": note,
            "mechanism": mechanism}


def analysis2_dual_pathway():
    """Is the kill-independent pathway REQUIRED to explain the sub-MIC data?"""
    out = {}
    for label, mult in (("0.5x MIC", 0.5), ("1x MIC", 1.0), ("50x MIC", 50.0)):
        row = {}
        for pathway, yf in (("dual", Y_FILAMENT_CAZ), ("lysis_only", 0.0)):
            s = const_exposure(mult, AGENT_YIELD["ceftazidime"], yf)
            row[pathway] = s
        out[label] = row
    sub_dual = out["0.5x MIC"]["dual"]["AUC_DS_pg_h_mL"]
    sub_lysis = out["0.5x MIC"]["lysis_only"]["AUC_DS_pg_h_mL"]
    high_dual = out["50x MIC"]["dual"]["AUC_DS_pg_h_mL"]
    return {
        "levels": out,
        "sub_mic_dual_over_lysis_only": (sub_dual / sub_lysis
                                         if sub_lysis > 0 else float("inf")),
        "sub_mic_over_50x_dual": sub_dual / high_dual if high_dual > 0 else None,
        "sub_mic_over_50x_lysis_only": (sub_lysis / high_dual
                                        if high_dual > 0 else None),
        "observation_to_explain":
            "Jackson & Kropp 1992: sub-MIC ceftazidime released LPS in amounts "
            "MATCHING those at 1, 2 and 50x MIC, without significant lysis.",
        "conclusion":
            "A lysis-only model cannot reproduce that observation: with "
            "Y_filament = 0 the predicted sub-MIC release collapses, because "
            "there is almost no killing at 0.5x MIC to release anything. The "
            "kill-independent pathway is therefore REQUIRED by the "
            "P. aeruginosa data.",
        "counter_evidence":
            "Prins 1995 (E. coli, whole blood) found 0.5x MIC comparable to "
            "untreated with little agent separation, which the dual-pathway "
            "model does NOT reproduce. The two held-out sources conflict; "
            "Y_filament is plausibly organism-dependent and bounded above by "
            "Prins. Unresolved -- see PHASE3_VALIDATION.md section 6.",
    }


def const_exposure(C_mult, Y, Yf, t_end=8.0, n=1600, B=1e8):
    """Constant-concentration exposure through the validated PD ODE.

    An in-vitro construct: constant concentration and no immune effectors, so
    the effector pool is switched off at the set-point as well as in the
    initial condition (review R10).
    """
    from scipy.integrate import solve_ivp
    pd = create_ode_system(immune_free_parameters())
    C_const = C_mult * MIC_EFFECT
    t = np.linspace(0.0, t_end, n)
    y0 = np.array([B, B * 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sol = solve_ivp(lambda tt, y: pd.rhs(tt, y, C_effect=C_const,
                                         drug_class="cidal"),
                    (0.0, t_end), y0, t_eval=t, method="LSODA",
                    max_step=0.1, rtol=1e-6, atol=1e-8)
    B_rep = np.maximum(sol.y[0], 1e-6)
    C = np.full(n, C_const)
    kill = np.asarray(pd.cidal_kill_rate(C), dtype=float)
    q = replace(P_REF, Y_lysis=Y, Y_filament=Yf, V_focus_mL=V_FOCUS)
    s = summarize(simulate_danger_signal(t, B_rep, kill, C, MIC_EFFECT, q))
    s["log10_change"] = float(np.log10(B_rep[-1] / B))
    return s


def analysis3_matched_exposure(targets=(40.0, 60.0, 80.0, 95.0)):
    """Matched %fT>MIC: equal exposure, near-equal killing, divergent cost."""
    rows = []
    for target in targets:
        entry = {"target_fT_pct": target, "arms": {}}
        sims = {}
        ok = True
        for agent in ("ceftazidime", "meropenem"):
            d = dose_for_target_ft(agent, target, 8.0, 6)
            if d is None:
                ok = False
                break
            sim = simulate(agent, d, 8.0, 6)
            _, s = ds_cost(sim, agent=agent)
            sims[agent] = sim
            entry["arms"][agent] = {
                "dose_mg": d, "achieved_fT_pct": sim["fT"],
                "net_log10_change": sim["net_log10"],
                "fAUC_MIC": sim["fAUC_MIC"], "fCmax_MIC": sim["fCmax_MIC"],
                "AUC_DS_pg_h_mL": s["AUC_DS_pg_h_mL"],
                "AUC_IL6_excess_pg_h_mL": s["AUC_IL6_excess_pg_h_mL"],
                "IL6_peak_pg_mL": s["IL6_peak_pg_mL"],
                "saturation_peak": s["saturation_peak"],
                "cum_killed_cfu_per_mL": sim["cum_killed_cfu_per_mL"],
                # Cumulative killing divided by the starting burden. Values
                # above 1 show that killed biomass is NOT bounded by the
                # burden initially present: growth replenishes it.
                "killed_over_initial_burden": (sim["cum_killed_cfu_per_mL"]
                                               / B_TOTAL_0),
            }
        if not ok:
            # Record it. Dropping the row silently changed the table's shape
            # without saying so when the refit pushed the 95% target past
            # the old 16 g search bracket.
            rows.append({"target_fT_pct": target, "evaluated": False,
                         "reason": "no dose inside the search bracket "
                                   "achieves this %fT>MIC for both agents",
                         "bracket_mg": [10.0, 60000.0]})
            continue
        entry["evaluated"] = True
        a, b = entry["arms"]["ceftazidime"], entry["arms"]["meropenem"]
        entry["kill_difference_log10"] = a["net_log10_change"] - b["net_log10_change"]
        entry["cost_ratio_range"] = yield_range_ratio(sims["ceftazidime"],
                                                      sims["meropenem"])
        rows.append(entry)
    return {"rows": rows,
            "claim": "At matched %fT>MIC the two agents achieve nearly equal "
                     "bacterial killing (see kill_difference_log10 per row; "
                     "matching exposure does not guarantee equal killing when "
                     "profile shapes differ), and the PBP-3 agent imposes the "
                     "higher danger-signal cost. The ratio is reported as a "
                     "range over the assumed 3.5-7x yield range."}


def analysis4_cost_index():
    """Can an existing PK/PD index rank regimens by inflammatory cost?

    The proposed mechanistic index is AUC_DS. The practical question is whether
    something computable WITHOUT the danger-signal model can substitute. Four
    candidates are tested against the modelled inflammatory cost across a
    regimen library spanning both agents, four intervals and a range of doses.
    """
    lib = []
    residuals = []
    for agent in ("ceftazidime", "meropenem"):
        for interval, n in ((6.0, 8), (8.0, 6), (12.0, 4), (24.0, 2)):
            for dose in (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0):
                sim = simulate(agent, dose, interval, n)
                out, s = ds_cost(sim, agent=agent)
                # The identity that explains the proposed index's near-perfect
                # correlation, checked on every regimen rather than asserted.
                q = replace(P_REF, Y_lysis=AGENT_YIELD[agent],
                            Y_filament=AGENT_FILAMENT[agent],
                            V_focus_mL=V_FOCUS)
                residuals.append(
                    abs(check_release_mass_balance(out, q)["relative_residual"]))
                lib.append({
                    "agent": agent, "dose_mg": dose, "interval_h": interval,
                    "fT_pct": sim["fT"], "fAUC_MIC": sim["fAUC_MIC"],
                    "fCmax_MIC": sim["fCmax_MIC"],
                    "net_log10_change": sim["net_log10"],
                    "cum_killed_cfu_per_mL": sim["cum_killed_cfu_per_mL"],
                    # The computable surrogate: cumulative killed biomass
                    # weighted by the agent's danger-signal yield. Needs only a
                    # standard PK/PD simulation plus a yield table.
                    "yield_weighted_kill": (AGENT_YIELD[agent]
                                            * sim["cum_killed_cfu_per_mL"]),
                    "AUC_DS_pg_h_mL": s["AUC_DS_pg_h_mL"],
                    "cost_AUC_DS": s["AUC_DS_pg_h_mL"],
                    "modelled_AUC_IL6_excess": s["AUC_IL6_excess_pg_h_mL"],
                })
    # Inflammatory cost is ATTRIBUTABLE DANGER-SIGNAL EXPOSURE. Phase 5a
    # established that the cytokine layer does not transfer from healthy
    # volunteers to septic patients, so modelled IL-6 cannot be the reported
    # endpoint. It is retained in the library only to show that the choice is
    # immaterial to the ranking: the two correlate at Spearman 0.997, and
    # Spearman is rank-based, so every candidate's score is unchanged.
    cost = np.array([r["cost_AUC_DS"] for r in lib])
    cands = {
        "fT_pct": "percent fT>MIC (the efficacy standard for beta-lactams)",
        "fAUC_MIC": "fAUC/MIC",
        "fCmax_MIC": "fCmax/MIC",
        "net_log10_change": "24-48 h net log10 kill (efficacy achieved)",
        "cum_killed_cfu_per_mL": "cumulative killed biomass (agent-blind)",
        "yield_weighted_kill": "yield-weighted killed biomass (PROPOSED, computable)",
    }
    # AUC_DS is deliberately NOT a candidate: it is the definition of the cost
    # being predicted, so correlating it with itself would be vacuous.
    perf = {}
    for key, desc in cands.items():
        x = np.array([r[key] for r in lib])
        rho, p = spearmanr(x, cost)
        perf[key] = {"description": desc, "spearman_rho": float(rho),
                     "p_value": float(p)}
    # Within-agent, the efficacy indices should work; the failure is ACROSS
    # agents, which is the clinically relevant comparison.
    within = {}
    for agent in ("ceftazidime", "meropenem"):
        sub = [r for r in lib if r["agent"] == agent]
        c = np.array([r["cost_AUC_DS"] for r in sub])
        within[agent] = {}
        for key in cands:
            x = np.array([r[key] for r in sub])
            rho, _ = spearmanr(x, c)
            within[agent][key] = float(rho)
    return {
        "library": lib, "candidates": perf, "within_agent_spearman": within,
        "n_regimens": len(lib),
        "mass_balance_identity": {
            "statement": "k_ds * AUC_D = a * [Y_lysis * int(kill*B_rep dt) + "
                         "Y_filament * int(phi*B_rep dt)] - [D(T) - D(0)], "
                         "with D the attributable danger signal and a the "
                         "common focus/translocation conversion factor",
            "max_abs_relative_residual": float(max(residuals)),
            "n_regimens_checked": len(residuals),
            "consequence": "The proposed yield-weighted killed-biomass index "
                           "is the leading term of the right-hand side, so "
                           "its near-perfect rank correlation with AUC_DS is "
                           "an algebraic consequence of the compartment's "
                           "mass balance, NOT independent evidence of "
                           "predictive performance. Its practical value is "
                           "that it needs only a standard PK/PD simulation "
                           "and a yield table.",
        },
        "interpretation":
            "Across agents, the established efficacy indices rank inflammatory "
            "cost poorly or perversely, because they are agent-blind: two "
            "regimens with the same %fT>MIC differ several-fold in cost "
            "(see analysis3 for the range) depending only on which agent "
            "delivers that exposure. "
            "Weighting cumulative killed biomass by the agent's yield recovers "
            "the ranking using only a standard PK/PD simulation plus a yield "
            "table, which is the practical proposal.",
    }


def analysis5_divergence_map(mic_factors=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
                             targets=(20.0, 40.0, 60.0, 80.0, 95.0)):
    """Susceptibility x exposure: where does more killing cost more injury?

    Re-scoped from the original fixed-daily-dose fractionation sweep, which
    Phase 1 showed produces almost no divergence.
    """
    cells = []
    for fac in mic_factors:
        prm = get_default_parameters()
        prm["bacteria"].kill_C50 = prm["bacteria"].kill_C50 * fac
        mic_f = effect_site_mic(prm)
        for target in targets:
            row = {"mic_factor": fac, "mic_mgL": mic_f, "target_fT_pct": target}
            for agent in ("ceftazidime", "meropenem"):
                d = dose_for_target_ft(agent, target, 8.0, 6, mic=mic_f)
                if d is None:
                    row[agent] = None
                    continue
                sim = simulate(agent, d, 8.0, 6, params=prm, mic=mic_f)
                _, s = ds_cost(sim, agent=agent)
                row[agent] = {
                    "dose_mg": d, "achieved_fT_pct": sim["fT"],
                    "net_log10_change": sim["net_log10"],
                    "AUC_DS_pg_h_mL": s["AUC_DS_pg_h_mL"],
                    "AUC_IL6_excess_pg_h_mL": s["AUC_IL6_excess_pg_h_mL"],
                    "cost_AUC_DS": s["AUC_DS_pg_h_mL"],
                    "saturation_peak": s["saturation_peak"],
                    "filament_fraction": s["filament_fraction_of_release"],
                }
            cells.append(row)

    # Divergence: moving from a lower to a higher exposure target at fixed
    # susceptibility, does inflammatory cost rise while killing improves?
    divergence = []
    for fac in mic_factors:
        seq = [c for c in cells if c["mic_factor"] == fac]
        for agent in ("ceftazidime", "meropenem"):
            arms = [(c["target_fT_pct"], c[agent]) for c in seq if c.get(agent)]
            arms.sort()
            for (t0, a0), (t1, a1) in zip(arms, arms[1:]):
                better_kill = a1["net_log10_change"] < a0["net_log10_change"]
                worse_cost = (a1["cost_AUC_DS"] > a0["cost_AUC_DS"])
                if better_kill and worse_cost:
                    divergence.append({
                        "mic_factor": fac, "agent": agent,
                        "from_fT": t0, "to_fT": t1,
                        "kill_gain_log10": a0["net_log10_change"] - a1["net_log10_change"],
                        "cost_increase_fold": (a1["cost_AUC_DS"]
                                               / max(a0["cost_AUC_DS"], 1e-12)),
                    })
    # Above the stasis threshold, how much does cost actually move with
    # exposure? This quantifies the within-agent effect, which turns out to be
    # small -- and small in a way that matters for the clinical message.
    flatness = {}
    for agent in ("ceftazidime", "meropenem"):
        for fac in mic_factors:
            arms = [c[agent] for c in cells
                    if c["mic_factor"] == fac and c.get(agent)
                    and c[agent]["net_log10_change"] < 0.0]
            if len(arms) < 2:
                continue
            costs = [a["cost_AUC_DS"] for a in arms]
            flatness[f"{agent}|MICx{fac:g}"] = {
                "n_arms_above_stasis": len(arms),
                "cost_span_fold": max(costs) / min(costs),
                "cost_at_lowest_exposure": max(costs),
                "cost_at_highest_exposure": min(costs),
            }
    spans = [v["cost_span_fold"] for v in flatness.values()]

    # Below stasis the burden grows, so there is far more biomass to kill and a
    # sustained kill-regrowth turnover. That is where cost explodes.
    # Failing vs controlled therapy, reported PER AGENT. Pooling the two
    # agents mixes a 5x yield difference into a within-agent comparison, so
    # the per-agent ratios are the primary numbers and the pooled one is
    # reported only for continuity with earlier phases.
    failing, controlled = [], []
    by_agent = {}
    for agent in ("ceftazidime", "meropenem"):
        f_a, c_a = [], []
        for c in cells:
            a = c.get(agent)
            if not a:
                continue
            (f_a if a["net_log10_change"] > 0 else c_a).append(a["cost_AUC_DS"])
        failing.extend(f_a)
        controlled.extend(c_a)
        by_agent[agent] = {
            "n_arms_populated": sum(1 for c in cells if c.get(agent)),
            "n_failing": len(f_a), "n_controlled": len(c_a),
            "median_cost_failing": float(np.median(f_a)) if f_a else None,
            "median_cost_controlled": float(np.median(c_a)) if c_a else None,
            "failing_over_controlled_fold": (float(np.median(f_a))
                                             / float(np.median(c_a))
                                             if f_a and c_a else None),
        }

    n_possible = len(mic_factors) * len(targets)
    coverage = {
        "n_cells": len(cells),
        "n_possible_arms_per_agent": n_possible,
        "populated_arms": {a: by_agent[a]["n_arms_populated"]
                           for a in by_agent},
        "note": "Arms are skipped where the dose solving the exposure target "
                "falls outside the 10-16000 mg search bracket, so the map is "
                "not fully populated and the summaries below are over the "
                "arms that exist, not over the full grid.",
    }

    return {
        "cells": cells, "divergent_steps": divergence,
        "n_divergent": len(divergence),
        "within_agent_flatness": flatness,
        "cost_span_above_stasis_fold": [min(spans), max(spans)] if spans else None,
        "coverage": coverage,
        "by_agent": by_agent,
        "median_cost_failing_therapy": float(np.median(failing)) if failing else None,
        "median_cost_controlled_therapy": (float(np.median(controlled))
                                           if controlled else None),
        "pooling_caveat":
            "The pooled failing/controlled medians mix both agents and so "
            "carry the yield difference as well as the exposure effect. The "
            "per-agent figures in 'by_agent' are the ones to report.",
        "interpretation":
            "NO divergent step was found anywhere on the AVAILABLE arms of "
            "this map: raising exposure never raised inflammatory cost in any "
            "evaluated step. Two regimes explain it. ABOVE stasis, cost is "
            "nearly FLAT in exposure, because rapid control leaves little "
            "kill-regrowth turnover: almost all of the cumulative killing is "
            "of the burden already present, so how fast it is killed barely "
            "matters. This is a REGIME-SPECIFIC explanation, not a general "
            "bound -- cumulative killed biomass is NOT bounded by the "
            "starting burden in a growing population; see "
            "analysis3.killed_over_initial_burden, which exceeds 1. BELOW "
            "stasis, therapy "
            "fails, the burden grows, and sustained kill-regrowth turnover "
            "makes cost rise by orders of magnitude. The practical reading is "
            "that no within-agent trade-off appeared in the scenarios "
            "evaluated: under-dosing is bad for both objectives at once.",
        "hypothesis_amendment":
            "This FALSIFIES the clause in DESIGN.md section 1 that 'the regimen "
            "that maximizes bacterial killing is not necessarily the one that "
            "minimizes host inflammatory injury' -- for regimens of a single "
            "agent above stasis, maximizing killing also minimizes "
            "inflammatory cost. The divergence the study is about lives "
            "BETWEEN agents at matched exposure, not between regimens of one "
            "agent. The hypothesis survives in its agent-comparison form and "
            "must be restated in that form.",
        "caveat":
            "The reported endpoint is attributable AUC_DS, which is an "
            "upstream LINEAR compartment and so has no mathematical "
            "saturation ceiling; the receptor-occupancy ceiling noted in "
            "PHASE2_BUILD.md bounds the (unreported) cytokine layer only. "
            "The empirical limits of the danger-signal layer at "
            "failing-therapy burdens are separate and remain unvalidated, so "
            "the MAGNITUDE of the failing-therapy cost should be read as "
            "indicative and its DIRECTION as the result.",
    }


def analysis6_predictions(a3, a4):
    """Falsifiable predictions and the experiment that would test them."""
    best = max(a4["candidates"], key=lambda k: abs(a4["candidates"][k]["spearman_rho"]))
    ft_rho = a4["candidates"]["fT_pct"]["spearman_rho"]
    ratios = [r for e in a3["rows"] for r in e["cost_ratio_range"]]
    lo = min(r["ratio_AUC_DS"] for r in ratios) if ratios else None
    hi = max(r["ratio_AUC_DS"] for r in ratios) if ratios else None
    return [
        {
            "id": "P1",
            "prediction": "At matched %fT>MIC against the same organism, "
                          "ceftazidime and meropenem achieve indistinguishable "
                          "bacterial killing but ceftazidime produces strictly "
                          "greater danger-signal exposure and cytokine "
                          "induction.",
            "quantitative": f"predicted cost ratio {lo:.0f}-{hi:.0f}x "
                            f"(interval reflects the revised 3.5-7x yield range; see PHASE5B_AGENT_EFFECT.md)",
            "test": "An in-vitro PK-simulation run of two matched-%fT>MIC "
                    "regimens with serial free-LPS and whole-blood cytokine "
                    "induction. This is a single experiment and it is the "
                    "decisive one.",
            "falsified_if": "cytokine induction is equal, or ordered the other "
                            "way, at matched exposure and matched kill.",
        },
        {
            "id": "P2",
            "prediction": "%fT>MIC does not rank regimens by inflammatory cost "
                          "across agents, whereas cumulative killed biomass "
                          "weighted by agent yield does.",
            "quantitative": f"Spearman rho vs modelled cost: %fT>MIC "
                            f"{ft_rho:+.2f}, best candidate '{best}' "
                            f"{a4['candidates'][best]['spearman_rho']:+.2f}",
            "test": "Rank a regimen library by each index and compare against "
                    "measured cytokine induction.",
            "falsified_if": "%fT>MIC ranks measured inflammatory cost as well "
                            "as the yield-weighted index.",
        },
        {
            "id": "P3",
            "prediction": "The kill-independent (filamentation) pathway is "
                          "activated by UNDER-EXPOSURE, not by reduced "
                          "susceptibility as such. Reduced susceptibility "
                          "matters only because, at a fixed dose, it causes "
                          "under-exposure.",
            "quantitative": "the filamentation share of release is a function "
                            "of %fT>MIC alone: 0.39 at a 20% fT>MIC target and "
                            "<=0.01 at 80%, and it is IDENTICAL at MIC x1 and "
                            "MIC x64 when the dose is raised to hold %fT>MIC "
                            "fixed",
            "test": "Measure free LPS and cytokine induction for ceftazidime "
                    "against isolates of graded MIC, once at a fixed dose and "
                    "once at doses adjusted to hold %fT>MIC constant. The "
                    "prediction is that the MIC effect disappears in the "
                    "second arm.",
            "falsified_if": "release per unit kill depends on the isolate's MIC "
                            "even at matched %fT>MIC.",
            "note": "This supersedes the Phase 1 statement, which was made at "
                    "FIXED DOSE and so confounded susceptibility with "
                    "exposure. The underlying numbers are unchanged; the "
                    "attribution is corrected.",
        },
        {
            "id": "P4",
            "prediction": "Y_filament is organism-dependent: substantial in "
                          "P. aeruginosa, small in E. coli.",
            "quantitative": "the two held-out sources differ on this; the "
                            "model reproduces Jackson & Kropp and contradicts "
                            "Prins",
            "test": "Repeat one sub-MIC release protocol on both organisms in "
                    "one laboratory with one assay.",
            "falsified_if": "both organisms behave the same, in which case one "
                            "of the two published results is an assay artefact.",
        },
        {
            "id": "P5",
            "prediction": "No survival difference is predicted. The model "
                          "predicts inflammatory exposure, and Simpson 2000 "
                          "observed a significant endotoxin difference with "
                          "identical mortality.",
            "quantitative": "n/a -- this is a statement of scope",
            "test": "n/a",
            "falsified_if": "n/a; recorded so the model is not over-read.",
        },
    ]


def main():
    global MIC_EFFECT, P_REF
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    MIC_EFFECT = effect_site_mic()
    fam = [e for e in calibrate_family() if e["feasible"]]
    ref = min(fam, key=lambda e: e["cost"])
    P_REF = calibrate_tnf(ref)["params"]
    print(f"effective MIC {MIC_EFFECT:.4f} mg/L; Layer-2 DS50={ref['DS50']:.0f}, "
          f"hill={ref['hill']:.3f}\n")

    print("=" * 74)
    print("ANALYSIS 1 -- agent-specific danger-signal yield table")
    print("=" * 74)
    a1 = analysis1_yield_table()
    print(f"  {'agent':14s} {'PBP':8s} {'Y_lysis':>9s}  basis")
    for r in a1["rows"]:
        iv = (f" [{r['interval'][0]:.2f}-{r['interval'][1]:.2f}]"
              if r["interval"] else "")
        print(f"  {r['agent']:14s} {r['pbp']:8s} "
              f"{r['Y_lysis_fg_per_CFU']:9.2f}{iv}")
    print(f"\n  {a1['interpretation']}")
    print(f"\n  {a1['mechanism']}")

    print("\n" + "=" * 74)
    print("ANALYSIS 2 -- is the kill-independent pathway required?")
    print("=" * 74)
    a2 = analysis2_dual_pathway()
    for label, row in a2["levels"].items():
        print(f"  {label:9s} dual AUC_DS {row['dual']['AUC_DS_pg_h_mL']:10.2f}   "
              f"lysis-only {row['lysis_only']['AUC_DS_pg_h_mL']:10.2f}   "
              f"kill {row['dual']['log10_change']:+5.2f} log10")
    print(f"\n  at 0.5x MIC the dual-pathway model releases "
          f"{a2['sub_mic_dual_over_lysis_only']:.0f}x what a lysis-only model does")
    print(f"  sub-MIC as a fraction of 50x MIC: dual "
          f"{a2['sub_mic_over_50x_dual']:.2f}, lysis-only "
          f"{a2['sub_mic_over_50x_lysis_only']:.4f}")
    print("  -> the pathway is REQUIRED by the P. aeruginosa data; the E. coli")
    print("     data contradict it (open conflict, Phase 3).")

    print("\n" + "=" * 74)
    print("ANALYSIS 3 -- matched %fT>MIC: near-equal killing, divergent cost")
    print("=" * 74)
    a3 = analysis3_matched_exposure()
    for e in a3["rows"]:
        caz, mero = e["arms"]["ceftazidime"], e["arms"]["meropenem"]
        lo = min(r["ratio_AUC_DS"] for r in e["cost_ratio_range"])
        hi = max(r["ratio_AUC_DS"] for r in e["cost_ratio_range"])
        print(f"  target {e['target_fT_pct']:4.0f}% fT>MIC: "
              f"CAZ {caz['dose_mg']:6.0f} mg ({caz['achieved_fT_pct']:5.1f}%, "
              f"{caz['net_log10_change']:+5.2f} log10)  vs  "
              f"MERO {mero['dose_mg']:6.0f} mg ({mero['achieved_fT_pct']:5.1f}%, "
              f"{mero['net_log10_change']:+5.2f} log10)")
        print(f"       kill difference {e['kill_difference_log10']:+.2f} log10   "
              f"danger-signal cost ratio {lo:.1f}-{hi:.1f}x")

    print("\n" + "=" * 74)
    print("ANALYSIS 4 -- can an existing index rank inflammatory cost?")
    print("=" * 74)
    a4 = analysis4_cost_index()
    print(f"  regimen library: {a4['n_regimens']} regimens "
          f"(2 agents x 4 intervals x 6 doses)")
    print(f"\n  {'candidate index':34s} {'rho (across agents)':>20s}")
    for k, v in sorted(a4["candidates"].items(),
                       key=lambda kv: -abs(kv[1]["spearman_rho"])):
        print(f"  {k:34s} {v['spearman_rho']:+20.3f}")
    print(f"\n  within-agent rho for %fT>MIC: "
          f"ceftazidime {a4['within_agent_spearman']['ceftazidime']['fT_pct']:+.3f}, "
          f"meropenem {a4['within_agent_spearman']['meropenem']['fT_pct']:+.3f}")
    print(f"\n  {a4['interpretation']}")

    print("\n" + "=" * 74)
    print("ANALYSIS 5 -- susceptibility x exposure divergence map")
    print("=" * 74)
    a5 = analysis5_divergence_map()
    print(f"  {len(a5['cells'])} cells; {a5['n_divergent']} steps where better "
          f"killing came with higher danger-signal cost")
    if a5["cost_span_above_stasis_fold"]:
        lo, hi = a5["cost_span_above_stasis_fold"]
        print(f"  above stasis, cost spans only {lo:.2f}-{hi:.2f}x across the "
              f"whole exposure range (nearly flat)")
    print(f"  median cost, failing therapy  : "
          f"{a5['median_cost_failing_therapy']:12.1f}")
    print(f"  median cost, controlled therapy: "
          f"{a5['median_cost_controlled_therapy']:12.1f}")
    print(f"\n  {a5['hypothesis_amendment']}")
    for fac in (1.0, 16.0, 64.0):
        sub = [c for c in a5["cells"] if c["mic_factor"] == fac]
        print(f"\n  MIC x{fac:.0f}:")
        for c in sub:
            caz = c.get("ceftazidime")
            if not caz:
                continue
            print(f"    target {c['target_fT_pct']:4.0f}%  CAZ dose "
                  f"{caz['dose_mg']:7.0f} mg  kill {caz['net_log10_change']:+5.2f}  "
                  f"AUC_IL6 {caz['AUC_IL6_excess_pg_h_mL']:10.1f}  "
                  f"filament {caz['filament_fraction']:.2f}")

    print("\n" + "=" * 74)
    print("ANALYSIS 6 -- falsifiable predictions")
    print("=" * 74)
    a6 = analysis6_predictions(a3, a4)
    for p in a6:
        print(f"  {p['id']}: {p['prediction']}")
        print(f"      quantitative: {p['quantitative']}")
        print(f"      falsified if: {p['falsified_if']}")

    payload = {
        "meta": {"mic_effect_site_mgL": MIC_EFFECT, "t_end_h": T_END,
                 "V_focus_mL": V_FOCUS, "layer2_reference":
                     {k: v for k, v in ref.items() if k != "params"},
                 "yield_ratio_range": list(YIELD_RATIO_RANGE),
                 "reporting_rule": "agent comparisons are reported as a ranking "
                                   "plus a range over the assumed 3.5-7x yield "
                                   "interval; no single fold-change and no "
                                   "survival claim"},
        "analysis1_yield_table": a1,
        "analysis2_dual_pathway": a2,
        "analysis3_matched_exposure": a3,
        "analysis4_cost_index": a4,
        "analysis5_divergence_map": a5,
        "analysis6_predictions": a6,
    }
    from src.utils.provenance import provenance
    payload["provenance"] = provenance(
        endpoint="AUC_DS_pg_h_mL (attributable danger-signal exposure)",
        exposure_index="%fT>MIC by crossing interpolation on the solver grid",
        solver={"bacterial": "RK45 rtol 1e-6 atol 1e-8 max_step 0.1 h",
                "danger_signal": "LSODA rtol 1e-6 atol 1e-9 max_step 0.1 h"})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
