"""Layer-2 calibration: the danger-signal to cytokine response, fitted to
human experimental endotoxemia.

The Phase 1 pilot ran on an *assumed* cytokine dose-response (DS50 = 100 pg/mL,
Hill coefficient 1.0) anchored on a single human endotoxemia dose. That was
enough to show the layer is not a binary detector, but not enough to build on:
the design document requires the cytokine layer to be calibrated to a measured
dose-response, with its operating point reported.

**Why human in vivo data rather than the baboon dose-response.** Haudek 2003
spans six orders of magnitude of LPS dose, which is attractive, but the paper
itself reports that baboons need ~1e4-fold more LPS than humans for equivalent
manifestations. It can therefore supply the *shape* but not an absolute human
scale. Hijma et al. gave two intravenous LPS doses to humans in one study, with
one assay and one population, and reported peak cytokine concentrations for
both -- which is exactly a two-point concentration-response on the scale this
model needs.

Calibration data (Hijma HJ et al., Brain Behav Immun 2020; 12 subjects per dose,
E. coli LPS, 2-minute infusion):

===========  ==========  ==========  =========
dose         TNF peak    IL-6 peak   t(peak)
===========  ==========  ==========  =========
1 ng/kg      66.49       65.55       1 h / 2 h
2 ng/kg      249.35      200.88      1 h / 2 h
===========  ==========  ==========  =========

The IL-6 ratio over a two-fold dose step is 200.88/65.55 = 3.06, i.e. a local
log-log slope of log2(3.06) = **1.62** -- the response is markedly steeper than
proportional. The pilot assumed 1.0. This matters, because in the unsaturated
regime the cytokine response goes as DS^h, so a steeper slope *amplifies*
agent-to-agent differences in danger-signal yield rather than transmitting them
one-for-one.

**Identifiability, stated honestly.** Two dose levels cannot separate the Hill
coefficient from DS50: a family of (hill, DS50) pairs reproduces the same local
slope, trading curvature against saturation. What the data *do* fix is
(i) the local log-log slope, (ii) the absolute production scale needed to reach
the observed peak, and (iii) a lower bound on DS50, since a DS50 near the
observed plasma concentrations would flatten the response below the measured
ratio. ``calibrate_family`` therefore returns the whole identified family
rather than a single point estimate, and every downstream conclusion is meant
to be repeated across it. That is the opposite of what was done with the old
host-damage layer, whose single unmeasured I50 turned out to decide everything.
"""

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from src.analysis.danger_signal import (
    DangerSignalParameters, hill_occupancy, integrate_ds_cytokine,
)

#: Hijma HJ, van Doorn ECH, Klaassen ES, Groeneveld GJ, Moerland M.
#: "Challenging the challenge: A randomized controlled trial evaluating the
#: inflammatory response and pain perception of healthy volunteers after
#: single-dose LPS administration, as a potential model for inflammatory pain
#: in early-phase drug development." Brain Behav Immun 2020;88:515-528.
#: doi:10.1016/j.bbi.2020.04.033. Values read from section 3.2 of the paper.
HIJMA_2020 = {
    "source": ("Hijma HJ et al. Brain Behav Immun 2020;88:515-528. "
               "doi:10.1016/j.bbi.2020.04.033"),
    "lps_source": "E. coli purified LPS, List Biological Laboratories Lot#94332B4",
    "n_per_dose": 12,
    "infusion_min": 2.0,
    "doses_ng_per_kg": [1.0, 2.0],
    "IL6_peak_pg_mL": [65.55, 200.88],
    "TNF_peak_pg_mL": [66.49, 249.35],
    "t_peak_IL6_h": 2.0,
    "t_peak_TNF_h": 1.0,
}

#: Dillingh MR et al. J Inflamm (Lond) 2014;11:28. doi:10.1186/s12950-014-0028-1
#: Single ascending doses 0.5/1/2 ng/kg, 6 LPS subjects per cohort. Used as an
#: independent qualitative check on the lower end: at 0.5 ng/kg the cytokine
#: response is still clearly distinguishable from placebo (0-6 h contrast
#: TNF +413%, IL-6 +288%), so the response threshold lies below the ~12 pg/mL
#: plasma concentration that dose implies. A 0.2 ng/kg dose elicited no
#: response (their ref 11), which brackets the foot of the curve.
DILLINGH_2014 = {
    "source": ("Dillingh MR et al. J Inflamm (Lond) 2014;11:28. "
               "doi:10.1186/s12950-014-0028-1"),
    "detectable_at_ng_per_kg": 0.5,
    "no_response_at_ng_per_kg": 0.2,
    "IL6_contrast_pct_0_6h_at_0_5": 288.0,
}

WEIGHT_KG = 70.0


def bolus_plasma_ds0(dose_ng_per_kg: float, p: DangerSignalParameters,
                     weight_kg: float = WEIGHT_KG) -> float:
    """Initial plasma danger-signal concentration (pg/mL) after an IV bolus.

    dose [ng/kg] * weight [kg] = ng into the plasma volume:
    1 ng in 1 L is 1e3 pg / 1e3 mL = 1 pg/mL, so pg/mL = ng_total / V_L.
    """
    return dose_ng_per_kg * weight_kg / p.V_ds_L


def simulate_endotoxemia(dose_ng_per_kg: float, p: DangerSignalParameters,
                         t_end: float = 10.0, n: int = 2001,
                         weight_kg: float = WEIGHT_KG) -> Dict[str, np.ndarray]:
    """Simulate an intravenous LPS bolus through the SAME cytokine layer used
    for the infection simulations (no bacteria, no release -- the danger signal
    is injected directly)."""
    t = np.linspace(0.0, t_end, n)
    src = np.zeros_like(t)
    DS0 = bolus_plasma_ds0(dose_ng_per_kg, p, weight_kg)
    DS, IL6, TNF, occ, tol = integrate_ds_cytokine(t, src, p, DS0=DS0)
    return {"t": t, "DS": DS, "IL6": IL6, "TNF": TNF, "saturation": occ,
            "tolerance": tol}


def predict_peaks(p: DangerSignalParameters,
                  doses: Optional[List[float]] = None) -> Dict[str, List[float]]:
    """Peak IL-6/TNF and their timing for each calibration dose."""
    doses = doses or HIJMA_2020["doses_ng_per_kg"]
    il6, tnf, t_il6, t_tnf, occ = [], [], [], [], []
    for d in doses:
        r = simulate_endotoxemia(d, p)
        i6, itnf = int(np.argmax(r["IL6"])), int(np.argmax(r["TNF"]))
        il6.append(float(r["IL6"][i6]))
        tnf.append(float(r["TNF"][itnf]))
        t_il6.append(float(r["t"][i6]))
        t_tnf.append(float(r["t"][itnf]))
        occ.append(float(np.max(r["saturation"])))
    return {"IL6_peak": il6, "TNF_peak": tnf, "t_peak_IL6": t_il6,
            "t_peak_TNF": t_tnf, "saturation_peak": occ}


def _scale_to_observed_peak(p: DangerSignalParameters,
                            dose: float, target_peak: float) -> float:
    """Solve for the IL-6 production scale that reproduces one observed peak.

    The IL-6 equation is linear in ``k_IL6_max``, but the peak is not an
    explicit function of it (the time of the peak shifts), so this is solved
    numerically rather than analytically. Monotone in k_IL6_max, hence a bracket
    search is safe.
    """
    def err(k):
        q = replace(p, k_IL6_max=float(k))
        r = simulate_endotoxemia(dose, q)
        return float(np.max(r["IL6"])) - target_peak

    lo, hi = 1e-3, 1e3
    while err(hi) < 0 and hi < 1e9:
        hi *= 10.0
    return float(brentq(err, lo, hi, xtol=1e-6, rtol=1e-8))


def calibrate_family(
    ds50_grid: Optional[List[float]] = None,
    k_ds_clear: float = 13.9,
    base: Optional[DangerSignalParameters] = None,
) -> List[Dict]:
    """Calibrate (hill, k_IL6_max, k_signal_off) at each DS50 on a grid.

    Four observables are fitted at each DS50 -- the absolute IL-6 peak at both
    doses and the time-to-peak at both doses -- with three free parameters, so
    each fit is slightly overdetermined and its residuals are meaningful rather
    than zero by construction.

    DS50 is NOT fitted, because two dose levels cannot separate it from the
    Hill coefficient. It is gridded instead, and every entry returned fits the
    human data; the spread across entries IS the Layer-2 uncertainty. Collapsing
    it to a point estimate is what went wrong with the old host-damage layer,
    whose single unmeasured I50 silently decided the entire result.
    """
    from scipy.optimize import least_squares

    ds50_grid = ds50_grid or [50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    base = base or DangerSignalParameters()
    doses = HIJMA_2020["doses_ng_per_kg"]
    obs = HIJMA_2020["IL6_peak_pg_mL"]
    t_obs = HIJMA_2020["t_peak_IL6_h"]

    out = []
    for ds50 in ds50_grid:
        def build(theta):
            hill, k_il6, k_sig = np.exp(theta)
            return replace(base, DS50=float(ds50), hill_ds=float(hill),
                           k_IL6_max=float(k_il6), k_signal_off=float(k_sig),
                           k_ds_clear=k_ds_clear)

        def resid(theta):
            pk = predict_peaks(build(theta), doses)
            r = []
            for i in range(len(doses)):
                r.append(np.log(max(pk["IL6_peak"][i], 1e-12) / obs[i]))
                # Time-to-peak weighted down: it constrains the transduction
                # rate but is read off a sampling schedule, so it is coarser
                # information than the concentrations.
                r.append(0.5 * np.log(max(pk["t_peak_IL6"][i], 1e-3) / t_obs))
            return r

        theta0 = np.log([1.7, 2e4, 0.5])
        sol = least_squares(resid, theta0, method="trf",
                            bounds=(np.log([0.3, 1e1, 0.02]),
                                    np.log([6.0, 1e10, 20.0])),
                            xtol=1e-10, ftol=1e-10)
        q = build(sol.x)
        pk = predict_peaks(q, doses)
        rel_err = [abs(pk["IL6_peak"][i] / obs[i] - 1.0) for i in range(len(doses))]
        out.append({
            "DS50": ds50,
            "feasible": bool(max(rel_err) < 0.10
                             and abs(pk["t_peak_IL6"][0] / t_obs - 1.0) < 0.30),
            "hill": float(np.exp(sol.x[0])),
            "k_IL6_max": float(np.exp(sol.x[1])),
            "k_signal_off": float(np.exp(sol.x[2])),
            "k_ds_clear": k_ds_clear,
            "pred_IL6_peak": pk["IL6_peak"],
            "obs_IL6_peak": list(obs),
            "max_rel_peak_error": float(max(rel_err)),
            "pred_t_peak_IL6_h": pk["t_peak_IL6"],
            "obs_t_peak_IL6_h": t_obs,
            "saturation_peak_at_2ng_kg": pk["saturation_peak"][1],
            "plasma_DS0_pg_mL": [bolus_plasma_ds0(d, q) for d in doses],
            "cost": float(sol.cost),
            "params": q,
        })
    return out


def observed_log_slope() -> float:
    """Local log-log slope of the human IL-6 response between the two doses."""
    obs1, obs2 = HIJMA_2020["IL6_peak_pg_mL"]
    d1, d2 = HIJMA_2020["doses_ng_per_kg"]
    base = DangerSignalParameters()
    r1, r2 = obs1 - base.IL6_baseline, obs2 - base.IL6_baseline
    return float(np.log(r2 / r1) / np.log(d2 / d1))


def calibrate_tnf(entry: Dict) -> Dict[str, object]:
    """Calibrate the TNF arm against the same human endotoxemia data.

    Phase 2 left TNF uncalibrated, which was acceptable only because every
    reported analysis used IL-6. Hijma reports TNF peaks of 66.49 and
    249.35 pg/mL at **1 h**, i.e. comparable in magnitude to IL-6 but about an
    hour earlier. Two parameters are fitted: the production ratio relative to
    IL-6 and the TNF elimination rate (which sets how much earlier it peaks).
    """
    from scipy.optimize import least_squares

    p = entry["params"]
    doses = HIJMA_2020["doses_ng_per_kg"]
    obs = HIJMA_2020["TNF_peak_pg_mL"]
    t_obs = HIJMA_2020["t_peak_TNF_h"]

    def build(theta):
        ratio, k_clear = np.exp(theta)
        return replace(p, TNF_IL6_ratio=float(ratio), k_TNF_clear=float(k_clear))

    def resid(theta):
        q = build(theta)
        r = []
        for d, o in zip(doses, obs):
            sim = simulate_endotoxemia(d, q)
            i = int(np.argmax(sim["TNF"]))
            r.append(np.log(max(float(sim["TNF"][i]), 1e-12) / o))
            r.append(0.5 * np.log(max(float(sim["t"][i]), 1e-3) / t_obs))
        return r

    sol = least_squares(resid, np.log([0.3, 0.3]), method="trf",
                        bounds=(np.log([0.01, 0.05]), np.log([20.0, 20.0])),
                        xtol=1e-10, ftol=1e-10)
    q = build(sol.x)
    pred, tpk = [], []
    for d in doses:
        sim = simulate_endotoxemia(d, q)
        i = int(np.argmax(sim["TNF"]))
        pred.append(float(sim["TNF"][i]))
        tpk.append(float(sim["t"][i]))
    return {
        "TNF_IL6_ratio": float(np.exp(sol.x[0])),
        "k_TNF_clear": float(np.exp(sol.x[1])),
        "pred_TNF_peak": pred, "obs_TNF_peak": list(obs),
        "pred_t_peak_TNF_h": tpk, "obs_t_peak_TNF_h": t_obs,
        "max_rel_error": float(max(abs(pr / o - 1.0)
                                   for pr, o in zip(pred, obs))),
        "params": q,
    }


def steady_state_tolerance(ds_pg_ml: float, p: DangerSignalParameters) -> float:
    """Tolerance reached by a subject held at a constant danger-signal level.

    dT/dt = k_tol_on * occ * (1 - T) - k_tol_off * T  =>
    T_ss = k_on*occ / (k_on*occ + k_off).
    """
    from src.analysis.danger_signal import hill_occupancy
    occ = float(hill_occupancy(ds_pg_ml, p))
    num = p.k_tol_on * occ
    return float(num / (num + p.k_tol_off)) if (num + p.k_tol_off) > 0 else 0.0


def calibrate_tolerance(p: DangerSignalParameters,
                        pretreatment_ds_pg_ml: float,
                        required_suppression_fold: float) -> Dict[str, object]:
    """Solve for k_tol_on from the suppression a septic patient shows.

    ``required_suppression_fold`` is how much smaller the therapy-attributable
    cytokine rise is in a chronically endotoxemic patient than the naive-
    subject calibration predicts. Setting tol_max = 1, the steady-state
    tolerance must satisfy (1 - T_ss) = 1 / fold.
    """
    from scipy.optimize import brentq
    from src.analysis.danger_signal import hill_occupancy

    target_T = 1.0 - 1.0 / float(required_suppression_fold)
    occ = float(hill_occupancy(pretreatment_ds_pg_ml, p))
    if occ <= 0:
        raise ValueError("pre-treatment danger signal gives zero occupancy")
    # T_ss = k_on*occ / (k_on*occ + k_off)  =>  k_on = k_off*T/(occ*(1-T))
    k_on = p.k_tol_off * target_T / (occ * (1.0 - target_T))
    q = replace(p, k_tol_on=float(k_on), tol_max=1.0)
    return {
        "k_tol_on": float(k_on),
        "k_tol_off": p.k_tol_off,
        "pretreatment_DS_pg_mL": pretreatment_ds_pg_ml,
        "pretreatment_occupancy": occ,
        "target_tolerance": target_T,
        "achieved_tolerance": steady_state_tolerance(pretreatment_ds_pg_ml, q),
        "suppression_fold": required_suppression_fold,
        "params": q,
    }


def tolerance_effect_on_calibration(p: DangerSignalParameters) -> Dict[str, object]:
    """Does the tolerance term disturb the healthy-volunteer calibration?

    It should not, and for a structural reason rather than a tuned one: an
    intravenous bolus is cleared from plasma within minutes, so it integrates
    almost no tolerance, whereas sustained endotoxemia integrates a lot. This
    function measures that, and it is the check that distinguishes a mechanism
    from a fudge factor.
    """
    naive = replace(p, k_tol_on=0.0)
    out = {}
    for dose, label in zip(HIJMA_2020["doses_ng_per_kg"], ("1ng", "2ng")):
        with_tol = simulate_endotoxemia(dose, p)
        without = simulate_endotoxemia(dose, naive)
        pk_w = float(np.max(with_tol["IL6"]))
        pk_o = float(np.max(without["IL6"]))
        out[label] = {
            "IL6_peak_with_tolerance": pk_w,
            "IL6_peak_without_tolerance": pk_o,
            "relative_change": float(pk_w / pk_o - 1.0),
            "peak_tolerance_reached": float(np.max(with_tol["tolerance"])),
        }
    return out


def check_lower_bound_consistency(entry: Dict) -> Dict[str, object]:
    """Does a calibrated entry respect the independent Dillingh constraints?

    Two qualitative facts must hold: 0.5 ng/kg produces a clearly detectable
    IL-6 rise, and 0.2 ng/kg produces essentially none. Both are from a
    different laboratory and are not used in the fit, so they act as a cheap
    held-out check on the foot of the curve.
    """
    p = entry["params"]
    lo = simulate_endotoxemia(DILLINGH_2014["no_response_at_ng_per_kg"], p)
    mid = simulate_endotoxemia(DILLINGH_2014["detectable_at_ng_per_kg"], p)
    rise_lo = float(np.max(lo["IL6"]) - p.IL6_baseline)
    rise_mid = float(np.max(mid["IL6"]) - p.IL6_baseline)
    # "Detectable" taken as a rise of at least 2 pg/mL over baseline (a
    # conservative reading of a 288% contrast on an assay with a ~1 pg/mL
    # lower limit); "no response" as a rise below that.
    return {
        "IL6_rise_at_0.2_ng_kg": rise_lo,
        "IL6_rise_at_0.5_ng_kg": rise_mid,
        "detectable_at_0.5": bool(rise_mid >= 2.0),
        "quiet_at_0.2": bool(rise_lo < rise_mid / 2.0),
        "monotone_foot": bool(rise_mid > rise_lo),
    }
