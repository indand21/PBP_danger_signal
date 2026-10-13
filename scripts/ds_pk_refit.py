"""Refit the pharmacokinetic layer, and say what it can and cannot identify.

The curve-level validation fails two of the targets the registry sets for it
(scripts/ds_validation_targets.py): the meropenem plasma RMSE and its
AUC(0-8h). Both follow from a terminal half-life that is too short, 0.81 h
against 1.15 h fitted from the digitised curve's own decay limb, which also
displaces the neutropenic-thigh stasis threshold.

WHAT IS ACTUALLY FITTABLE HERE. Despite its class name, the PK layer is a
ONE-compartment model: run_simulation uses CL, Vc, Ka and Kp, and Vp and Q do
not enter (see TwoCompartmentPKModel's docstring). So only CL and Vc are
identifiable from a plasma curve, and only those two are fitted. Vp and Q are
left alone because changing an inert parameter would be theatre.

THE TWO AGENTS ARE NOT IN THE SAME POSITION.

  meropenem    has a digitised plasma curve (Abulfathi 2021, 1 g IV), so CL
               and Vc are FITTED to it and scored against the registry's own
               acceptance targets.
  ceftazidime  has no digitised curve in this repository. Its parameters are
               ANCHORED to the measured values in the source the study cites
               (Leroy 1984, normal subjects, n=5: t1/2 1.57 h, apparent Vd
               0.230 L/kg), not fitted. That is a weaker basis and is
               reported as such rather than presented as a fit.

Writes results/danger_signal/pk_refit.json. Changes no default; adopting is
the author's decision, as with the bacterial refit.
"""
import json
import os
import sys

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.analysis.external_validation as ev  # noqa: E402
from src.core.parameters import get_drug_pk_parameters  # noqa: E402

OUT = os.path.join("results", "danger_signal", "pk_refit.json")
WEIGHT = 70.0
DATASET = "meropenem_pk_1g"


def _predict(ds, CL, Vc_per_kg):
    """Plasma curve at a candidate (CL, Vc), leaving everything else alone."""
    base = get_drug_pk_parameters

    def patched(name):
        p = base(name)
        if name.lower() == "meropenem":
            import dataclasses
            return dataclasses.replace(p, CL=CL, Vc=Vc_per_kg)
        return p

    ev.get_drug_pk_parameters = patched
    try:
        return np.asarray(ev.predict(ds.load()), dtype=float)
    finally:
        ev.get_drug_pk_parameters = base


def _score(ds, pred):
    """The registry's three targets, plus the terminal slope behind them."""
    d = ds.load()
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
    lim = (t >= 2.0) & (obs > 0) & (pred > 0)
    k = float(-np.polyfit(t[lim], np.log(pred[lim]), 1)[0])
    return {
        "rmse_log10": rmse, "rmse_target_met": bool(rmse < 0.15),
        "cmax_relative_error": cmax, "cmax_target_met": bool(abs(cmax) <= 0.20),
        "auc_0_8h_relative_error": auc, "auc_target_met": bool(abs(auc) <= 0.15),
        "terminal_half_life_h": 0.693 / k,
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = {d.id: d for d in ev.load_registry()}[DATASET]
    d = ds.load()
    obs = np.asarray(d.y_obs, dtype=float)
    t = np.asarray(d.t_obs, dtype=float)
    keep = obs > 0

    cur = get_drug_pk_parameters("meropenem")
    before = _score(ds, _predict(ds, cur.CL, cur.Vc))

    def objective(x):
        CL, Vc = float(np.exp(x[0])), float(np.exp(x[1]))
        if not (1.0 < CL < 60.0 and 0.05 < Vc < 1.5):
            return 1e3
        pred = _predict(ds, CL, Vc)
        r = (np.log10(np.maximum(pred[keep], 1e-3))
             - np.log10(np.maximum(obs[keep], 1e-3)))
        return float(np.sqrt(np.mean(r ** 2)))

    best = minimize(objective, x0=np.log([cur.CL, cur.Vc]),
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-8})
    CL_fit, Vc_fit = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    after = _score(ds, _predict(ds, CL_fit, Vc_fit))

    # Observed terminal slope, the quantity the fit has to reproduce.
    lim = (t >= 2.0) & (obs > 0)
    k_obs = float(-np.polyfit(t[lim], np.log(obs[lim]), 1)[0])

    # Ceftazidime: anchored, not fitted. A one-compartment model reproduces a
    # measured half-life exactly when V = CL * t_half / ln 2.
    cz = get_drug_pk_parameters("ceftazidime")
    leroy_t_half, leroy_vd = 1.57, 0.230
    vc_anchor = cz.CL * leroy_t_half / 0.693 / WEIGHT

    payload = {
        "what_is_identifiable":
            "The PK layer is one-compartment in use: run_simulation reads CL, "
            "Vc, Ka and Kp, and Vp and Q do not enter. Only CL and Vc are "
            "fitted; Vp and Q are left at their current values because they "
            "are inert.",
        "meropenem": {
            "basis": "fitted to the digitised plasma curve",
            "dataset": DATASET,
            "source": "Abulfathi AA et al. Front Pharmacol 2021;12:637618",
            "current": {"CL_L_per_h": cur.CL, "Vc_L_per_kg": cur.Vc,
                        "Vc_L_at_70kg": cur.Vc * WEIGHT, **before},
            "fitted": {"CL_L_per_h": CL_fit, "Vc_L_per_kg": Vc_fit,
                       "Vc_L_at_70kg": Vc_fit * WEIGHT, **after},
            "observed_terminal_half_life_h": 0.693 / k_obs,
            "published_popPK_for_comparison": {
                "CL_L_per_h": 11.8, "V1_L": 14.2, "Q_L_per_h": 3.26,
                "V2_L": 3.12,
                "note": "Abulfathi's own two-compartment estimates, recorded "
                        "in the registry. Not directly comparable to a "
                        "one-compartment Vc, but the steady-state volume "
                        "17.3 L is the relevant scale."},
            "mouton_review_for_comparison": {
                "Vss_L": 21.0, "t_half_h": 1.0,
                "source": "Mouton & van den Anker 1995, Clin Pharmacokinet "
                          "28:275-286"},
        },
        "ceftazidime": {
            "basis": "ANCHORED to the cited source, not fitted: this "
                     "repository holds no digitised ceftazidime curve",
            "source": "Leroy A et al. Antimicrob Agents Chemother "
                      "1984;25:638-642 (PMID 6375562), normal subjects n=5",
            "measured": {"t_half_h": leroy_t_half,
                         "apparent_Vd_L_per_kg": leroy_vd,
                         "central_volume_L_per_kg": 0.127,
                         "protein_binding_pct": 17},
            "current": {"CL_L_per_h": cz.CL, "Vc_L_per_kg": cz.Vc,
                        "implied_t_half_h": 0.693 * cz.Vc * WEIGHT / cz.CL},
            "anchored": {"CL_L_per_h": cz.CL, "Vc_L_per_kg": vc_anchor,
                         "implied_t_half_h": leroy_t_half,
                         "note": "V = CL * t_half / ln 2 at the measured "
                                 "half-life; lands at %.3f L/kg against "
                                 "Leroy's apparent Vd of %.3f"
                                 % (vc_anchor, leroy_vd)},
        },
    }
    payload["summary"] = {
        "meropenem_targets_met_before": sum(
            [before["rmse_target_met"], before["cmax_target_met"],
             before["auc_target_met"]]),
        "meropenem_targets_met_after": sum(
            [after["rmse_target_met"], after["cmax_target_met"],
             after["auc_target_met"]]),
        "caveat":
            "A one-compartment model cannot reproduce a biexponential "
            "profile. If the fitted curve still misses a target, that is the "
            "structure failing, not the parameters, and the fix is a real "
            "two-compartment solution rather than a better search.",
    }

    from src.utils.provenance import provenance
    payload["provenance"] = provenance(
        endpoint="plasma concentration, mg/L",
        exposure_index="log10 RMSE against the digitised curve",
        solver={"pk": "analytic one-compartment with effect site"})
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=str)

    print("PK refit -- meropenem, fitted to %s" % DATASET)
    print("=" * 66)
    print("                       CL (L/h)   Vc (L/kg)   t1/2 (h)   RMSE")
    print("  current              %8.2f %11.3f %10.2f %6.3f"
          % (cur.CL, cur.Vc, before["terminal_half_life_h"],
             before["rmse_log10"]))
    print("  fitted               %8.2f %11.3f %10.2f %6.3f"
          % (CL_fit, Vc_fit, after["terminal_half_life_h"],
             after["rmse_log10"]))
    print("  observed curve                              %10.2f"
          % (0.693 / k_obs))
    print()
    for lab, s in (("current", before), ("fitted", after)):
        print("  %-8s RMSE %.3f %-7s Cmax %+.1f%% %-7s AUC %+.1f%% %s"
              % (lab, s["rmse_log10"],
                 "met" if s["rmse_target_met"] else "NOT MET",
                 s["cmax_relative_error"] * 100,
                 "met" if s["cmax_target_met"] else "NOT MET",
                 s["auc_0_8h_relative_error"] * 100,
                 "met" if s["auc_target_met"] else "NOT MET"))
    print()
    print("  ceftazidime (no curve; anchored to Leroy 1984 t1/2 %.2f h):"
          % leroy_t_half)
    print("    Vc %.3f -> %.3f L/kg, implied t1/2 %.2f -> %.2f h"
          % (cz.Vc, vc_anchor,
             0.693 * cz.Vc * WEIGHT / cz.CL, leroy_t_half))
    print("\nwrote " + OUT)


if __name__ == "__main__":
    main()
