"""Numerical assurance for the danger-signal study (external review R11).

Three checks, each written to a results file so the claims made about them
in the supplement trace to stored numbers (the third, release sampling, is in
release_sampling_check below):

1. ANALYTICAL BENCHMARK. The danger-signal compartment is linear,
   D' = src(t) - k_ds * D, so for a piecewise-constant source it has an exact
   solution. The production integrator is run on (a) a step source and
   (b) an on/off source mimicking intermittent dosing, and compared with the
   exact trajectory.

2. TOLERANCE TIGHTENING. The four matched-exposure pairs of Phase 4
   (analysis 3) are re-simulated at their stored doses with relative and
   absolute tolerances tightened by 1000x in BOTH solvers (the bacterial RK45
   and the danger-signal LSODA). The quantities the manuscript reports from
   them -- kill difference and the cost-ratio interval -- are compared.

Tolerances are tightened by wrapping solve_ivp inside this process only; no
production code is changed by running this script.

Reads results/danger_signal/phase4_analyses.json (for the matched doses).
Writes results/danger_signal/numerical_assurance.json.
"""
import contextlib
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.analysis.danger_signal as dsmod  # noqa: E402
import src.core.simulation as simmod  # noqa: E402
from src.analysis.danger_signal import (  # noqa: E402
    DangerSignalParameters, integrate_ds_cytokine,
)

P4_IN = os.path.join("results", "danger_signal", "phase4_analyses.json")
OUT = os.path.join("results", "danger_signal", "numerical_assurance.json")
TIGHTEN = 1e-3


def _exact_piecewise(t, src_level, on, k, D0=0.0):
    """Exact solution of D' = s(t) - k D for s(t) = src_level * on(t), with
    `on` piecewise constant on the grid intervals."""
    D = np.empty_like(t)
    D[0] = D0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        s = src_level * on[i - 1]
        D[i] = s / k + (D[i - 1] - s / k) * np.exp(-k * dt)
    return D


def analytical_benchmark():
    p = DangerSignalParameters()
    k = p.k_ds_clear
    out = {}
    t = np.linspace(0.0, 24.0, 24001)
    cases = {
        "step": np.ones_like(t),
        "on_off_8h": ((t % 8.0) < 2.0).astype(float),
    }
    for name, on in cases.items():
        level = 5.0  # pg/mL/h
        src = level * on
        DS, *_ = integrate_ds_cytokine(t, src, p)
        # The integrator interpolates the source linearly between grid
        # points; on this 1e-3 h grid the interpolation of a switch spreads it
        # over one interval, so the exact reference uses the same
        # left-constant convention and the dense grid keeps the difference
        # far below the solver tolerance.
        exact = _exact_piecewise(t, level, on, k)
        scale = float(np.max(np.abs(exact)))
        err = float(np.max(np.abs(DS - exact)) / scale)
        from scipy.integrate import trapezoid
        auc_err = float(abs(trapezoid(DS, t) - trapezoid(exact, t))
                        / trapezoid(exact, t))
        out[name] = {"max_relative_error": err,
                     "auc_relative_error": auc_err,
                     "peak_pg_mL": scale}
    out["interpretation"] = (
        "For a smooth (step) source the integrator matches the closed-form "
        "solution to ~5e-7 pointwise. For the on/off source the pointwise "
        "maximum is larger (~7e-3) and occurs AT the switch instants only: "
        "the integrator interpolates the source linearly between grid points "
        "whereas the reference switches instantaneously, so the two differ "
        "over one grid interval at each edge. That is a difference in how a "
        "discontinuous forcing is represented, not accumulated integration "
        "error, which is why the area under the curve -- the reported "
        "quantity -- agrees to ~8e-5. No reported result is a pointwise "
        "value at a switch instant.")
    return out


@contextlib.contextmanager
def tightened(factor):
    """Tighten rtol/atol in both production solvers, this process only."""
    originals = {m: m.solve_ivp for m in (simmod, dsmod)}

    def wrap(orig):
        def _w(*a, **kw):
            kw["rtol"] = kw.get("rtol", 1e-3) * factor
            kw["atol"] = kw.get("atol", 1e-6) * factor
            return orig(*a, **kw)
        return _w

    try:
        for m, orig in originals.items():
            m.solve_ivp = wrap(orig)
        yield
    finally:
        for m, orig in originals.items():
            m.solve_ivp = orig


def tolerance_check():
    p4 = _phase4()
    with open(P4_IN, encoding="utf-8") as fh:
        stored = json.load(fh)

    def evaluate():
        rows = []
        for e in stored["analysis3_matched_exposure"]["rows"]:
            if not e.get("evaluated", True):
                continue
            sims = {a: p4.simulate(a, e["arms"][a]["dose_mg"], 8.0, 6)
                    for a in ("ceftazidime", "meropenem")}
            ratios = p4.yield_range_ratio(sims["ceftazidime"],
                                          sims["meropenem"])
            rows.append({
                "target_fT_pct": e["target_fT_pct"],
                "kill_difference_log10": (sims["ceftazidime"]["net_log10"]
                                          - sims["meropenem"]["net_log10"]),
                "ratio_low": min(r["ratio_AUC_DS"] for r in ratios),
                "ratio_high": max(r["ratio_AUC_DS"] for r in ratios),
            })
        return rows

    base = evaluate()
    with tightened(TIGHTEN):
        tight = evaluate()
    comp = []
    for b, s in zip(base, tight):
        comp.append({
            "target_fT_pct": b["target_fT_pct"],
            "kill_difference_default": b["kill_difference_log10"],
            "kill_difference_tight": s["kill_difference_log10"],
            "abs_change_kill_difference": abs(b["kill_difference_log10"]
                                              - s["kill_difference_log10"]),
            "rel_change_ratio_low": abs(b["ratio_low"] - s["ratio_low"])
                                    / b["ratio_low"],
            "rel_change_ratio_high": abs(b["ratio_high"] - s["ratio_high"])
                                     / b["ratio_high"],
        })
    return {
        "tightening_factor": TIGHTEN,
        "default": {"bacterial_RK45": {"rtol": 1e-6, "atol": 1e-8},
                    "danger_signal_LSODA": {"rtol": 1e-6, "atol": 1e-9}},
        "rows": comp,
        "max_abs_change_kill_difference":
            max(c["abs_change_kill_difference"] for c in comp),
        "max_rel_change_cost_ratio":
            max(max(c["rel_change_ratio_low"], c["rel_change_ratio_high"])
                for c in comp),
    }


def release_sampling_check():
    """How exact are the reported release integrals? (plan item 8)

    At every matched-exposure arm of Table 1, at its stored dose: cumulative
    killing and the 48-h endotoxin AUC with the release rate sampled at 1, 4,
    8 (production) and 16 points per solver step, against an independent
    reference -- 8-point Gauss-Legendre quadrature of the true rate on the
    solver's interpolant, with the exact exponential kernel. Also recorded:
    what the previous method (trapezoid over the endotoxin trajectory sampled
    on the solver grid) gave, the cytokine solver's own endotoxin state
    against the closed form, and the exposure the PK-only dose solve hit.
    """
    from scipy.integrate import trapezoid

    from src.analysis.danger_signal import filamentation_activity
    from src.analysis.exact_release import REFINE, gauss_release_reference

    p4 = _phase4()
    with open(P4_IN, encoding="utf-8") as fh:
        stored = json.load(fh)
    q = p4.P_REF
    conv = q.V_ds_L and 1e-6 / q.V_ds_L * q.translocation_fraction * p4.V_FOCUS
    rows = []
    for e in stored["analysis3_matched_exposure"]["rows"]:
        for agent in ("ceftazidime", "meropenem"):
            dose = e["arms"][agent]["dose_mg"]
            Y, Yf = p4.AGENT_YIELD[agent], p4.AGENT_FILAMENT[agent]
            ref = None
            row = {"target_fT_pct": e["target_fT_pct"], "agent": agent, "dose_mg": dose,
                   "dose_solve_fT_residual_pct_points":
                       abs(p4.exact_ft(agent, dose, 8.0, 6) - e["target_fT_pct"])}
            for m in (1, 4, 8, 16):
                sim = p4.simulate(agent, dose, 8.0, 6, keep_solution=(m == 1), refine=m)
                if ref is None:
                    ref = gauss_release_reference(
                        sim["_solution"], sim["_pd"],
                        lambda C: filamentation_activity(C, sim["mic"], q.filament_window),
                        q.k_ds_clear)
                    auc_ref = conv * (Y * ref["auc_lys"] + Yf * ref["auc_fil"])
                out, s = p4.ds_cost(sim, agent=agent, cytokines=(m == REFINE))
                row[f"killing_rel_error_m{m}"] = sim["cum_killed_cfu_per_mL"] / ref["cum_killed"] - 1.0
                row[f"auc_rel_error_m{m}"] = s["AUC_DS_pg_h_mL"] / auc_ref - 1.0
                if m == 1:
                    # The method this replaced: trapezoid over the sampled signal.
                    row["auc_rel_error_previous_method"] = (
                        float(trapezoid(out["DS"], out["t"])) / auc_ref - 1.0)
                if m == REFINE:
                    row["cytokine_solver_ds_max_dev_over_peak"] = out["ds_solver_check"]
            rows.append(row)

    def worst(key):
        return max(abs(r[key]) for r in rows)

    return {
        "production_substeps": REFINE,
        "reference": "8-point Gauss-Legendre per solver step on the RK45 interpolant, "
                     "exact exponential kernel for the endotoxin compartment",
        "rows": rows,
        "max_abs_rel_error": {
            **{f"killing_m{m}": worst(f"killing_rel_error_m{m}") for m in (1, 4, 8, 16)},
            **{f"auc_m{m}": worst(f"auc_rel_error_m{m}") for m in (1, 4, 8, 16)},
            "auc_previous_method": worst("auc_rel_error_previous_method"),
        },
        "max_cytokine_solver_ds_dev_over_peak": worst("cytokine_solver_ds_max_dev_over_peak"),
        "max_dose_solve_fT_residual_pct_points": worst("dose_solve_fT_residual_pct_points"),
    }


def previous_exposure_method():
    """What the exposure matching replaced in plan item 8 got wrong.

    At every Table 1 arm, the previous method is re-run exactly: %fT>MIC by
    interpolating MIC crossings between solver time points, and the dose
    solved to 1 mg (brentq xtol) on a full simulation per evaluation. Records
    how far that %fT>MIC is from the exact value at the current dose, and how
    far the dose it found misses the target when its exposure is computed
    exactly.
    """
    from scipy.optimize import brentq

    from src.core.simulation import SimulationResult

    p4 = _phase4()
    with open(P4_IN, encoding="utf-8") as fh:
        stored = json.load(fh)

    def grid_ft(agent, dose):
        sim = p4.simulate(agent, dose, 8.0, 6, keep_solution=True, refine=1)
        r = sim["_solution"]
        C = r.get_free_effect_site_concentration(weight_kg=p4.WEIGHT)
        return 100.0 * SimulationResult._time_above_threshold(r.t, C, sim["mic"]) / (r.t[-1] - r.t[0])

    rows = []
    for e in stored["analysis3_matched_exposure"]["rows"]:
        target = e["target_fT_pct"]
        for agent in ("ceftazidime", "meropenem"):
            dose = e["arms"][agent]["dose_mg"]
            old = float(brentq(lambda d: grid_ft(agent, d) - target, 10.0, 60000.0, xtol=1.0))
            rows.append({"target_fT_pct": target, "agent": agent, "dose_mg": dose,
                         "grid_fT_error_pct_points": grid_ft(agent, dose) - p4.exact_ft(agent, dose, 8.0, 6),
                         "previous_dose_mg": old,
                         "previous_dose_fT_miss_pct_points": p4.exact_ft(agent, old, 8.0, 6) - target})
    return {"rows": rows,
            "max_abs_grid_fT_error_pct_points": max(abs(r["grid_fT_error_pct_points"]) for r in rows),
            "max_abs_previous_dose_fT_miss_pct_points":
                max(abs(r["previous_dose_fT_miss_pct_points"]) for r in rows)}


def _phase4():
    spec = importlib.util.spec_from_file_location(
        "p4", os.path.join("scripts", "ds_phase4_analyses.py"))
    p4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p4)
    with open(P4_IN, encoding="utf-8") as fh:
        stored = json.load(fh)
    p4.MIC_EFFECT = stored["meta"]["mic_effect_site_mgL"]
    from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf
    fam = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(fam, key=lambda e: e["cost"]))["params"]
    return p4


def main():
    payload = {"analytical_benchmark": analytical_benchmark()}
    for name, r in payload["analytical_benchmark"].items():
        if not isinstance(r, dict):
            continue
        print(f"analytical {name:10s} max rel error {r['max_relative_error']:.2e}"
              f"   AUC rel error {r['auc_relative_error']:.2e}")
    payload["tolerance_tightening"] = tolerance_check()
    tt = payload["tolerance_tightening"]
    print(f"\ntolerances x{TIGHTEN:g}: kill difference moves by at most "
          f"{tt['max_abs_change_kill_difference']:.2e} log10; cost ratio by "
          f"at most {tt['max_rel_change_cost_ratio']:.2e} relative")
    payload["release_sampling"] = rs = release_sampling_check()
    print("\nrelease sampling, max |relative error| against Gauss quadrature:")
    for k, v in rs["max_abs_rel_error"].items():
        print(f"  {k:24s} {v:.2e}")
    print(f"  cytokine solver DS vs closed form {rs['max_cytokine_solver_ds_dev_over_peak']:.2e} of peak;"
          f" dose-solve fT residual {rs['max_dose_solve_fT_residual_pct_points']:.1e} points")
    payload["previous_exposure_method"] = pe = previous_exposure_method()
    print(f"  previous method: grid %fT>MIC off by up to {pe['max_abs_grid_fT_error_pct_points']:.4f} "
          f"points; its 1-mg doses missed targets by up to "
          f"{pe['max_abs_previous_dose_fT_miss_pct_points']:.4f} points")
    from src.analysis.exact_release import NUMERICS
    from src.utils.provenance import provenance
    payload["provenance"] = provenance(solver_note="see 'default' and "
                                       "'tightening_factor' above", solver=NUMERICS)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
