"""Replay Figure 2A-C's full 24-h schedules: where the bacterial model fails.

Nussbaumer-Proll et al. (2022) exposed P. aeruginosa ATCC-27853 (MIC 1 mg/L)
to constant meropenem concentrations of 0.25-16 x MIC and removed the drug by
centrifugation: panel A once (0-9.6 h, 40% T>MIC), panel B twice (0-4.8 and
12-16.8 h), panel C three times (0-3.2, 8-11.2, 16-19.2 h), sampling to 24 h.
Earlier analyses used only the first on-drug window. This script replays the
whole schedule of every readable arm (data/external_validation/
meropenem_figure2_schedules.csv): the regrowth after removal and the
response to repeated exposure.

Model. The immune-free reduction of the study's bacterial layer, as in
scripts/ds_multiarm_refit.py: replicating cells grow logistically and are
killed at k_max C^h/(C50^h + C^h); they switch into a non-killed persister
state at k_pers (0.01/h, adopted) and return at 0.05/h. Concentration is the
arm's constant level inside each exposure window and zero outside it. Each
panel's growth rate, starting burden and carrying capacity are fitted to its
two untreated controls alone and then held fixed.

Kill models compared, all on the same growth:
  adopted   the study's parameters, nothing fitted to these data
  refit     k_max, C50 and h refitted, the persistence mechanism unchanged
  tolerant  the refit plus ONE extension: while drug is present, replicating
            cells also switch into the tolerant state at k_tol x the kill
            fraction (drug-dependent switching), one extra parameter.
  persister_kill  the refit plus a different single extension, tested
            separately: persisters are killed at phi x the replicating
            cells' rate (phi in [0, 1]; 0 is the adopted structure). The
            current structure cannot take counts below the untouched
            persister pool, about 10^2 CFU/mL, while several panel B arms
            fall below the 10 CFU/mL floor within 2 h.

Prediction tests, for refit and tolerant:
  first-window   fitted to each panel's first exposure window only, then
                 asked to predict everything after the first removal;
  leave-one-out  fitted to two panels' full schedules, then asked to predict
                 the third panel's.

Observation operator. Readable and overlapping markers are point readings
(log10 CFU/mL, +/-0.3 reading allowance, not an SD). A line that leaves the
plot floor is a reading below 10 CFU/mL: it contributes only if the model
predicts above the floor. A marker hidden in a cluster or behind another is
an interval reading and contributes only outside the interval. The second
diamond series of panel B, which cannot be assigned to an arm, is excluded.

Fits are independent and run in a process pool (--workers, default all
cores but one); results do not depend on the number of workers.

Writes results/danger_signal/broth_replay.json.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.parameters import get_default_parameters  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

DATA = ROOT / "data/external_validation/meropenem_figure2_schedules.csv"
OUT = ROOT / "results/danger_signal/broth_replay.json"
SCHEDULES = {"A": ((0.0, 9.6),), "B": ((0.0, 4.8), (12.0, 16.8)),
             "C": ((0.0, 3.2), (8.0, 11.2), (16.0, 19.2))}
#: A sample plotted at a removal time is taken at it; allow the plotted
#: position to fall this far past the window end and still count as on drug.
SAMPLE_TOLERANCE_H = 0.15
FLOOR = 1.0                      # log10 CFU/mL, the plotted axis floor
READING_ALLOWANCE = 0.3
T_END = 24.0
PERS0 = 100.0                    # CFU/mL, as scripts/ds_multiarm_refit.py
REACTIVATION = 0.05
_B = get_default_parameters()["bacteria"]
K_PERS = float(_B.k_pers)
ADOPTED = {"k_kill_max": float(_B.k_kill_max), "kill_C50": float(_B.kill_C50),
           "kill_hill": float(_B.kill_hill), "k_tol": 0.0, "phi": 0.0}
KILL_KEYS = ("k_kill_max", "kill_C50", "kill_hill", "k_tol", "phi")
EXTRA = {"refit": None, "tolerant": ("k_tol", (0.0, 5.0), (0.0, 0.5)),
         "persister_kill": ("phi", (0.0, 1.0), (0.0, 0.5))}
#: Bounds on the log scale, as the multi-arm refit; k_tol on a linear scale.
KILL_LOW = np.log([0.05, 0.02, 0.25])
KILL_HIGH = np.log([12.0, 20.0, 64.0])
FITTED = ("refit", "tolerant", "persister_kill")
GROWTH_LOW = np.array([np.log(0.05), 3.0, 7.0])
GROWTH_HIGH = np.array([np.log(3.0), 6.0, 10.5])
KILL_STARTS = ((2.0, 0.7, 2.0), (1.6, 0.94, 4.0), (4.0, 0.4, 8.0))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load():
    with DATA.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    obs = []
    for r in rows:
        if r["quality"] == "unidentified_series":
            continue
        kind = ("point" if r["value"] else
                "below_floor" if r["upper"] and not r["lower"] else "interval")
        obs.append({"panel": r["panel"], "arm": r["arm"],
                    "conc": float(r["concentration_mgL"]) if r["concentration_mgL"] else 0.0,
                    "t": float(r["time_h"]), "kind": kind, "quality": r["quality"],
                    "value": float(r["value"]) if r["value"] else None,
                    "lower": float(r["lower"]) if r["lower"] else None,
                    "upper": float(r["upper"]) if r["upper"] else None,
                    "control": r["arm"] in ("GC-native", "GC-centrifuged", "shared-baseline")})
    return obs


def phase(panel, t):
    windows = SCHEDULES[panel]
    if t <= windows[0][1] + SAMPLE_TOLERANCE_H:
        return "first_exposure"
    for start, end in windows[1:]:
        if start < t <= end + SAMPLE_TOLERANCE_H:
            return "later_exposure"
    return "drug_free"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def simulate_panel(panel, growth, kill, concs, times):
    """log10 total CFU/mL for each arm (concs) at each time, piecewise in the schedule."""
    g, log_b0, log_bmax = growth
    k, c50, h, k_tol, phi = kill
    concs = np.asarray(concs, float)
    n = len(concs)
    frac = np.zeros(n)
    nz = concs > 0
    frac[nz] = expit(h * (np.log(concs[nz]) - np.log(c50)))
    bmax = 10.0 ** log_bmax
    windows = SCHEDULES[panel]
    edges = sorted({0.0, T_END} | {a for a, _ in windows} | {b for _, b in windows})
    state = np.r_[np.full(n, 10.0 ** log_b0 - PERS0), np.full(n, PERS0)]
    times = sorted(set(float(t) for t in times))
    out = {}
    if 0.0 in times:
        out[0.0] = np.log10(state[:n] + state[n:])
    for a, b in zip(edges[:-1], edges[1:]):
        on = any(w0 <= a < w1 for w0, w1 in windows)
        kill_rate = k * frac if on else np.zeros(n)
        switch = K_PERS + (k_tol * frac if on else 0.0)

        def rhs(_, y):
            rep, pers = y[:n], y[n:]
            grow = g * (1.0 - (rep + pers) / bmax) * rep
            to_pers = switch * rep - REACTIVATION * pers
            return np.r_[grow - kill_rate * rep - to_pers, to_pers - phi * kill_rate * pers]

        te = [t for t in times if a < t <= b]
        sol = solve_ivp(rhs, (a, b), state, method="LSODA", t_eval=sorted(set(te + [b])),
                        rtol=1e-8, atol=1e-6, max_step=0.25)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(f"panel {panel}: integration failed on [{a}, {b}]")
        y = np.maximum(sol.y, 0.0)
        for j, t in enumerate(sol.t):
            if any(abs(t - x) < 1e-12 for x in te):
                out[t] = np.log10(np.maximum(y[:n, j] + y[n:, j], 1e-10))
        state = y[:, -1]
    return out


def predict(obs, growths, kill):
    """Model value for each observation, panel by panel."""
    pred = np.empty(len(obs))
    for panel in sorted({o["panel"] for o in obs}):
        idx = [i for i, o in enumerate(obs) if o["panel"] == panel]
        arms = sorted({(obs[i]["arm"], obs[i]["conc"]) for i in idx})
        col = {a: j for j, a in enumerate(arms)}
        sims = simulate_panel(panel, growths[panel], kill, [c for _, c in arms],
                              [obs[i]["t"] for i in idx])
        for i in idx:
            pred[i] = sims[obs[i]["t"]][col[(obs[i]["arm"], obs[i]["conc"])]]
    return pred


def residuals(obs, pred):
    """Signed residual per observation under its observation operator."""
    r = np.empty(len(obs))
    for i, (o, p) in enumerate(zip(obs, pred)):
        if o["kind"] == "point":
            r[i] = p - o["value"]
        elif o["kind"] == "below_floor":
            r[i] = max(0.0, p - o["upper"])
        else:
            r[i] = max(0.0, o["lower"] - p) * -1.0 if p < o["lower"] else max(0.0, p - o["upper"])
    return r


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_growth(obs, panel):
    ctl = [o for o in obs if o["panel"] == panel and o["control"] and o["kind"] == "point"]

    def res(x):
        g = (np.exp(x[0]), x[1], x[2])
        p = predict(ctl, {panel: g}, (1.0, 1.0, 1.0, 0.0, 0.0))
        return residuals(ctl, p)

    best = None
    for start in ((np.log(0.6), 4.1, 9.0), (np.log(1.2), 4.5, 8.5), (np.log(0.3), 4.0, 9.5)):
        sol = least_squares(res, np.clip(start, GROWTH_LOW + 1e-6, GROWTH_HIGH - 1e-6),
                            bounds=(GROWTH_LOW, GROWTH_HIGH), diff_step=1e-4, xtol=1e-9, ftol=1e-9)
        if best is None or sol.cost < best.cost:
            best = sol
    x = best.x
    return (float(np.exp(x[0])), float(x[1]), float(x[2])), float(np.sqrt(np.mean(best.fun ** 2)))


def kill_from(z, model):
    k, c50, h = np.exp(z[:3])
    out = dict(zip(KILL_KEYS, (float(k), float(c50), float(h), 0.0, 0.0)))
    if EXTRA[model]:
        out[EXTRA[model][0]] = float(z[3])
    return tuple(out[key] for key in KILL_KEYS)


def fit_task(args):
    """One kill-model fit (all starts) on the observations selected by ``label``."""
    label, model, growths = args
    obs = load()
    train = [o for o in obs if not o["control"] and selector(label, "train")(o)]
    extra = EXTRA[model]
    low = np.r_[KILL_LOW, [extra[1][0]] if extra else []]
    high = np.r_[KILL_HIGH, [extra[1][1]] if extra else []]

    def res(z):
        return residuals(train, predict(train, growths, kill_from(z, model)))

    best = None
    for k, c50, h in KILL_STARTS:
        for kt in (extra[2] if extra else (None,)):
            z0 = np.r_[np.log([k, c50, h]), [kt] if extra else []]
            sol = least_squares(res, np.clip(z0, low + 1e-6, high - 1e-6), bounds=(low, high),
                                diff_step=1e-4, xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=300)
            if best is None or sol.cost < best.cost:
                best = sol
    kill = kill_from(best.x, model)
    names = ("k_kill_max", "kill_C50", "kill_hill") + ((extra[0],) if extra else ())
    at_bound = [n for n, z, lo, hi in zip(names, best.x, low, high) if min(z - lo, hi - z) < 1e-3]
    return label, model, {"kill": dict(zip(KILL_KEYS, kill)),
                          "n_train": len(train), "train_rmse": float(np.sqrt(np.mean(best.fun ** 2))),
                          "bounds_reached": at_bound, "success": bool(best.success)}


def selector(label, part):
    """Which drug-arm observations a fit trains on, or is tested on."""
    if label == "all":
        return lambda o: True
    if label == "first_window":
        return (lambda o: phase(o["panel"], o["t"]) == "first_exposure") if part == "train" else \
               (lambda o: phase(o["panel"], o["t"]) != "first_exposure")
    if label.startswith("leave_out_"):
        held = label[-1]
        return (lambda o: o["panel"] != held) if part == "train" else (lambda o: o["panel"] == held)
    raise ValueError(label)


# --------------------------------------------------------------------------
# reading the results
# --------------------------------------------------------------------------
def score(obs, pred, which=None):
    r = residuals(obs, pred)
    sel = [i for i, o in enumerate(obs) if which is None or which(o)]
    pts = [i for i in sel if obs[i]["kind"] == "point"]
    cens = [i for i in sel if obs[i]["kind"] != "point"]
    rp = r[pts]
    return {"n_points": len(pts), "rmse_points": float(np.sqrt(np.mean(rp ** 2))) if pts else None,
            "bias_points": float(np.mean(rp)) if pts else None,
            "max_abs_point_residual": float(np.max(np.abs(rp))) if pts else None,
            "n_points_beyond_allowance": int(np.sum(np.abs(rp) > READING_ALLOWANCE)),
            "n_censored_or_interval": len(cens),
            "n_censored_or_interval_violated": int(np.sum(np.abs(r[cens]) > READING_ALLOWANCE)),
            "rmse_all": float(np.sqrt(np.mean(r[sel] ** 2))) if sel else None}


def by_phase(obs, pred, extra=lambda o: True):
    return {ph: score(obs, pred, lambda o, ph=ph: not o["control"] and extra(o)
                      and phase(o["panel"], o["t"]) == ph)
            for ph in ("first_exposure", "drug_free", "later_exposure")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    t0 = time.time()
    obs = load()
    drug = [o for o in obs if not o["control"]]

    growths, growth_rmse = {}, {}
    for panel in SCHEDULES:
        growths[panel], growth_rmse[panel] = fit_growth(obs, panel)
    print("growth:", {p: [round(v, 3) for v in g] for p, g in growths.items()}, flush=True)

    labels = ["all", "first_window", "leave_out_A", "leave_out_B", "leave_out_C"]
    tasks = [(lab, model, growths) for lab in labels for model in FITTED]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(fit_task, tasks))
    else:
        results = list(map(fit_task, tasks))
    fits = {(lab, model): res for lab, model, res in results}

    def kill_tuple(d):
        return tuple(d[k] for k in KILL_KEYS)

    models = {"adopted": ADOPTED, **{m: fits[("all", m)]["kill"] for m in FITTED}}
    full = {}
    for name, kill in models.items():
        pred = predict(obs, growths, kill_tuple(kill))
        full[name] = {"kill": kill, "overall": score(obs, pred, lambda o: not o["control"]),
                      "by_phase": by_phase(obs, pred),
                      "by_panel": {p: score(obs, pred, lambda o, p=p: not o["control"] and o["panel"] == p)
                                   for p in SCHEDULES},
                      "predictions": [{"panel": o["panel"], "arm": o["arm"], "conc_mgL": o["conc"],
                                       "time_h": o["t"], "kind": o["kind"], "observed": o["value"],
                                       "lower": o["lower"], "upper": o["upper"], "phase": phase(o["panel"], o["t"]),
                                       "predicted": float(p), "residual": float(r)}
                                      for o, p, r in zip(obs, pred, residuals(obs, pred))]}

    tests = {}
    for lab in labels[1:]:
        tests[lab] = {}
        held = selector(lab, "test")
        for model in FITTED:
            f = fits[(lab, model)]
            pred = predict(obs, growths, kill_tuple(f["kill"]))
            tests[lab][model] = {**f, "held_out": score(obs, pred, lambda o: not o["control"] and held(o)),
                                 "held_out_by_phase": by_phase(obs, pred, held)}

    def held_rmse(model):
        return {lab: tests[lab][model]["held_out"]["rmse_points"] for lab in tests}

    # Re-exposure, predicted from the first window: how far off, and whether
    # one arm carries it. Panel C's 2xMIC arm grows through its third exposure.
    def reexposure(model):
        pred = predict(obs, growths, kill_tuple(fits[("first_window", model)]["kill"]))
        r = residuals(obs, pred)
        later = [(o, ri) for o, ri in zip(obs, r) if not o["control"] and o["kind"] == "point"
                 and phase(o["panel"], o["t"]) == "later_exposure"]
        rest = [ri for o, ri in later if not (o["panel"] == "C" and o["conc"] == 2.0)]
        return {"n": len(later), "bias": float(np.mean([ri for _, ri in later])),
                "n_predicted_below_observed": int(sum(ri < 0 for _, ri in later)),
                "bias_without_C_2x": float(np.mean(rest)), "n_without_C_2x": len(rest)}

    def below_floor_violated(model):
        pred = predict(obs, growths, kill_tuple(models[model]))
        r = residuals(obs, pred)
        return int(sum(1 for o, ri in zip(obs, r) if o["kind"] == "below_floor" and ri > READING_ALLOWANCE))

    c2 = {o["t"]: o["value"] for o in obs if o["panel"] == "C" and o["conc"] == 2.0 and o["kind"] == "point"}

    headline = {
        "n_observations": len(obs), "n_drug_arm_observations": len(drug),
        "n_point_readings_drug_arms": sum(o["kind"] == "point" for o in drug),
        "n_below_floor": sum(o["kind"] == "below_floor" for o in drug),
        "n_interval": sum(o["kind"] == "interval" for o in drug),
        "growth": {p: {"rate_per_h": g[0], "log10_initial": g[1], "log10_capacity": g[2],
                       "control_rmse": growth_rmse[p]} for p, g in growths.items()},
        "rmse_by_model": {m: v["overall"]["rmse_points"] for m, v in full.items()},
        "bias_by_model_and_phase": {m: {ph: v["by_phase"][ph]["bias_points"] for ph in v["by_phase"]}
                                    for m, v in full.items()},
        "rmse_by_model_and_phase": {m: {ph: v["by_phase"][ph]["rmse_points"] for ph in v["by_phase"]}
                                    for m, v in full.items()},
        "beyond_allowance_by_model": {m: v["overall"]["n_points_beyond_allowance"] for m, v in full.items()},
        "censored_violated_by_model": {m: v["overall"]["n_censored_or_interval_violated"] for m, v in full.items()},
        "held_out_rmse": {m: held_rmse(m) for m in FITTED},
        "held_out_censored_violated": {m: {lab: tests[lab][m]["held_out"]["n_censored_or_interval_violated"]
                                           for lab in tests} for m in FITTED},
        "held_out_bias_first_window": {m: {ph: tests["first_window"][m]["held_out_by_phase"][ph]["bias_points"]
                                           for ph in ("drug_free", "later_exposure")}
                                       for m in FITTED},
        "first_window_reexposure": {m: reexposure(m) for m in FITTED},
        "below_floor_readings_violated": {m: below_floor_violated(m) for m in models},
        "C_2x_on_drug_third_exposure": {f"{t:g}": c2[t] for t in sorted(c2) if 16.0 <= t <= 19.2 + SAMPLE_TOLERANCE_H},
        "C_2x_on_drug_first_exposure": {f"{t:g}": c2[t] for t in sorted(c2) if t <= 3.2 + SAMPLE_TOLERANCE_H},
        "minimum_predicted_log10_drug_arms": {m: float(min(p["predicted"] for p in full[m]["predictions"]
                                                           if p["conc_mgL"] > 0)) for m in full},
        "extension_parameter_all_data": {"k_tol": fits[("all", "tolerant")]["kill"]["k_tol"],
                                         "phi": fits[("all", "persister_kill")]["kill"]["phi"]},
        "bounds_reached": {f"{lab}|{m}": fits[(lab, m)]["bounds_reached"] for lab, m in fits},
    }
    payload = {
        "design": {"schedules_h": {p: [list(w) for w in ws] for p, ws in SCHEDULES.items()},
                   "floor_log10": FLOOR, "reading_allowance_log10": READING_ALLOWANCE,
                   "k_pers_per_h": K_PERS, "reactivation_per_h": REACTIVATION,
                   "persisters_initial_cfu_per_mL": PERS0, "adopted_kill": ADOPTED,
                   "kill_starts": [list(s) for s in KILL_STARTS],
                   "extensions": {m: {"parameter": e[0], "bounds": list(e[1]), "starts": list(e[2])}
                                  for m, e in EXTRA.items() if e}},
        "headline": headline, "full_fits": full, "prediction_tests": tests,
        "input_sha256": {DATA.name: hashlib.sha256(DATA.read_bytes()).hexdigest()},
        "provenance": provenance(note="Full-schedule replay of Figure 2A-C; diagnostic, nothing adopted"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(headline, indent=1), flush=True)
    print(f"elapsed {time.time() - t0:.0f} s on {args.workers} worker(s)", flush=True)


if __name__ == "__main__":
    main()
