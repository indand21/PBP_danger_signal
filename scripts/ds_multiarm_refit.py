"""Fit readable Figure 2A arms in absolute mg/L without an MIC constraint.

Run: python scripts/ds_multiarm_refit.py [--perturbations 40]

This exploratory calibration does not adopt defaults or regenerate Table 1.
The historical thigh data already informed the defaults and are a diagnostic,
not a newly held-out validation set. They do not enter this fit.

The two bacterial states used here are an exact reduction of the current
immune-free cidal ODE: SCVs stay zero and all other states have no feedback
into bacteria. Each run checks this against the full nine-state model.
Profiles and bounded reading perturbations are sensitivity analyses, not CIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.parameters import get_default_parameters, immune_free_parameters
from src.core.pd_model import BacterialPopulationODE
from src.utils.provenance import provenance

DATA = ROOT / "data/external_validation/meropenem_figure2a_multiarm.csv"
SOURCE = ROOT / "data/external_validation/source_figures/figure2_embedded0.png"
OUT = ROOT / "results/danger_signal/multiarm_refit.json"
FIG = ROOT / "results/figures/multiarm_refit"
NAMES = ["k_kill_max", "kill_C50", "kill_hill", "k_growth_broth", "log10_initial_total"]
LOW = np.array([np.log(.05), np.log(.02), np.log(.25), np.log(.1), 4.3])
HIGH = np.array([np.log(12), np.log(20), np.log(64), np.log(2), 5.3])
READING_ERROR = .3


def load_points():
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    used = [r for r in rows if r["use_in_fit"] == "1"]
    # One common baseline, not seven or nine pseudoreplicated copies.
    c = np.array([0.] + [float(r["concentration_mgL"]) for r in used])
    t = np.array([0.] + [float(r["time_h"]) for r in used])
    y = np.array([4.8] + [float(r["value"]) for r in used])
    labels = ["shared-baseline"] + [r["arm"] for r in used]
    assert np.isfinite(y).all() and t.max() <= 9.6
    return rows, c, t, y, labels


def decode(x):
    return np.r_[np.exp(x[:4]), x[4]]


def encode(values):
    return np.r_[np.log(values[:4]), values[4]]


def predict(x, c, t, *, end=None):
    k, c50, h, g, logb = decode(x)
    base = get_default_parameters()["bacteria"]
    conc, inv = np.unique(c, return_inverse=True)
    kill = np.zeros_like(conc)
    nz = conc > 0
    kill[nz] = k * expit(h * (np.log(conc[nz]) - np.log(c50)))
    n = len(conc)
    total0 = 10 ** logb
    state0 = np.r_[np.full(n, total0 - 100.), np.full(n, 100.)]

    def rhs(_, state):
        rep, pers = state[:n], state[n:]
        growth = g * (1 - (rep + pers) / base.B_max) * rep
        transfer = base.k_pers * rep - .05 * pers
        return np.r_[growth - kill * rep - transfer, transfer]

    t_end = max(float(t.max()), float(end or 0), .001)
    sol = solve_ivp(rhs, (0, t_end), state0, method="LSODA", dense_output=True,
                    rtol=2e-8, atol=1e-6, max_step=.5)
    if not sol.success or not np.isfinite(sol.y).all() or sol.y.min() < -1e-5:
        raise RuntimeError("Bacterial integration failed or became negative")
    vals = sol.sol(t)
    total = vals[inv, np.arange(len(t))] + vals[n + inv, np.arange(len(t))]
    return np.log10(np.maximum(total, 1e-10))


def fit(c, t, y, starts, fixed=None):
    fixed = fixed or {}
    free = [i for i in range(5) if i not in fixed]
    best = None
    for start in starts:
        x0 = np.clip(np.array(start, copy=True), LOW + 1e-7, HIGH - 1e-7)
        for i, v in fixed.items():
            x0[i] = v

        def expand(z):
            x = x0.copy()
            x[free] = z
            return x

        def residual(z):
            return predict(expand(z), c, t) - y

        res = least_squares(residual, x0[free], bounds=(LOW[free], HIGH[free]),
                            diff_step=1e-4, ftol=1e-8, xtol=1e-8,
                            gtol=1e-7, max_nfev=220)
        x = expand(res.x)
        err = predict(x, c, t) - y
        item = {"x": x, "sse": float(err @ err),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "max_abs_residual": float(np.max(np.abs(err))),
                "optimizer_success": bool(res.success)}
        if best is None or item["sse"] < best["sse"]:
            best = item
    return best


def describe(item):
    vals = decode(item["x"])
    return {**{k: v for k, v in item.items() if k != "x"},
            "parameters": dict(zip(NAMES, map(float, vals))),
            "bounds_reached": [NAMES[i] for i, x in enumerate(item["x"])
                               if min(x - LOW[i], HIGH[i] - x) < .005]}


def full_model_check(x, c, t):
    params = immune_free_parameters(get_default_parameters())
    k, c50, h, g, b = decode(x)
    for name, value in zip(NAMES[:3], [k, c50, h]):
        setattr(params["bacteria"], name, value)
    params["bacteria"].k_growth = g
    model = BacterialPopulationODE(params)
    initial = np.array([10**b - 100, 100, 0, 0, 0, 0, 0, 0, 0.])
    full = np.empty(len(t))
    for ci in np.unique(c):
        use = c == ci
        sol = solve_ivp(lambda ti, yi: model.rhs(ti, yi, ci, "cidal"),
                        (0, max(t.max(), .001)), initial, method="LSODA",
                        rtol=1e-8, atol=1e-8, max_step=.1, dense_output=True)
        assert sol.success
        full[use] = np.log10(sol.sol(t[use])[:3].sum(axis=0))
    gap = float(np.max(np.abs(full - predict(x, c, t))))
    assert gap < 1e-5, gap
    # Concentration/C50 symmetry must survive this implementation too.
    shifted = x.copy()
    shifted[1] += np.log(3)
    scaling_gap = float(np.max(np.abs(predict(shifted, c*3, t) - predict(x, c, t))))
    assert scaling_gap < 1e-6, scaling_gap
    return {"max_full_ODE_discrepancy_log10": gap,
            "joint_concentration_C50_scaling_discrepancy_log10": scaling_gap}


def stasis(x, g=None, initial=5e5, initial_persister=100):
    # Same bacterial mechanism, 24 h endpoint; use standard inoculum and growth.
    # initial_persister=100 is fixed in predict and explicit here for provenance.
    assert initial_persister == 100
    xs = x.copy()
    xs[4] = np.log10(initial + initial_persister)
    if g is not None:
        xs[3] = np.log(g)
    def delta(logc):
        return float(predict(xs, np.array([np.exp(logc)]), np.array([24.]))[0] - xs[4])
    lo, hi = np.log(1e-5), np.log(1e4)
    if delta(lo) * delta(hi) >= 0:
        return None
    return float(np.exp(brentq(delta, lo, hi, xtol=1e-7)))


def diagnostic_thigh(x):
    import src.analysis.external_validation as ev
    original = ev.get_default_parameters
    k, c50, h, _, _ = decode(x)
    def patched():
        p = original()
        for name, v in zip(NAMES[:3], [k, c50, h]):
            setattr(p["bacteria"], name, v)
        return p
    ev.get_default_parameters = patched
    try:
        ds = next(d for d in ev.load_registry() if d.id == "meropenem_neutropenic_thigh")
        idx, delta, mic = ev.dose_response_curve(ds)
        return {"stasis_fT_pct": ev.stasis_index(idx, delta, 0),
                "one_log_fT_pct": ev.stasis_index(idx, delta, -1),
                "effective_mic_mgL": mic,
                "published_stasis_fT_pct": 17.3,
                "growth_h_inv": original()["bacteria"].k_growth,
                "note": "Historical calibration source; diagnostic only. Existing "
                        "index-driven human-PK engine, not a mechanistic murine PK fit."}
    finally:
        ev.get_default_parameters = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbations", type=int, default=40)
    ap.add_argument("--skip-thigh", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rows, c, t, y, labels = load_points()
    base = get_default_parameters()["bacteria"]
    starts = [encode([base.k_kill_max, base.kill_C50, base.kill_hill, .8, 4.8])]
    starts += [encode([k, .7, h, .8, 4.8]) for k in [1.5, 3., 6.] for h in [1., 4., 12.]]
    # Distinct response-transition locations avoid missing the low-C50 branch
    # when h becomes steep and finite-difference gradients flatten out.
    starts += [encode([2., cc, hh, .75, 4.8])
               for cc, hh in [(.25, 4.), (.25, 12.), (1., 12.), (5., .7)]]
    best = fit(c, t, y, starts)
    print("PRIMARY", json.dumps(describe(best)), flush=True)
    adopted_x = encode([base.k_kill_max, base.kill_C50, base.kill_hill, base.k_growth, 4.8])
    old_nuisance = fit(c, t, y, [best["x"]],
                       fixed={i: adopted_x[i] for i in range(3)})
    verification = {"primary": full_model_check(best["x"], c, t),
                    "adopted": full_model_check(adopted_x, c, t)}

    profiles = {}
    for parameter, i, grid in [("kill_C50", 1, np.geomspace(.02, 20, 42)),
                                ("kill_hill", 2, np.geomspace(.25, 64, 42))]:
        grid = np.unique(np.r_[grid, decode(best["x"])[i], [4.,8.] if i==2 else [.936]])
        # Traverse outwards from the optimum; use the best and previous solutions.
        jobs = sorted(grid, key=lambda v: abs(np.log(v)-best["x"][i]))
        warm = best["x"]
        profile = []
        for v in jobs:
            extra = ([encode([2., cc, v, .75, 4.8]) for cc in [.25,1.,5.]] if i==2
                     else [encode([2.,v,hh,.75,4.8]) for hh in [.7,4.,12.]])
            f = fit(c, t, y, [best["x"], warm]+extra, fixed={i: np.log(v)})
            warm = f["x"]
            profile.append({"value": float(v), **describe(f)})
        profiles[parameter] = sorted(profile, key=lambda r: r["value"])
        print("PROFILE", parameter, "done", flush=True)

    scenarios = {"primary": describe(best), "adopted_PD_with_refitted_broth_nuisance": describe(old_nuisance)}
    for h in [4.,8.]:
        f = fit(c,t,y,starts+[best["x"]],fixed={2:np.log(h)})
        scenarios[f"fixed_h_{int(h)}"] = describe(f)
    scenario_fits = {}
    for control, vals in [("native", [5.2, 7.6]), ("centrifuged", [5.5, 8.0])]:
        cc, tt, yy = np.r_[c, [.5, .5]], np.r_[t, [2., 9.6]], np.r_[y, vals]
        f = fit(cc, tt, yy, starts+[best["x"]])
        scenarios[f"0.5x_assumed_at_{control}_control"] = describe(f)
        scenario_fits[f"0.5x_assumed_at_{control}_control"] = f
    keep = c != .25
    f = fit(c[keep], t[keep], y[keep], starts+[best["x"]])
    scenarios["omit_0.25x_influence_check"] = describe(f)
    scenario_fits["omit_0.25x_influence_check"] = f
    # Readings of 2x and 4x at 2 h are ordered against a monotone concentration
    # response; quantify their influence without silently swapping the symbols.
    keep = np.array([not (lab in ["2xMIC", "4xMIC"] and ti == 2)
                     for lab, ti in zip(labels, t)])
    f = fit(c[keep], t[keep], y[keep], [best["x"], starts[0]])
    scenarios["omit_nonmonotone_2h_pair"] = describe(f)
    scenario_fits["omit_nonmonotone_2h_pair"] = f

    rng = np.random.default_rng(840692)
    perturb = []
    for j in range(args.perturbations):
        # Uniform independent bounded reading error is a sensitivity assumption.
        yp = y + rng.uniform(-READING_ERROR, READING_ERROR, len(y))
        f = fit(c, t, yp, [best["x"]])
        perturb.append(describe(f))
        if (j+1) % 10 == 0:
            print("READING PERTURBATIONS", j+1, flush=True)
    ranges = {name: [float(min(p["parameters"][name] for p in perturb)),
                     float(max(p["parameters"][name] for p in perturb))]
              for name in NAMES} if perturb else {}
    profile_ranges = {}
    for name, prof in profiles.items():
        profile_ranges[name] = {}
        for margin in [.05, .1, .3]:
            eligible = [r["value"] for r in prof if r["rmse"] <= best["rmse"]+margin]
            profile_ranges[name][f"RMSE_plus_{margin}"] = [min(eligible), max(eligible)]

    errors = predict(best["x"], c, t) - y
    result = {"question": "Does readable multi-concentration Figure 2A information constrain C50 and h?",
              "status": "exploratory calibration; defaults not adopted",
              "source": "Nussbaumer-Proll 2022, doi:10.3389/fphar.2022.840692, Figure 2A",
              "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "source_image_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "n_observations": len(y), "n_readable_drug_arms": len(set(c[c>0])),
              "n_free_parameters": 5, "endpoint_times_h": [0,2,9.6],
              "excluded": "24h washout and unresolved 0.5x endpoints; t0 counted once",
              "fixed_parameters": {"B_max": base.B_max, "k_pers": base.k_pers,
                                    "persister_reactivation_h_inv": .05, "initial_persister": 100},
              "reading_error": {"allowance_log10": READING_ERROR,
                                "interpretation": "bounded sensitivity allowance; not SD or confidence level"},
              "primary": describe(best), "scenarios": scenarios,
              "profiles": profiles, "profile_RMSE_sensitivity_ranges": profile_ranges,
              "reading_perturbations": {"seed":840692, "n": args.perturbations,
                                        "distribution_assumption": "independent uniform [-0.3,+0.3] log10",
                                        "ranges_not_confidence_intervals": ranges, "fits": perturb},
              "predictions": [{"arm": lab, "concentration_mgL":float(ci), "time_h":float(ti),
                                "observed":float(yi), "predicted":float(yi+ei), "residual":float(ei)}
                               for lab, ci, ti, yi, ei in zip(labels,c,t,y,errors)],
              "stasis_mgL": {"at_fitted_broth_growth": stasis(best["x"]),
                              "at_adopted_growth_0.5": stasis(best["x"], g=.5),
                              "measured_MIC_reference_not_fit_constraint": 1.0},
              "verification": verification,
              "provenance": provenance(note="Absolute-unit multiarm broth fit, no MIC anchor or thigh residuals")}
    if not args.skip_thigh:
        result["thigh_diagnostic"] = {"adopted": diagnostic_thigh(adopted_x),
                                     "multiarm_primary": diagnostic_thigh(best["x"])}
        shifted = best["x"].copy()
        shifted[1] += np.log(2)
        result["thigh_diagnostic"]["multiarm_C50_doubled"] = diagnostic_thigh(shifted)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    colors = dict(zip(["0.25xMIC","1xMIC","2xMIC","4xMIC","8xMIC","16xMIC","GC-native","GC-centrifuged"],
                      plt.get_cmap("tab10").colors[:8]))
    fig, ax = plt.subplots(figsize=(8.2,5.2), layout="constrained")
    ts = np.linspace(0,9.6,160)
    for lab in colors:
        use = np.array(labels) == lab
        ci = c[use][0]
        ax.errorbar(t[use],y[use],yerr=.3,fmt="o",color=colors[lab],capsize=2,ms=4)
        ax.plot(ts,predict(best["x"],np.full(len(ts),ci),ts),color=colors[lab],label=lab)
    ax.errorbar([0],[4.8],yerr=.3,fmt="s",color="black",capsize=2,label="Common baseline")
    ax.set(xlabel="Time on drug (h)",ylabel="log10 CFU/mL",xlim=(-.2,9.9),ylim=(2,9),
           title=f"Figure 2A: absolute-unit fit (RMSE {best['rmse']:.2f} log10)")
    fig.supxlabel("Bars show +/-0.3 log10 reading allowance; not sampling uncertainty",fontsize=9)
    ax.legend(ncol=3,fontsize=8,loc="upper left")
    fig.savefig(FIG/"multiarm_fit.png",dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout="constrained")
    for ax,(name,prof) in zip(axes,profiles.items()):
        ax.plot([p["value"] for p in prof],[p["rmse"] for p in prof],"o-",ms=3)
        for margin,style in [(.05,":"),(.1,"--"),(.3,"-.")]:
            ax.axhline(best["rmse"]+margin,color="gray",linestyle=style,label=f"best + {margin}")
        ax.axvline(decode(best["x"])[NAMES.index(name)],color="black",alpha=.4)
        ax.set(xscale="log",xlabel=name+(" (mg/L)" if name=="kill_C50" else ""),ylabel="Profile RMSE (log10)")
        ax.legend(fontsize=7)
    fig.suptitle("Sensitivity profiles; horizontal bands are not confidence limits",fontsize=10)
    fig.savefig(FIG/"multiarm_profiles.png",dpi=180); plt.close(fig)

    im = plt.imread(SOURCE)
    fig, ax = plt.subplots(figsize=(12,4.4),layout="constrained")
    ax.imshow(im); ax.set(xlim=(0,970),ylim=(355,0)); ax.axis("off")
    for row in rows:
        if row["use_in_fit"] != "1": continue
        xx,yy = float(row["pixel_x"]),float(row["pixel_y"])
        lab = row["arm"]
        ax.plot(xx,yy,"o",mfc="none",mec=colors[lab],ms=8,mew=1.2)
        tx = 255 if float(row["time_h"])==2 else 520
        rank = list(colors).index(lab)
        ax.annotate(f"{lab}: {float(row['value']):.1f}",(xx,yy),xytext=(tx,35+rank*34),
                    color=colors[lab],fontsize=8,arrowprops={"arrowstyle":"-","color":colors[lab],"lw":.6})
    fig.savefig(FIG/"figure2a_readings.png",dpi=160); plt.close(fig)
    print("SAVED", OUT, flush=True)
    print("SENSITIVITY", json.dumps(profile_ranges), flush=True)
    print("THIGH", json.dumps(result.get("thigh_diagnostic")), flush=True)


if __name__ == "__main__":
    main()
