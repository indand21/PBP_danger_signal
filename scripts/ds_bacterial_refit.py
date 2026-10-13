"""Refit the bacterial layer against data that also constrain kill_C50 (review R7).

WHY THE EARLIER FIT COULD NOT CONSTRAIN kill_C50
------------------------------------------------
The cidal kill rate depends on concentration only through the ratio
C / kill_C50:

    kill(C) = k_kill_max * (C/C50)^h / (1 + (C/C50)^h)

so every concentration-dependent quantity the model produces is invariant
under a joint rescaling of C and kill_C50. In particular the model's own
effective MIC -- defined in ``external_validation.effective_mic`` as the
constant concentration giving 24 h stasis -- is EXACTLY proportional to
kill_C50.

The previous calibration (parent repository, ``scripts/pathb_calibrate.py``)
evaluated both cidal datasets on an MIC-normalised scale: the time-kill
exposure was set to ``mic_multiple * effective_mic(...)`` and the
neutropenic-thigh index was %fT>MIC against that same effective MIC. Under
that normalisation C/C50 at every evaluated point is independent of C50, so
the objective is flat in C50 and no amount of data of that kind can identify
it. kill_C50 = 0.6 mg/L was therefore never fitted; it was chosen, and the
comment in ``parameters.py`` claiming it "sets the effective MIC ~1 mg/L" is
wrong -- the ratio is 0.667, so it sets the effective MIC to 0.40 mg/L.

Stage 1 below measures that flatness instead of asserting it.

WHAT CONSTRAINS kill_C50
------------------------
Two pieces of information, both already in the repository and both discarded
by the MIC normalisation:

1. The time-kill experiment was run at a KNOWN ABSOLUTE concentration,
   4.0 mg/L (Nussbaumer-Proll 2022, Fig 2A: P. aeruginosa ATCC-27853,
   MIC 1 mg/L, "4xMIC"). ``registry.json`` records
   ``concentration_mgL = 4.0`` and the digitisation note says it was put
   there for exactly this purpose. Used in absolute units, the observed kill
   at 4.0 mg/L fixes C/C50 and hence, given (k_kill_max, kill_hill), fixes
   C50.

2. The MIC of that isolate, 1 mg/L, is itself a measurement. The model's
   stasis concentration under broth-like growth is the same quantity, so
   requiring the two to agree pins the concentration scale directly. Because
   stasis is exactly proportional to C50, that requirement is a closed-form
   solution for C50, not a search:

       C50* = MIC_observed / r(k_kill_max, kill_hill, k_growth),
       r := C_stasis / C50   (computed once per parameter triple)

THE FITS REPORTED
-----------------
  Fit A  the two datasets, time-kill in ABSOLUTE units, C50 free.
         What the kill data alone say about C50.
  Fit B  Fit A constrained so the model's broth stasis concentration equals
         the isolate's measured MIC of 1 mg/L. One fewer free parameter and
         no arbitrary weighting between concentration and CFU residuals.

Drug-effect parameters (k_kill_max, kill_hill, kill_C50) are shared; growth
is a per-dataset nuisance, because broth and neutropenic-thigh growth are
not the same quantity -- the convention of the earlier fit, kept so the
objectives are comparable.

Objective: mean of the per-dataset RMSEs in log10 CFU, as before. Profiles
are reported with the digitisation uncertainty of the underlying figures
(+/-0.3 log10) marked, so "indistinguishable given the data" can be read off
rather than inferred from a p-value the digitised data do not support.

Writes results/danger_signal/bacterial_refit.json. Does NOT change any
parameter default: adopting a refit moves every number in the study and is
the author's decision.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.provenance import provenance

OUT = os.path.join("results", "danger_signal", "bacterial_refit.json")

# The calibration isolate's measured MIC, and the absolute exposure used.
MIC_OBSERVED_MGL = 1.0
TIMEKILL_CONC_MGL = 4.0
DIGITISATION_LOG10 = 0.3

TIMEKILL_ID = "meropenem_timekill"
THIGH_ID = "meropenem_neutropenic_thigh"

# Shared drug-effect grids. 3.0 is the value in use and 1.59 the earlier
# MIC-normalised fit; both are on the grid so the comparison is exact rather
# than interpolated. 0.6 likewise on the C50 grid.
K_KILL = sorted({float(f"{x:.4f}") for x in np.geomspace(0.5, 6.0, 16)}
                | {1.59, 3.0})
#: Extended past 5 because the first run put the optimum at 5, the edge of
#: the original grid; an optimum on a boundary is not an optimum.
HILL = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
C50 = sorted({float(f"{x:.4f}") for x in np.geomspace(0.1, 12.0, 28)} | {0.6})

# Per-dataset nuisance growth.
G_VITRO = [0.3, 0.5, 0.7, 0.9, 1.2]
G_VIVO = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


# --------------------------------------------------------------------------
# workers
# --------------------------------------------------------------------------

def _patched(**overrides):
    """external_validation reads get_default_parameters from its own namespace."""
    import src.analysis.external_validation as ev
    from src.core.parameters import get_default_parameters as base

    def f():
        p = base()
        for k, v in overrides.items():
            setattr(p["bacteria"], k, v)
        return p

    ev.get_default_parameters = f
    return ev


def _dataset(ev, ds_id, absolute=False):
    ds = [d for d in ev.load_registry() if d.id == ds_id][0].load()
    if absolute:
        # Drop the MIC normalisation so the registry's absolute
        # concentration_mgL drives the prediction.
        ds.scenario = {k: v for k, v in ds.scenario.items() if k != "mic_multiple"}
    return ds


def _rmse(ev, ds):
    return float(ev.compute_metrics(ds.y_obs, ev.predict(ds), ds.y_space).rmse)


#: kill_C50 is pinned here rather than left at whatever src/core/parameters.py
#: currently holds. The thigh objective is invariant to kill_C50 in exact
#: arithmetic -- that is stage 1's result -- but the engine resolves the
#: effective MIC on a 25-point geometric grid with interpolation, which leaves
#: about 1e-3 of numerical wobble in the RMSE. On a surface this flat that is
#: enough to relocate the reported argmin, so leaving it to the ambient default
#: made this script's output depend on the parameters it is used to choose.
#: Any fixed value serves; 1.0 mg/L is used.
C50_REFERENCE_FOR_THIGH = 1.0


def w_thigh(args):
    """RMSE of the neutropenic-thigh dose-response. Independent of C50."""
    k, h, g = args
    ev = _patched(k_kill_max=k, kill_hill=h, k_growth=g,
                  kill_C50=C50_REFERENCE_FOR_THIGH)
    return (k, h, g), _rmse(ev, _dataset(ev, THIGH_ID))


def w_timekill(args):
    """RMSE of the time-kill at the ABSOLUTE experimental concentration."""
    k, h, c50, g = args
    ev = _patched(k_kill_max=k, kill_hill=h, kill_C50=c50, k_growth=g)
    return (k, h, c50, g), _rmse(ev, _dataset(ev, TIMEKILL_ID, absolute=True))


def w_timekill_normalised(args):
    """RMSE of the same dataset as the earlier fit used it: in MIC multiples."""
    k, h, c50, g = args
    ev = _patched(k_kill_max=k, kill_hill=h, kill_C50=c50, k_growth=g)
    return (k, h, c50, g), _rmse(ev, _dataset(ev, TIMEKILL_ID, absolute=False))


def w_stasis_ratio(args):
    """C_stasis / kill_C50 at this (k, h, growth): the scale factor.

    Exact, because kill depends on C only through C/C50. Computed at C50 = 1
    by bisection on the 24 h net change of a broth culture with no immune
    effectors -- the quantity an MIC assay measures.
    """
    k, h, g = args
    ev = _patched(k_kill_max=k, kill_hill=h, kill_C50=1.0, k_growth=g)
    from src.core.pd_model import BacterialPopulationODE
    from src.core.parameters import immune_free_parameters
    # "No immune effectors" has to be said at the set-point, not only in the
    # initial condition that _delta_log10_const_conc supplies: since the R10
    # repair the pool is homeostatic, so a zeroed pool with the default
    # set-point would be recruited and add immune killing to a broth MIC.
    pd_model = BacterialPopulationODE(
        immune_free_parameters(ev.get_default_parameters()))

    def delta(c):
        return ev._delta_log10_const_conc(pd_model, c, "cidal", 5e5, t_end=24.0)

    lo, hi = 1e-3, 1e3
    if delta(lo) < 0.0 or delta(hi) > 0.0:
        return (k, h, g), None  # no stasis crossing in range
    for _ in range(25):
        mid = np.sqrt(lo * hi)
        if delta(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (k, h, g), float(np.sqrt(lo * hi))


def _run(fn, jobs, label):
    print(f"[{label}] {len(jobs)} evaluations ...", flush=True)
    with Pool() as pool:
        return dict(pool.map(fn, jobs))


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def stage1_flatness():
    """Measure how flat the MIC-normalised objective is in C50, and how far
    from flat the absolute-unit one is. The premise of everything below."""
    k, h, g = 3.0, 4.0, 0.5
    jobs = [(k, h, c, g) for c in C50]
    norm = _run(w_timekill_normalised, jobs, "stage 1 MIC-normalised")
    absol = _run(w_timekill, jobs, "stage 1 absolute")
    rows = [{"kill_C50": c,
             "rmse_mic_normalised": norm[(k, h, c, g)],
             "rmse_absolute": absol[(k, h, c, g)]} for c in C50]
    nr = [r["rmse_mic_normalised"] for r in rows]
    ar = [r["rmse_absolute"] for r in rows]
    span = {"c50_range_fold": max(C50) / min(C50),
            "mic_normalised_rmse_span": max(nr) - min(nr),
            "absolute_rmse_span": max(ar) - min(ar)}
    print(f"  over a {span['c50_range_fold']:.0f}-fold C50 range: "
          f"MIC-normalised RMSE span {span['mic_normalised_rmse_span']:.4f}, "
          f"absolute {span['absolute_rmse_span']:.4f}")
    return {"at": {"k_kill_max": k, "kill_hill": h, "k_growth": g},
            "rows": rows, "summary": span}


def stage1b_noise_floor():
    """How low can ANY deterministic model drive each dataset's RMSE?

    The thigh dataset is a digitized scatter cloud pooling six A. baumannii
    strains: several Delta-log10 values sit at the same %fT>MIC. No function
    of the index can beat the within-bin spread about the conditional mean.
    If the fitted RMSE is at that floor, the dataset carries no information
    about which kill parameterisation is right -- it only penalises ones that
    miss the cloud. Reported so the objective's absolute size can be read.
    """
    import src.analysis.external_validation as ev
    ds = [d for d in ev.load_registry() if d.id == THIGH_ID][0].load()
    x = np.asarray(ds.t_obs, dtype=float)
    y = np.asarray(ds.y_obs, dtype=float)

    bins = {}
    for width in (5.0, 10.0, 15.0):
        edges = np.arange(x.min(), x.max() + width, width)
        idx = np.digitize(x, edges)
        res, nb = [], 0
        for b in np.unique(idx):
            m = idx == b
            if m.sum() >= 2:
                nb += 1
                res.extend(y[m] - y[m].mean())
        bins[f"bin_width_{int(width)}_pct"] = {
            "usable_bins": nb,
            "irreducible_rmse_log10": float(np.sqrt(np.mean(np.asarray(res) ** 2))),
        }

    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    lin = A @ coef
    r2 = 1 - np.sum((y - lin) ** 2) / np.sum((y - y.mean()) ** 2)

    out = {"thigh_within_bin": bins,
           "thigh_straight_line": {
               "rmse_log10": float(np.sqrt(np.mean((y - lin) ** 2))),
               "r2": float(r2),
               "published_r2": 0.74,
               "slope_per_pct": float(coef[0]),
               "intercept": float(coef[1])},
           "timekill_n": 3,
           "timekill_digitisation_log10": DIGITISATION_LOG10}
    floor = min(v["irreducible_rmse_log10"] for v in bins.values())
    print(f"  thigh irreducible RMSE >= {floor:.3f} log10; "
          f"straight line {out['thigh_straight_line']['rmse_log10']:.3f} "
          f"(R2 {r2:.3f} vs published 0.74)")
    return out


def stage2_tables():
    thigh = _run(w_thigh, list(itertools.product(K_KILL, HILL, G_VIVO)),
                 "stage 2 thigh")
    tk = _run(w_timekill, list(itertools.product(K_KILL, HILL, C50, G_VITRO)),
              "stage 2 time-kill (absolute)")
    ratio = _run(w_stasis_ratio, list(itertools.product(K_KILL, HILL, G_VITRO)),
                 "stage 2 stasis ratio")
    return thigh, tk, ratio


def _best_thigh(thigh, k, h):
    cands = [(g, thigh[(k, h, g)]) for g in G_VIVO]
    return min(cands, key=lambda c: c[1])


def fit_a(thigh, tk):
    """C50 free."""
    rows = []
    for k, h in itertools.product(K_KILL, HILL):
        g_vivo, r_thigh = _best_thigh(thigh, k, h)
        for c50 in C50:
            g_vitro, r_tk = min(((g, tk[(k, h, c50, g)]) for g in G_VITRO),
                                key=lambda c: c[1])
            rows.append({"k_kill_max": k, "kill_hill": h, "kill_C50": c50,
                         "g_vitro": g_vitro, "g_vivo": g_vivo,
                         "rmse_timekill": r_tk, "rmse_thigh": r_thigh,
                         "objective": float(np.mean([r_tk, r_thigh]))})
    rows.sort(key=lambda r: r["objective"])
    return rows


def fit_b(thigh, tk, ratio):
    """C50 pinned by the measured MIC: C50 = MIC_observed / stasis_ratio.

    The pinned value depends on the nuisance broth growth, so g_vitro is no
    longer free to float independently -- each (k, h, g_vitro) gives one C50,
    and the time-kill RMSE is evaluated there by interpolating the C50 grid in
    log space.
    """
    log_c50 = np.log(C50)
    rows = []
    for k, h in itertools.product(K_KILL, HILL):
        g_vivo, r_thigh = _best_thigh(thigh, k, h)
        best = None
        for g in G_VITRO:
            r = ratio[(k, h, g)]
            if r is None:
                continue
            c50_star = MIC_OBSERVED_MGL / r
            if not (C50[0] <= c50_star <= C50[-1]):
                continue
            curve = [tk[(k, h, c, g)] for c in C50]
            r_tk = float(np.interp(np.log(c50_star), log_c50, curve))
            cand = {"k_kill_max": k, "kill_hill": h, "kill_C50": float(c50_star),
                    "g_vitro": g, "g_vivo": g_vivo, "stasis_ratio": r,
                    "rmse_timekill": r_tk, "rmse_thigh": r_thigh,
                    "objective": float(np.mean([r_tk, r_thigh]))}
            if best is None or cand["objective"] < best["objective"]:
                best = cand
        if best:
            rows.append(best)
    rows.sort(key=lambda r: r["objective"])
    return rows


def profile(rows, key):
    """Lowest objective at each value of `key`, with the minimum subtracted."""
    vals = sorted({r[key] for r in rows})
    out = []
    for v in vals:
        sub = [r for r in rows if r[key] == v]
        best = min(sub, key=lambda r: r["objective"])
        out.append({key: v, "objective": best["objective"],
                    "k_kill_max": best["k_kill_max"],
                    "kill_hill": best["kill_hill"],
                    "kill_C50": best["kill_C50"]})
    floor = min(o["objective"] for o in out)
    for o in out:
        o["excess_over_best"] = o["objective"] - floor
    return out


def within(prof, key, thresholds=(0.02, 0.05, 0.10, DIGITISATION_LOG10)):
    """Range of `key` whose objective stays within each threshold of the best.

    Reported at several thresholds rather than one, because a single
    tolerance would hide the shape: an interval that is the whole grid at
    every threshold means the data do not constrain the parameter, and that
    is a result, not a failure of the search.
    """
    out = {}
    for t in thresholds:
        ok = [p[key] for p in prof if p["excess_over_best"] <= t]
        out[f"within_{t:g}"] = [min(ok), max(ok)] if ok else None
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    s1 = stage1_flatness()
    s1b = stage1b_noise_floor()
    thigh, tk, ratio = stage2_tables()

    a = fit_a(thigh, tk)
    b = fit_b(thigh, tk, ratio)

    prof_a_c50 = profile(a, "kill_C50")
    prof_a_k = profile(a, "k_kill_max")
    prof_b_k = profile(b, "k_kill_max")

    in_use = {"k_kill_max": 3.0, "kill_C50": 0.6, "kill_hill": 4.0}
    in_use_row = min(
        a, key=lambda r: (abs(r["kill_hill"] - 4.0),
                          abs(np.log(r["k_kill_max"] / 3.0))
                          + abs(np.log(r["kill_C50"] / 0.6))))

    payload = {
        "question": "Can kill_C50 be identified, and what do the data give?",
        "stage1_identifiability": s1,
        "stage1b_noise_floor": s1b,
        "fit_a_c50_free": {"best": a[:8], "n_rows": len(a)},
        "fit_b_c50_pinned_by_measured_mic": {"best": b[:8], "n_rows": len(b)},
        "profiles": {
            "fit_a_over_kill_C50": prof_a_c50,
            "fit_a_over_k_kill_max": prof_a_k,
            "fit_b_over_k_kill_max": prof_b_k,
        },
        "intervals": {
            "note": "range of the parameter whose objective stays within each "
                    "threshold (log10 RMSE) of the best fit; the figures' own "
                    f"digitisation error is {DIGITISATION_LOG10} log10",
            "grid_span": {"kill_C50": [C50[0], C50[-1]],
                          "k_kill_max": [K_KILL[0], K_KILL[-1]]},
            "fit_a_kill_C50": within(prof_a_c50, "kill_C50"),
            "fit_a_k_kill_max": within(prof_a_k, "k_kill_max"),
            "fit_b_k_kill_max": within(prof_b_k, "k_kill_max"),
        },
        "argmin_stability": {
            "note": "The objective surface is flat enough that the reported "
                    "argmin is not a stable quantity, while the objective "
                    "VALUE is. Running this script under NumPy 2.5.2 rather "
                    "than the pinned 2.4.6, with everything else identical, "
                    "moved the fit A optimum from (1.351, hill 8, C50 0.100) "
                    "to (1.590, hill 6, C50 2.038) while the objective moved "
                    "by 4.8e-4. Quote the objective and the profiles, not the "
                    "argmin's coordinates.",
            "objective_reproducibility": 5e-4,
        },
        "boundary_check": {
            "note": "an optimum on a grid edge is not an optimum",
            "fit_a_best_kill_hill": a[0]["kill_hill"],
            "fit_b_best_kill_hill": b[0]["kill_hill"],
            "kill_hill_grid": [HILL[0], HILL[-1]],
            "fit_a_hill_on_boundary": a[0]["kill_hill"] in (HILL[0], HILL[-1]),
            "fit_b_hill_on_boundary": b[0]["kill_hill"] in (HILL[0], HILL[-1]),
            "fit_a_k_on_boundary": a[0]["k_kill_max"] in (K_KILL[0], K_KILL[-1]),
            "fit_b_k_on_boundary": b[0]["k_kill_max"] in (K_KILL[0], K_KILL[-1]),
        },
        "parameters_in_use": {
            **in_use,
            "nearest_grid_row": in_use_row,
            "excess_over_fit_a_best": in_use_row["objective"] - a[0]["objective"],
        },
        "inputs": {
            "timekill": {
                "id": TIMEKILL_ID, "used_in": "absolute mg/L",
                "concentration_mgL": TIMEKILL_CONC_MGL,
                "isolate_mic_mgL": MIC_OBSERVED_MGL,
                "n": 3,
                "source": "Nussbaumer-Proll 2022 Front Pharmacol 13:840692 Fig 2A",
            },
            "thigh": {
                "id": THIGH_ID, "used_in": "%fT>MIC vs model effective MIC",
                "n": 47,
                "source": "Sabet 2020 AAC 64(4):e02388-19 Fig 7",
            },
            "mic_anchor": {
                "value_mgL": MIC_OBSERVED_MGL,
                "role": "pins the concentration scale in fit B",
                "model_counterpart": "24 h stasis concentration, broth growth, "
                                     "no immune effectors, 5e5 CFU/mL inoculum",
            },
        },
        "grids": {"k_kill_max": K_KILL, "kill_hill": HILL, "kill_C50": C50,
                  "g_vitro": G_VITRO, "g_vivo": G_VIVO},
    }
    payload["provenance"] = provenance(
        endpoint="log10 CFU RMSE across two datasets",
        note="bacterial-layer refit with kill_C50 identified (review R7)")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print()
    print("=" * 74)
    print("FIT A  (kill_C50 free, time-kill in absolute units)")
    for r in a[:3]:
        print(f"  k_kill_max={r['k_kill_max']:.2f} kill_hill={r['kill_hill']:.0f} "
              f"kill_C50={r['kill_C50']:.3f}  obj={r['objective']:.3f} "
              f"(tk {r['rmse_timekill']:.3f} / thigh {r['rmse_thigh']:.3f})")
    print()
    print("FIT B  (kill_C50 pinned so model stasis = measured MIC 1 mg/L)")
    for r in b[:3]:
        print(f"  k_kill_max={r['k_kill_max']:.2f} kill_hill={r['kill_hill']:.0f} "
              f"kill_C50={r['kill_C50']:.3f}  obj={r['objective']:.3f} "
              f"(tk {r['rmse_timekill']:.3f} / thigh {r['rmse_thigh']:.3f})")
    print()
    iv = payload["intervals"]
    print("range still within a given objective tolerance of the best fit")
    print(f"  grid spans      C50 {iv['grid_span']['kill_C50']}, "
          f"k {iv['grid_span']['k_kill_max']}")
    for t in ("within_0.02", "within_0.05", "within_0.1", "within_0.3"):
        print(f"  {t:<12}  C50 {iv['fit_a_kill_C50'][t]}, "
              f"k {iv['fit_a_k_kill_max'][t]}")
    bc = payload["boundary_check"]
    print(f"  best kill_hill: fit A {bc['fit_a_best_kill_hill']:.0f} "
          f"(boundary: {bc['fit_a_hill_on_boundary']}), "
          f"fit B {bc['fit_b_best_kill_hill']:.0f} "
          f"(boundary: {bc['fit_b_hill_on_boundary']})")
    print()
    print(f"in use (3.0 / 0.6 / 4.0): objective "
          f"{in_use_row['objective']:.3f}, "
          f"{payload['parameters_in_use']['excess_over_fit_a_best']:+.3f} vs best")
    print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
