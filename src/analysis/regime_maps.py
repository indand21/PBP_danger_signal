"""Regime-map computations for the cidal/static host-damage study.

Builds the threshold surface (immune capacity x persister rate, across exposure),
the hyperinflammatory trade-off map (immune capacity x inflammation susceptibility),
and robustness sweeps, all from the Phase 2a strategy-margin engine. Functions
return plain dicts of numpy arrays for JSON serialisation by the driver script.
"""
import numpy as np

from src.analysis.strategy_margin import (
    margin_at_point, margin_grid, run_one, zero_crossing,
)

# Representative immune phenotypes for the Figure 2 / Section 3.1 trajectories:
# (label, N_eff(0), initial-condition overrides). The hyperinflammatory phenotype
# additionally starts from elevated baseline cytokines (IL-6, TNF). This is the
# single definition shared by the peak computation below and the figure script.
REPRESENTATIVE_PHENOTYPES = [
    ("neutropenic", 1e5, None),
    ("immunosuppressed", 5e6, None),
    ("immunocompetent", 1e7, None),
    ("hyperinflammatory", 5e7, {"IL6": 100, "TNF": 50}),
]


def compute_threshold_surface(n_eff_values, k_pers_values, exposures):
    """Delta over (n_eff x k_pers) at each exposure, plus the Delta=0 immune boundary."""
    n_eff_values = np.asarray(n_eff_values, dtype=float)
    k_pers_values = np.asarray(k_pers_values, dtype=float)
    n_exp = len(exposures)
    delta = np.zeros((n_exp, len(n_eff_values), len(k_pers_values)))
    boundary = np.zeros((n_exp, len(k_pers_values)))
    for e, exp in enumerate(exposures):
        delta[e] = margin_grid(n_eff_values, k_pers_values, k_infl=0.03, exposure_scale=exp)
        for j in range(len(k_pers_values)):
            boundary[e, j] = zero_crossing(n_eff_values, delta[e, :, j])
    # Fraction of the sampled grid where static is preferred (Delta < 0) at each
    # exposure. This is the headline "10/14/19/29%" exposure sweep in the main
    # text (Section 3.2); emitted here so the manuscript consistency audit can
    # assert the reported percentages directly against this source-of-truth file.
    static_fraction_by_exposure = (delta < 0.0).mean(axis=(1, 2))
    return {
        "n_eff": n_eff_values, "k_pers": k_pers_values,
        "exposures": np.asarray(exposures, dtype=float),
        "delta": delta, "boundary_n_eff": boundary,
        "static_fraction_by_exposure": static_fraction_by_exposure,
    }


def compute_tradeoff_map(n_eff_values, k_infl_values):
    """Delta over (n_eff x k_infl); mask where static is preferred; decomposition."""
    n_eff_values = np.asarray(n_eff_values, dtype=float)
    k_infl_values = np.asarray(k_infl_values, dtype=float)
    delta = np.zeros((len(n_eff_values), len(k_infl_values)))
    for i, n_eff in enumerate(n_eff_values):
        for j, k_infl in enumerate(k_infl_values):
            delta[i, j] = margin_at_point(n_eff, 0.01, k_infl).delta
    prefer_static = delta < 0.0
    # Fraction of the sampled (n_eff x k_infl) grid where static is preferred;
    # the headline "14% of the sampled regimes" in the main text (Section 3.3).
    # Emitted so the consistency audit can assert it against this file.
    static_fraction = float(prefer_static.mean())
    # Decomposition at a FIXED representative hyperinflammatory operating point
    # (n_eff = 5e7, the immune level of the hyperinflammatory phenotype, and the
    # most inflammation-susceptible setting). Evaluated directly at 5e7 rather than
    # at the nearest grid node so the reported decomposition is independent of grid
    # resolution; this is the regime the section characterises (sampling mid-immunity
    # would illustrate a point where the regime's conclusion does not hold).
    ref = margin_at_point(5e7, 0.01, float(k_infl_values[-1]))
    decomposition = {
        "path_static": ref.path_static, "infl_static": ref.infl_static,
        "path_cidal": ref.path_cidal, "infl_cidal": ref.infl_cidal,
        "delta": ref.delta,
        "n_eff": 5e7, "k_infl": float(k_infl_values[-1]),
    }
    return {
        "n_eff": n_eff_values, "k_infl": k_infl_values,
        "delta": delta, "prefer_static": prefer_static,
        "static_fraction": static_fraction, "decomposition": decomposition,
    }


def compute_representative_peaks(k_pers=0.01, k_infl=0.03):
    """Peak host damage per phenotype per drug class (Figure 2 / Section 3.1).

    Emitted so the headline representative-trajectory peaks (cidal approximately
    0.30 in every phenotype; static approximately 1.08/1.08/1.06/0.09) are tied to
    a source-of-truth file for the consistency audit.
    """
    peaks = {}
    for label, n_eff, ic_ovr in REPRESENTATIVE_PHENOTYPES:
        peaks[label] = {}
        for dc in ("cidal", "static"):
            r = run_one(n_eff=n_eff, k_pers=k_pers, k_infl=k_infl,
                        drug_class=dc, init_overrides=ic_ovr)
            _, dh = r.get_host_damage()
            peaks[label][dc] = float(np.max(dh))
    return {"peaks": peaks, "k_pers": k_pers, "k_infl": k_infl}


def compute_robustness(reference, scv_midpoints, horizons):
    """Delta at a reference point vs SCV-switch midpoint, vs horizon, and terminal outcome."""
    n_eff, k_pers, k_infl = reference
    delta_vs_scv = [margin_at_point(n_eff, k_pers, k_infl, scv_midpoint=m).delta
                    for m in scv_midpoints]
    delta_vs_horizon = [margin_at_point(n_eff, k_pers, k_infl, t_span=tuple(h)).delta
                        for h in horizons]
    delta_terminal = margin_at_point(n_eff, k_pers, k_infl).delta_terminal
    return {
        "reference": np.asarray(reference, dtype=float),
        "scv_midpoints": np.asarray(scv_midpoints, dtype=float),
        "delta_vs_scv": np.asarray(delta_vs_scv, dtype=float),
        "horizons": np.asarray(horizons, dtype=float),
        "delta_vs_horizon": np.asarray(delta_vs_horizon, dtype=float),
        "delta_terminal": float(delta_terminal),
    }
