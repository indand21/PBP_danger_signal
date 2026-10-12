"""Tests for regime-map computations (tiny grids)."""
import numpy as np
from src.analysis.regime_maps import (
    compute_threshold_surface, compute_tradeoff_map, compute_robustness,
)


def test_threshold_surface_shapes():
    n_eff = np.array([1e5, 1e6, 1e7])
    k_pers = np.array([0.005, 0.02])
    exposures = [1.0, 2.0]
    out = compute_threshold_surface(n_eff, k_pers, exposures)
    assert out["delta"].shape == (2, 3, 2)          # (exposure, n_eff, k_pers)
    assert out["boundary_n_eff"].shape == (2, 2)     # (exposure, k_pers)
    assert np.all(np.isfinite(out["delta"]))


def test_tradeoff_map_shapes_and_mask():
    n_eff = np.array([1e5, 1e7, 5e7])
    k_infl = np.array([0.01, 0.05])
    out = compute_tradeoff_map(n_eff, k_infl)
    assert out["delta"].shape == (3, 2)
    assert out["prefer_static"].shape == (3, 2)
    assert out["prefer_static"].dtype == bool
    assert set(out["decomposition"]) >= {"path_static", "infl_static", "path_cidal", "infl_cidal"}


def test_robustness_shapes():
    out = compute_robustness(
        reference=(1e7, 0.01, 0.03),
        scv_midpoints=[0.2, 0.3, 0.4],
        horizons=[(0, 48), (0, 96)],
    )
    assert len(out["delta_vs_scv"]) == 3
    assert len(out["delta_vs_horizon"]) == 2
    assert np.isfinite(out["delta_terminal"])
