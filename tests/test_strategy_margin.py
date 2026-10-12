"""Tests for the static-vs-cidal strategy-margin engine."""
import numpy as np
import math

from src.analysis.strategy_margin import (
    run_one, margin_at_point, margin_grid, decompose_damage, MarginResult,
    zero_crossing,
)
from src.core.parameters import HostDamageParameters


def test_run_one_returns_positive_peak_damage():
    r = run_one(n_eff=1e7, k_pers=0.01, k_infl=0.03, drug_class="cidal")
    assert r.peak_host_damage() > 0


def test_margin_at_point_fields_finite():
    m = margin_at_point(n_eff=1e7, k_pers=0.01, k_infl=0.03)
    assert isinstance(m, MarginResult)
    assert np.isfinite(m.delta)
    assert m.peak_static > 0 and m.peak_cidal > 0
    assert np.isclose(m.delta, m.peak_static - m.peak_cidal)


def test_cidal_drives_more_inflammatory_injury():
    """Cidal therapy (alpha_cidal=3 vs alpha_static=1) accrues more inflammation injury."""
    m = margin_at_point(n_eff=1e7, k_pers=0.01, k_infl=0.03)
    assert m.infl_cidal > m.infl_static


def test_decompose_damage_nonnegative():
    r = run_one(n_eff=1e7, k_pers=0.01, k_infl=0.03, drug_class="cidal")
    path, infl = decompose_damage(r, HostDamageParameters(k_infl=0.03))
    assert path >= 0
    assert infl >= 0


def test_margin_grid_shape_and_finite():
    n_eff_values = np.array([1e5, 1e7])
    k_pers_values = np.array([0.005, 0.02, 0.05])
    grid = margin_grid(n_eff_values, k_pers_values, k_infl=0.03)
    assert grid.shape == (2, 3)
    assert np.all(np.isfinite(grid))


def test_zero_crossing_interpolates():
    x = [0.0, 1.0, 2.0, 3.0]
    d = [2.0, 1.0, -1.0, -2.0]     # crosses between x=1 and x=2
    assert abs(zero_crossing(x, d) - 1.5) < 1e-9


def test_zero_crossing_none_returns_nan():
    assert math.isnan(zero_crossing([0, 1, 2], [1.0, 2.0, 3.0]))


def test_margin_result_has_terminal_fields():
    m = margin_at_point(n_eff=1e7, k_pers=0.01, k_infl=0.03)
    assert np.isfinite(m.terminal_static) and np.isfinite(m.terminal_cidal)
    assert np.isclose(m.delta_terminal, m.terminal_static - m.terminal_cidal)


def test_scv_midpoint_override_accepted():
    r = run_one(n_eff=1e7, k_pers=0.01, k_infl=0.03, drug_class="static", scv_midpoint=0.5)
    assert r.peak_host_damage() >= 0
