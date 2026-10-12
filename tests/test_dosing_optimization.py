"""
Tests for the dosing optimization module (dosing_optimization.py).

Covers:
- OptimizationConstraints creation
- OptimizationResult creation
- Objective functions (burden, resistance, multi)
- PK/PD target computation (fT>MIC, AUC/MIC)
- Grid search optimization
- SciPy optimization
- Dosing strategy comparison
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.dosing_optimization import (
    OptimizationConstraints,
    OptimizationResult,
    objective_minimize_burden,
    objective_minimize_resistance,
    objective_multi,
    compute_ft_mic,
    compute_auc_mic,
    optimize_dosing_grid,
    optimize_dosing_scipy,
    quick_optimize,
    compare_dosing_strategies,
)
from src.core.simulation import run_simulation
from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_ic():
    return {
        "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
        "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
    }


@pytest.fixture
def sample_result(default_ic):
    """Run a standard simulation for PK/PD metric testing."""
    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)
    pk_params = get_drug_pk_parameters("meropenem")
    pk_model = TwoCompartmentPKModel(
        CL=pk_params.CL, Vc=pk_params.Vc, Vp=pk_params.Vp,
        Q=pk_params.Q, Ka=pk_params.Ka, Kp=pk_params.Kp,
        effect_site_model=True,
    )
    regimen = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=12)
    return run_simulation(
        pk_model=pk_model, regimen=regimen, pd_model=pd_model,
        initial_conditions=default_ic, t_span=(0, 120),
        drug_class="cidal", weight_kg=70.0,
    )


# ---------------------------------------------------------------------------
# OptimizationConstraints tests
# ---------------------------------------------------------------------------

class TestOptimizationConstraints:

    def test_defaults(self):
        c = OptimizationConstraints()
        assert c.dose_min == 100.0
        assert c.dose_max == 4000.0
        assert c.interval_min == 4.0
        assert c.interval_max == 24.0

    def test_custom(self):
        c = OptimizationConstraints(dose_min=250, dose_max=2000)
        assert c.dose_min == 250.0
        assert c.dose_max == 2000.0


# ---------------------------------------------------------------------------
# OptimizationResult tests
# ---------------------------------------------------------------------------

class TestOptimizationResult:

    def test_defaults(self):
        r = OptimizationResult()
        assert r.success is False
        assert r.objective_value == np.inf

    def test_custom(self):
        r = OptimizationResult(
            optimal_dose=1000.0,
            optimal_interval=8.0,
            success=True,
        )
        assert r.optimal_dose == 1000.0
        assert r.success is True


# ---------------------------------------------------------------------------
# Objective function tests
# ---------------------------------------------------------------------------

class TestObjectiveFunctions:

    def test_burden_objective(self, default_ic):
        params = np.array([1000.0, 8.0, 12.0])
        value = objective_minimize_burden(
            params, "meropenem", "cidal", default_ic,
        )
        assert np.isfinite(value)
        assert value < 10  # Should achieve some killing

    def test_resistance_objective(self, default_ic):
        params = np.array([1000.0, 8.0, 12.0])
        value = objective_minimize_resistance(
            params, "meropenem", "cidal", default_ic,
        )
        assert np.isfinite(value)
        assert 0 <= value <= 1

    def test_multi_objective(self, default_ic):
        params = np.array([1000.0, 8.0, 12.0])
        value = objective_multi(
            params, "meropenem", "cidal", default_ic,
        )
        assert np.isfinite(value)

    def test_burden_invalid_params(self, default_ic):
        # Very high dose should still return a value (not crash)
        params = np.array([10000.0, 8.0, 12.0])
        value = objective_minimize_burden(
            params, "meropenem", "cidal", default_ic,
        )
        assert np.isfinite(value)


# ---------------------------------------------------------------------------
# PK/PD target tests
# ---------------------------------------------------------------------------

class TestPKPDTargets:

    def test_ft_mic(self, sample_result):
        ft = compute_ft_mic(sample_result, MIC=1.0)
        assert 0 <= ft <= 100

    def test_ft_mic_decreases_with_mic(self, sample_result):
        ft_low = compute_ft_mic(sample_result, MIC=1.0)
        ft_high = compute_ft_mic(sample_result, MIC=1000.0)
        assert ft_high < ft_low  # Higher MIC → lower fT>MIC

    def test_auc_mic(self, sample_result):
        auc = compute_auc_mic(sample_result, MIC=1.0)
        assert auc > 0


# ---------------------------------------------------------------------------
# Grid search tests
# ---------------------------------------------------------------------------

class TestGridSearch:

    @pytest.mark.slow
    def test_grid_search_burden(self, default_ic):
        result = optimize_dosing_grid(
            drug_name="meropenem",
            drug_class="cidal",
            objective="burden",
            initial_conditions=default_ic,
            n_dose_points=3,
            n_interval_points=3,
            verbose=False,
        )
        assert result.success is True
        assert result.optimal_dose > 0
        assert result.optimal_interval > 0
        assert result.simulation_result is not None

    @pytest.mark.slow
    def test_grid_search_multi(self, default_ic):
        result = optimize_dosing_grid(
            drug_name="meropenem",
            drug_class="cidal",
            objective="multi",
            initial_conditions=default_ic,
            n_dose_points=3,
            n_interval_points=3,
            verbose=False,
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# SciPy optimization tests
# ---------------------------------------------------------------------------

class TestSciPyOptimization:

    @pytest.mark.slow
    def test_scipy_burden(self, default_ic):
        result = optimize_dosing_scipy(
            drug_name="meropenem",
            drug_class="cidal",
            objective="burden",
            initial_conditions=default_ic,
            verbose=False,
        )
        assert result.success is True
        assert result.optimal_dose > 0
        assert result.simulation_result is not None


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:

    @pytest.mark.slow
    def test_quick_optimize_grid(self):
        result = quick_optimize(
            drug_name="meropenem",
            objective="burden",
            method="grid",
            verbose=False,
        )
        assert result.success is True

    @pytest.mark.slow
    def test_compare_dosing_strategies(self):
        results = compare_dosing_strategies(
            drug_name="meropenem",
            drug_class="cidal",
        )
        assert len(results) == 5
        for name, result in results.items():
            assert result is not None
