"""
Tests for the sensitivity analysis module (sensitivity_analysis.py).

Covers:
- Problem construction
- Metric functions
- Parameter application from sample vectors
- Single-sample evaluation
- Full SA run (small sample size for speed)
- Visualization (smoke test)
"""

import numpy as np
import pytest
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.sensitivity_analysis import (
    build_sa_problem,
    DEFAULT_SA_BOUNDS,
    METRICS,
    metric_auc_bacterial_burden,
    metric_peak_bacterial_burden,
    metric_final_bacterial_burden,
    metric_auc_il6,
    metric_peak_il6,
    metric_peak_resistance_fraction,
    _apply_sample_to_params,
    _run_single_sample,
    run_sensitivity_analysis,
    plot_sobol_indices,
    quick_sa,
)
from src.core.parameters import get_default_parameters
from src.core.simulation import run_simulation
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_ic():
    """Default initial conditions for SA simulations."""
    return {
        "B_rep": 1e5,
        "B_pers": 1e2,
        "B_SCV": 0,
        "N_eff": 1e7,
        "Damage": 0,
        "IL6": 10,
        "TNF": 5,
    }


@pytest.fixture
def meropenem_regimen():
    """Standard meropenem dosing regimen."""
    return DosingRegimen(
        dose_mg=1000.0,
        interval_hours=8.0,
        start_time=0.0,
        n_doses=12,
        infusion_duration_min=60,
    )


@pytest.fixture
def base_sim_result(default_ic, meropenem_regimen):
    """Run a baseline simulation and return the SimulationResult."""
    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)
    pk_model = TwoCompartmentPKModel(
        CL=15.0, Vc=0.25, Vp=1.0, Q=2.0, Ka=0.0, Kp=0.4,
        effect_site_model=True,
    )
    return run_simulation(
        pk_model=pk_model,
        regimen=meropenem_regimen,
        pd_model=pd_model,
        initial_conditions=default_ic,
        t_span=(0, 96),
        drug_class="cidal",
        weight_kg=70.0,
    )


# ---------------------------------------------------------------------------
# Test: build_sa_problem
# ---------------------------------------------------------------------------

class TestBuildSAProblem:

    def test_default_problem(self):
        """Building with no args should use all DEFAULT_SA_BOUNDS params."""
        problem = build_sa_problem()
        assert problem["num_vars"] == len(DEFAULT_SA_BOUNDS)
        assert set(problem["names"]) == set(DEFAULT_SA_BOUNDS.keys())

    def test_subset_problem(self):
        """Building with a subset of names should produce a smaller problem."""
        subset = ["k_growth", "k_pers", "mu_mut"]
        problem = build_sa_problem(param_names=subset)
        assert problem["num_vars"] == 3
        assert problem["names"] == subset

    def test_custom_bounds(self):
        """Custom bounds should be used instead of defaults."""
        custom = {"alpha": [0.0, 1.0], "beta": [0.5, 2.0]}
        problem = build_sa_problem(bounds=custom)
        assert problem["num_vars"] == 2
        assert problem["bounds"] == [[0.0, 1.0], [0.5, 2.0]]

    def test_empty_subset_raises(self):
        """Requesting names not in bounds should raise ValueError."""
        with pytest.raises(ValueError, match="No valid parameter names"):
            build_sa_problem(param_names=["nonexistent_param_xyz"])

    def test_partial_overlap(self):
        """Only valid names should be included when some are invalid."""
        names = ["k_growth", "bogus", "k_pers"]
        problem = build_sa_problem(param_names=names)
        assert problem["names"] == ["k_growth", "k_pers"]

    def test_bounds_shape(self):
        """Each bound should be a [lower, upper] pair."""
        problem = build_sa_problem()
        for b in problem["bounds"]:
            assert len(b) == 2
            assert b[0] < b[1]


# ---------------------------------------------------------------------------
# Test: Metric functions
# ---------------------------------------------------------------------------

class TestMetricFunctions:

    def test_auc_burden_positive(self, base_sim_result):
        """AUC of bacterial burden should be a finite positive number."""
        val = metric_auc_bacterial_burden(base_sim_result)
        assert np.isfinite(val)
        assert val > 0

    def test_peak_burden_positive(self, base_sim_result):
        val = metric_peak_bacterial_burden(base_sim_result)
        assert np.isfinite(val)
        assert val > 0

    def test_final_burden_finite(self, base_sim_result):
        val = metric_final_bacterial_burden(base_sim_result)
        assert np.isfinite(val)

    def test_auc_il6_finite(self, base_sim_result):
        val = metric_auc_il6(base_sim_result)
        assert np.isfinite(val)

    def test_peak_il6_positive(self, base_sim_result):
        val = metric_peak_il6(base_sim_result)
        assert np.isfinite(val)
        assert val >= 0

    def test_peak_resistance_fraction(self, base_sim_result):
        val = metric_peak_resistance_fraction(base_sim_result)
        assert np.isfinite(val)
        assert 0.0 <= val <= 1.0

    def test_all_metrics_registered(self):
        """All expected metric names should be in METRICS dict."""
        expected = {
            "auc_burden", "peak_burden", "final_burden",
            "auc_il6", "peak_il6", "peak_resistance", "peak_host_damage",
        }
        assert set(METRICS.keys()) == expected


# ---------------------------------------------------------------------------
# Test: _apply_sample_to_params
# ---------------------------------------------------------------------------

class TestApplySampleToParams:

    def test_returns_models(self):
        """Should return (pd_model, pk_model, regimen) tuple."""
        problem = build_sa_problem(["k_growth", "k_pers"])
        sample = np.array([0.5, 0.01])
        base = get_default_parameters()
        pd_model, pk_model, regimen = _apply_sample_to_params(
            sample, problem, base, "meropenem", 70.0
        )
        assert isinstance(pd_model, BacterialPopulationODE)
        assert isinstance(pk_model, TwoCompartmentPKModel)
        assert isinstance(regimen, DosingRegimen)

    def test_values_applied_to_pd(self):
        """PD parameter values from the sample should be applied."""
        problem = build_sa_problem(["k_growth", "k_pers"])
        sample = np.array([0.77, 0.033])
        base = get_default_parameters()
        pd_model, _, _ = _apply_sample_to_params(
            sample, problem, base, "meropenem", 70.0
        )
        assert pd_model.p_bact.k_growth == pytest.approx(0.77)
        assert pd_model.p_bact.k_pers == pytest.approx(0.033)

    def test_values_applied_to_pk(self):
        """PK parameter values from the sample should be applied."""
        problem = build_sa_problem(["CL", "Vc", "Kp"])
        sample = np.array([20.0, 0.5, 0.8])
        base = get_default_parameters()
        _, pk_model, _ = _apply_sample_to_params(
            sample, problem, base, "meropenem", 70.0
        )
        assert pk_model.CL == pytest.approx(20.0)
        assert pk_model.Vc == pytest.approx(0.5)
        assert pk_model.Kp == pytest.approx(0.8)

    def test_does_not_mutate_base(self):
        """Modifying a sample should not affect the base parameter dict."""
        problem = build_sa_problem(["k_growth"])
        base = get_default_parameters()
        original_k = base["bacteria"].k_growth
        _apply_sample_to_params(np.array([999.0]), problem, base, "meropenem", 70.0)
        assert base["bacteria"].k_growth == original_k


# ---------------------------------------------------------------------------
# Test: _run_single_sample
# ---------------------------------------------------------------------------

class TestRunSingleSample:

    def test_returns_float(self):
        """Single sample evaluation should return a float metric value."""
        problem = build_sa_problem(["k_growth"])
        sample = np.array([0.5])
        base = get_default_parameters()
        val = _run_single_sample(
            sample, problem, base,
            drug_name="meropenem", drug_class="cidal",
            weight_kg=70.0,
            initial_conditions={
                "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
                "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
            },
            t_span=(0, 96),
            metric_fn=metric_auc_bacterial_burden,
        )
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_different_samples_give_different_results(self):
        """Two very different k_growth values should produce different AUC."""
        problem = build_sa_problem(["k_growth"])
        base = get_default_parameters()
        ic = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }
        val_low = _run_single_sample(
            np.array([0.2]), problem, base,
            "meropenem", "cidal", 70.0, ic, (0, 96),
            metric_auc_bacterial_burden,
        )
        val_high = _run_single_sample(
            np.array([0.9]), problem, base,
            "meropenem", "cidal", 70.0, ic, (0, 96),
            metric_auc_bacterial_burden,
        )
        assert val_low != val_high


# ---------------------------------------------------------------------------
# Test: run_sensitivity_analysis (small run)
# ---------------------------------------------------------------------------

class TestRunSensitivityAnalysis:

    @pytest.mark.slow
    def test_basic_sa_run(self):
        """Full SA run on a small subset should complete and return expected keys."""
        subset = ["k_growth", "k_pers"]
        result = run_sensitivity_analysis(
            param_names=subset,
            drug_name="meropenem",
            drug_class="cidal",
            metric="auc_burden",
            n_samples=16,
            calc_second_order=True,
            seed=42,
            print_progress=False,
        )
        assert "Si" in result
        assert "problem" in result
        assert "Y" in result
        assert "metric_name" in result
        assert result["problem"]["num_vars"] == 2
        # Sobol evaluations: n_samples * (2D + 2) = 16 * (4+2) = 96
        assert len(result["Y"]) == 16 * (2 * 2 + 2)

    @pytest.mark.slow
    def test_sa_s1_range(self):
        """First-order indices should be in [0, 1] (approximately)."""
        result = run_sensitivity_analysis(
            param_names=["k_growth", "k_pers"],
            metric="auc_burden",
            n_samples=16,
            calc_second_order=True,
            print_progress=False,
        )
        for s1 in result["Si"]["S1"]:
            # S1 can be slightly negative due to numerical noise; clamp check
            assert s1 >= -0.1

    @pytest.mark.slow
    def test_sa_custom_metric(self):
        """SA should accept a custom metric function."""
        result = run_sensitivity_analysis(
            param_names=["k_growth"],
            metric_fn=metric_peak_il6,
            n_samples=16,
            calc_second_order=False,
            print_progress=False,
        )
        assert result["metric_name"] == "custom"
        assert len(result["Y"]) > 0

    @pytest.mark.slow
    def test_sa_invalid_metric_raises(self):
        """Requesting a nonexistent metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            run_sensitivity_analysis(
                param_names=["k_growth"],
                metric="nonexistent_metric_xyz",
                n_samples=8,
                print_progress=False,
            )

    @pytest.mark.slow
    def test_sa_all_params(self):
        """SA on all default parameters should run without error."""
        result = run_sensitivity_analysis(
            param_names=None,  # all defaults
            metric="auc_burden",
            n_samples=8,  # very small for speed
            calc_second_order=False,
            print_progress=False,
        )
        assert result["problem"]["num_vars"] == len(DEFAULT_SA_BOUNDS)


# ---------------------------------------------------------------------------
# Test: Visualization (smoke tests)
# ---------------------------------------------------------------------------

class TestPlotting:

    @pytest.mark.slow
    def test_plot_sobol_indices_runs(self, tmp_path):
        """plot_sobol_indices should not raise and should save a file."""
        result = run_sensitivity_analysis(
            param_names=["k_growth", "k_pers"],
            metric="auc_burden",
            n_samples=16,
            calc_second_order=True,
            print_progress=False,
        )
        save_path = str(tmp_path / "sobol_test.png")
        plot_sobol_indices(result, save_path=save_path, show=False)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


# ---------------------------------------------------------------------------
# Test: quick_sa
# ---------------------------------------------------------------------------

class TestQuickSA:

    @pytest.mark.slow
    def test_quick_sa_runs(self):
        """quick_sa should complete and return a valid result dict."""
        result = quick_sa(
            n_samples=8,
            drug_name="meropenem",
            drug_class="cidal",
            metric="auc_burden",
            param_subset=["k_growth", "k_pers"],
        )
        assert "Si" in result
        assert "Y" in result
        assert len(result["Y"]) > 0
