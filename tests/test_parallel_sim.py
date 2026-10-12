"""
Tests for the parallel simulation engine (parallel_sim.py).

Covers:
- VirtualPatient creation
- Cohort generation
- Single-patient simulation
- Sequential cohort execution
- Parallel cohort execution
- Result aggregation
- Metric extraction
- Report serialization
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.parallel_sim import (
    VirtualPatient,
    PatientResult,
    generate_cohort,
    _simulate_patient,
    run_cohort_sequential,
    run_cohort_parallel,
    run_cohort,
    aggregate_results,
    results_to_dataframe,
    default_metrics,
    plot_cohort_summary,
    plot_kinetics_overlay,
)


# ---------------------------------------------------------------------------
# VirtualPatient tests
# ---------------------------------------------------------------------------

class TestVirtualPatient:

    def test_default_creation(self):
        p = VirtualPatient()
        assert p.patient_id == 0
        assert p.weight_kg == 70.0
        assert p.drug_name == "meropenem"
        assert p.drug_class == "cidal"

    def test_custom_creation(self):
        p = VirtualPatient(
            patient_id=5,
            weight_kg=85.0,
            immune_level=1e3,
            initial_burden=1e6,
            drug_name="doxycycline",
            drug_class="static",
        )
        assert p.patient_id == 5
        assert p.weight_kg == 85.0
        assert p.immune_level == 1e3
        assert p.drug_class == "static"

    def test_param_overrides(self):
        p = VirtualPatient(
            param_overrides={"bacteria": {"k_growth": 0.8}}
        )
        assert p.param_overrides["bacteria"]["k_growth"] == 0.8


# ---------------------------------------------------------------------------
# Cohort generation tests
# ---------------------------------------------------------------------------

class TestGenerateCohort:

    def test_cohort_size(self):
        patients = generate_cohort(n_patients=10)
        assert len(patients) == 10

    def test_cohort_ids_unique(self):
        patients = generate_cohort(n_patients=20)
        ids = [p.patient_id for p in patients]
        assert len(set(ids)) == 20

    def test_cohort_drug_name(self):
        patients = generate_cohort(n_patients=5, drug_name="doxycycline", drug_class="static")
        for p in patients:
            assert p.drug_name == "doxycycline"
            assert p.drug_class == "static"

    def test_cohort_weight_range(self):
        patients = generate_cohort(n_patients=100, weight_range=(60.0, 90.0))
        weights = [p.weight_kg for p in patients]
        assert min(weights) >= 60.0
        assert max(weights) <= 90.0

    def test_cohort_reproducible(self):
        p1 = generate_cohort(n_patients=5, seed=42)
        p2 = generate_cohort(n_patients=5, seed=42)
        for a, b in zip(p1, p2):
            assert a.weight_kg == b.weight_kg
            assert a.initial_burden == b.initial_burden

    def test_cohort_neutropenic(self):
        patients = generate_cohort(
            n_patients=100,
            include_neutropenic=True,
            neutropenic_fraction=0.5,
            seed=42,
        )
        # Some patients should have very low immune levels
        low_immune = [p for p in patients if p.immune_level < 1e4]
        assert len(low_immune) > 0


# ---------------------------------------------------------------------------
# Single-patient simulation tests
# ---------------------------------------------------------------------------

class TestSimulatePatient:

    def test_basic_simulation(self):
        patient = VirtualPatient(patient_id=0, drug_name="meropenem", drug_class="cidal")
        result = _simulate_patient(patient)
        assert result.success is True
        assert result.sim_result is not None
        assert result.metrics is not None

    def test_static_simulation(self):
        patient = VirtualPatient(patient_id=0, drug_name="doxycycline", drug_class="static")
        result = _simulate_patient(patient)
        assert result.success is True

    def test_neutropenic_simulation(self):
        patient = VirtualPatient(
            patient_id=0,
            drug_name="meropenem",
            drug_class="cidal",
            immune_level=1e3,
        )
        result = _simulate_patient(patient)
        assert result.success is True

    def test_metrics_extracted(self):
        patient = VirtualPatient(patient_id=0)
        result = _simulate_patient(patient)
        assert "final_burden_log10" in result.metrics
        assert "peak_il6" in result.metrics
        assert np.isfinite(result.metrics["final_burden_log10"])


# ---------------------------------------------------------------------------
# Sequential execution tests
# ---------------------------------------------------------------------------

class TestSequentialCohort:

    def test_sequential_run(self):
        patients = generate_cohort(n_patients=3, seed=42)
        results = run_cohort_sequential(patients)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_sequential_with_callback(self):
        patients = generate_cohort(n_patients=3, seed=42)
        progress_log = []

        def callback(done, total):
            progress_log.append((done, total))

        run_cohort_sequential(patients, progress_callback=callback)
        assert len(progress_log) == 3
        assert progress_log[-1] == (3, 3)


# ---------------------------------------------------------------------------
# Parallel execution tests
# ---------------------------------------------------------------------------

class TestParallelCohort:

    @pytest.mark.slow
    def test_parallel_run(self):
        patients = generate_cohort(n_patients=4, seed=42)
        results = run_cohort_parallel(patients, max_workers=2)
        assert len(results) == 4
        assert all(r.success for r in results)

    @pytest.mark.slow
    def test_parallel_matches_sequential(self):
        """Parallel and sequential should produce similar (not identical) results."""
        patients = generate_cohort(n_patients=4, seed=42)
        seq_results = run_cohort_sequential(patients)
        par_results = run_cohort_parallel(patients, max_workers=2)

        # Both should succeed
        assert all(r.success for r in seq_results)
        assert all(r.success for r in par_results)

        # Metrics should be similar (same patient specs)
        for s, p in zip(seq_results, par_results):
            for key in s.metrics:
                assert abs(s.metrics[key] - p.metrics[key]) < 1e-6, \
                    f"Metric {key} differs: seq={s.metrics[key]}, par={p.metrics[key]}"


# ---------------------------------------------------------------------------
# run_cohort dispatcher tests
# ---------------------------------------------------------------------------

class TestRunCohort:

    def test_sequential_mode(self):
        patients = generate_cohort(n_patients=2, seed=42)
        results = run_cohort(patients, parallel=False)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_single_patient(self):
        patients = [VirtualPatient(patient_id=0)]
        results = run_cohort(patients, parallel=True)
        # With 1 patient, should use sequential even if parallel=True
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregateResults:

    def test_basic_aggregation(self):
        patients = generate_cohort(n_patients=5, seed=42)
        results = run_cohort_sequential(patients)
        agg = aggregate_results(results)

        assert agg["n_total"] == 5
        assert agg["n_success"] == 5
        assert agg["n_failed"] == 0
        assert agg["success_rate"] == 1.0
        assert "final_burden_log10" in agg["metrics_mean"]
        assert np.isfinite(agg["metrics_mean"]["final_burden_log10"])

    def test_aggregation_with_failures(self):
        # Create a mix of successful and failed results
        results = [
            PatientResult(patient_id=0, success=True, metrics={"val": 1.0}),
            PatientResult(patient_id=1, success=False, error_message="test error"),
            PatientResult(patient_id=2, success=True, metrics={"val": 3.0}),
        ]
        agg = aggregate_results(results)

        assert agg["n_total"] == 3
        assert agg["n_success"] == 2
        assert agg["n_failed"] == 1
        assert agg["success_rate"] == pytest.approx(2 / 3)
        assert agg["metrics_mean"]["val"] == pytest.approx(2.0)

    def test_aggregation_all_failed(self):
        results = [
            PatientResult(patient_id=0, success=False, error_message="err"),
        ]
        agg = aggregate_results(results)
        assert agg["n_success"] == 0
        assert agg["success_rate"] == 0.0
        assert agg["metrics_mean"] == {}


# ---------------------------------------------------------------------------
# DataFrame conversion tests
# ---------------------------------------------------------------------------

class TestDataFrame:

    def test_to_dataframe(self):
        patients = generate_cohort(n_patients=3, seed=42)
        results = run_cohort_sequential(patients)
        df = results_to_dataframe(results)

        assert len(df) == 3
        assert "patient_id" in df.columns
        assert "success" in df.columns
        assert "final_burden_log10" in df.columns


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------

class TestPlotting:

    @pytest.mark.slow
    def test_plot_cohort_summary(self, tmp_path):
        patients = generate_cohort(n_patients=5, seed=42)
        results = run_cohort_sequential(patients)
        path = str(tmp_path / "cohort_summary.png")
        plot_cohort_summary(results, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    @pytest.mark.slow
    def test_plot_kinetics_overlay(self, tmp_path):
        patients = generate_cohort(n_patients=5, seed=42)
        results = run_cohort_sequential(patients)
        path = str(tmp_path / "kinetics.png")
        plot_kinetics_overlay(results, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Default metrics tests
# ---------------------------------------------------------------------------

class TestDefaultMetrics:

    def test_metrics_keys(self):
        patient = VirtualPatient(patient_id=0)
        result = _simulate_patient(patient)
        assert result.success
        expected_keys = {
            "final_burden_log10",
            "peak_burden_log10",
            "min_burden_log10",
            "peak_il6",
            "final_il6",
            "peak_scv_fraction",
            "auc_burden_log10",
        }
        assert set(result.metrics.keys()) == expected_keys

    def test_metrics_finite(self):
        patient = VirtualPatient(patient_id=0)
        result = _simulate_patient(patient)
        for key, val in result.metrics.items():
            assert np.isfinite(val), f"Metric {key} is not finite: {val}"
