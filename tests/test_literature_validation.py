"""
Tests for the literature validation pipeline (literature_validation.py).

Covers:
- LiteratureBenchmark dataclass
- ValidationResult dataclass
- Metric extractors
- Scenario runner
- Validation engine
- Report serialization
"""

import numpy as np
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.literature_validation import (
    LiteratureBenchmark,
    ValidationResult,
    BENCHMARKS,
    validate_benchmark,
    extract_final_burden,
    extract_peak_il6,
    extract_time_to_3log_kill,
    extract_auc_burden,
    _run_scenario,
    run_full_validation,
    results_to_dict,
    save_validation_report,
    plot_validation_summary,
)


# ---------------------------------------------------------------------------
# LiteratureBenchmark tests
# ---------------------------------------------------------------------------

class TestLiteratureBenchmark:

    def test_in_range(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0,
        )
        assert bench.in_range(3.0) is True
        assert bench.in_range(0.5) is False
        assert bench.in_range(6.0) is False

    def test_in_range_boundaries(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0,
        )
        assert bench.in_range(1.0) is True  # inclusive
        assert bench.in_range(5.0) is True

    def test_midpoint_default(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0,
        )
        assert bench.midpoint == 4.0

    def test_midpoint_explicit(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0, literature_value=3.0,
        )
        assert bench.midpoint == 3.0

    def test_distance_from_range_below(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0,
        )
        assert bench.distance_from_range(1.0) == pytest.approx(-1.0)

    def test_distance_from_range_above(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0,
        )
        assert bench.distance_from_range(8.0) == pytest.approx(2.0)

    def test_distance_from_range_inside(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0,
        )
        assert bench.distance_from_range(4.0) == pytest.approx(0.0)

    def test_relative_error(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=2.0, literature_max=6.0, literature_value=4.0,
        )
        assert bench.relative_error(4.0) == pytest.approx(0.0)
        assert bench.relative_error(5.0) == pytest.approx(0.25)
        assert bench.relative_error(2.0) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# ValidationResult tests
# ---------------------------------------------------------------------------

class TestValidationResult:

    def test_to_dict(self):
        bench = LiteratureBenchmark(
            name="test", description="Test metric", unit="CFU",
            literature_min=1.0, literature_max=5.0, source="Test source",
        )
        vr = ValidationResult(
            benchmark=bench,
            model_value=3.0,
            in_range=True,
            relative_error=0.0,
            status="PASS",
        )
        d = vr.to_dict()
        assert d["name"] == "test"
        assert d["model_value"] == 3.0
        assert d["status"] == "PASS"
        assert d["in_range"] is True


# ---------------------------------------------------------------------------
# validate_benchmark tests
# ---------------------------------------------------------------------------

class TestValidateBenchmark:

    def test_pass_when_in_range(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0, literature_value=3.0,
        )
        vr = validate_benchmark(bench, 3.0)
        assert vr.status == "PASS"
        assert vr.in_range is True

    def test_fail_when_far_outside(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0, literature_value=3.0,
        )
        vr = validate_benchmark(bench, 100.0)
        assert vr.status == "FAIL"
        assert vr.in_range is False

    def test_warn_when_slightly_outside(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0, literature_value=3.0,
        )
        # 5.5 is outside range, relative error = (5.5-3)/3 = 0.83 > 0.5
        # 4.5 is outside range, relative error = (4.5-3)/3 = 0.5 == threshold
        # Use 4.4: rel_err = (4.4-3)/3 = 0.467 < 0.5 → WARN
        vr = validate_benchmark(bench, 5.4, warn_threshold=1.0)
        assert vr.status == "WARN"


# ---------------------------------------------------------------------------
# Pre-defined benchmark tests
# ---------------------------------------------------------------------------

class TestPredefinedBenchmarks:

    def test_all_benchmarks_exist(self):
        """All expected benchmark keys should be defined."""
        expected = {
            "burden_cidal_final",
            "burden_static_final",
            "il6_peak_cidal",
            "il6_peak_static",
            "il6_baseline",
            "time_to_3log_kill_cidal",
            "cidal_static_burden_ratio",
            "burden_neutropenic_cidal",
        }
        assert set(BENCHMARKS.keys()) == expected

    def test_all_benchmarks_valid(self):
        """Each benchmark should have min < max."""
        for name, bench in BENCHMARKS.items():
            assert bench.literature_min <= bench.literature_max, f"{name}: min > max"


# ---------------------------------------------------------------------------
# Metric extractor tests
# ---------------------------------------------------------------------------

class TestMetricExtractors:

    @pytest.fixture
    def sim_result(self):
        """Run a quick cidal simulation for metric extraction."""
        return _run_scenario(
            drug_name="meropenem",
            drug_class="cidal",
            initial_burden=1e5,
            t_span=(0, 48),
            n_doses=6,
        )

    def test_final_burden(self, sim_result):
        val = extract_final_burden(sim_result)
        assert np.isfinite(val)
        # With meropenem, burden should decrease
        assert val < 5.0  # below initial 5.0 log10

    def test_peak_il6(self, sim_result):
        val = extract_peak_il6(sim_result)
        assert np.isfinite(val)
        assert val >= 0

    def test_time_to_3log_kill(self, sim_result):
        val = extract_time_to_3log_kill(sim_result)
        assert np.isfinite(val)
        assert val >= 0
        assert val <= 48  # within simulation window

    def test_auc_burden(self, sim_result):
        val = extract_auc_burden(sim_result)
        assert np.isfinite(val)


# ---------------------------------------------------------------------------
# Scenario runner tests
# ---------------------------------------------------------------------------

class TestRunScenario:

    def test_cidal_scenario(self):
        result = _run_scenario(drug_name="meropenem", drug_class="cidal", t_span=(0, 24))
        _, B = result.get_bacterial_burden()
        assert len(B) > 0
        assert np.all(np.isfinite(B))

    def test_static_scenario(self):
        result = _run_scenario(drug_name="doxycycline", drug_class="static", t_span=(0, 24))
        _, B = result.get_bacterial_burden()
        assert len(B) > 0
        assert np.all(np.isfinite(B))

    def test_neutropenic_scenario(self):
        result = _run_scenario(
            drug_name="meropenem", drug_class="cidal",
            immune_level=1e3, t_span=(0, 24),
        )
        _, B = result.get_bacterial_burden()
        assert len(B) > 0


# ---------------------------------------------------------------------------
# Full validation tests (slow)
# ---------------------------------------------------------------------------

class TestFullValidation:

    @pytest.mark.slow
    def test_run_full_validation(self):
        """Full validation should produce results for all benchmarks."""
        results = run_full_validation(verbose=False)
        assert len(results) == 8
        for name, vr in results.items():
            assert isinstance(vr, ValidationResult)
            assert vr.status in ("PASS", "WARN", "FAIL")

    @pytest.mark.slow
    def test_cidal_burden_passes(self):
        """Cidal drug should produce low final burden (within literature range)."""
        results = run_full_validation(verbose=False)
        vr = results["burden_cidal_final"]
        # Should be in the range 0.5-2.1 log10
        assert vr.status in ("PASS", "WARN"), \
            f"Cidal burden {vr.model_value:.2f} log10 outside expected range"


# ---------------------------------------------------------------------------
# Report serialization tests
# ---------------------------------------------------------------------------

class TestReportSerialization:

    def test_results_to_dict(self):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0,
        )
        vr = ValidationResult(
            benchmark=bench, model_value=3.0,
            in_range=True, relative_error=0.0, status="PASS",
        )
        d = results_to_dict({"test": vr})
        assert "test" in d
        assert d["test"]["status"] == "PASS"

    def test_save_validation_report(self, tmp_path):
        bench = LiteratureBenchmark(
            name="test", description="test", unit="unit",
            literature_min=1.0, literature_max=5.0, source="test",
        )
        vr = ValidationResult(
            benchmark=bench, model_value=3.0,
            in_range=True, relative_error=0.0, status="PASS",
        )
        path = str(tmp_path / "report.json")
        save_validation_report({"test": vr}, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["pass"] == 1
        assert data["results"]["test"]["model_value"] == 3.0


# ---------------------------------------------------------------------------
# Plotting smoke test
# ---------------------------------------------------------------------------

class TestPlotValidation:

    @pytest.mark.slow
    def test_plot_runs(self, tmp_path):
        """plot_validation_summary should not raise and should save a file."""
        results = run_full_validation(verbose=False)
        path = str(tmp_path / "validation_plot.png")
        plot_validation_summary(results, save_path=path, show=False)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
