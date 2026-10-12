"""
Tests for the sequential therapy module (sequential_therapy.py).

Covers:
- TreatmentPhase creation and properties
- SequentialProtocol creation and phase lookup
- Pre-defined protocols (stepdown, cycling, de-escalation)
- Sequential simulation execution
- Monotherapy comparison
- Benefit metrics computation
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.therapy.sequential_therapy import (
    TreatmentPhase,
    SequentialProtocol,
    create_stepdown_protocol,
    create_cycling_protocol,
    create_deescalation_protocol,
    run_sequential_simulation,
    compare_sequential_vs_monotherapy,
    compute_sequential_benefit,
    quick_stepdown,
)


# ---------------------------------------------------------------------------
# TreatmentPhase tests
# ---------------------------------------------------------------------------

class TestTreatmentPhase:

    def test_defaults(self):
        phase = TreatmentPhase()
        assert phase.drug_name == "meropenem"
        assert phase.drug_class == "cidal"
        assert phase.dose_mg == 1000.0

    def test_custom(self):
        phase = TreatmentPhase(
            drug_name="doxycycline",
            drug_class="static",
            dose_mg=200,
            duration_hours=96.0,
        )
        assert phase.drug_name == "doxycycline"
        assert phase.duration_hours == 96.0

    def test_n_doses(self):
        phase = TreatmentPhase(
            interval_hours=8.0,
            duration_hours=72.0,
        )
        assert phase.n_doses == 9  # 72/8 = 9

    def test_n_doses_minimum(self):
        phase = TreatmentPhase(
            interval_hours=24.0,
            duration_hours=12.0,
        )
        assert phase.n_doses == 1  # Minimum 1 dose


# ---------------------------------------------------------------------------
# SequentialProtocol tests
# ---------------------------------------------------------------------------

class TestSequentialProtocol:

    def test_defaults(self):
        protocol = SequentialProtocol()
        assert protocol.name == "sequential"
        assert len(protocol.phases) == 0

    def test_total_duration(self):
        phases = [
            TreatmentPhase(duration_hours=72.0),
            TreatmentPhase(duration_hours=96.0),
        ]
        protocol = SequentialProtocol(phases=phases)
        assert protocol.total_duration == 168.0

    def test_get_phase_at_time(self):
        phases = [
            TreatmentPhase(phase_id=0, duration_hours=72.0),
            TreatmentPhase(phase_id=1, duration_hours=96.0),
        ]
        protocol = SequentialProtocol(phases=phases)

        assert protocol.get_phase_at_time(0).phase_id == 0
        assert protocol.get_phase_at_time(36).phase_id == 0
        assert protocol.get_phase_at_time(72).phase_id == 1
        assert protocol.get_phase_at_time(100).phase_id == 1

    def test_get_phase_at_time_out_of_range(self):
        phases = [TreatmentPhase(duration_hours=72.0)]
        protocol = SequentialProtocol(phases=phases)
        assert protocol.get_phase_at_time(100) is None


# ---------------------------------------------------------------------------
# Pre-defined protocol tests
# ---------------------------------------------------------------------------

class TestPredefinedProtocols:

    def test_stepdown_protocol(self):
        protocol = create_stepdown_protocol()
        assert protocol.name == "iv_to_oral_stepdown"
        assert len(protocol.phases) == 2
        assert protocol.phases[0].drug_name == "meropenem"
        assert protocol.phases[1].drug_name == "doxycycline"
        assert protocol.total_duration == 168.0

    def test_stepdown_custom(self):
        protocol = create_stepdown_protocol(
            iv_drug="linezolid",
            oral_drug="ciprofloxacin",
            iv_duration=48.0,
            oral_duration=120.0,
        )
        assert protocol.phases[0].drug_name == "linezolid"
        assert protocol.phases[1].drug_name == "ciprofloxacin"
        assert protocol.total_duration == 168.0

    def test_cycling_protocol(self):
        drugs = [
            {"drug_name": "meropenem", "drug_class": "cidal"},
            {"drug_name": "doxycycline", "drug_class": "static"},
        ]
        protocol = create_cycling_protocol(drugs, n_cycles=2)
        assert protocol.name == "antibiotic_cycling"
        assert len(protocol.phases) == 4  # 2 drugs × 2 cycles

    def test_deescalation_protocol(self):
        protocol = create_deescalation_protocol()
        assert protocol.name == "de_escalation"
        assert len(protocol.phases) == 2
        assert protocol.phases[0].drug_name == "meropenem"
        assert protocol.phases[1].drug_name == "doxycycline"


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSequentialSimulation:

    @pytest.mark.slow
    def test_run_sequential_simulation(self):
        protocol = create_stepdown_protocol(
            iv_duration=48.0,
            oral_duration=48.0,
        )
        result = run_sequential_simulation(protocol)
        assert result is not None
        assert len(result.t) > 0
        assert "B_rep" in result.state_names

    @pytest.mark.slow
    def test_sequential_simulation_burden_decreases(self):
        protocol = create_stepdown_protocol(
            iv_duration=48.0,
            oral_duration=48.0,
        )
        result = run_sequential_simulation(protocol)
        _, B = result.get_bacterial_burden()
        # Burden should decrease with treatment
        assert B[-1] < B[0]

    @pytest.mark.slow
    def test_compare_sequential_vs_monotherapy(self):
        protocol = create_stepdown_protocol(
            iv_duration=48.0,
            oral_duration=48.0,
        )
        results = compare_sequential_vs_monotherapy(protocol)
        assert "sequential" in results
        assert "monotherapy" in results

    @pytest.mark.slow
    def test_compute_sequential_benefit(self):
        protocol = create_stepdown_protocol(
            iv_duration=48.0,
            oral_duration=48.0,
        )
        results = compare_sequential_vs_monotherapy(protocol)
        metrics = compute_sequential_benefit(results)
        assert "burden_reduction" in metrics
        assert "final_burden_sequential" in metrics
        assert "final_burden_monotherapy" in metrics
        assert "peak_il6_sequential" in metrics
        assert "peak_il6_monotherapy" in metrics


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestQuickStepdown:

    @pytest.mark.slow
    def test_quick_stepdown(self):
        metrics = quick_stepdown(
            iv_drug="meropenem",
            oral_drug="doxycycline",
            iv_duration=48.0,
            oral_duration=48.0,
        )
        assert "burden_reduction" in metrics
        assert np.isfinite(metrics["burden_reduction"])
