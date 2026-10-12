"""
End-to-end integration tests for the QSP Antibiotic Model.

Tests the complete workflow:
1. Parameter loading and configuration
2. Single drug simulation
3. Sequential therapy simulation
4. Combination therapy simulation
5. Virtual patient cohort generation and execution
6. Sensitivity analysis
7. Literature validation
8. Dosing optimization
9. Checkpoint save/load
10. CLI interface

These tests verify that all modules work together correctly.
"""

import numpy as np
import pytest
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation, SimulationResult
from src.analysis.sensitivity_analysis import run_sensitivity_analysis, build_sa_problem
from src.analysis.literature_validation import run_full_validation, BENCHMARKS
from src.utils.parallel_sim import generate_cohort, run_cohort_sequential, aggregate_results
from src.utils.checkpoint import CheckpointManager
from src.therapy.resistance_evolution import (
    ResistanceEvolutionModel, ResistanceParameters, simulate_resistance_evolution,
)
from src.therapy.combination_therapy import (
    DrugInCombination, CombinationTherapyModel, InteractionParameters,
)
from src.therapy.sequential_therapy import create_stepdown_protocol, run_sequential_simulation
from src.analysis.dosing_optimization import optimize_dosing_grid, OptimizationConstraints


# ---------------------------------------------------------------------------
# End-to-end workflow tests
# ---------------------------------------------------------------------------

class TestEndToEndWorkflow:
    """Complete workflow integration tests."""

    @pytest.fixture
    def initial_conditions(self):
        return {
            "B_rep": 1e5,
            "B_pers": 1e2,
            "B_SCV": 0,
            "N_eff": 1e7,
            "Damage": 0,
            "IL6": 10,
            "TNF": 5,
        }

    def test_full_workflow_cidal(self, initial_conditions):
        """Complete workflow with cidal drug (meropenem)."""
        # 1. Load parameters
        params = get_default_parameters()
        assert "bacteria" in params
        assert "immune" in params
        assert "cytokine" in params

        # 2. Build PD model
        pd_model = BacterialPopulationODE(params)
        assert pd_model is not None

        # 3. Build PK model
        pk_params = get_drug_pk_parameters("meropenem")
        pk_model = TwoCompartmentPKModel(
            CL=pk_params.CL,
            Vc=pk_params.Vc,
            Vp=pk_params.Vp,
            Q=pk_params.Q,
            Ka=pk_params.Ka,
            Kp=pk_params.Kp,
            effect_site_model=True,
        )

        # 4. Create dosing regimen
        regimen = DosingRegimen(
            dose_mg=1000.0,
            interval_hours=8.0,
            start_time=0.0,
            n_doses=12,
            infusion_duration_min=60.0,
        )

        # 5. Run simulation
        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=(0, 120),
            drug_class="cidal",
            weight_kg=70.0,
        )

        # 6. Verify results
        assert isinstance(result, SimulationResult)
        t, B = result.get_bacterial_burden()
        assert len(t) > 0
        assert np.all(np.isfinite(B))
        # Cidal therapy achieves deep on-treatment killing (a nadir far below
        # baseline during dosing). The terminal burden may relapse above baseline
        # once dosing stops, because the drug-tolerant persister reservoir
        # reactivates and regrows (the horizon-dependence characterised in the
        # manuscript); the clinically meaningful readout is the on-treatment kill.
        assert B.min() < B[0] / 100  # >= 2-log on-treatment kill

        # 7. Extract metrics
        _, _, il6 = result.get_cytokines()
        _, frac_scv = result.get_resistance_fraction()
        assert np.all(np.isfinite(il6))
        assert np.all(frac_scv >= 0)
        assert np.all(frac_scv <= 1)

    def test_full_workflow_static(self, initial_conditions):
        """Complete workflow with static drug (doxycycline)."""
        params = get_default_parameters()
        pd_model = BacterialPopulationODE(params)

        pk_params = get_drug_pk_parameters("doxycycline")
        pk_model = TwoCompartmentPKModel(
            CL=pk_params.CL,
            Vc=pk_params.Vc,
            Vp=pk_params.Vp,
            Q=pk_params.Q,
            Ka=pk_params.Ka,
            Kp=pk_params.Kp,
            effect_site_model=True,
        )

        regimen = DosingRegimen(
            dose_mg=200.0,
            interval_hours=12.0,
            start_time=0.0,
            n_doses=8,
            infusion_duration_min=0.0,  # oral
        )

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=(0, 120),
            drug_class="static",
            weight_kg=70.0,
        )

        assert isinstance(result, SimulationResult)
        t, B = result.get_bacterial_burden()
        assert len(t) > 0
        assert np.all(np.isfinite(B))

    def test_full_workflow_neutropenic(self, initial_conditions):
        """Complete workflow with neutropenic patient."""
        initial_conditions["N_eff"] = 1e3  # Neutropenic

        params = get_default_parameters()
        pd_model = BacterialPopulationODE(params)

        pk_params = get_drug_pk_parameters("meropenem")
        pk_model = TwoCompartmentPKModel(
            CL=pk_params.CL,
            Vc=pk_params.Vc,
            Vp=pk_params.Vp,
            Q=pk_params.Q,
            Ka=pk_params.Ka,
            Kp=pk_params.Kp,
            effect_site_model=True,
        )

        regimen = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=12)

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=(0, 120),
            drug_class="cidal",
            weight_kg=70.0,
        )

        assert isinstance(result, SimulationResult)
        _, B = result.get_bacterial_burden()
        # Even in neutropenia, cidal therapy achieves deep on-treatment killing;
        # the terminal burden may relapse as the tolerant persister reservoir
        # regrows after dosing stops, so the on-treatment nadir is the readout.
        assert B.min() < B[0] / 100  # >= 2-log on-treatment kill


class TestSequentialTherapyIntegration:
    """Integration tests for sequential therapy."""

    @pytest.mark.slow
    def test_stepdown_protocol(self):
        """IV-to-oral step-down therapy should work end-to-end."""
        protocol = create_stepdown_protocol(
            iv_drug="meropenem",
            oral_drug="doxycycline",
            iv_duration=72.0,
            oral_duration=96.0,
        )

        result = run_sequential_simulation(protocol)
        assert isinstance(result, SimulationResult)
        assert len(result.t) > 0

        _, B = result.get_bacterial_burden()
        assert np.all(np.isfinite(B))
        # Treatment should reduce burden
        assert B[-1] < B[0]


class TestCombinationTherapyIntegration:
    """Integration tests for combination therapy."""

    @pytest.mark.slow
    def test_combination_bliss_independence(self):
        """Bliss Independence model should compute combined effect correctly."""
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]

        interaction = InteractionParameters.from_FIC(0.75)
        model = CombinationTherapyModel(drugs, interaction)

        # Test with equal effects
        effects = [0.5, 0.5]
        combined = model.compute_combined_effect(effects)

        # Bliss Independence: E = 1 - (1-0.5)(1-0.5) = 0.75
        assert abs(combined - 0.75) < 0.1  # Allow for interaction modifier


class TestResistanceEvolutionIntegration:
    """Integration tests for resistance evolution."""

    @pytest.mark.slow
    def test_resistance_under_drug_pressure(self):
        """Resistance should evolve under sustained drug pressure."""
        result = simulate_resistance_evolution(
            drug_concentration=0.5,  # In selection window for MIC=1.0
            initial_burden=1e8,
            duration_hours=168.0,
            dt=0.1,
            mu_resistance=1e-5,
        )

        assert "resistant_fraction" in result
        assert "MIC_effective" in result

        # Resistance should increase under drug pressure
        assert result["resistant_fraction"][-1] > 0


class TestCohortIntegration:
    """Integration tests for virtual patient cohort."""

    @pytest.mark.slow
    def test_cohort_generation_and_simulation(self):
        """Virtual patient cohort should generate and simulate correctly."""
        patients = generate_cohort(
            n_patients=10,
            drug_name="meropenem",
            drug_class="cidal",
            seed=42,
        )

        assert len(patients) == 10

        results = run_cohort_sequential(patients)
        assert len(results) == 10
        assert all(r.success for r in results)

        agg = aggregate_results(results)
        assert agg["n_success"] == 10
        assert agg["success_rate"] == 1.0
        assert "final_burden_log10" in agg["metrics_mean"]


class TestSensitivityAnalysisIntegration:
    """Integration tests for sensitivity analysis."""

    @pytest.mark.slow
    def test_sensitivity_analysis_workflow(self):
        """SA should run and produce valid Sobol indices."""
        result = run_sensitivity_analysis(
            param_names=["k_growth", "k_pers"],
            metric="auc_burden",
            n_samples=16,
            calc_second_order=True,
            print_progress=False,
        )

        assert "Si" in result
        assert "problem" in result
        assert result["problem"]["num_vars"] == 2

        Si = result["Si"]
        assert len(Si["S1"]) == 2
        assert len(Si["ST"]) == 2


class TestCheckpointIntegration:
    """Integration tests for checkpoint save/load."""

    def test_checkpoint_workflow(self, tmp_path):
        """Checkpoint save/load should preserve simulation state."""
        # Run simulation
        params = get_default_parameters()
        pd_model = BacterialPopulationODE(params)

        pk_params = get_drug_pk_parameters("meropenem")
        pk_model = TwoCompartmentPKModel(
            CL=pk_params.CL,
            Vc=pk_params.Vc,
            Vp=pk_params.Vp,
            Q=pk_params.Q,
            Ka=pk_params.Ka,
            Kp=pk_params.Kp,
            effect_site_model=True,
        )

        regimen = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=12)

        ic = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=ic,
            t_span=(0, 120),
            drug_class="cidal",
            weight_kg=70.0,
        )

        # Save checkpoint
        ckpt_dir = str(tmp_path / "checkpoints")
        mgr = CheckpointManager(ckpt_dir)
        ckpt_id = mgr.save_simulation(result, description="Integration test")

        # Load checkpoint
        loaded = mgr.load_simulation(ckpt_id)
        np.testing.assert_array_equal(loaded.t, result.t)
        np.testing.assert_array_equal(loaded.y, result.y)
        assert loaded.state_names == result.state_names


class TestDosingOptimizationIntegration:
    """Integration tests for dosing optimization."""

    @pytest.mark.slow
    def test_optimization_workflow(self):
        """Dosing optimization should find a valid regimen."""
        result = optimize_dosing_grid(
            drug_name="meropenem",
            drug_class="cidal",
            objective="burden",
            n_dose_points=3,
            n_interval_points=3,
            verbose=False,
        )

        assert result.success is True
        assert result.optimal_dose > 0
        assert result.optimal_interval > 0
        assert result.simulation_result is not None

        # Optimal regimen should achieve good bacterial killing
        _, B = result.simulation_result.get_bacterial_burden()
        assert B[-1] < 1e5  # Should achieve significant reduction


class TestCLIIntegration:
    """Integration tests for CLI interface."""

    def test_cli_simulate(self):
        """CLI simulate command should work."""
        from cli import main as cli_main
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cli_main(["simulate", "--drug", "meropenem", "--dose", "1000", "--interval", "8"])

        output = f.getvalue()
        assert "Running simulation" in output
        assert "Final burden" in output

    def test_cli_validate(self):
        """CLI validate command should work."""
        from cli import main as cli_main
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cli_main(["validate"])

        output = f.getvalue()
        assert "LITERATURE VALIDATION REPORT" in output


class TestCrossModuleIntegration:
    """Tests that verify modules work together."""

    @pytest.mark.slow
    def test_resistance_affects_optimization(self):
        """Resistance evolution should influence dosing optimization."""
        # Run optimization with resistance-sensitive objective
        result = optimize_dosing_grid(
            drug_name="meropenem",
            drug_class="cidal",
            objective="multi",
            n_dose_points=3,
            n_interval_points=3,
            verbose=False,
        )

        assert result.success is True
        # Multi-objective should consider resistance
        assert result.objective_value < 1e6

    @pytest.mark.slow
    def test_combination_better_than_monotherapy(self):
        """Combination therapy should outperform monotherapy for some scenarios."""
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]

        interaction = InteractionParameters.from_FIC(0.5)  # Synergistic
        model = CombinationTherapyModel(drugs, interaction)

        # Individual effects
        effects = [0.6, 0.4]
        combined = model.compute_combined_effect(effects)

        # Combined should be better than either alone
        assert combined > max(effects)

    def test_all_drugs_available(self):
        """All expected drugs should be available in the system."""
        expected_drugs = ["meropenem", "doxycycline", "linezolid", "ciprofloxacin"]
        for drug in expected_drugs:
            pk_params = get_drug_pk_parameters(drug)
            assert pk_params is not None
            assert pk_params.CL > 0
            assert pk_params.Vc > 0


class TestLiteratureValidationIntegration:
    """Integration tests for literature validation."""

    @pytest.mark.slow
    def test_validation_produces_results(self):
        """Literature validation should produce results for all benchmarks."""
        results = run_full_validation(verbose=False)
        assert len(results) == len(BENCHMARKS)

        for name, vr in results.items():
            assert vr.status in ("PASS", "WARN", "FAIL")
            assert np.isfinite(vr.model_value)
