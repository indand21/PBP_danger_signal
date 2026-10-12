"""
Tests for the combination therapy module (combination_therapy.py).

Covers:
- DrugInCombination creation
- InteractionParameters from FIC
- CombinationTherapyModel effect computation
- Bliss Independence model
- FIC index computation
- Monotherapy vs combination comparison
- Combination benefit metrics
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.therapy.combination_therapy import (
    DrugInCombination,
    InteractionParameters,
    CombinationTherapyModel,
    run_combination_simulation,
    compare_monotherapy_vs_combination,
    compute_combination_benefit,
    quick_combination,
)


# ---------------------------------------------------------------------------
# DrugInCombination tests
# ---------------------------------------------------------------------------

class TestDrugInCombination:

    def test_defaults(self):
        drug = DrugInCombination(drug_name="meropenem")
        assert drug.drug_name == "meropenem"
        assert drug.drug_class == "cidal"
        assert drug.dose_mg == 1000.0

    def test_custom(self):
        drug = DrugInCombination(
            drug_name="doxycycline",
            drug_class="static",
            dose_mg=200,
            interval_hours=12,
        )
        assert drug.drug_name == "doxycycline"
        assert drug.drug_class == "static"
        assert drug.dose_mg == 200


# ---------------------------------------------------------------------------
# InteractionParameters tests
# ---------------------------------------------------------------------------

class TestInteractionParameters:

    def test_defaults(self):
        params = InteractionParameters()
        assert params.FIC_index == 1.0
        assert params.interaction_type == "additive"

    def test_from_FIC_synergy(self):
        params = InteractionParameters.from_FIC(0.3)
        assert params.interaction_type == "synergy"
        assert params.FIC_index == 0.3
        assert params.synergy_factor < 1.0

    def test_from_FIC_additive(self):
        params = InteractionParameters.from_FIC(0.75)
        assert params.interaction_type == "additive"
        assert params.FIC_index == 0.75

    def test_from_FIC_indifference(self):
        params = InteractionParameters.from_FIC(1.5)
        assert params.interaction_type == "indifference"
        assert params.FIC_index == 1.5

    def test_from_FIC_antagonism(self):
        params = InteractionParameters.from_FIC(3.0)
        assert params.interaction_type == "antagonism"
        assert params.FIC_index == 3.0
        assert params.antagonism_factor > 1.0


# ---------------------------------------------------------------------------
# CombinationTherapyModel tests
# ---------------------------------------------------------------------------

class TestCombinationTherapyModel:

    @pytest.fixture
    def model(self):
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]
        return CombinationTherapyModel(drugs)

    @pytest.fixture
    def synergy_model(self):
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]
        interaction = InteractionParameters.from_FIC(0.3)
        return CombinationTherapyModel(drugs, interaction)

    @pytest.fixture
    def antagonism_model(self):
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]
        interaction = InteractionParameters.from_FIC(3.0)
        return CombinationTherapyModel(drugs, interaction)

    def test_requires_two_drugs(self):
        with pytest.raises(ValueError, match="at least 2 drugs"):
            CombinationTherapyModel([DrugInCombination(drug_name="meropenem")])

    def test_combined_effect_additive(self, model):
        """Additive: combined effect should be 1 - (1-E1)(1-E2)."""
        effects = [0.5, 0.5]
        combined = model.compute_combined_effect(effects)
        expected = 1.0 - (1.0 - 0.5) * (1.0 - 0.5)  # = 0.75
        assert abs(combined - expected) < 0.01

    def test_combined_effect_both_zero(self, model):
        effects = [0.0, 0.0]
        combined = model.compute_combined_effect(effects)
        assert combined == pytest.approx(0.0)

    def test_combined_effect_both_one(self, model):
        effects = [1.0, 1.0]
        combined = model.compute_combined_effect(effects)
        assert combined == pytest.approx(1.0)

    def test_combined_effect_one_drug(self, model):
        effects = [0.8, 0.0]
        combined = model.compute_combined_effect(effects)
        assert combined == pytest.approx(0.8, abs=0.01)

    def test_synergy_amplifies(self, synergy_model):
        """Synergistic interaction should amplify the combined effect."""
        effects = [0.5, 0.5]
        combined = synergy_model.compute_combined_effect(effects)
        # Additive would be 0.75, synergy should be higher
        assert combined > 0.75

    def test_antagonism_reduces(self, antagonism_model):
        """Antagonistic interaction should reduce the combined effect."""
        effects = [0.5, 0.5]
        combined = antagonism_model.compute_combined_effect(effects)
        # Additive would be 0.75, antagonism should be lower
        assert combined < 0.75

    def test_individual_effects(self, model):
        concentrations = [2.0, 1.0]
        MICs = [1.0, 1.0]
        effects = model.compute_individual_effects(concentrations, MICs)
        assert len(effects) == 2
        assert all(0 <= e <= 1 for e in effects)

    def test_individual_effects_zero_conc(self, model):
        concentrations = [0.0, 0.0]
        MICs = [1.0, 1.0]
        effects = model.compute_individual_effects(concentrations, MICs)
        assert all(e == pytest.approx(0.0) for e in effects)

    def test_individual_effects_high_conc(self, model):
        concentrations = [100.0, 100.0]
        MICs = [1.0, 1.0]
        effects = model.compute_individual_effects(concentrations, MICs)
        # High C/MIC → effect close to 1
        assert all(e > 0.9 for e in effects)

    def test_FIC_index(self, model):
        concentrations = [1.0, 1.0]
        MICs = [1.0, 1.0]
        FIC = model.compute_FIC_index(concentrations, MICs)
        assert FIC == pytest.approx(2.0)  # 1/1 + 1/1 = 2

    def test_FIC_index_different_conc(self, model):
        concentrations = [0.5, 0.25]
        MICs = [1.0, 1.0]
        FIC = model.compute_FIC_index(concentrations, MICs)
        assert FIC == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestCombinationSimulation:

    @pytest.mark.slow
    def test_run_combination_simulation(self):
        drugs = [
            DrugInCombination(drug_name="meropenem", drug_class="cidal"),
            DrugInCombination(drug_name="doxycycline", drug_class="static"),
        ]
        result = run_combination_simulation(
            drugs=drugs,
            t_span=(0, 48),
        )
        assert result is not None
        assert len(result.t) > 0
        assert "B_rep" in result.state_names

    @pytest.mark.slow
    def test_compare_monotherapy_vs_combination(self):
        drug1 = DrugInCombination(drug_name="meropenem", drug_class="cidal")
        drug2 = DrugInCombination(drug_name="doxycycline", drug_class="static")

        results = compare_monotherapy_vs_combination(
            drug1, drug2, t_span=(0, 48),
        )

        assert "drug1_alone" in results
        assert "drug2_alone" in results
        assert "combination" in results

        # At least combination should succeed
        assert results["combination"] is not None

    @pytest.mark.slow
    def test_combination_benefit(self):
        drug1 = DrugInCombination(drug_name="meropenem", drug_class="cidal")
        drug2 = DrugInCombination(drug_name="doxycycline", drug_class="static")

        results = compare_monotherapy_vs_combination(
            drug1, drug2, t_span=(0, 48),
        )

        metrics = compute_combination_benefit(results)
        assert "burden_reduction_vs_drug1" in metrics
        assert "burden_reduction_vs_drug2" in metrics
        assert "synergy_score" in metrics


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestQuickCombination:

    @pytest.mark.slow
    def test_quick_combination(self):
        metrics = quick_combination(
            drug1_name="meropenem",
            drug2_name="doxycycline",
            FIC=0.75,
            t_span=(0, 48),
        )
        assert "synergy_score" in metrics
        assert np.isfinite(metrics["synergy_score"])
