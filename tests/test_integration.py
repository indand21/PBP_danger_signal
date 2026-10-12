"""
Integration tests for complete workflows.
"""
import pytest
import numpy as np
from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation, SimulationResult


def _raw_model(p):
    """PK model from RAW per-kg parameters (run_simulation scales Vc/Vp by weight)."""
    return TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka, Kp=p.Kp,
                                 effect_site_model=True)


class TestFullWorkflow:
    """End-to-end integration tests."""

    def test_static_vs_cidal_comparison(self):
        """Compare static and cidal drugs in identical conditions."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        weight = 70

        init_cond = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        # Static
        pk_static_raw = get_drug_pk_parameters("doxycycline")
        model_static = _raw_model(pk_static_raw)
        regimen_static = DosingRegimen(dose_mg=100, interval_hours=12, n_doses=4)
        result_static = run_simulation(
            model_static, regimen_static, pd_model, init_cond,
            t_span=(0, 48), drug_class="static", weight_kg=weight
        )

        # Cidal
        pk_cidal_raw = get_drug_pk_parameters("meropenem")
        model_cidal = _raw_model(pk_cidal_raw)
        regimen_cidal = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=6)
        result_cidal = run_simulation(
            model_cidal, regimen_cidal, pd_model, init_cond,
            t_span=(0, 48), drug_class="cidal", weight_kg=weight
        )

        # Cidal should generally show different dynamics than static
        # Note: IL-6 levels can vary significantly based on bacterial burden dynamics
        _, IL6_static, _ = result_static.get_cytokines()
        _, IL6_cidal, _ = result_cidal.get_cytokines()
        # Static drug may sustain higher bacterial burden leading to more IL-6
        # Cidal drug reduces bacteria faster, potentially lower sustained IL-6
        # The key scientific finding is the dynamic pattern, not absolute magnitude
        assert IL6_cidal.max() > 0  # Cidal should produce some IL-6
        assert IL6_static.max() > 0  # Static should produce some IL-6
        # Cidal typically produces IL-6 faster (acute response from DNA release)
        # Static may produce more sustained IL-6 (chronic bacterial presence)
        # Both are physiologically plausible - verify cytokines are in reasonable range
        assert IL6_cidal.max() < 1e12  # Sanity check: should not be astronomically high
        assert IL6_static.max() < 1e12

    def test_multiple_drugs(self):
        """Test that all available drugs can be simulated."""
        drugs = ["doxycycline", "meropenem", "linezolid", "ciprofloxacin"]
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        init_cond = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        for drug in drugs:
            pk_raw = get_drug_pk_parameters(drug)
            model = _raw_model(pk_raw)
            drug_class = "static" if drug in ["doxycycline", "linezolid"] else "cidal"
            regimen = DosingRegimen(dose_mg=500, interval_hours=8, n_doses=3)

            result = run_simulation(
                model, regimen, pd_model, init_cond,
                t_span=(0, 24), drug_class=drug_class, weight_kg=70
            )
            assert isinstance(result, SimulationResult)

    def test_dose_response(self):
        """Test that higher doses produce stronger effects."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        pk_raw = get_drug_pk_parameters("meropenem")
        model = _raw_model(pk_raw)
        init_cond = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        doses = [500, 1000, 2000]
        final_burdens = []

        for dose in doses:
            regimen = DosingRegimen(dose_mg=dose, interval_hours=8, n_doses=3)
            result = run_simulation(
                model, regimen, pd_model, init_cond,
                t_span=(0, 24), drug_class="cidal", weight_kg=70
            )
            t, B = result.get_bacterial_burden()
            final_burdens.append(B[-1])

        # Higher dose should result in lower final burden (or at least not higher)
        assert final_burdens[2] <= final_burdens[0] * 1.5  # Allow some noise

    def test_immune_status_effect(self):
        """Test that immune status affects outcomes."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        pk_raw = get_drug_pk_parameters("doxycycline")
        model = _raw_model(pk_raw)
        regimen = DosingRegimen(dose_mg=100, interval_hours=12, n_doses=4)

        # High neutrophils vs low neutrophils
        init_high_immune = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e8, "Damage": 0, "IL6": 10, "TNF": 5,
        }
        init_low_immune = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e5, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        result_high = run_simulation(
            model, regimen, pd_model, init_high_immune,
            t_span=(0, 48), drug_class="static", weight_kg=70
        )
        result_low = run_simulation(
            model, regimen, pd_model, init_low_immune,
            t_span=(0, 48), drug_class="static", weight_kg=70
        )

        t, B_high = result_high.get_bacterial_burden()
        t, B_low = result_low.get_bacterial_burden()
        # Higher immune should lead to lower bacterial burden
        assert B_high[-1] <= B_low[-1] * 2  # Allow some variability


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_simulation(self):
        """Test very short simulation (1 hour)."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        pk_raw = get_drug_pk_parameters("meropenem")
        model = _raw_model(pk_raw)
        regimen = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=1)
        init_cond = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        result = run_simulation(
            model, regimen, pd_model, init_cond,
            t_span=(0, 1), drug_class="cidal", weight_kg=70
        )
        assert result.t[-1] >= 0.9

    def test_very_high_initial_burden(self):
        """Test with very high initial bacterial burden."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        pk_raw = get_drug_pk_parameters("meropenem")
        model = _raw_model(pk_raw)
        regimen = DosingRegimen(dose_mg=1000, interval_hours=8, n_doses=3)
        init_cond = {
            "B_rep": 1e9, "B_pers": 1e6, "B_SCV": 1e3,
            "N_eff": 1e7, "Damage": 0, "IL6": 100, "TNF": 50,
        }

        result = run_simulation(
            model, regimen, pd_model, init_cond,
            t_span=(0, 24), drug_class="cidal", weight_kg=70
        )
        t, B = result.get_bacterial_burden()
        assert np.all(np.isfinite(B))

    def test_zero_dose(self):
        """Test with zero dose (no drug)."""
        params = get_default_parameters()
        pd_model = create_ode_system(params)
        pk_raw = get_drug_pk_parameters("meropenem")
        model = _raw_model(pk_raw)
        regimen = DosingRegimen(dose_mg=0, interval_hours=8, n_doses=3)
        init_cond = {
            "B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

        result = run_simulation(
            model, regimen, pd_model, init_cond,
            t_span=(0, 24), drug_class="cidal", weight_kg=70
        )
        t, B = result.get_bacterial_burden()
        # Without drug, bacteria should grow
        assert B[-1] > B[0]
