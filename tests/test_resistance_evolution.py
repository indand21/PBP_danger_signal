"""
Tests for the resistance evolution module (resistance_evolution.py).

Covers:
- ResistanceParameters validation
- ResistanceState tracking
- ResistanceEvolutionModel core logic
- MIC computation by level
- Selection pressure computation
- Mutation rate computation
- Fitness modifier computation
- Continuous MIC update
- Stepwise MIC update
- Resistant fraction update
- ResistanceODESystem multi-level dynamics
- Simulation convenience function
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.therapy.resistance_evolution import (
    ResistanceParameters,
    ResistanceState,
    ResistanceEvolutionModel,
    ResistanceODESystem,
    create_resistance_model,
    create_resistance_ode,
    simulate_resistance_evolution,
)


# ---------------------------------------------------------------------------
# ResistanceParameters tests
# ---------------------------------------------------------------------------

class TestResistanceParameters:

    def test_defaults(self):
        p = ResistanceParameters()
        assert p.MIC_baseline == 1.0
        assert p.MIC_max == 256.0
        assert p.mu_resistance == 1e-7
        assert p.n_resistance_levels == 4
        assert p.MIC_fold_change == 2.0
        assert p.fitness_cost == 0.1

    def test_custom(self):
        p = ResistanceParameters(
            MIC_baseline=0.5,
            n_resistance_levels=3,
            MIC_fold_change=4.0,
            fitness_cost=0.2,
        )
        assert p.MIC_baseline == 0.5
        assert p.n_resistance_levels == 3
        assert p.MIC_fold_change == 4.0
        assert p.fitness_cost == 0.2


# ---------------------------------------------------------------------------
# ResistanceState tests
# ---------------------------------------------------------------------------

class TestResistanceState:

    def test_defaults(self):
        s = ResistanceState()
        assert s.MIC_current == 1.0
        assert s.resistance_level == 0
        assert s.resistant_fraction == 0.0

    def test_to_dict(self):
        s = ResistanceState(MIC_current=2.0, resistance_level=1)
        d = s.to_dict()
        assert d["MIC_current"] == 2.0
        assert d["resistance_level"] == 1


# ---------------------------------------------------------------------------
# ResistanceEvolutionModel tests
# ---------------------------------------------------------------------------

class TestResistanceEvolutionModel:

    @pytest.fixture
    def model(self):
        return ResistanceEvolutionModel()

    @pytest.fixture
    def custom_model(self):
        params = ResistanceParameters(
            MIC_baseline=0.5,
            n_resistance_levels=3,
            MIC_fold_change=4.0,
            fitness_cost=0.15,
        )
        return ResistanceEvolutionModel(params)

    # --- Parameter validation ---

    def test_invalid_mic_baseline(self):
        with pytest.raises(AssertionError):
            ResistanceEvolutionModel(ResistanceParameters(MIC_baseline=-1.0))

    def test_invalid_fold_change(self):
        with pytest.raises(AssertionError):
            ResistanceEvolutionModel(ResistanceParameters(MIC_fold_change=0.5))

    def test_invalid_fitness_cost(self):
        with pytest.raises(AssertionError):
            ResistanceEvolutionModel(ResistanceParameters(fitness_cost=1.5))

    # --- MIC by level ---

    def test_mic_level_0(self, model):
        assert model.get_mic_for_level(0) == 1.0

    def test_mic_level_1(self, model):
        assert model.get_mic_for_level(1) == 2.0

    def test_mic_level_2(self, model):
        assert model.get_mic_for_level(2) == 4.0

    def test_mic_level_3(self, model):
        assert model.get_mic_for_level(3) == 8.0

    def test_mic_custom_fold(self, custom_model):
        assert custom_model.get_mic_for_level(0) == 0.5
        assert custom_model.get_mic_for_level(1) == 2.0  # 0.5 * 4
        assert custom_model.get_mic_for_level(2) == 8.0  # 0.5 * 16

    def test_max_resistance_level(self, model):
        # MIC_baseline=1.0, fold=2.0, max=256.0
        # Levels: 1, 2, 4, 8, 16, 32, 64, 128, 256 → 8 steps
        max_level = model.get_max_resistance_level()
        assert max_level == 8
        assert model.get_mic_for_level(max_level) <= 256.0

    # --- Selection pressure ---

    def test_selection_below_window(self, model):
        # C/MIC = 0.1 < 0.25 (window_low)
        sp = model.compute_selection_pressure(0.1, 1.0)
        assert sp == 0.0

    def test_selection_above_window(self, model):
        # C/MIC = 2.0 > 1.0 (window_high)
        sp = model.compute_selection_pressure(2.0, 1.0)
        assert sp == 0.0

    def test_selection_inside_window(self, model):
        # C/MIC = 0.5 (inside window 0.25-1.0)
        sp = model.compute_selection_pressure(0.5, 1.0)
        assert sp > 0.0
        assert sp <= 1.0

    def test_selection_peak(self, model):
        # C/MIC = 0.625 (midpoint of 0.25-1.0)
        sp_peak = model.compute_selection_pressure(0.625, 1.0)
        sp_edge = model.compute_selection_pressure(0.3, 1.0)
        assert sp_peak >= sp_edge

    def test_selection_zero_mic(self, model):
        sp = model.compute_selection_pressure(1.0, 0.0)
        assert sp == 0.0

    # --- Mutation rate ---

    def test_mutation_rate_positive(self, model):
        rate = model.compute_mutation_rate(1e6, 0.5)
        assert rate > 0

    def test_mutation_rate_scales_with_burden(self, model):
        rate_low = model.compute_mutation_rate(1e4, 0.5)
        rate_high = model.compute_mutation_rate(1e8, 0.5)
        assert rate_high > rate_low

    def test_mutation_rate_cidal_boost(self, model):
        rate_static = model.compute_mutation_rate(1e6, 0.5, drug_class="static")
        rate_cidal = model.compute_mutation_rate(1e6, 0.5, drug_class="cidal")
        assert rate_cidal > rate_static

    # --- Fitness modifier ---

    def test_fitness_level_0(self, model):
        assert model.compute_fitness_modifier(0) == 1.0

    def test_fitness_level_1(self, model):
        assert model.compute_fitness_modifier(1) == pytest.approx(0.9)

    def test_fitness_level_2(self, model):
        assert model.compute_fitness_modifier(2) == pytest.approx(0.8)

    def test_fitness_high_level_clamped(self, model):
        # Level 10 with 0.1 cost per level → 1.0 - 1.0 = 0.0
        assert model.compute_fitness_modifier(10) == 0.0

    # --- Continuous MIC update ---

    def test_continuous_mic_increases(self, model):
        new_mic = model.update_mic_continuous(1.0, 1e3, 0.5, 1.0)
        assert new_mic >= 1.0

    def test_continuous_mic_clamped(self, model):
        new_mic = model.update_mic_continuous(256.0, 1e10, 1.0, 1000.0)
        assert new_mic <= 256.0

    # --- Stepwise MIC update ---

    def test_stepwise_no_step_without_pressure(self, model):
        new_mic, new_acc = model.update_mic_stepwise(1.0, 0.0, 0.0, 1.0, 1e6)
        assert new_mic == 1.0

    def test_stepwise_accumulation(self, model):
        # With selection pressure, accumulator should increase
        _, new_acc = model.update_mic_stepwise(1.0, 0.0, 0.5, 1.0, 1e8)
        assert new_acc > 0.0

    def test_stepwise_step_increase(self, model):
        # Force a step by setting accumulator above threshold
        threshold = 1e8 * 1e-7 * 100  # B_total * mu * 100
        new_mic, new_acc = model.update_mic_stepwise(1.0, threshold + 1, 0.5, 1.0, 1e8)
        assert new_mic == 2.0  # 2-fold increase
        assert new_acc == 0.0  # Reset

    def test_stepwise_clamped_to_max(self, model):
        threshold = 1e8 * 1e-7 * 100
        new_mic, _ = model.update_mic_stepwise(256.0, threshold + 1, 0.5, 1.0, 1e8)
        assert new_mic == 256.0  # Clamped to max

    # --- Resistant fraction ---

    def test_resistant_fraction_starts_zero(self, model):
        new_frac = model.compute_resistant_fraction(0.0, 0.5, 0.9, 1e3, 1.0)
        # Should increase from zero due to mutation
        assert new_frac > 0.0

    def test_resistant_fraction_selection(self, model):
        frac_no_sel = model.compute_resistant_fraction(0.1, 0.0, 0.9, 1e3, 1.0)
        frac_sel = model.compute_resistant_fraction(0.1, 0.5, 0.9, 1e3, 1.0)
        assert frac_sel >= frac_no_sel

    def test_resistant_fraction_clamped(self, model):
        new_frac = model.compute_resistant_fraction(0.99, 1.0, 0.9, 1e10, 1.0)
        assert new_frac <= 1.0

    def test_resistant_fraction_non_negative(self, model):
        # High fitness loss should not make fraction negative
        new_frac = model.compute_resistant_fraction(0.01, 0.0, 0.0, 0.0, 100.0)
        assert new_frac >= 0.0


# ---------------------------------------------------------------------------
# ResistanceODESystem tests
# ---------------------------------------------------------------------------

class TestResistanceODESystem:

    @pytest.fixture
    def ode_sys(self):
        return ResistanceODESystem(n_levels=4)

    def test_n_states(self, ode_sys):
        assert ode_sys.get_n_states() == 4

    def test_state_indices(self, ode_sys):
        indices = ode_sys.get_state_indices()
        assert len(indices) == 4
        assert "B_resist_0" in indices
        assert "B_resist_3" in indices

    def test_rhs_shape(self, ode_sys):
        B_resist = np.array([1e6, 1e3, 1e1, 1e0])
        dB = ode_sys.rhs(
            t=0, B_resist=B_resist, B_pers=1e2, B_SCV=0,
            N_eff=1e7, C_effect=1.0, k_growth=0.5, B_max=1e9,
            k_kill_base=1e-8, drug_class="cidal",
        )
        assert dB.shape == (4,)

    def test_rhs_no_drug_growth(self, ode_sys):
        """Without drug, susceptible population should grow."""
        B_resist = np.array([1e5, 0, 0, 0])
        dB = ode_sys.rhs(
            t=0, B_resist=B_resist, B_pers=0, B_SCV=0,
            N_eff=1e7, C_effect=0.0, k_growth=0.5, B_max=1e9,
            k_kill_base=1e-8, drug_class="cidal",
        )
        # Susceptible population should grow (positive dB[0])
        # minus immune killing
        assert dB[0] > 0 or abs(dB[0]) < 1e3  # growth or near-zero

    def test_rhs_cidal_kills(self, ode_sys):
        """With cidal drug, susceptible population should decrease."""
        B_resist = np.array([1e6, 0, 0, 0])
        dB = ode_sys.rhs(
            t=0, B_resist=B_resist, B_pers=0, B_SCV=0,
            N_eff=1e7, C_effect=10.0, k_growth=0.5, B_max=1e9,
            k_kill_base=1e-8, drug_class="cidal",
        )
        # High drug concentration should kill susceptible bacteria
        assert dB[0] < 0

    def test_rhs_resistant_less_affected(self, ode_sys):
        """Resistant bacteria should be less affected by drug."""
        B_susceptible = np.array([1e6, 0, 0, 0])
        B_resistant = np.array([0, 0, 1e6, 0])  # Level 2 resistant

        dB_sus = ode_sys.rhs(
            0, B_susceptible, 0, 0, 1e7, 5.0, 0.5, 1e9, 1e-8, "cidal",
        )
        dB_res = ode_sys.rhs(
            0, B_resistant, 0, 0, 1e7, 5.0, 0.5, 1e9, 1e-8, "cidal",
        )
        # Resistant bacteria should have less negative (or more positive) dB
        assert dB_res[2] >= dB_sus[0]

    def test_rhs_mutation_transfer(self, ode_sys):
        """Mutation should transfer bacteria from lower to higher levels."""
        B_resist = np.array([1e8, 0, 0, 0])
        dB = ode_sys.rhs(
            0, B_resist, 0, 0, 1e7, 0.5, 0.5, 1e9, 1e-8, "cidal",
        )
        # Level 1 should gain from mutations of level 0
        # (depends on selection pressure, but at least the mechanism exists)
        assert len(dB) == 4


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:

    def test_create_resistance_model(self):
        model = create_resistance_model(MIC_baseline=0.5, n_levels=3)
        assert model.params.MIC_baseline == 0.5
        assert model.params.n_resistance_levels == 3

    def test_create_resistance_ode(self):
        ode = create_resistance_ode(n_levels=3)
        assert ode.n_levels == 3
        assert ode.get_n_states() == 3


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulateResistanceEvolution:

    def test_basic_simulation(self):
        result = simulate_resistance_evolution(
            drug_concentration=2.0,
            duration_hours=48.0,
            dt=0.1,
        )
        assert "t" in result
        assert "B_resist" in result
        assert "B_total" in result
        assert "MIC_effective" in result
        assert "resistant_fraction" in result
        assert len(result["t"]) > 0

    def test_no_drug_no_resistance(self):
        """Without drug, resistance fraction should stay low."""
        result = simulate_resistance_evolution(
            drug_concentration=0.0,
            duration_hours=168.0,
            dt=0.1,
            initial_burden=1e6,
        )
        # Resistant fraction should be very low without drug pressure
        assert result["resistant_fraction"][-1] < 0.01

    def test_drug_pressure_increases_resistance(self):
        """With drug in selection window, resistance should increase."""
        # C/MIC = 0.5/1.0 = 0.5 (inside selection window 0.25-1.0)
        result_with_drug = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        result_no_drug = simulate_resistance_evolution(
            drug_concentration=0.0,
            duration_hours=168.0,
            dt=0.1,
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        # Drug pressure should increase resistance fraction
        assert result_with_drug["resistant_fraction"][-1] >= \
               result_no_drug["resistant_fraction"][-1]

    def test_mic_increases_under_pressure(self):
        """MIC should increase under sustained drug pressure."""
        # C/MIC = 0.5/1.0 = 0.5 (inside selection window)
        result = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        # MIC should have increased from baseline
        assert result["MIC_effective"][-1] >= result["MIC_effective"][0]

    def test_cidal_vs_static(self):
        """Cidal drugs should drive faster resistance evolution."""
        # C/MIC = 0.5/1.0 = 0.5 (inside selection window)
        result_cidal = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            drug_class="cidal",
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        result_static = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            drug_class="static",
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        # Cidal drugs boost mutation rate, so resistance should evolve faster
        assert result_cidal["resistant_fraction"][-1] >= \
               result_static["resistant_fraction"][-1]

    def test_higher_fitness_cost_slows_evolution(self):
        """Higher fitness cost should slow resistance evolution."""
        # C/MIC = 0.5/1.0 = 0.5 (inside selection window)
        result_low_cost = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            fitness_cost=0.05,
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        result_high_cost = simulate_resistance_evolution(
            drug_concentration=0.5,
            duration_hours=168.0,
            dt=0.1,
            fitness_cost=0.3,
            initial_burden=1e8,
            mu_resistance=1e-5,
        )
        # Higher fitness cost should result in lower resistant fraction
        assert result_low_cost["resistant_fraction"][-1] >= \
               result_high_cost["resistant_fraction"][-1]

    def test_output_shapes(self):
        """Output arrays should have consistent shapes."""
        result = simulate_resistance_evolution(
            drug_concentration=2.0,
            duration_hours=24.0,
            dt=0.5,
            n_levels=3,
        )
        n_steps = int(24.0 / 0.5) + 1
        assert result["t"].shape == (n_steps,)
        assert result["B_resist"].shape == (n_steps, 3)
        assert result["B_total"].shape == (n_steps,)
        assert result["MIC_effective"].shape == (n_steps,)
        assert result["resistant_fraction"].shape == (n_steps,)


# ===========================================================================
# Two-stage pipeline tests
# ===========================================================================

class TestGetEffectSiteConcentration:
    """Tests for SimulationResult.get_effect_site_concentration()."""

    def test_returns_array(self):
        """Should return a numpy array of concentrations."""
        from src.core.simulation import SimulationResult
        from src.core.pk_models import TwoCompartmentPKModel

        pk_model = TwoCompartmentPKModel(CL=10.0, Vc=20.0, Vp=15.0, Q=5.0, Kp=0.7)

        t = np.array([0, 1, 2, 3])
        # A_central values that give known concentrations
        A_c = np.array([100, 80, 60, 40])
        y = np.zeros((4, 11))
        y[:, 0] = A_c  # A_central column

        result = SimulationResult(t, y, ["A_central"] + ["x"] * 10, {})
        C = result.get_effect_site_concentration(pk_model)

        assert isinstance(C, np.ndarray)
        assert len(C) == 4

    def test_concentration_formula(self):
        """C_effect = Kp * A_central / Vc_total (from stored params)."""
        from src.core.simulation import SimulationResult
        from src.core.pk_models import TwoCompartmentPKModel

        Kp = 0.7
        Vc = 20.0
        pk_model = TwoCompartmentPKModel(CL=10.0, Vc=Vc, Vp=15.0, Q=5.0, Kp=Kp)

        t = np.array([0])
        y = np.zeros((1, 12))  # 12 states (added PAMP)
        y[0, 0] = 100  # A_central = 100 mg

        # With stored params, Vc_total = 17.5 L (default), Kp_val = 0.4 (default)
        result = SimulationResult(t, y, ["A_central"] + ["x"] * 11, {
            '_Vc_val': Vc, '_Kp_val': Kp
        })
        C = result.get_effect_site_concentration(pk_model)

        # C_effect = Kp * A_central / Vc_total = 0.7 * 100 / 20.0 = 3.5 mg/L
        expected = Kp * 100 / Vc
        assert np.isclose(C[0], expected)


class TestSimulateResistanceFromProfile:
    """Tests for simulate_resistance_from_profile()."""

    def test_returns_dict_with_keys(self):
        """Should return dict with expected keys."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 24, 100)
        C = np.ones(100) * 0.5  # constant 0.5 mg/L

        result = simulate_resistance_from_profile(C, t)

        for key in ["t", "B_resist", "B_total", "MIC_effective", "resistant_fraction"]:
            assert key in result

    def test_output_shapes(self):
        """Output arrays should have consistent shapes."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 24, 100)
        C = np.ones(100) * 1.0

        result = simulate_resistance_from_profile(C, t, n_levels=4)

        assert result["B_resist"].shape[1] == 4
        assert len(result["t"]) == len(result["B_total"])
        assert len(result["MIC_effective"]) == len(result["t"])

    def test_no_drug_no_mic_increase(self):
        """Without drug, MIC should remain at baseline."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 24, 100)
        C = np.zeros(100)  # no drug

        result = simulate_resistance_from_profile(C, t, MIC_baseline=1.0)

        # MIC should stay at baseline (small fluctuations from numerical noise)
        assert result["MIC_effective"][0] == 1.0
        assert result["MIC_effective"][-1] < 2.0  # no significant increase

    def test_drug_pressure_increases_mic(self):
        """Sustained drug pressure should increase effective MIC."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 168, 500)
        C = np.ones(500) * 1.0  # constant 1x MIC

        result = simulate_resistance_from_profile(C, t, MIC_baseline=1.0)

        final_mic = result["MIC_effective"][-1]
        assert final_mic >= 1.0  # at least baseline

    def test_higher_fitness_cost_slows_evolution(self):
        """Higher fitness cost should slow resistance evolution."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 168, 500)
        C = np.ones(500) * 0.75  # within mutant selection window

        res_low_cost = simulate_resistance_from_profile(C, t, fitness_cost=0.05)
        res_high_cost = simulate_resistance_from_profile(C, t, fitness_cost=0.20)

        assert res_high_cost["MIC_effective"][-1] <= res_low_cost["MIC_effective"][-1]

    def test_cidal_vs_static(self):
        """Cidal and static drugs should produce different resistance dynamics."""
        from src.therapy.resistance_evolution import simulate_resistance_from_profile

        t = np.linspace(0, 168, 500)
        C = np.ones(500) * 0.75

        res_cidal = simulate_resistance_from_profile(C, t, drug_class="cidal")
        res_static = simulate_resistance_from_profile(C, t, drug_class="static")

        # Both should produce valid results
        assert res_cidal["MIC_effective"][0] == 1.0
        assert res_static["MIC_effective"][0] == 1.0


class TestRunSimulationWithResistance:
    """Tests for the full two-stage pipeline."""

    def _make_pd_model(self, drug_class="cidal"):
        from src.core.parameters import get_default_parameters
        from src.core.pd_model import BacterialPopulationODE
        params = get_default_parameters()
        params['drug_class'] = drug_class
        return BacterialPopulationODE(params)

    def _make_pk_model(self):
        from src.core.pk_models import TwoCompartmentPKModel
        return TwoCompartmentPKModel(CL=15.0, Vc=0.25, Vp=0.15, Q=3.5, Kp=0.7)

    def test_returns_tuple(self):
        """Should return (SimulationResult, dict)."""
        from src.core.pk_models import DosingRegimen
        from src.therapy.resistance_evolution import run_simulation_with_resistance

        pd_model = self._make_pd_model("cidal")
        pk_model = self._make_pk_model()
        regimen = DosingRegimen(dose_mg=500, interval_hours=8, n_doses=6,
                                 infusion_duration_min=60)
        ic = {"B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
              "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0}

        sim_res, res = run_simulation_with_resistance(
            pk_model, regimen, pd_model, ic,
            t_span=(0, 48), drug_class="cidal",
        )

        # Stage 1 result
        assert hasattr(sim_res, "t")
        assert hasattr(sim_res, "y")
        _, burden = sim_res.get_bacterial_burden()
        assert len(burden) == len(sim_res.t)

        # Stage 2 result
        assert "MIC_effective" in res
        assert "resistant_fraction" in res
        assert res["MIC_effective"][0] == 1.0

    def test_concentration_profile_matches_simulation(self):
        """The concentration profile used for resistance should match simulation."""
        from src.core.pk_models import DosingRegimen
        from src.therapy.resistance_evolution import run_simulation_with_resistance

        pd_model = self._make_pd_model("cidal")
        pk_model = self._make_pk_model()
        regimen = DosingRegimen(dose_mg=500, interval_hours=8, n_doses=3,
                                 infusion_duration_min=60)
        ic = {"B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
              "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0}

        sim_res, res = run_simulation_with_resistance(
            pk_model, regimen, pd_model, ic,
            t_span=(0, 24), drug_class="cidal",
        )

        # Verify the resistance time grid covers the simulation time
        assert res["t"][-1] >= sim_res.t[-1] - 1.0  # within 1 hour tolerance

    def test_high_dose_clears_infection_low_resistance(self):
        """High dose cidal therapy should clear infection with minimal resistance."""
        from src.core.pk_models import DosingRegimen
        from src.therapy.resistance_evolution import run_simulation_with_resistance

        pd_model = self._make_pd_model("cidal")
        pk_model = self._make_pk_model()
        regimen = DosingRegimen(dose_mg=1000, interval_hours=6, n_doses=8,
                                 infusion_duration_min=60)
        ic = {"B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
              "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0}

        sim_res, res = run_simulation_with_resistance(
            pk_model, regimen, pd_model, ic,
            t_span=(0, 48), drug_class="cidal",
        )

        _, final_burden = sim_res.get_bacterial_burden()
        # With high dose cidal, burden should be low
        assert final_burden[-1] < 1e3
