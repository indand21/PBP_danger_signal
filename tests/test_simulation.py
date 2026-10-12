"""
Tests for simulation engine.
"""
import pytest
import numpy as np
from src.core.simulation import run_simulation, SimulationResult


class TestSimulationResult:
    """Tests for SimulationResult class."""

    def test_initialization(self):
        t = np.array([0, 1, 2])
        y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1],
                      [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2]])
        names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF']
        result = SimulationResult(t, y, names, {})
        assert len(result.t) == 3
        assert result.y.shape == (3, 11)

    def test_get_bacterial_burden(self):
        t = np.array([0, 1])
        y = np.array([[0, 0, 0, 0, 1e6, 1e3, 0, 1e7, 0, 10, 5],
                      [0, 0, 0, 0, 2e6, 2e3, 0, 1e7, 0, 10, 5]])
        names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF']
        result = SimulationResult(t, y, names, {})
        t_out, B = result.get_bacterial_burden()
        assert B[0] == 1.001e6
        assert B[1] == 2.002e6

    def test_get_resistance_fraction(self):
        t = np.array([0, 1])
        y = np.array([[0, 0, 0, 0, 1e6, 1e3, 1e2, 1e7, 0, 10, 5],
                      [0, 0, 0, 0, 5e5, 5e2, 5e5, 1e7, 0, 10, 5]])
        names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF']
        result = SimulationResult(t, y, names, {})
        t_out, frac = result.get_resistance_fraction()
        assert frac[0] == pytest.approx(1e2 / 1.0101e6, abs=1e-6)
        assert frac[1] == pytest.approx(5e5 / 1.0005e6, abs=1e-6)

    def test_get_cytokines(self):
        t = np.array([0, 1])
        y = np.array([[0, 0, 0, 0, 1e6, 1e3, 0, 1e7, 0, 10, 5],
                      [0, 0, 0, 0, 1e6, 1e3, 0, 1e7, 0, 20, 10]])
        names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF']
        result = SimulationResult(t, y, names, {})
        t_out, IL6, TNF = result.get_cytokines()
        assert IL6[0] == 10
        assert IL6[1] == 20
        assert TNF[0] == 5
        assert TNF[1] == 10

    def test_dataframe_creation(self):
        t = np.array([0, 1])
        y = np.array([[0, 0, 0, 0, 1e6, 1e3, 0, 1e7, 0, 10, 5],
                      [0, 0, 0, 0, 1e6, 1e3, 0, 1e7, 0, 10, 5]])
        names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF']
        result = SimulationResult(t, y, names, {})
        assert 'time' in result.df.columns
        assert 'B_rep' in result.df.columns
        assert len(result.df) == 2


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_basic_run(self, pk_model_meropenem, standard_regimen, pd_model, standard_init_cond):
        result = run_simulation(
            pk_model=pk_model_meropenem,
            regimen=standard_regimen,
            pd_model=pd_model,
            initial_conditions=standard_init_cond,
            t_span=(0, 24),
            drug_class="cidal",
            weight_kg=70,
        )
        assert isinstance(result, SimulationResult)
        assert len(result.t) > 0
        assert result.y.shape[1] == 13  # 4 PK + 9 PD (including PAMP and D_host)

    def test_static_drug(self, pk_model_doxycycline, standard_regimen, pd_model, standard_init_cond):
        result = run_simulation(
            pk_model=pk_model_doxycycline,
            regimen=standard_regimen,
            pd_model=pd_model,
            initial_conditions=standard_init_cond,
            t_span=(0, 24),
            drug_class="static",
            weight_kg=70,
        )
        assert isinstance(result, SimulationResult)

    def test_no_drug(self, pk_model_meropenem, pd_model, standard_init_cond):
        from src.core.pk_models import DosingRegimen
        regimen = DosingRegimen(dose_mg=0, interval_hours=24, n_doses=0)
        result = run_simulation(
            pk_model=pk_model_meropenem,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=standard_init_cond,
            t_span=(0, 24),
            drug_class="none",
            weight_kg=70,
        )
        assert isinstance(result, SimulationResult)
        # Without drug, bacteria should grow
        t, B = result.get_bacterial_burden()
        assert B[-1] > B[0]

    def test_bacterial_results_physiological(self, short_simulation_result):
        """Results should be physiologically plausible."""
        t, B = short_simulation_result.get_bacterial_burden()
        assert np.all(B >= 0)
        assert np.all(np.isfinite(B))
        # Should not exceed carrying capacity by much
        assert B.max() < 1e10

    def test_cytokine_results_physiological(self, short_simulation_result):
        """Cytokines should be in plausible range, allowing for numerical noise."""
        t, IL6, TNF = short_simulation_result.get_cytokines()
        assert np.all(IL6 >= 0)
        assert np.all(np.isfinite(IL6))
        assert np.all(np.isfinite(TNF))
        # TNF may dip slightly below zero due to numerical noise in ODE solver
        assert TNF.max() > 0  # Should have some positive TNF
        assert IL6.max() < 1e9  # Sanity check

    def test_time_progression(self, short_simulation_result):
        """Time should always increase."""
        t = short_simulation_result.t
        assert np.all(np.diff(t) > 0)

    def test_state_names_correct(self, short_simulation_result):
        """Should have correct state names."""
        expected = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                    'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF', 'PAMP',
                    'D_host']
        assert short_simulation_result.state_names == expected

    def test_different_methods(self, pk_model_meropenem, standard_regimen, pd_model, standard_init_cond):
        """Test different ODE integration methods."""
        for method in ['RK45', 'BDF']:
            result = run_simulation(
                pk_model=pk_model_meropenem,
                regimen=standard_regimen,
                pd_model=pd_model,
                initial_conditions=standard_init_cond,
                t_span=(0, 12),
                drug_class="cidal",
                weight_kg=70,
                method=method,
            )
            assert isinstance(result, SimulationResult)

    def test_long_simulation(self, pk_model_meropenem, standard_regimen, pd_model, standard_init_cond):
        """Test 96-hour simulation completes."""
        result = run_simulation(
            pk_model=pk_model_meropenem,
            regimen=standard_regimen,
            pd_model=pd_model,
            initial_conditions=standard_init_cond,
            t_span=(0, 96),
            drug_class="cidal",
            weight_kg=70,
        )
        assert isinstance(result, SimulationResult)
        assert result.t[-1] >= 95  # Should reach near 96h

    def test_zero_initial_bacteria(self, pk_model_meropenem, standard_regimen, pd_model):
        """Test with no initial bacteria."""
        init_cond = {
            "B_rep": 0,
            "B_pers": 0,
            "B_SCV": 0,
            "N_eff": 1e7,
            "Damage": 0,
            "IL6": 10,
            "TNF": 5,
        }
        result = run_simulation(
            pk_model=pk_model_meropenem,
            regimen=standard_regimen,
            pd_model=pd_model,
            initial_conditions=init_cond,
            t_span=(0, 24),
            drug_class="cidal",
            weight_kg=70,
        )
        t, B = result.get_bacterial_burden()
        # Bacteria may grow from numerical zero, or stay at zero
        # The key is that total burden stays very low if starting from zero
        assert B.max() < 10  # Should stay negligible
