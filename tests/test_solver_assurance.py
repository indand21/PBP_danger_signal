"""Guard tests for integration failure handling (external review R11).

The review found that run_simulation() printed a warning when the solver
reported failure and returned the incomplete result anyway, and that the
danger-signal integrator read its output without checking success at all.
Downstream summaries -- AUCs, ratios, medians -- would then be computed from a
truncated or non-finite trajectory that looks like any other.

A failed or non-physical integration must now stop the analysis, loudly.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel
from src.core.simulation import check_integration, run_simulation

IC = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
      "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0}


def _pk():
    p = get_drug_pk_parameters("meropenem")
    return TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka,
                                 Kp=p.Kp, effect_site_model=True)


def _reg():
    return DosingRegimen(dose_mg=1000.0, interval_hours=8.0, start_time=0,
                         n_doses=1, infusion_duration_min=30)


def _sol(t, y, success=True, message="ok"):
    return SimpleNamespace(t=np.asarray(t, float), y=np.asarray(y, float),
                           success=success, message=message)


class TestCheckIntegration:
    def test_accepts_a_complete_finite_nonnegative_solution(self):
        check_integration(_sol([0, 1, 2], [[1.0, 0.5, 0.2]]), t_end=2.0,
                          where="unit")

    def test_rejects_a_solver_reported_failure(self):
        with pytest.raises(RuntimeError, match="did not succeed"):
            check_integration(_sol([0, 1], [[1.0, 0.5]], success=False,
                                   message="step size too small"),
                              t_end=1.0, where="unit")

    def test_rejects_a_trajectory_that_stops_early(self):
        with pytest.raises(RuntimeError, match="stopped at"):
            check_integration(_sol([0, 1], [[1.0, 0.5]]), t_end=2.0,
                              where="unit")

    def test_rejects_non_finite_states(self):
        with pytest.raises(RuntimeError, match="non-finite"):
            check_integration(_sol([0, 1, 2], [[1.0, np.nan, 0.2]]),
                              t_end=2.0, where="unit")

    def test_rejects_material_negativity_in_a_population_state(self):
        with pytest.raises(RuntimeError, match="negative"):
            check_integration(_sol([0, 1, 2], [[1e6, 5e5, -10.0]]),
                              t_end=2.0, where="unit",
                              nonnegative_rows=[0])

    def test_tolerates_negativity_at_rounding_level(self):
        """Clipping in the right-hand side keeps states near zero, not at it;
        a few ulps below zero on a 1e6 scale is not a physical defect."""
        check_integration(_sol([0, 1, 2], [[1e6, 5e5, -1e-9]]), t_end=2.0,
                          where="unit", nonnegative_rows=[0])

    def test_tolerates_integration_noise_on_a_state_that_is_zero(self):
        """A population that starts at zero and stays there undershoots by
        integration noise: -3e-5 CFU/mL was observed for a zero inoculum. A
        purely relative tolerance cannot accept that, because the state's
        own scale is the noise itself."""
        check_integration(_sol([0, 1, 2], [[0.0, -3.12e-5, 0.0]]), t_end=2.0,
                          where="unit", nonnegative_rows=[0])

    def test_still_rejects_material_negativity_near_zero(self):
        """The absolute floor must stay far below anything physical."""
        with pytest.raises(RuntimeError, match="negative"):
            check_integration(_sol([0, 1, 2], [[0.0, -0.5, 0.0]]), t_end=2.0,
                              where="unit", nonnegative_rows=[0])

    def test_message_names_the_failing_layer(self):
        with pytest.raises(RuntimeError, match="bacterial PD"):
            check_integration(_sol([0, 1], [[1.0, 0.5]], success=False),
                              t_end=1.0, where="bacterial PD")


class TestRunSimulationFailsLoudly:
    def test_a_failed_integration_raises_instead_of_returning(self):
        """Previously this printed a warning and returned the partial result."""
        pd = create_ode_system(get_default_parameters())
        pd.rhs = lambda t, y, **kw: np.full(len(y), np.nan)
        with pytest.raises(RuntimeError):
            run_simulation(_pk(), _reg(), pd, dict(IC), t_span=(0, 4.0),
                           drug_class="cidal", weight_kg=70.0)

    def test_a_normal_run_still_succeeds(self):
        pd = create_ode_system(get_default_parameters())
        r = run_simulation(_pk(), _reg(), pd, dict(IC), t_span=(0, 4.0),
                           drug_class="cidal", weight_kg=70.0)
        assert r.t[-1] == pytest.approx(4.0)


class TestDangerSignalIntegratorFailsLoudly:
    def test_a_failed_integration_raises(self):
        """Previously the danger-signal layer read sol.y without checking
        sol.success at all."""
        from src.analysis.danger_signal import (
            DangerSignalParameters, integrate_ds_cytokine,
        )
        t = np.linspace(0.0, 4.0, 200)
        src = np.full_like(t, np.nan)
        with pytest.raises(RuntimeError):
            integrate_ds_cytokine(t, src, DangerSignalParameters())
