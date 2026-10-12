"""The protein-binding convention (external review R4).

$K_p$ is a TOTAL-concentration tissue partition, so ``Kp * C_plasma`` is a
total effect-site concentration and the pharmacologically active quantity is
``fu`` times it. Binding must therefore be applied exactly once, where drug
meets bacteria, and the reported free-drug indices must describe that same
concentration.

Previously the study scripts applied ``fu`` nowhere: all three "free" indices
and the kill function were computed from the total concentration. Applying it
only to the reported indices would be worse than either, because the matched
comparison solves doses to equalise the reported index: with different ``fu``
per agent (0.90 ceftazidime, 0.98 meropenem) the agents would then receive
different total exposures and would no longer kill equally, which is the
property the whole comparison rests on.
"""
import numpy as np
import pytest

from src.core.parameters import (
    get_default_parameters, get_drug_pd_parameters, get_drug_pk_parameters,
)
from src.core.pd_model import create_ode_system
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel
from src.core.simulation import run_simulation

WEIGHT = 70.0
IC = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
      "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0}


def _pk(drug="meropenem"):
    p = get_drug_pk_parameters(drug)
    return TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka,
                                 Kp=p.Kp, effect_site_model=True)


def _run(fu, dose_mg=500.0, t_end=24.0, drug="meropenem"):
    reg = DosingRegimen(dose_mg=dose_mg, interval_hours=8.0, start_time=0,
                        n_doses=3, infusion_duration_min=30)
    return run_simulation(_pk(drug), reg,
                          create_ode_system(get_default_parameters()),
                          dict(IC), t_span=(0, t_end), drug_class="cidal",
                          weight_kg=WEIGHT, fraction_unbound=fu)


class TestBindingReachesTheKillFunction:
    def test_lower_unbound_fraction_kills_less(self):
        """The point of the convention: only free drug acts."""
        full = _run(1.0)
        half = _run(0.5)
        _, b_full = full.get_bacterial_burden()
        _, b_half = half.get_bacterial_burden()
        assert b_half[-1] > b_full[-1]

    def test_unbound_fraction_of_one_is_the_old_behaviour(self):
        """Backwards compatible: fu = 1 must reproduce the unadjusted run."""
        reg = DosingRegimen(dose_mg=500.0, interval_hours=8.0, start_time=0,
                            n_doses=3, infusion_duration_min=30)
        plain = run_simulation(_pk(), reg,
                               create_ode_system(get_default_parameters()),
                               dict(IC), t_span=(0, 24.0), drug_class="cidal",
                               weight_kg=WEIGHT)
        explicit = _run(1.0)
        _, a = plain.get_bacterial_burden()
        _, b = explicit.get_bacterial_burden()
        assert np.allclose(a, b, rtol=1e-12)


class TestBindingIsAppliedExactlyOnce:
    def test_effect_site_concentration_stays_total(self):
        """``get_effect_site_concentration`` is the TOTAL concentration, so a
        caller that multiplies by fu is not applying it twice.

        The two runs kill differently and so land on different adaptive
        grids, which samples the peaks at different points. The grid-robust
        statement is that the total exposure and the underlying PK are
        untouched by fu.
        """
        from scipy.integrate import trapezoid
        r_full = _run(1.0)
        r_half = _run(0.5)
        auc_full = trapezoid(r_full.get_effect_site_concentration(), r_full.t)
        auc_half = trapezoid(r_half.get_effect_site_concentration(), r_half.t)
        # 1e-3 covers trapezoid error between the two grids. If fu leaked
        # into the total it would halve the AUC, a 500-fold larger effect.
        assert auc_half == pytest.approx(auc_full, rel=1e-3)
        for k in ("_CL_val", "_Vc_val", "_Kp_val"):
            assert r_full.params[k] == r_half.params[k]

    def test_free_concentration_is_fu_times_total(self):
        r = _run(0.5)
        assert np.allclose(r.get_free_effect_site_concentration(),
                           0.5 * r.get_effect_site_concentration(), rtol=1e-12)

    def test_stored_unbound_fraction_is_recorded(self):
        assert _run(0.9).params["_fraction_unbound"] == 0.9


class TestReportedIndicesUseFreeDrug:
    def test_phase4_reports_the_free_index_not_the_total_one(self):
        """The reported %fT>MIC must be computed from free concentration."""
        import importlib.util
        import pathlib
        path = (pathlib.Path(__file__).resolve().parent.parent / "scripts"
                / "ds_phase4_analyses.py")
        spec = importlib.util.spec_from_file_location("p4", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mic = 0.4011
        sim = mod.simulate("ceftazidime", 300.0, 8.0, 6, mic=mic)
        fu = get_drug_pd_parameters("ceftazidime").fraction_unbound
        from src.core.simulation import SimulationResult
        t = sim["t"]
        free = fu * sim["C_eff_total"]
        expected = 100.0 * SimulationResult._time_above_threshold(t, free, mic) \
            / (t[-1] - t[0])
        assert sim["fT"] == pytest.approx(expected, abs=1e-9)

    def test_free_index_is_lower_than_the_total_one(self):
        """fu < 1, so the honest index must be smaller than the one the study
        previously reported."""
        import importlib.util
        import pathlib
        path = (pathlib.Path(__file__).resolve().parent.parent / "scripts"
                / "ds_phase4_analyses.py")
        spec = importlib.util.spec_from_file_location("p4", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mic = 0.4011
        sim = mod.simulate("ceftazidime", 300.0, 8.0, 6, mic=mic)
        from src.core.simulation import SimulationResult
        t = sim["t"]
        total_based = 100.0 * SimulationResult._time_above_threshold(
            t, sim["C_eff_total"], mic) / (t[-1] - t[0])
        assert sim["fT"] < total_based
