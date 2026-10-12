"""Guard tests for the pharmacokinetic representation used by run_simulation().

The external review (R3) found two inconsistent PK representations in one
simulation. Killing was driven by an analytical ONE-compartment solution
(k = CL/Vc, times Kp), while four "two-compartment" ODE states were integrated
alongside it from equations that do not conserve drug: with no input they
could CREATE drug mass. Those states were exported in every result, and shared
the adaptive solver's error control with the bacterial states.

The study now uses one representation, the one-compartment solution, and the
exported drug states are derived from it. These tests pin the properties that
make that representation physically coherent: the exported amounts reproduce
the concentration that drives killing, drug mass is conserved, and the
solution has the analytical limits a one-compartment model must have.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel
from src.core.simulation import run_simulation

WEIGHT_KG = 70.0
IC = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
      "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0}


def _pk(drug="meropenem", Ka=None):
    p = get_drug_pk_parameters(drug)
    return TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q,
                                 Ka=p.Ka if Ka is None else Ka, Kp=p.Kp,
                                 effect_site_model=True)


def _run(pk, dose_mg=1000.0, interval_h=8.0, n_doses=1, inf_min=30,
         t_end=8.0):
    reg = DosingRegimen(dose_mg=dose_mg, interval_hours=interval_h,
                        start_time=0, n_doses=n_doses,
                        infusion_duration_min=inf_min)
    r = run_simulation(pk_model=pk, regimen=reg,
                       pd_model=create_ode_system(get_default_parameters()),
                       initial_conditions=dict(IC), t_span=(0, t_end),
                       drug_class="cidal", weight_kg=WEIGHT_KG)
    return r, reg


def _state(r, name):
    return r.y[:, r.state_names.index(name)]


def _input_to(t, reg):
    """Cumulative drug administered by each time t, mg."""
    dur = reg.infusion_duration_min / 60.0
    out = np.zeros_like(t)
    for td in reg.get_dose_times():
        tau = np.clip(t - td, 0.0, None)
        out += reg.dose_mg * (np.clip(tau / dur, 0.0, 1.0) if dur > 1e-3
                              else (tau > 0).astype(float))
    return out


class TestOneRepresentation:
    def test_exported_amount_reproduces_the_driving_concentration(self):
        """The amount the result exports must be the amount whose
        concentration drives killing. Before the fix they were unrelated."""
        pk = _pk()
        r, _ = _run(pk)
        Vc = r.params["_Vc_val"]
        from_state = pk.Kp * _state(r, "A_central") / Vc
        driving = r.get_effect_site_concentration(pk_model=pk,
                                                  weight_kg=WEIGHT_KG)
        assert np.allclose(from_state, driving, rtol=1e-9, atol=1e-12)

    def test_there_is_no_peripheral_compartment(self):
        """A one-compartment model has no peripheral drug. Exporting a nonzero
        peripheral amount would imply a second compartment that nothing uses."""
        r, _ = _run(_pk(), n_doses=3, t_end=24.0)
        assert np.all(_state(r, "A_peripheral") == 0.0)


class TestDrugMassBalance:
    def test_iv_mass_is_conserved(self):
        """input(t) = A_central(t) + CL * integral(C dt). The old equations
        failed this: with no input they gained ~12.8 mg/h."""
        pk = _pk()
        r, reg = _run(pk, n_doses=3, t_end=24.0)
        t = r.t
        C = _state(r, "A_central") / r.params["_Vc_val"]
        eliminated = pk.CL * np.concatenate(
            [[0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * np.diff(t))])
        balance = _input_to(t, reg) - _state(r, "A_central") - eliminated
        assert np.max(np.abs(balance)) < 1e-2 * reg.dose_mg

    def test_oral_mass_is_conserved_through_the_absorption_depot(self):
        pk = _pk(Ka=1.2)
        r, reg = _run(pk, n_doses=2, t_end=16.0, inf_min=0)
        t = r.t
        C = _state(r, "A_central") / r.params["_Vc_val"]
        eliminated = pk.CL * np.concatenate(
            [[0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * np.diff(t))])
        balance = (_input_to(t, reg) - _state(r, "A_absorption")
                   - _state(r, "A_central") - eliminated)
        assert np.max(np.abs(balance)) < 1e-2 * reg.dose_mg

    def test_no_drug_appears_before_the_first_dose(self):
        pk = _pk()
        reg = DosingRegimen(dose_mg=1000.0, interval_hours=8.0, start_time=2.0,
                            n_doses=1, infusion_duration_min=30)
        r = run_simulation(pk_model=pk, regimen=reg,
                           pd_model=create_ode_system(get_default_parameters()),
                           initial_conditions=dict(IC), t_span=(0, 6.0),
                           drug_class="cidal", weight_kg=WEIGHT_KG)
        early = r.t < 2.0
        assert np.all(_state(r, "A_central")[early] == 0.0)


class TestAnalyticalLimits:
    def test_post_infusion_decay_has_the_one_compartment_half_life(self):
        pk = _pk()
        r, _ = _run(pk, t_end=8.0)
        k = pk.CL / r.params["_Vc_val"]
        t = r.t
        A = _state(r, "A_central")
        m = (t > 1.0) & (t < 6.0)
        slope = np.polyfit(t[m], np.log(A[m]), 1)[0]
        assert slope == pytest.approx(-k, rel=1e-4)

    def test_steady_state_average_concentration(self):
        """Average concentration over a dosing interval at steady state is
        dose / (CL * tau) for any linear one-compartment model."""
        pk = _pk()
        tau, dose = 8.0, 1000.0
        r, _ = _run(pk, dose_mg=dose, interval_h=tau, n_doses=12, t_end=96.0)
        t = r.t
        C = _state(r, "A_central") / r.params["_Vc_val"]
        last = (t >= 88.0) & (t <= 96.0)
        c_avg = trapezoid(C[last], t[last]) / (t[last][-1] - t[last][0])
        assert c_avg == pytest.approx(dose / (pk.CL * tau), rel=2e-3)
