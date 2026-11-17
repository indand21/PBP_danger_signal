"""The exact endotoxin integrals, the analytical PK they rest on, and the
refined release sampling (plan item 8; src/analysis/exact_release.py)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.analysis.exact_release import (
    linear_source_ds, mass_balance_residual, refined_times, refined_trajectory, time_above,
)
from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import DosingRegimen, TwoCompartmentPKModel
from src.core.simulation import effect_site_concentration, run_simulation

K = 13.9


def _src(t):
    return 50.0 * (np.sin(2 * np.pi * t / 8.0) > 0) * (1 + np.cos(t)) + 2.0


def test_closed_form_matches_an_ode_solve_of_the_same_linear_source():
    t = np.sort(np.concatenate([np.linspace(0, 24, 200), [0.013, 3.3333]]))
    s = _src(t)
    D, area, released = linear_source_ds(t, s, K)

    def rhs(tt, y):
        si = np.interp(tt, t, s)
        return [si - K * y[0], y[0]]

    sol = solve_ivp(rhs, (0, 24), [0.0, 0.0], t_eval=t, rtol=1e-11, atol=1e-13,
                    max_step=float(np.min(np.diff(t))), method="LSODA")
    assert np.allclose(D, sol.y[0], rtol=1e-7, atol=1e-9)
    assert np.allclose(area, sol.y[1], rtol=1e-7, atol=1e-9)
    assert mass_balance_residual(D, area, released, K) < 1e-13


def test_small_steps_use_a_continuous_series():
    """Steps with k h < 1e-3 switch to Taylor series; both sides of the switch
    must give the same answer as a step split in two."""
    for h in (9.9e-5, 1.01e-4):
        t = np.array([0.0, h])
        s = np.array([3.0, 7.0])
        D1, a1, _ = linear_source_ds(t, s, K, D0=1.0)
        t2 = np.array([0.0, h / 2, h])
        D2, a2, _ = linear_source_ds(t2, np.interp(t2, t, s), K, D0=1.0)
        assert D1[-1] == pytest.approx(D2[-1], rel=1e-12)
        assert a1[-1] == pytest.approx(a2[-1], rel=1e-10)


def test_auc_is_linear_in_the_release():
    t = np.linspace(0, 12, 300)
    a, b = _src(t), np.exp(-t)
    _, A, _ = linear_source_ds(t, a, K)
    _, B, _ = linear_source_ds(t, b, K)
    _, AB, _ = linear_source_ds(t, 2.5 * a + 0.4 * b, K)
    assert AB[-1] == pytest.approx(2.5 * A[-1] + 0.4 * B[-1], rel=1e-13)


def _sim(dose=1000.0, inf_min=30, dense=True, agent="meropenem"):
    p = get_drug_pk_parameters(agent)
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka, Kp=p.Kp,
                                effect_site_model=True)
    reg = DosingRegimen(dose_mg=dose, interval_hours=8.0, start_time=0, n_doses=3,
                        infusion_duration_min=inf_min)
    pd = create_ode_system(get_default_parameters())
    r = run_simulation(pkm, reg, pd, dict(B_rep=1e7, B_pers=1e4, B_SCV=0, N_eff=1e7, Damage=0,
                                          IL6=10, TNF=5, PAMP=0, D_host=0),
                       t_span=(0, 24.0), drug_class="cidal", weight_kg=70.0,
                       fraction_unbound=0.98, dense_output=dense)
    return r, pd, reg, p


def test_vectorized_pk_equals_the_concentration_that_drove_killing():
    for inf_min in (30, 0):
        r, _, reg, p = _sim(inf_min=inf_min)
        grid = r.get_effect_site_concentration(weight_kg=70.0)
        vec = effect_site_concentration(r.t, reg, p.CL, p.Vc * 70.0, p.Kp, p.Ka)
        assert np.allclose(vec, grid, rtol=1e-13, atol=0.0)
        assert np.allclose(r.free_concentration_at(r.t), 0.98 * grid, rtol=1e-13, atol=0.0)


def test_keeping_the_interpolant_does_not_change_the_solve():
    a, *_ = _sim(dense=False)
    b, *_ = _sim(dense=True)
    assert np.array_equal(a.t, b.t) and np.array_equal(a.y, b.y)


def test_refined_release_converges_at_second_order():
    r, pd, *_ = _sim()
    vals = []
    for m in (2, 4, 8, 16):
        f = refined_trajectory(r, pd, m)
        assert len(f["t"]) == (len(r.t) - 1) * m + 1
        assert np.array_equal(f["t"][::m], r.t)
        assert np.array_equal(f["B_rep"][::m], r.y[:, r.state_names.index("B_rep")])
        vals.append(np.trapezoid(f["kill"] * f["B_rep"], f["t"]))
    diffs = np.abs(np.diff(vals))
    assert all(3.0 < c / f < 5.0 for c, f in zip(diffs, diffs[1:])), vals


def test_refined_times_keep_the_grid():
    t = np.array([0.0, 0.1, 0.25, 1.0])
    rt = refined_times(t, 4)
    assert np.allclose(rt[::4], t) and np.all(np.diff(rt) > 0)


def test_time_above_matches_the_analytical_answer():
    k, c0, mic = 0.7, 10.0, 1.3

    def conc(t):
        t = np.asarray(t, float)
        return np.where(t >= 2.0, c0 * np.exp(-k * (t - 2.0)), 0.0)

    hours = time_above(conc, mic, 0.0, 12.0, breakpoints=[2.0])
    assert hours == pytest.approx(np.log(c0 / mic) / k, abs=1e-10)


def test_concentration_exposure_is_exact_for_an_infusion():
    """One 0.5-h infusion, one compartment: closed-form AUC to 6 h and peak."""
    from src.analysis.exact_release import concentration_exposure
    k, R0, Vc, T, tinf = 0.6, 2000.0, 15.0, 6.0, 0.5

    def conc(t):
        t = np.asarray(t, float)
        rise = R0 / (k * Vc) * (1 - np.exp(-k * np.minimum(t, tinf)))
        return np.where(t <= tinf, rise, rise * np.exp(-k * (t - tinf)))

    c_end = R0 / (k * Vc) * (1 - np.exp(-k * tinf))
    auc_exact = (R0 / (k * Vc) * (tinf - (1 - np.exp(-k * tinf)) / k)
                 + c_end * (1 - np.exp(-k * (T - tinf))) / k)
    auc, cmax = concentration_exposure(conc, 0.0, T, breakpoints=[tinf])
    assert auc == pytest.approx(auc_exact, rel=1e-12)
    assert cmax == pytest.approx(c_end, rel=1e-14)


def test_exact_ft_agrees_with_a_very_fine_grid():
    r, _, reg, _ = _sim()
    from src.analysis.exact_release import dosing_breakpoints
    mic = 2.0
    exact = time_above(r.free_concentration_at, mic, 0.0, 24.0, dosing_breakpoints(reg, 24.0))
    tf = np.linspace(0.0, 24.0, 2_000_001)
    fine = float(np.sum(r.free_concentration_at(tf)[:-1] >= mic) * (tf[1] - tf[0]))
    assert exact == pytest.approx(fine, abs=2e-5)
