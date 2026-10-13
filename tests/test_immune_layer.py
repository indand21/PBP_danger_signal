"""The immune effector layer after the R10 repair.

The defect these guard against: recruitment used to be

    recruit = k_prod * B/(EC50_immune + B)      # k_prod = 0.5, units 1/h

in an equation whose other term is k_deg_immune * N_eff in cells/mL/h. At
N_eff_0 = 1e7 the loss is 5e5 cells/mL/h and recruitment at most 0.5, so the
pool could only decay and the largest level recruitment could sustain was
10 cells/mL. Nothing in the model complained, because every consequence was
a plausible-looking number.

The repair makes the pool homeostatic and burden-recruited:

    dN/dt = k_deg_immune * (target(B) - N)
    target(B) = N_eff_0 * (1 + (recruit_fold_max - 1) * B/(EC50 + B))

The tests below pin the three properties that make it a response rather than
a pool -- the right units, the right fixed points, monotonicity in burden --
and the two invariants the rest of the repository depends on: an immune-free
run stays immune-free, and the effective MIC is a drug property.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.core.parameters import (get_default_parameters,
                                 immune_free_parameters)
from src.core.pd_model import BacterialPopulationODE


def _state(B_rep=1e6, N_eff=1e7):
    # B_rep, B_pers, B_SCV, N_eff, Damage, IL6, TNF, PAMP, D_host
    return np.array([B_rep, 1e3, 0.0, N_eff, 0.0, 10.0, 5.0, 0.0, 0.0])


def _dN(model, y):
    return model.rhs(0.0, y, C_effect=0.0, drug_class="cidal")[3]


class TestScale:
    """Recruitment must be on the effector state's own scale."""

    def test_production_balances_loss_at_baseline(self):
        """With no infection the pool is stationary at N_eff_0.

        This is the property the old term could not have: its production was
        six orders of magnitude below the loss it had to balance.
        """
        p = get_default_parameters()
        model = BacterialPopulationODE(p)
        # B = 0 exactly: the rhs floors B_rep at 1e-6, far below EC50_immune,
        # so the recruited increment is negligible and the target is N_eff_0.
        y = _state(B_rep=0.0, N_eff=p["immune"].N_eff_0)
        y[1] = 0.0
        assert abs(_dN(model, y)) < 1e-3 * p["immune"].k_deg_immune * p["immune"].N_eff_0

    def test_production_is_in_cells_per_ml_per_hour(self):
        """Production must scale with N_eff_0, not sit at a bare rate constant.

        Doubling the baseline must double the production an empty pool sees.
        A term in 1/h would not move at all.
        """
        p = get_default_parameters()
        empty = _state(B_rep=0.0, N_eff=0.0)
        empty[1] = 0.0
        base = _dN(BacterialPopulationODE(p), empty)
        p2 = get_default_parameters()
        p2["immune"].N_eff_0 *= 2.0
        doubled = _dN(BacterialPopulationODE(p2), empty)
        assert base > 0
        assert doubled == pytest.approx(2.0 * base, rel=1e-9)


class TestFixedPoints:
    def test_uninfected_host_settles_at_baseline(self):
        """A depleted pool in an uninfected host recovers to N_eff_0.

        The burden is held at zero rather than integrated: the rhs floors
        B_rep at 1e-6 and lets it replicate, so a 400 h run of the full
        system would grow an infection and recruit against it.
        """
        p = get_default_parameters()
        model = BacterialPopulationODE(p)

        def rhs(t, N):
            y = _state(B_rep=0.0, N_eff=float(N[0]))
            y[1] = 0.0
            return [_dN(model, y)]
        sol = solve_ivp(rhs, (0.0, 400.0), [0.3 * p["immune"].N_eff_0],
                        method="LSODA", rtol=1e-9, atol=1e-9)
        assert sol.y[0, -1] == pytest.approx(p["immune"].N_eff_0, rel=1e-3)

    def test_saturating_burden_recruits_to_the_plateau(self):
        """At B >> EC50 the pool settles at recruit_fold_max x N_eff_0."""
        p = get_default_parameters()
        imm = p["immune"]
        model = BacterialPopulationODE(p)
        B = 1e4 * imm.EC50_immune
        # Hold the burden fixed so only the effector equation evolves.
        def rhs(t, N):
            y = _state(B_rep=B, N_eff=float(N[0]))
            return [_dN(model, y)]
        sol = solve_ivp(rhs, (0.0, 400.0), [imm.N_eff_0], method="LSODA",
                        rtol=1e-9, atol=1e-9)
        assert sol.y[0, -1] == pytest.approx(
            imm.recruit_fold_max * imm.N_eff_0, rel=1e-3)

    def test_response_is_monotone_in_burden(self):
        p = get_default_parameters()
        model = BacterialPopulationODE(p)
        # At a fixed effector level, a larger burden must recruit harder.
        rates = [_dN(model, _state(B_rep=B, N_eff=p["immune"].N_eff_0))
                 for B in (1e3, 1e5, 1e7, 1e9)]
        assert all(b > a for a, b in zip(rates, rates[1:]))

    def test_recruitment_is_bounded_by_the_plateau(self):
        """An unbounded response would let any burden produce any killing."""
        p = get_default_parameters()
        imm = p["immune"]
        model = BacterialPopulationODE(p)
        y = _state(B_rep=1e12, N_eff=imm.recruit_fold_max * imm.N_eff_0)
        assert _dN(model, y) <= 0.0


class TestImmuneFreeInvariant:
    """An immune-free condition must stay immune-free.

    Both datasets the bacterial layer is calibrated against are immune-free
    by construction -- a broth time-kill and a *neutropenic* thigh model -- so
    if homeostasis could refill a zeroed pool, the repair would silently
    re-immunise the calibration.
    """

    def test_zero_pool_stays_zero(self):
        p = immune_free_parameters()
        model = BacterialPopulationODE(p)
        y0 = _state(B_rep=1e6, N_eff=0.0)
        sol = solve_ivp(lambda t, y: model.rhs(t, y, 0.0, "cidal"),
                        (0.0, 48.0), y0, method="LSODA",
                        rtol=1e-8, atol=1e-10)
        assert np.all(np.abs(sol.y[3]) < 1e-9)

    def test_initial_condition_alone_is_not_enough(self):
        """Documents why callers must set the set-point, not just y0.

        With the default set-point a zeroed pool is refilled, which is the
        trap every immune-free call site in this repository now has to avoid.
        """
        model = BacterialPopulationODE(get_default_parameters())
        y0 = _state(B_rep=1e6, N_eff=0.0)
        sol = solve_ivp(lambda t, y: model.rhs(t, y, 0.0, "cidal"),
                        (0.0, 48.0), y0, method="LSODA",
                        rtol=1e-8, atol=1e-10)
        assert sol.y[3, -1] > 1e6

    def test_helper_does_not_mutate_the_caller(self):
        p = get_default_parameters()
        before = p["immune"].N_eff_0
        immune_free_parameters(p)
        assert p["immune"].N_eff_0 == before


class TestImmuneStatusSweepsStillSweep:
    """Modules that vary immune status must carry it in the set-point.

    Three modules express a host's immune status as the N_eff initial
    condition alone. Homeostasis relaxes any such pool back to the set-point
    with a time constant of 1/k_deg_immune = 20 h, so left unfixed every
    point of these sweeps would converge on the same immunocompetent
    baseline and the immune axis would quietly collapse -- the runs would
    still succeed and still look plausible.
    """

    def test_strategy_margin_responds_to_immune_status(self):
        from src.analysis.strategy_margin import run_one
        lo = run_one(1e3, 0.01, 0.03, "cidal", t_span=(0, 48))
        hi = run_one(1e8, 0.01, 0.03, "cidal", t_span=(0, 48))
        n_lo = lo.y[:, lo.state_names.index("N_eff")]
        n_hi = hi.y[:, hi.state_names.index("N_eff")]
        # Each must stay on its own scale rather than meeting in the middle.
        assert n_lo[-1] < 1e5
        assert n_hi[-1] > 1e7
        _, b_lo = lo.get_bacterial_burden()
        _, b_hi = hi.get_bacterial_burden()
        assert b_hi[-1] < b_lo[-1]


class TestEffectiveMicIsADrugProperty:
    def test_effective_mic_does_not_depend_on_the_immune_variant(self):
        """The effective MIC is resolved immune-free, so immune parameters
        must not move it. Before the repair this held only because the
        effector pool was numerically irrelevant."""
        from src.analysis import external_validation as ev
        scenario = {"initial_burden": 1e6}
        base = ev.effective_mic(drug_class="cidal", scenario=scenario)
        assert np.isfinite(base) and base > 0
        # effective_mic builds its own immune-free parameters internally, so
        # a different recruited plateau in the defaults must not reach it.
        import src.core.parameters as params_mod
        original = params_mod.ImmuneParameters.recruit_fold_max
        try:
            params_mod.ImmuneParameters.recruit_fold_max = 5.0
            again = ev.effective_mic(drug_class="cidal", scenario=scenario)
        finally:
            params_mod.ImmuneParameters.recruit_fold_max = original
        assert again == pytest.approx(base, rel=1e-12)
