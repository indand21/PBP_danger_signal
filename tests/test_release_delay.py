import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts.ds_endpoint_windows import exact_window_aucs
from scripts.ds_release_delay import delayed_endotoxin

ROOT = Path(__file__).resolve().parents[1]
DELAY = ROOT / "results/danger_signal/release_delay.json"
PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"

T = np.array([0.0, 0.3, 0.5, 1.7, 4.0, 9.0, 12.0])
LYS = np.array([0.0, 8.0, 2.0, 5.0, 0.5, 3.0, 0.0])
FIL = np.array([0.0, 0.0, 1.0, 1.5, 0.2, 0.0, 0.0])


def test_one_compartment_matches_closed_form_for_constant_release():
    # A' = a - c A, D' = c A - k D, zero start:
    # integral_0^T D = (a/k) [T - (k (1 - e^{-cT})/c - c (1 - e^{-kT})/k) / (k - c)]
    a, k, mtt = 3.0, 13.9, 7.63
    c = 1.0 / mtt
    t = np.linspace(0.0, 48.0, 5)          # coarse on purpose: the method is exact
    run = delayed_endotoxin(t, np.full_like(t, a), np.zeros_like(t), k,
                            mtt=mtt, n=1, windows=(1.0, 24.0, 48.0))
    for Tw in (1.0, 24.0, 48.0):
        want = a / k * (Tw - (k * (1 - np.exp(-c * Tw)) / c
                              - c * (1 - np.exp(-k * Tw)) / k) / (k - c))
        assert run["auc"][Tw] == pytest.approx(want, rel=1e-10)


def test_zero_delay_is_the_existing_exact_integrator():
    k = 13.9
    run = delayed_endotoxin(T, LYS, FIL, k, mtt=0.0, windows=(1.0, 4.0, 12.0))
    ref = exact_window_aucs(T, LYS + FIL, k, (1.0, 4.0, 12.0))
    for w in (1.0, 4.0, 12.0):
        assert run["auc"][w] == pytest.approx(ref[w], rel=1e-10)


@pytest.mark.parametrize("n,delay_filament", [(1, False), (3, False), (1, True)])
def test_chain_matches_an_independent_ode_solution(n, delay_filament):
    k, mtt = 2.5, 1.7
    ktr = n / mtt

    def rhs(tt, y):
        ul, uf = np.interp(tt, T, LYS), np.interp(tt, T, FIL)
        A, D = y[:n], y[n]
        dA = np.empty(n)
        dA[0] = ul + (uf if delay_filament else 0.0) - ktr * A[0]
        dA[1:] = ktr * (A[:-1] - A[1:])
        dD = ktr * A[-1] + (0.0 if delay_filament else uf) - k * D
        return [*dA, dD, D]

    sol = solve_ivp(rhs, (0.0, 12.0), np.zeros(n + 2), t_eval=[1.0, 4.0, 12.0],
                    rtol=1e-11, atol=1e-13, max_step=0.01)
    run = delayed_endotoxin(T, LYS, FIL, k, mtt=mtt, n=n,
                            delay_filament=delay_filament, windows=(1.0, 4.0, 12.0))
    for j, w in enumerate((1.0, 4.0, 12.0)):
        assert run["auc"][w] == pytest.approx(sol.y[n + 1, j], rel=1e-7)


def test_mass_balance_closes_and_total_exposure_ignores_the_delay():
    k = 13.9
    for mtt in (0.0, 0.5, 7.63):
        run = delayed_endotoxin(T, LYS, FIL, k, mtt=mtt, n=1, windows=(12.0,))
        assert run["mass_balance_rel_residual"] < 1e-10
        # Exposure to infinity is released / k whatever the delay; the window
        # loses exactly what is still in transit (and in plasma) at its end.
        total = run["auc"][12.0] + (run["D_end"] + run["in_transit_end"]) / k
        assert total == pytest.approx(run["released"] / k, rel=1e-10)


@pytest.mark.skipif(not DELAY.exists(), reason="release_delay.json not generated")
def test_delay_results_were_made_from_the_current_phase4_results():
    d = json.loads(DELAY.read_text(encoding="utf-8"))
    assert d["input_sha256"][PHASE4.name] == hashlib.sha256(PHASE4.read_bytes()).hexdigest()


@pytest.mark.skipif(not DELAY.exists(), reason="release_delay.json not generated")
def test_delay_results_reproduce_phase4_and_carry_their_checks():
    d = json.loads(DELAY.read_text(encoding="utf-8"))
    num = d["numerical_check"]
    assert num["max_rel_diff_reproduced_vs_stored_48h_AUC"] <= num["reproduction_tolerance"]
    assert num["max_rel_diff_zero_delay_vs_independent_exact"] < 1e-9
    assert num["max_mass_balance_rel_residual"] < 1e-9
    assert d["delay_structure"]["thorsted_mtt_h"] == 7.63
    assert d["delay_structure"]["filament_term"].startswith("kept immediate")
    # What the paper claims survives the delay: the 48-hour endpoint shows no
    # trade-off under any delay, and the matched-exposure ratios barely move.
    every = d["headline"]["over_every_delay"]
    assert every["divergent_48h_steps_max"] == 0
    assert d["headline"]["table1_max_abs_rel_change_pct_any_delay"] < 1.0
    # And what it says does not: with any delay no step raises the peak.
    assert every["divergent_peak_steps_max"] == 0
