import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts.ds_endpoint_windows import exact_window_aucs, window_auc

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "results/danger_signal/endpoint_windows.json"
PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"


def test_exact_window_auc_matches_closed_form_for_constant_release():
    # D' = a - k D, D(0) = 0  =>  integral_0^T D = (a/k) (T - (1 - e^{-kT})/k)
    a, k = 3.0, 2.5
    t = np.linspace(0.0, 10.0, 7)          # coarse on purpose: the method is exact
    got = exact_window_aucs(t, np.full_like(t, a), k, (1.0, 2.5, 10.0))
    for T in (1.0, 2.5, 10.0):              # 1.0 and 2.5 are not grid points
        want = a / k * (T - (1 - np.exp(-k * T)) / k)
        assert got[T] == pytest.approx(want, rel=1e-12)


def test_exact_window_auc_is_exact_for_piecewise_linear_release():
    t = np.array([0.0, 0.3, 0.5, 1.7, 4.0])
    src = np.array([0.0, 8.0, 2.0, 5.0, 0.0])
    k = 1.3

    def rhs(tt, y):
        return [np.interp(tt, t, src) - k * y[0], y[0]]

    sol = solve_ivp(rhs, (0, 4.0), [0.0, 0.0], t_eval=[1.0, 4.0],
                    rtol=1e-11, atol=1e-13, max_step=0.01)
    got = exact_window_aucs(t, src, k, (1.0, 4.0))
    assert got[1.0] == pytest.approx(sol.y[1, 0], rel=1e-7)
    assert got[4.0] == pytest.approx(sol.y[1, 1], rel=1e-7)


def test_window_auc_interpolates_at_a_window_end_off_the_grid():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 2.0, 2.0])
    assert window_auc(t, y, 1.5) == pytest.approx(1.0 + 2.0 * 0.5)
    assert window_auc(t, y, 2.0) == pytest.approx(3.0)


@pytest.mark.skipif(not WINDOWS.exists(), reason="endpoint_windows.json not generated")
def test_window_results_were_made_from_the_current_phase4_results():
    # If phase 4 is rerun, the window re-reading is stale until regenerated.
    w = json.loads(WINDOWS.read_text(encoding="utf-8"))
    current = hashlib.sha256(PHASE4.read_bytes()).hexdigest()
    assert w["input_sha256"][PHASE4.name] == current


@pytest.mark.skipif(not WINDOWS.exists(), reason="endpoint_windows.json not generated")
def test_window_results_reproduce_phase4_and_carry_their_checks():
    w = json.loads(WINDOWS.read_text(encoding="utf-8"))
    num = w["numerical_check"]
    assert num["max_rel_diff_reproduced_vs_stored_48h_AUC"] <= num["reproduction_tolerance"]
    assert num["step_directions_agree_with_exact"] is True
    # The 48-hour endpoint is the paper's; it must still show no trade-off.
    assert w["divergent_steps_by_endpoint"]["0-48h"] == 0
    # The susceptibility sweep is a replicate: spreads at solver tolerance.
    assert w["susceptibility_invariance"]["max_spread"]["AUC_DS"] < 0.02
