import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import trapezoid

from scripts.ds_kill_matched import (EXTREMUM_MIN_PROMINENCE, IMMEDIATE, MATCH_RTOL,
                                     SCAN_MG, Arms, decompose, scan, solve_equal_killing)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/danger_signal/kill_matched.json"


# --------------------------------------------------------------------------
# the factorization, on synthetic release inputs
# --------------------------------------------------------------------------
def stub_p4():
    q = SimpleNamespace(DS_baseline=0.0, k_ds_clear=13.9, V_ds_L=3.0,
                        translocation_fraction=0.01)
    return SimpleNamespace(P_REF=q, V_FOCUS=10.0, Y_FILAMENT_CAZ=1.5,
                           AGENT_YIELD={"ceftazidime": 50.0, "meropenem": 10.0})


def stub_arm(arms, lys, fil, t, consistent=True):
    c = arms.p4.V_FOCUS * arms.to_conc
    killed = trapezoid(lys, t) / c
    return {"_t": t, "_lys": lys, "_fil": fil, "_auc": {},
            "cum_killed_cfu_per_mL": killed if consistent else 1.3 * killed}


def test_factors_multiply_to_the_ratio_and_timing_reads_late_release():
    arms = Arms(stub_p4())
    t = np.linspace(0.0, 48.0, 961)
    early = stub_arm(arms, np.exp(-t), 0.3 * np.exp(-((t - 2.0) / 0.5) ** 2), t)
    late = stub_arm(arms, np.exp(-((t - 47.0) / 1.0) ** 2), np.zeros_like(t), t)
    delayed = {"label": "n1_mtt7.63", "mtt_h": 7.63, "n": 1, "delay_filament": False}
    for cfg in (IMMEDIATE, delayed):
        for frac in (0.0, 1.0):
            d = decompose(arms, early, late, cfg, 3.5, frac)
            assert d["identity_rel_residual"] < 1e-12
            assert d["factors"]["yield"] == 3.5
    # Release late in the window is partly outside the 48-h AUC, so the
    # late arm's tau is lower and the timing factor favours the early arm.
    d = decompose(arms, early, late, IMMEDIATE, 3.5)
    assert d["tau"]["meropenem"] < d["tau"]["ceftazidime"] <= 1.0
    assert d["factors"]["timing"] > 1.0
    # A delay holds more of the late release in transit at 48 h.
    assert (decompose(arms, early, late, delayed, 3.5)["factors"]["timing"]
            > d["factors"]["timing"])


def test_identity_fails_when_killing_and_release_disagree():
    # The check has teeth: break released = c K on one arm and it shows.
    arms = Arms(stub_p4())
    t = np.linspace(0.0, 48.0, 481)
    a = stub_arm(arms, np.exp(-t), np.zeros_like(t), t)
    b = stub_arm(arms, np.exp(-t / 2), np.zeros_like(t), t, consistent=False)
    assert decompose(arms, a, b, IMMEDIATE, 3.5)["identity_rel_residual"] > 0.1


# --------------------------------------------------------------------------
# solving for equal killing, on synthetic killing curves
# --------------------------------------------------------------------------
class CurveArms:
    def __init__(self, fn):
        self.fn = fn

    def killing(self, agent, dose):
        return float(self.fn(dose))

    def sim(self, agent, dose):
        return {"fT": 0.0, "net_log10": 0.0}


def test_both_branches_of_a_unimodal_curve_are_found():
    # Rise, peak, fall to a plateau: the shape the model's killing has.
    fn = lambda d: 1.3 + 80.0 * np.exp(-((np.log(d) - np.log(40.0)) / 1.0) ** 2)
    arms = CurveArms(fn)
    sc = scan(arms, "x")
    assert len(sc["extrema"]) == 1
    roots, _ = solve_equal_killing(arms, "x", sc, 2.0)
    assert len(roots) == 2
    assert roots[0] < 40.0 < roots[1]
    assert all(abs(fn(r) / 2.0 - 1.0) < 1e-9 for r in roots)
    # Below the plateau only the rising branch reaches the target.
    assert len(solve_equal_killing(arms, "x", sc, 1.0)[0]) == 0


def test_a_pair_of_roots_between_scan_points_is_found_by_the_extremum():
    d = np.array(SCAN_MG)
    i = 80
    centre = np.log(d[i]) + 0.2 * (np.log(d[i + 1]) - np.log(d[i]))
    width = 0.6 * (np.log(d[i + 1]) - np.log(d[i]))
    fn = lambda x: 1.0 + np.exp(-((np.log(x) - centre) / width) ** 2)
    arms = CurveArms(fn)
    sc = scan(arms, "x")
    peak = max(e["cum_killed_cfu_per_mL"] for e in sc["extrema"])
    target = 0.5 * (fn(d[i]) + peak)        # above every scan point, below the peak
    assert max(sc["cum_killed_cfu_per_mL"]) < target < peak
    roots, added = solve_equal_killing(arms, "x", sc, target)
    assert added == 2 and len(roots) == 2


def test_jitter_below_the_prominence_floor_is_not_shape():
    rng = np.random.default_rng(0)
    noise = {float(x): 1.0 + 1e-5 * rng.standard_normal() for x in SCAN_MG}
    fn = lambda d: (1.3 + 80.0 * np.exp(-((np.log(d) - np.log(40.0)) / 1.0) ** 2)) * noise.get(float(d), 1.0)
    sc = scan(CurveArms(fn), "x")
    assert len(sc["extrema"]) == 1
    assert sc["jitter_extrema"]["max_prominence_rel"] < EXTREMUM_MIN_PROMINENCE


# --------------------------------------------------------------------------
# the stored results
# --------------------------------------------------------------------------
@pytest.mark.skipif(not OUT.exists(), reason="kill_matched.json not generated")
def test_results_were_made_from_the_current_inputs():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for name, digest in d["input_sha256"].items():
        path = ROOT / "results/danger_signal" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(not OUT.exists(), reason="kill_matched.json not generated")
def test_results_carry_their_checks():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    num = d["numerical_check"]
    assert num["max_identity_rel_residual"] < 1e-12
    assert num["max_mass_balance_rel_residual"] < 1e-10
    assert num["max_rel_diff_zero_delay_vs_exact_window_aucs"] < 1e-10
    assert num["max_rel_diff_table1_ratio_vs_stored_trapezoid"] < 5e-3
    assert num["max_rel_killing_mismatch_at_solutions"] < MATCH_RTOL
    assert num["abs_diff_pct_table1_delay_change_vs_release_delay"] < 1e-9
    assert num["n_scan_extrema"] == {"ceftazidime": 1, "meropenem": 1}
    # Each target and direction: one controlled and one failing solution.
    for block in d["kill_matched"].values():
        for m in block.values():
            branches = sorted(s["branch"] for s in m["solutions"])
            assert branches == ["above_peak", "below_peak"]
            for s in m["solutions"]:
                assert s["arm"]["controlled"] == (s["branch"] == "above_peak")
