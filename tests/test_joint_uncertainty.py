import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ds_joint_uncertainty import (ASSUMED, BASELINE, DELTA, FACTORS, LOG_RSTAR_CAP,
                                          NAMES, log_r_star, morris_design, value, verdict)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/danger_signal/joint_uncertainty.json"


def test_morris_trajectories_move_one_factor_at_a_time_once_each():
    trajectories = morris_design(np.random.default_rng(0))
    for tr in trajectories:
        u = np.array(tr["u"])
        assert u.shape == (len(FACTORS) + 1, len(FACTORS))
        assert u.min() >= -1e-12 and u.max() <= 1 + 1e-12
        moved = []
        for a, b, step in zip(u[:-1], u[1:], tr["steps"]):
            changed = np.nonzero(np.abs(b - a) > 1e-12)[0]
            assert len(changed) == 1
            assert abs(abs(b[changed[0]] - a[changed[0]]) - DELTA) < 1e-12
            assert NAMES[changed[0]] == step["factor"]
            moved.append(step["factor"])
        assert sorted(moved) == sorted(NAMES)


def test_every_default_lies_inside_its_factor_range():
    for f in FACTORS:
        if "levels" in f:
            assert f["baseline"] in f["levels"]
        else:
            assert f["lo"] <= f["baseline"] <= f["hi"]
            assert value(f, 0.0) == pytest.approx(f["lo"])
            assert value(f, 1.0) == pytest.approx(f["hi"])
    # The levels the screen uses for the interval are the doses' real intervals.
    interval = FACTORS[NAMES.index("interval_h")]
    assert [value(interval, u) for u in (0, 1 / 3, 2 / 3, 1)] == list(interval["levels"])
    assert BASELINE["interval_h"] == 8.0


def test_verdict_and_cap():
    assert verdict(ASSUMED[0] - 1e-9) == "holds"
    assert verdict(ASSUMED[1] + 1e-9) == "reverses"
    # No finite ratio: meropenem's own sub-MIC release outweighs ceftazidime's.
    assert verdict(None) == "reverses"
    assert log_r_star(None) == LOG_RSTAR_CAP
    assert log_r_star(10.0) == pytest.approx(1.0)


@pytest.mark.skipif(not OUT.exists(), reason="joint_uncertainty.json not generated")
def test_joint_results_were_made_from_the_current_inputs():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for name, digest in d["input_sha256"].items():
        path = ROOT / "results/danger_signal" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(not OUT.exists(), reason="joint_uncertainty.json not generated")
def test_joint_results_carry_their_checks():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    num = d["numerical_check"]
    # The default scenario at the stored doses IS Table 1.
    assert num["table1_r_star_max_rel_diff_vs_kill_matched"] < 1e-9
    # Solving the doses again lands within the stored solver's tolerance.
    lo, hi = num["baseline_solved_over_stored_dose_range"]
    assert 0.995 < lo <= hi < 1.005
    assert num["max_mass_balance_rel_residual"] < 1e-10
    h = d["headline"]
    assert sum(h["verdict_counts"].values()) == h["n_attainable"]
    assert h["n_attainable"] + h["n_unattainable"] == h["n_cell_targets"]
