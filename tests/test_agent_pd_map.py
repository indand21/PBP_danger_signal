import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ds_agent_pd_map import (ASSUMED, broth_net_change, tsuji_data,
                                     tsuji_digest, verdict)
from scripts.ds_multiarm_refit import encode, predict

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "results/danger_signal/agent_pd_map.json"


def test_broth_model_is_the_refit_model_at_the_default_switching_rate():
    # The Tsuji check must use the same broth structure the PD refit used.
    pd = {"k_kill_max": 1.7, "kill_C50": 0.8, "kill_hill": 3.0, "k_pers": 0.01}
    growth, log10_b0 = 0.9, 5.0
    x = encode([pd["k_kill_max"], pd["kill_C50"], pd["kill_hill"], growth, log10_b0])
    for conc in (0.0, 0.1, 1.0, 10.0):
        want = predict(x, np.array([conc]), np.array([6.0]))[0] - log10_b0
        assert broth_net_change(pd, conc, growth) == pytest.approx(want, abs=1e-6)


def test_switching_into_persisters_slows_net_killing():
    base = {"k_kill_max": 2.0, "kill_C50": 1.0, "kill_hill": 4.0, "k_pers": 0.01}
    more = dict(base, k_pers=0.16)
    assert broth_net_change(more, 10.0, 1.0) > broth_net_change(base, 10.0, 1.0)


def test_verdict_reads_r_star_against_the_assumed_range():
    assert verdict(ASSUMED[0] - 1e-9) == "holds"
    assert verdict(0.5 * (ASSUMED[0] + ASSUMED[1])) == "depends_on_yield"
    assert verdict(ASSUMED[1] + 1e-9) == "reverses"


@pytest.mark.skipif(not MAP.exists(), reason="agent_pd_map.json not generated")
def test_map_was_made_from_the_current_inputs():
    d = json.loads(MAP.read_text(encoding="utf-8"))
    for name, digest in d["input_sha256"].items():
        path = ROOT / "results/danger_signal" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    # Only the Tsuji values the analysis reads, so unrelated registry edits
    # do not mark the map stale.
    assert tsuji_digest(tsuji_data()) == d["tsuji_data_sha256"]


@pytest.mark.skipif(not MAP.exists(), reason="agent_pd_map.json not generated")
def test_map_carries_its_checks():
    d = json.loads(MAP.read_text(encoding="utf-8"))
    num = d["numerical_check"]
    # The meropenem refit set that carries the adopted kill parameters must
    # reproduce the main map cell for cell.
    assert num["max_rel_diff_envelope_twin_vs_main_map"] < 1e-9
    # Both maps contain the shared-PD cell; it must be the same simulation.
    assert num["max_rel_diff_mirror_shared_cell_vs_main_shared_cell"] < 1e-9
    # Exact integration against ds_joint_robustness's trapezoid values.
    assert num["max_rel_diff_shared_cell_r_star_vs_joint_robustness_trapezoid"] < 0.01
    # Tsuji's three single-time points per agent do not identify the PD.
    assert d["tsuji_check"]["identifiable"] is False
    # The exposure reference is fixed, never re-derived per scenario.
    assert d["design"]["mic_reference_mgL"] == 1.0
