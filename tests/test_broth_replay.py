import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ds_broth_replay import (DATA, FLOOR, SCHEDULES, load, phase, predict,
                                     residuals, simulate_panel)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/danger_signal/broth_replay.json"
PROVENANCE = ROOT / "data/external_validation/meropenem_figure2_schedules.provenance.json"
NO_DRUG = (0.0, 1.0, 1.0, 0.0, 0.0)          # zero maximum kill rate


def test_readings_file_matches_its_provenance_and_excludes_the_unassigned_series():
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == prov["csv_sha256"]
    with DATA.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert any(r["quality"] == "unidentified_series" for r in rows)
    assert all(o["arm"] != "unidentified-diamond" for o in load())
    # A reading below the plot floor carries the floor as its upper bound only.
    for r in rows:
        if r["quality"] == "below_axis_floor":
            assert r["value"] == "" and r["lower"] == "" and float(r["upper"]) == FLOOR


def test_phases_follow_the_published_schedules():
    assert phase("A", 9.6) == "first_exposure" and phase("A", 24.0) == "drug_free"
    assert phase("B", 4.3) == "first_exposure"
    assert phase("B", 12.0) == "drug_free"          # sampled before the second dose
    assert phase("B", 16.3) == "later_exposure"
    assert phase("C", 3.3) == "first_exposure"      # sampled at the first removal
    assert [phase("C", t) for t in (8.0, 10.0, 18.0, 24.0)] == \
        ["drug_free", "later_exposure", "later_exposure", "drug_free"]


def test_without_drug_every_arm_grows_alike_toward_capacity():
    growth = (0.9, 4.2, 8.5)
    sims = simulate_panel("B", growth, NO_DRUG, [0.0, 4.0], [0.0, 6.0, 24.0])
    assert np.allclose(sims[6.0][0], sims[6.0][1])
    assert sims[0.0][0] == pytest.approx(4.2, abs=1e-9)
    assert sims[0.0][0] < sims[6.0][0] < sims[24.0][0] <= 8.5 + 1e-6


def test_drug_acts_only_inside_the_exposure_windows():
    growth = (0.9, 4.2, 8.5)
    kill = (3.0, 0.9, 2.0, 0.0, 0.0)
    t = np.linspace(0.0, 24.0, 97)
    sims = simulate_panel("C", growth, kill, [8.0], t)
    y = np.array([sims[x][0] for x in t])
    for a, b in SCHEDULES["C"]:
        inside = (t > a + 0.5) & (t < b)
        assert np.all(np.diff(y[inside]) < 0)          # killing on drug
    gap = (t > 3.2 + 0.5) & (t < 8.0)
    assert np.all(np.diff(y[gap]) > 0)                 # regrowth off drug


def test_observation_operator_counts_floor_and_interval_readings_only_when_violated():
    obs = [{"kind": "below_floor", "upper": FLOOR, "lower": None, "value": None},
           {"kind": "interval", "lower": 6.7, "upper": 7.9, "value": None},
           {"kind": "point", "value": 3.0, "lower": None, "upper": None}]
    assert list(residuals(obs, np.array([0.5, 7.0, 3.2]))) == pytest.approx([0.0, 0.0, 0.2])
    assert list(residuals(obs, np.array([1.4, 6.2, 2.9]))) == pytest.approx([0.4, -0.5, -0.1])


@pytest.mark.skipif(not OUT.exists(), reason="broth_replay.json not generated")
def test_replay_results_match_their_inputs_and_hold_their_reading():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    assert d["input_sha256"][DATA.name] == hashlib.sha256(DATA.read_bytes()).hexdigest()
    h = d["headline"]
    # Refitting improves on the adopted parameters, but no model clears the
    # reading allowance for every reading.
    assert h["rmse_by_model"]["refit"] < h["rmse_by_model"]["adopted"]
    assert min(h["beyond_allowance_by_model"].values()) > 0
    # The predictions the stored file carries are the ones the model makes now.
    obs = load()
    g = {p: (v["rate_per_h"], v["log10_initial"], v["log10_capacity"]) for p, v in h["growth"].items()}
    kill = d["full_fits"]["refit"]["kill"]
    pred = predict(obs, g, tuple(kill[k] for k in ("k_kill_max", "kill_C50", "kill_hill", "k_tol", "phi")))
    stored = [p["predicted"] for p in d["full_fits"]["refit"]["predictions"]]
    assert np.allclose(pred, stored, rtol=0, atol=1e-8)
