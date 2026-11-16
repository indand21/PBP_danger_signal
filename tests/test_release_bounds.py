import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.ds_release_bounds import JACK, YAM, bounds, ratio, rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/danger_signal/release_bounds.json"
PROV = ROOT / "data/danger_signal/yamaguchi1999_pao1_release.provenance.json"


def lps(data, agent, x, t):
    return next(float(r["value"]) for r in data if r["agent"] == agent and float(r["concentration_xMIC"]) == x
                and float(r["time_h"]) == t and r["series"] == "free_lps_ng_mL")


def test_digitized_yamaguchi_matches_the_values_the_paper_states():
    data = rows(YAM)
    assert lps(data, "ceftazidime", 1.0, 4.0) == pytest.approx(493, rel=0.02)
    assert lps(data, "piperacillin", 1.0, 4.0) == pytest.approx(515, rel=0.02)
    mean = sum(lps(data, "isepamicin", 1.0, t) for t in (2.0, 4.0, 6.0)) / 3
    assert abs(mean - 28) <= 2
    assert json.loads(PROV.read_text(encoding="utf-8"))["csv_sha256"] == hashlib.sha256(YAM.read_bytes()).hexdigest()


def test_jackson_transcription_is_complete_and_starts_every_arm_at_time_zero():
    data = rows(JACK)
    assert {r["table"] for r in data} == {"1", "2", "4", "5"}
    for tab in ("1", "2", "4", "5"):
        for agent in ("control", "imipenem", "ceftazidime"):
            assert any(r["table"] == tab and r["agent"] == agent and float(r["time_h"]) == 0
                       and r["series"] == "log10_cfu_mL" and r["value"] for r in data)


def test_bounds_bracket_the_central_reading_and_the_ratio_bound_is_conservative():
    a = bounds(500.0, 1e6, 1e3, 1.4, 0.0, 25.0)
    b = bounds(25.0, 1e6, 1e2, 1.4, 0.0, 25.0)
    assert a["killed_lo"] <= a["killed_hi"]
    assert a["per_killed_lo"] <= a["per_killed_central"] <= a["per_killed_hi"]
    assert ratio(a, b, "lower") <= ratio(a, b, "central")
    # No net loss of cells: the killed biomass may be zero, so per-killed
    # release has no finite upper bound.
    grown = bounds(3000.0, 1e6, 2e6, 80.0, 0.0, 25.0)
    assert grown["killed_lo"] == 0 and grown["per_killed_hi"] is None


@pytest.mark.skipif(not OUT.exists(), reason="release_bounds.json not generated")
def test_release_bounds_results_match_inputs_and_are_ordered():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for name, digest in d["input_sha256"].items():
        assert hashlib.sha256((ROOT / "data/danger_signal" / name).read_bytes()).hexdigest() == digest
    for study in ("yamaguchi", "jackson"):
        for v in d["ratios"][study].values():
            if v["lower_bound"] is not None and v["central"] is not None:
                assert v["lower_bound"] <= v["central"] + 1e-12
