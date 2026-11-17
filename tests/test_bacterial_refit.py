"""The bacterial-refit values the supplement (S2.2) quotes, held to their claims."""
import json
from pathlib import Path

import pytest

from src.core.parameters import get_default_parameters

OUT = Path(__file__).resolve().parents[1] / "results/danger_signal/bacterial_refit.json"
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="bacterial_refit.json not generated")


def _load():
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_the_anchored_c50_puts_stasis_on_the_measured_mic():
    """The adopted C50 is the closed-form anchor, rounded to three decimals:
    its stasis concentration must land on the isolate's MIC to that rounding."""
    q = _load()["quoted_in_text"]["anchor_check"]
    bact = get_default_parameters()["bacteria"]
    assert q["adopted"]["kill_C50"] == pytest.approx(bact.kill_C50)
    assert q["adopted"]["k_kill_max"] == pytest.approx(bact.k_kill_max)
    assert q["stasis_concentration_mgL"] == pytest.approx(q["adopted"]["kill_C50"] * q["stasis_over_C50"], rel=1e-12)
    assert q["stasis_concentration_mgL"] == pytest.approx(q["measured_mic_mgL"], rel=1e-3)
    # Pinning C50 to the MIC costs almost nothing in fit.
    assert 0.0 <= q["anchor_cost_log10_rmse"] < 1e-3


def test_old_kill_rate_points_are_the_profile_points_at_three_per_hour():
    d = _load()
    q = d["quoted_in_text"]
    for key, prof in (("k_kill_max_3_c50_free", "fit_a_over_k_kill_max"),
                      ("k_kill_max_3_c50_pinned", "fit_b_over_k_kill_max")):
        point = next(e for e in d["profiles"][prof] if e["k_kill_max"] == 3.0)
        for field in ("kill_C50", "objective", "excess_over_best"):
            assert q[key][field] == point[field], (key, field)


def test_the_old_kill_rate_fits_the_kill_data_but_not_the_mic():
    """S2.2's argument: with C50 free, 3.0/h is close to the optimum; with
    C50 pinned by the MIC it is far from it."""
    q = _load()["quoted_in_text"]
    assert q["k_kill_max_3_c50_free"]["excess_over_best"] < 0.02
    assert q["k_kill_max_3_c50_pinned"]["excess_over_best"] > 0.2
    assert q["k_kill_max_3_c50_free"]["kill_C50"] > 3 * q["k_kill_max_3_c50_pinned"]["kill_C50"]
