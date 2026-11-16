import hashlib
import json
from pathlib import Path

import pytest

import scripts.ds_cytokine_ridge as ridge
import src.analysis.cytokine_calibration as cc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/danger_signal/cytokine_ridge.json"


def test_the_between_grid_points_refit_leaves_the_module_as_it_found_it(monkeypatch):
    # The diagnostic swaps the peak reader for one refit only; the adopted
    # calibration must be untouched afterwards.
    original = cc.predict_peaks
    monkeypatch.setattr(cc, "calibrate_family", lambda: cc.predict_peaks)
    assert ridge.family_between_grid_points() is ridge.predict_peaks_between_grid_points
    assert cc.predict_peaks is original


@pytest.mark.skipif(not OUT.exists(), reason="cytokine_ridge.json not generated")
def test_ridge_results_were_made_from_the_current_inputs():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    for name, digest in d["input_sha256"].items():
        path = ROOT / "results/danger_signal" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


@pytest.mark.skipif(not OUT.exists(), reason="cytokine_ridge.json not generated")
def test_the_ridge_is_flat_and_the_endpoint_does_not_see_it():
    h = json.loads(OUT.read_text(encoding="utf-8"))["headline"]
    # Same cost, different parameters: a ridge, not a different optimum.
    assert h["cost_rel_diff"] < 1e-12
    assert abs(h["param_rel_diff_refit_vs_stored"]["k_signal_off"]) > 1e-5
    # The endotoxin AUC is upstream of the cytokine layer.
    assert h["table1_max_rel_diff_refit_vs_stored"]["AUC_DS_pg_h_mL"] < 1e-5
    # Read between grid points, every DS50 fits; the grid-read screen rejects two.
    assert h["between_grid_points"]["n_feasible"] == h["between_grid_points"]["n_grid"]
    assert h["grid_read"]["n_feasible"] < h["between_grid_points"]["n_grid"]
