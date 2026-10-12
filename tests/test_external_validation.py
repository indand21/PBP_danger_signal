"""
Tests for the curve-level external-validation engine
(src/analysis/external_validation.py).

These cover the machinery (metrics math, predictor execution, registry loading,
role-based independence), not the scientific agreement — that depends on
digitized data the user supplies, and is asserted only loosely here.
"""
import numpy as np
import pytest

from src.analysis.external_validation import (
    DigitizedDataset,
    compute_metrics,
    dose_response_curve,
    effective_mic,
    load_registry,
    predict_pk,
    predict_timekill,
    stasis_index,
    validate_dataset,
)


# --- Metrics math -----------------------------------------------------------

def test_metrics_self_consistency():
    y = np.array([6.0, 4.0, 3.0, 2.5])
    m = compute_metrics(y, y, "log10")
    assert m.n == 4
    assert m.rmse == pytest.approx(0.0)
    assert m.bias == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)


def test_metrics_constant_offset():
    y = np.array([6.0, 4.0, 3.0, 2.5])
    m = compute_metrics(y, y + 0.5, "log10")
    assert m.rmse == pytest.approx(0.5)
    assert m.bias == pytest.approx(0.5)  # pred - obs
    assert m.mae == pytest.approx(0.5)


def test_metrics_ignores_nan():
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    pred = np.array([1.0, 2.0, 3.0, np.nan])
    m = compute_metrics(obs, pred, "linear")
    assert m.n == 2  # only the two finite-finite pairs


# --- Predictors run and return sane shapes ---------------------------------

def _mk(kind, drug, drug_class, scenario, y_space):
    ds = DigitizedDataset(
        id=f"t_{drug}_{kind}", kind=kind, drug=drug, drug_class=drug_class,
        role="validation", csv="none.csv", y_unit="u", y_space=y_space,
        source="test", scenario=scenario,
    )
    return ds


def test_timekill_cidal_kills():
    ds = _mk("timekill", "meropenem", "cidal",
             {"concentration_mgL": 8.0, "initial_burden": 1e6, "immune": 0.0},
             "log10")
    ds.t_obs = np.array([0.0, 2.0, 6.0, 24.0])
    y = predict_timekill(ds)
    assert y.shape == (4,)
    assert y[0] == pytest.approx(np.log10(1e6 + 1e2), abs=0.05)
    assert y[-1] < y[0]  # bactericidal: net reduction


def test_timekill_static_does_not_collapse():
    ds = _mk("timekill", "doxycycline", "static",
             {"concentration_mgL": 2.0, "initial_burden": 1e6, "immune": 0.0},
             "log10")
    ds.t_obs = np.array([0.0, 6.0, 24.0])
    y = predict_timekill(ds)
    # bacteriostatic: stays near the inoculum, no multi-log crash
    assert y[-1] > y[0] - 1.0


def test_pk_profile_peaks_and_declines():
    ds = _mk("pk", "meropenem", "cidal",
             {"dose_mg": 1000, "interval_hours": 8, "n_doses": 1,
              "infusion_min": 30, "weight_kg": 70}, "linear")
    ds.t_obs = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    c = predict_pk(ds)
    assert c.shape == (5,)
    assert np.all(c >= 0)
    assert c[0] > c[-1]  # declines after peak


# --- Registry + independence -----------------------------------------------

def test_registry_loads_and_has_roles():
    datasets = load_registry()
    assert len(datasets) >= 1
    roles = {d.role for d in datasets}
    assert roles <= {"calibration", "validation"}


def test_no_data_dataset_reports_no_data():
    # Header-only stub CSVs in the registry should validate as NO_DATA, never crash.
    datasets = load_registry()
    for ds in datasets:
        if not ds.has_data:
            res = validate_dataset(ds)
            assert res.status == "NO_DATA"


# --- Dose-response (neutropenic-thigh PK/PD index) mode --------------------

# Small grids keep these fast while still exercising the full path.
_FAST = {"n_grid": 6, "mic_search_n": 9, "t_end": 24}


def _dr(drug, drug_class, index):
    sc = dict(_FAST, index=index, interval_hours=(6 if index == "ft_mic" else 12),
              initial_burden=1e6, infusion_min=(30 if drug == "meropenem" else 0))
    return DigitizedDataset(
        id=f"dr_{drug}", kind="dose_response", drug=drug, drug_class=drug_class,
        role="validation", csv="none.csv", y_unit="dlog", y_space="log10",
        source="test", scenario=sc)


def test_effective_mic_positive():
    mic = effective_mic("cidal", dict(_FAST, initial_burden=1e6))
    assert mic > 0 and np.isfinite(mic)


def test_dose_response_cidal_kills_and_is_monotonic():
    ig, dg, mic = dose_response_curve(_dr("meropenem", "cidal", "ft_mic"))
    assert mic > 0
    assert dg.min() < 0  # achieves net kill at high exposure
    assert dg.max() > 0  # net growth at low exposure
    # kill increases with index (delta decreases): monotonic non-increasing
    assert np.all(np.diff(dg) <= 1e-6)


def test_dose_response_static_never_kills():
    # Central immune-dependence claim: static drug achieves stasis at best.
    ig, dg, mic = dose_response_curve(_dr("doxycycline", "static", "fauc_mic"))
    assert dg.min() >= -0.1  # no meaningful net kill anywhere


def test_stasis_index_interpolates_crossing():
    idx = np.array([0.0, 10.0, 20.0, 30.0])
    delta = np.array([2.0, 1.0, -1.0, -2.0])  # crosses 0 between 10 and 20
    s = stasis_index(idx, delta, target=0.0)
    assert 10.0 < s < 20.0
    assert s == pytest.approx(15.0, abs=0.01)


def test_stasis_index_none_when_no_crossing():
    idx = np.array([0.0, 10.0, 20.0])
    delta = np.array([2.0, 1.5, 1.0])  # never reaches 0
    assert stasis_index(idx, delta, target=0.0) is None
