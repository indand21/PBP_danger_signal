"""Sobol sensitivity for the host-damage layer."""
from src.analysis.sensitivity_analysis import (
    DEFAULT_SA_BOUNDS, _PARAM_MAP, METRICS, metric_peak_host_damage,
    run_sensitivity_analysis,
)


def test_host_damage_params_registered():
    for name in ("k_infl", "k_path", "k_heal"):
        assert name in DEFAULT_SA_BOUNDS
        assert name in _PARAM_MAP
        assert _PARAM_MAP[name][0] == "damage"


def test_metric_peak_host_damage_registered_and_positive(short_simulation_result):
    assert "peak_host_damage" in METRICS
    assert metric_peak_host_damage(short_simulation_result) > 0


def test_sobol_runs_with_host_damage_metric():
    out = run_sensitivity_analysis(
        param_names=["k_pers", "k_infl"],
        n_samples=8,  # tiny, just exercises the pipeline
        metric="peak_host_damage",
        drug_class="cidal",
        print_progress=False,
    )
    assert "problem" in out
    assert out["problem"]["num_vars"] == 2
