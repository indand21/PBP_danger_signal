"""Tests for the ported PK/PD-index reporting + effect-mode taxonomy.

These cover the additive IJAA-facing layer (AntibioticPDParameters /
get_drug_pd_parameters and SimulationResult.get_pkpd_indices). They must not
change the calibrated dynamics -- only report unbound Craig indices on top of an
existing simulation.
"""
import numpy as np
import pytest

from src.core.parameters import AntibioticPDParameters, get_drug_pd_parameters
from src.analysis.strategy_margin import run_one


def test_time_above_threshold_crossings():
    """Crossing interpolation must count the time ACTUALLY above the threshold.

    Regression test for an inverted crossing branch: a linear ramp 0->10 over 1 h
    with threshold 2 is above from t=0.2 to t=1.0, i.e. 0.8 h (not 0.2 h), and a
    falling ramp 10->0 is above from t=0 to t=0.8, also 0.8 h.
    """
    from src.core.simulation import SimulationResult as SR
    rising = SR._time_above_threshold(np.array([0.0, 1.0]), np.array([0.0, 10.0]), 2.0)
    falling = SR._time_above_threshold(np.array([0.0, 1.0]), np.array([10.0, 0.0]), 2.0)
    assert abs(rising - 0.8) < 1e-9, rising
    assert abs(falling - 0.8) < 1e-9, falling
    # fully above / fully below
    assert abs(SR._time_above_threshold(np.array([0.0, 1.0]), np.array([5.0, 6.0]), 2.0) - 1.0) < 1e-9
    assert SR._time_above_threshold(np.array([0.0, 1.0]), np.array([0.1, 0.2]), 2.0) == 0.0


def test_taxonomy_profiles():
    assert get_drug_pd_parameters("meropenem").effect_mode == "time_dependent"
    assert get_drug_pd_parameters("meropenem").drug_class == "cidal"
    assert get_drug_pd_parameters("ciprofloxacin").effect_mode == "exposure_dependent"
    assert get_drug_pd_parameters("vancomycin").effect_mode == "exposure_dependent"
    assert get_drug_pd_parameters("doxycycline").drug_class == "static"
    assert get_drug_pd_parameters("doxycycline").effect_mode == "static"
    with pytest.raises(ValueError):
        get_drug_pd_parameters("not-a-drug")


def test_antibiotic_pd_validate():
    AntibioticPDParameters().validate()  # defaults are valid
    with pytest.raises(ValueError):
        AntibioticPDParameters(effect_mode="bogus").validate()
    with pytest.raises(ValueError):
        AntibioticPDParameters(mic=0).validate()
    with pytest.raises(ValueError):
        AntibioticPDParameters(fraction_unbound=1.5).validate()


def test_pkpd_indices_basic():
    r = run_one(n_eff=1e7, k_pers=0.01, k_infl=0.03, drug_class="cidal")
    idx = r.get_pkpd_indices(mic=1.0, fraction_unbound=0.98)
    assert set(idx) == {"fT_above_MIC_pct", "fAUC_MIC", "fCmax_MIC"}
    assert 0.0 <= idx["fT_above_MIC_pct"] <= 100.0
    assert idx["fAUC_MIC"] > 0.0
    assert idx["fCmax_MIC"] > 0.0
    # A higher MIC can only reduce (never increase) %fT>MIC.
    idx_hi = r.get_pkpd_indices(mic=4.0, fraction_unbound=0.98)
    assert idx_hi["fT_above_MIC_pct"] <= idx["fT_above_MIC_pct"] + 1e-9
    assert idx_hi["fCmax_MIC"] < idx["fCmax_MIC"]


def test_pkpd_indices_requires_mic():
    r = run_one(n_eff=1e7, k_pers=0.01, k_infl=0.03, drug_class="cidal")
    with pytest.raises(ValueError):
        r.get_pkpd_indices()  # no MIC supplied and none stored


def test_time_dependent_fractionation_signature():
    """More-fractionated dosing gives higher %fT>MIC (the time-dependent signature)."""
    from src.core.parameters import get_default_parameters, get_drug_pk_parameters
    from src.core.pd_model import create_ode_system
    from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
    from src.core.simulation import run_simulation

    p = get_drug_pk_parameters("meropenem")
    pk = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka, Kp=p.Kp,
                               effect_site_model=True)
    ic = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
          "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0}
    daily = 3000.0
    ft = {}
    for interval, n in [(24, 4), (6, 16)]:
        params = get_default_parameters()
        reg = DosingRegimen(dose_mg=daily / (24 / interval), interval_hours=interval,
                            start_time=0, n_doses=n, infusion_duration_min=30)
        r = run_simulation(pk_model=pk, regimen=reg,
                           pd_model=create_ode_system(params),
                           initial_conditions=ic, t_span=(0, 96),
                           drug_class="cidal", weight_kg=70.0)
        ft[interval] = r.get_pkpd_indices(mic=1.0, fraction_unbound=0.98)["fT_above_MIC_pct"]
    assert ft[6] > ft[24]  # q6 spends more time above MIC than q24
