"""Guard tests for the PK volume-scaling convention.

run_simulation() scales pk_model.Vc and pk_model.Vp by weight_kg internally
(Vc_val = pk_model.Vc * weight_kg). Callers must therefore build the PK model from
the RAW per-kg parameters returned by get_drug_pk_parameters(), not from
normalize_pk_parameters() output, which is already weight-scaled. Passing the
normalized values double-scales Vc (meropenem 17.5 L -> 1225 L at 70 kg), giving an
elimination half-life of ~57 h instead of ~0.8 h.

These tests fail if that double-scaling is reintroduced.
"""
import numpy as np

from src.analysis.strategy_margin import _make_pk
from src.core.parameters import (
    get_default_parameters,
    get_drug_pk_parameters,
    normalize_pk_parameters,
)
from src.core.pd_model import create_ode_system
from src.core.pk_models import DosingRegimen
from src.core.simulation import run_simulation

WEIGHT_KG = 70.0


def _single_dose(pk_model, t_end=8.0):
    """One 1 g meropenem dose as a 30-min infusion, cidal, 70 kg."""
    regimen = DosingRegimen(dose_mg=1000, interval_hours=8, start_time=0, n_doses=1,
                            infusion_duration_min=30)
    ic = {"B_rep": 1e6, "B_pers": 1e3, "B_SCV": 0, "N_eff": 1e7,
          "Damage": 0, "IL6": 10, "TNF": 5, "PAMP": 0, "D_host": 0}
    return run_simulation(pk_model=pk_model, regimen=regimen,
                          pd_model=create_ode_system(get_default_parameters()),
                          initial_conditions=ic, t_span=(0, t_end),
                          drug_class="cidal", weight_kg=WEIGHT_KG)


def _half_life(result, pk_model):
    return np.log(2) / (pk_model.CL / result.params["_Vc_val"])


def test_strategy_margin_pk_half_life_is_realistic():
    """The manuscript's analysis path must give a meropenem half-life near 1 h."""
    pk = _make_pk()
    t_half = _half_life(_single_dose(pk), pk)
    assert 0.5 <= t_half <= 1.5, f"meropenem t1/2 = {t_half:.2f} h; double-scaled Vc gives ~57 h"


def test_conftest_meropenem_fixture_half_life_is_realistic(pk_model_meropenem):
    """The shared test fixture must follow the same convention."""
    t_half = _half_life(_single_dose(pk_model_meropenem), pk_model_meropenem)
    assert 0.5 <= t_half <= 1.5, f"fixture t1/2 = {t_half:.2f} h; double-scaled Vc gives ~57 h"


def test_stored_central_volume_is_weight_scaled_once():
    pk_raw = get_drug_pk_parameters("meropenem")
    result = _single_dose(_make_pk())
    assert abs(result.params["_Vc_val"] - pk_raw.Vc * WEIGHT_KG) < 1e-9


def test_drug_has_cleared_by_8_hours():
    """With t1/2 ~0.8 h the effect-site concentration at 8 h is <5% of its peak
    (with the double-scaling bug it would still be ~90% of peak)."""
    pk = _make_pk()
    result = _single_dose(pk)
    t_uniform = np.linspace(0.0, 8.0, 801)
    # Recompute the analytic effect-site profile on a uniform grid from the stored values.
    k = result.params["_CL_val"] / result.params["_Vc_val"]
    inf_h = 0.5
    r0 = 1000.0 / inf_h
    c_end = r0 / (k * result.params["_Vc_val"]) * (1 - np.exp(-k * inf_h))
    c = np.where(t_uniform <= inf_h,
                 r0 / (k * result.params["_Vc_val"]) * (1 - np.exp(-k * t_uniform)),
                 c_end * np.exp(-k * (t_uniform - inf_h))) * result.params["_Kp_val"]
    assert c[-1] / c.max() < 0.05, f"C(8 h) is {100 * c[-1] / c.max():.1f}% of peak"


def test_normalized_parameters_would_double_scale():
    """Documents the bug: normalized Vc passed to run_simulation is scaled a second time."""
    pk_norm = normalize_pk_parameters(get_drug_pk_parameters("meropenem"), WEIGHT_KG)
    t_half_if_double_scaled = np.log(2) / (pk_norm["CL"] / (pk_norm["Vc"] * WEIGHT_KG))
    assert t_half_if_double_scaled > 10.0
