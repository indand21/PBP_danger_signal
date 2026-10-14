import copy
import numpy as np
import pytest

from src.analysis.external_validation import parameters_for_scenario,reference_mic
from src.analysis.pd_context import growth_from_control,untreated_delta,CONTROL_ELAPSED_H,load_thigh_strains
from src.core.parameters import get_default_parameters


def test_growth_override_does_not_redefine_measured_mic_or_mutate_defaults():
    sc={"mic_reference_mgL":2.,"bacterial_parameters":{"k_growth":.2}}
    before=get_default_parameters()["bacteria"].k_growth
    p=parameters_for_scenario(sc)
    assert p["bacteria"].k_growth==.2
    p["bacteria"].k_growth=.8
    assert reference_mic("cidal",sc)==2.
    assert get_default_parameters()["bacteria"].k_growth==before


def test_missing_or_invalid_mic_cannot_silently_use_model_stasis():
    for sc in [{},{"mic_reference_mgL":0},{"mic_reference_mgL":float("nan")}]:
        with pytest.raises(ValueError):reference_mic("cidal",sc)


def test_context_rejects_an_inert_misspelled_parameter():
    with pytest.raises(ValueError):parameters_for_scenario({"bacterial_parameters":{"k_growht":.2}})


def test_growth_uses_elapsed_control_time_not_treated_endpoint():
    assert CONTROL_ELAPSED_H==10.
    g=growth_from_control(1.5,1e7,10.)
    assert untreated_delta(g,1e7,10.)==pytest.approx(1.5,abs=1e-6)
    assert untreated_delta(g,1e7,24.)>2.


def test_mean_published_threshold_uses_strain_table_not_pooled_curve():
    rows=load_thigh_strains()
    assert len(rows)==6
    assert np.mean([r["stasis_ft_pct"] for r in rows])==pytest.approx(17.35)
    assert rows[-2]["two_log_ft_pct"] is None
