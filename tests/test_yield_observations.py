import numpy as np
import pytest
from src.analysis.yield_observations import observations,comparable_ratio,cumulative_drug_killing


def test_sampling_mean_cannot_be_substituted_for_a_six_hour_point():
    obs=observations()["yamaguchi_isepamicin"]
    assert obs.predict([0,2,4,6],[0,10,20,60])==30.
    assert obs.predict([0,2,4,6],[0,10,20,60])!=60.


def test_incompatible_time_estimands_are_not_pooled_as_a_yield_ratio():
    obs=observations()
    with pytest.raises(ValueError):comparable_ratio(obs["yamaguchi_ceftazidime"],obs["yamaguchi_isepamicin"])
    assert comparable_ratio(obs["trautmann_ceftazidime"],obs["trautmann_meropenem"])==pytest.approx(.62/.13)


def test_positive_killed_biomass_can_coexist_with_zero_net_cfu_change():
    t=np.linspace(0,4,41);b=np.full(len(t),1e6);kill=np.full(len(t),.5)
    assert b[-1]-b[0]==0
    assert cumulative_drug_killing(t,b,kill)==pytest.approx(2e6)
