import numpy as np
import pytest
from src.analysis.yield_observations import (observations,comparable_ratio,
                                             cumulative_drug_killing,kill_matched_ratio)


def test_sampling_mean_cannot_be_substituted_for_a_six_hour_point():
    obs=observations()["yamaguchi_isepamicin"]
    assert obs.predict([0,2,4,6],[0,10,20,60])==30.
    assert obs.predict([0,2,4,6],[0,10,20,60])!=60.


def test_incompatible_time_estimands_are_not_pooled_as_a_yield_ratio():
    obs=observations()
    with pytest.raises(ValueError):comparable_ratio(obs["yamaguchi_ceftazidime"],obs["yamaguchi_isepamicin"])
    assert comparable_ratio(obs["trautmann_ceftazidime"],obs["trautmann_meropenem"])==pytest.approx(.62/.13)


def test_a_ratio_at_equal_mic_multiple_is_not_a_ratio_at_equal_killing():
    obs=observations()
    # Same MIC multiple, but the agents did not kill equally there: 0.48 log10
    # apart is inside the default tolerance, 2.26 log10 apart is not.
    ratio,gap=kill_matched_ratio(obs["tsuji_ceftazidime_1x"],obs["tsuji_meropenem_1x"])
    assert gap==pytest.approx(.48) and ratio==pytest.approx(250.9/38.28)
    with pytest.raises(ValueError):
        kill_matched_ratio(obs["tsuji_ceftazidime_1x"],obs["tsuji_imipenem_0.1x"])


def test_matching_on_killing_requires_viable_counts_beside_the_endpoint():
    obs=observations()
    # Trautmann and Yamaguchi report LPS without the matched counts, so no
    # kill-matched ratio can be formed from them however tempting the numbers.
    with pytest.raises(ValueError):
        kill_matched_ratio(obs["trautmann_ceftazidime"],obs["trautmann_meropenem"])


def test_positive_killed_biomass_can_coexist_with_zero_net_cfu_change():
    t=np.linspace(0,4,41);b=np.full(len(t),1e6);kill=np.full(len(t),.5)
    assert b[-1]-b[0]==0
    assert cumulative_drug_killing(t,b,kill)==pytest.approx(2e6)
