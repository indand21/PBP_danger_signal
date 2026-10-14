import pytest
from scripts.ds_joint_robustness import tipping_ratio
from src.analysis.danger_signal import DangerSignalParameters, simulate_danger_signal, summarize
import numpy as np


def test_tipping_point_includes_both_release_pathways():
    critical=tipping_ratio(2.,3.,4.,5.,1.)
    assert critical==pytest.approx(20/13)
    assert (5*2+1*3)/((5/critical)*4)==pytest.approx(1.)
    assert tipping_ratio(2.,3.,4.,5.,0.)==pytest.approx(2.)


def test_release_auc_is_linear_even_when_cytokine_response_is_nonlinear():
    t=np.linspace(0,4,81)
    b=1e5*np.exp(-.2*t)
    k=np.full_like(t,.4)
    c=np.full_like(t,.5)
    def auc(y,yf):
        q=DangerSignalParameters(Y_lysis=y,Y_filament=yf)
        return summarize(simulate_danger_signal(t,b,k,c,1.,q))['AUC_DS_pg_h_mL']
    assert auc(2.,3.)==pytest.approx(2*auc(1.,0.)+3*auc(0.,1.),rel=2e-5)
