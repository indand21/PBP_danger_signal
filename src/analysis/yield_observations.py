"""Observation operators for LPS evidence; observations are not latent yields.

Free LPS at one time, a mean over times, whole-blood TNF and plasma peaks
cannot be pooled as measurements of fg/CFU killed. This module deliberately
keeps the observed quantity and assay separate from model release parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class ReleaseObservation:
    agent: str
    value: float
    unit: str
    quantity: str
    assay: str
    organism: str
    times_h: tuple[float, ...]
    operator: str = "point"
    concentration_xMIC: float = 1.

    def predict(self, times, values):
        """Apply the published sampling operator without extrapolation."""
        t,y=np.asarray(times,float),np.asarray(values,float)
        if t.ndim!=1 or len(t)!=len(y) or len(t)<2 or np.any(np.diff(t)<=0):
            raise ValueError("Prediction times must be increasing and match the series")
        q=np.asarray(self.times_h,float)
        if not np.isfinite(y).all() or q.min()<t[0] or q.max()>t[-1]:
            raise ValueError("Observation lies outside the finite predicted trajectory")
        sampled=np.interp(q,t,y)
        if self.operator=="point" and len(q)==1:return float(sampled[0])
        if self.operator=="mean" and len(q)>1:return float(np.mean(sampled))
        raise ValueError("Use one time for a point, multiple times for a sampling mean")

    def same_estimand(self, other):
        return all(getattr(self,k)==getattr(other,k) for k in
                   ["unit","quantity","assay","organism","times_h","operator","concentration_xMIC"])


def comparable_ratio(numerator, denominator):
    """A ratio of matched observations, explicitly not a per-killed-cell yield."""
    if not numerator.same_estimand(denominator):
        raise ValueError("Cannot form an estimand-matched ratio from different observation operators")
    if denominator.value<=0:raise ValueError("Ratio denominator must be positive")
    return numerator.value/denominator.value


def cumulative_drug_killing(times, replicating_burden, kill_rate):
    """Killed biomass is the integrated drug-kill flux, not net CFU loss."""
    t,b,k=map(lambda x:np.asarray(x,float),(times,replicating_burden,kill_rate))
    if len(t)<2 or t.shape!=b.shape or t.shape!=k.shape or np.any(np.diff(t)<=0):
        raise ValueError("Killing trajectories must share increasing time points")
    if not all(np.isfinite(v).all() for v in [t,b,k]) or np.any(b<0) or np.any(k<0):
        raise ValueError("Killing flux inputs must be finite and nonnegative")
    return float(trapezoid(b*k,t))


def observations():
    """Published endpoints with their own experimental observation operators."""
    def tr(agent,value):
        return ReleaseObservation(agent,value,"mg/L","free LPS","capture ELISA",
                                  "E. coli Bort",(8.,),concentration_xMIC=50.)
    def ya(agent,value,times,operator="point"):
        return ReleaseObservation(agent,value,"ng/mL","free LPS","Limulus colour KY",
                                  "P. aeruginosa PAO1",times,operator)
    return {"trautmann_ceftazidime":tr("ceftazidime",.62),
            "trautmann_meropenem":tr("meropenem",.13),
            "trautmann_imipenem":tr("imipenem",.09),
            "yamaguchi_ceftazidime":ya("ceftazidime",493.,(4.,)),
            "yamaguchi_piperacillin":ya("piperacillin",515.,(4.,)),
            "yamaguchi_isepamicin":ya("isepamicin",28.,(2.,4.,6.),"mean")}
