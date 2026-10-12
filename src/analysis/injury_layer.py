"""Layer 3: injury / outcome, as an explicitly ORDINAL layer.

The study design commits to validating this layer on *direction, not
magnitude*: the available clinical evidence (Prins 1995, n=30; Toky 2005,
murine) can support a statement that one agent produces more inflammatory
injury than another, and nothing finer. The honest way to model that is not to
invent an injury function and report its numbers as if they meant something,
but to ask whether the **ranking** of therapies is stable across every
defensible injury function.

So this module supplies a *family* of monotone maps from cytokine exposure to
injury, and a test of whether the ranking they induce is the same for all of
them. A ranking that survives the whole family is a result about the biology.
A ranking that depends on which map is chosen is a result about the map, and
must not be reported.

This is also the direct remedy for how the previous host-damage layer failed.
There, a single saturating injury term with one unmeasured half-saturation
constant was used to produce *quantitative* margins; the constant turned out to
pin the term past saturation, making every agent identical. Here saturation is
one member of the family rather than the whole model, and
``saturation_diagnostics`` reports where each map is operating.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp, trapezoid


@dataclass
class InjuryParameters:
    """Parameters of the injury accrual/recovery ODE."""

    k_injury: float = 1.0
    """Scale of cytokine-driven injury accrual, per hour (arbitrary units --
    only ranks are interpreted)."""

    k_heal: float = 0.10
    """Recovery rate, per hour (t1/2 ~7 h)."""

    cyto_ref: float = 100.0
    """Reference cytokine concentration, pg/mL, used to make the argument of
    each monotone map dimensionless. Set near the middle of the observed human
    endotoxemia range so the maps are compared on a sensible scale."""


#: Monotone maps g: [0, inf) -> [0, inf), g(0) = 0, g increasing. Each encodes
#: a different, defensible assumption about how inflammatory exposure converts
#: into injury. If the therapy ranking is the same under all of them, the
#: ranking does not depend on this unknown.
INJURY_MAPS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    # Injury proportional to cytokine exposure.
    "linear": lambda x: x,
    # Diminishing returns without a ceiling.
    "sqrt": lambda x: np.sqrt(x),
    "log1p": lambda x: np.log1p(x),
    # Supralinear: high cytokine levels are disproportionately damaging.
    "quadratic": lambda x: x ** 2,
    # Saturating -- the shape the OLD model used, included deliberately so that
    # if it is the one map that breaks the ranking, that fact is visible.
    "hill1": lambda x: x / (1.0 + x),
    "hill2": lambda x: x ** 2 / (1.0 + x ** 2),
    # A threshold-like map: little injury until exposure is substantial.
    "soft_threshold": lambda x: np.clip(x - 0.5, 0.0, None),
}


def injury_trajectory(t: np.ndarray, cytokine: np.ndarray,
                      map_name: str = "linear",
                      params: InjuryParameters = None) -> Dict[str, np.ndarray]:
    """Integrate injury accrual and recovery driven by a cytokine trajectory.

    ``cytokine`` should be the THERAPY-ATTRIBUTABLE cytokine concentration
    (the rise over the healthy baseline), in pg/mL.
    """
    p = params or InjuryParameters()
    g = INJURY_MAPS[map_name]
    t = np.asarray(t, dtype=float)
    drive = p.k_injury * np.asarray(g(np.asarray(cytokine, dtype=float)
                                     / p.cyto_ref), dtype=float)

    def interp(tt):
        return float(np.interp(tt, t, drive, left=drive[0], right=drive[-1]))

    # max_step is REQUIRED, not a refinement. The drive is an interpolated
    # forcing term, and for a thresholded map it is exactly zero until the
    # cytokine exceeds the threshold. With an unbounded step the solver sees a
    # zero right-hand side at t=0, takes one enormous step and integrates
    # straight over the pulse, returning zero injury for an arm that in fact
    # has the largest exposure -- which looks exactly like a model result and
    # is not one. 0.1 h matches the resolution of the driving trajectories.
    sol = solve_ivp(lambda tt, y: [interp(tt) - p.k_heal * max(y[0], 0.0)],
                    (t[0], t[-1]), [0.0], t_eval=t, method="LSODA",
                    max_step=0.1, rtol=1e-6, atol=1e-10)
    D = np.maximum(sol.y[0], 0.0)
    return {"t": t, "injury": D, "drive": drive,
            "peak": float(np.max(D)), "auc": float(trapezoid(D, t))}


def rank_by_injury(arms: Dict[str, Dict[str, np.ndarray]],
                   map_name: str = "linear",
                   metric: str = "peak",
                   params: InjuryParameters = None) -> List[str]:
    """Rank arms from least to most injury under one monotone map.

    ``arms`` maps an arm label to a dict with ``t`` and ``IL6_excess``.
    """
    scores = {}
    for label, out in arms.items():
        r = injury_trajectory(out["t"], out["IL6_excess"], map_name, params)
        scores[label] = r[metric]
    return sorted(scores, key=lambda k: scores[k])


def check_rank_invariance(arms: Dict[str, Dict[str, np.ndarray]],
                          metric: str = "peak",
                          params: InjuryParameters = None,
                          maps: Sequence[str] = None,
                          tol: float = 1e-9) -> Dict[str, object]:
    """Is the therapy ranking the same under every monotone injury map?

    This is the acceptance test for Layer 3. Passing it licenses an ordinal
    claim ("agent A inflicts more inflammatory injury than agent B") without
    committing to an injury function. Failing it means the ranking is an
    artefact of the map and must not be reported -- which is exactly the error
    the previous model made.
    """
    maps = list(maps or INJURY_MAPS.keys())
    rankings, scores = {}, {}
    for m in maps:
        per_arm = {}
        for label, out in arms.items():
            r = injury_trajectory(out["t"], out["IL6_excess"], m, params)
            per_arm[label] = {"peak": r["peak"], "auc": r["auc"]}
        scores[m] = per_arm
        rankings[m] = sorted(per_arm, key=lambda k: per_arm[k][metric])

    # Invariance is judged on PAIRWISE STRICT orderings, not on the sorted
    # order. Sorting is stable, so two arms with equal scores come back in
    # insertion order and look like a definite ranking; a map that cannot
    # separate two arms would then masquerade as agreeing or disagreeing with
    # the others depending on dictionary order. A tie is an absence of
    # evidence, so it is recorded as such instead.
    labels = list(arms)
    conflicts, pair_orders = [], {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            seen = set()
            for m in maps:
                xa, xb = scores[m][a][metric], scores[m][b][metric]
                scale = max(abs(xa), abs(xb))
                if scale == 0 or abs(xa - xb) <= tol * scale:
                    seen.add("~")
                else:
                    seen.add("<" if xa < xb else ">")
            strict = seen - {"~"}
            pair_orders[f"{a}|{b}"] = sorted(seen)
            if len(strict) > 1:
                conflicts.append(f"{a}|{b}")

    # A map that assigns the same score to every arm cannot discriminate at
    # all. That is not agreement -- it is a degenerate map, and it is reported
    # so it cannot prop up an invariance claim silently.
    degenerate = []
    for m in maps:
        vals = [scores[m][l][metric] for l in labels]
        hi = max(abs(v) for v in vals) or 0.0
        if hi == 0.0 or (max(vals) - min(vals)) <= tol * hi:
            degenerate.append(m)

    informative = [m for m in maps if m not in degenerate]
    distinct = {tuple(rankings[m]) for m in informative}
    consensus = list(next(iter(distinct))) if len(distinct) == 1 else None
    return {
        "metric": metric,
        "maps": maps,
        "rankings": {k: list(v) for k, v in rankings.items()},
        "scores": scores,
        "pair_orders": pair_orders,
        "conflicting_pairs": conflicts,
        "degenerate_maps": degenerate,
        "informative_maps": informative,
        "invariant": bool(not conflicts and informative),
        "n_distinct_rankings": len(distinct),
        "consensus_ranking": consensus,
    }


def saturation_diagnostics(arms: Dict[str, Dict[str, np.ndarray]],
                           params: InjuryParameters = None) -> Dict[str, Dict]:
    """Where is each saturating map operating?

    Reported for the same reason the cytokine layer's occupancy is reported:
    the previous model's failure was invisible precisely because nobody printed
    this number. For the saturating maps, a value approaching 1 means that map
    has become a binary detector and cannot distinguish arms.
    """
    p = params or InjuryParameters()
    out = {}
    for label, arm in arms.items():
        x = np.asarray(arm["IL6_excess"], dtype=float) / p.cyto_ref
        out[label] = {
            "max_x_over_ref": float(np.max(x)),
            "hill1_peak_occupancy": float(np.max(x / (1.0 + x))),
            "hill2_peak_occupancy": float(np.max(x ** 2 / (1.0 + x ** 2))),
        }
    return out
