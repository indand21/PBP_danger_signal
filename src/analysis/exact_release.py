"""Exact endotoxin integrals on a solved bacterial trajectory (plan item 8).

The danger-signal (endotoxin) compartment is linear,

    D' = src(t) - k D,

so for a release rate that is linear over each step it has a closed-form
update, and so does the integral of D. This module provides that update and
the two things it needs to be accurate:

* **Release sampled inside each solver step.** The bacterial solver's grid
  steps are up to 0.1 h, and a rate sampled only there and joined by straight
  lines misses curvature: cumulative killing came out up to 8e-4 off, and the
  trapezoid AUC of the sampled endotoxin trajectory 0.2% low. Here each step
  is split into ``REFINE`` sub-steps and the rate is evaluated EXACTLY at
  every sub-point from the solver's continuous interpolant (which leaves the
  trajectory itself unchanged) and the analytical drug concentration. The
  remaining error falls as the square of the sub-step count: about 1e-5 at
  ``REFINE = 8`` against quadrature on the interpolant, and
  ``scripts/ds_numerical_assurance.py`` records it.
* **Integrals that agree by construction.** Cumulative killing, released
  danger signal, the endotoxin trajectory and its AUC are all exact for the
  same piecewise-linear rate, so the mass balance
  ``released = k * AUC + D(T)`` holds to rounding and the item-6 factor
  identity (released lysis = yield x killing) stays exact.

Because D is linear in the release, the AUC for any pair of yields is a
linear combination of two unit-yield AUCs, so a bacterial trajectory is
solved once and reused across yield scenarios without the cytokine solve.
"""
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

#: Sub-steps per solver step at which release rates are evaluated exactly.
REFINE = 8

#: How the danger-signal layer is computed, for provenance blocks.
NUMERICS = {
    "bacterial": "RK45 rtol 1e-6 atol 1e-8 max_step 0.1 h, continuous interpolant kept",
    "release_rates": f"exact at {REFINE} sub-steps per solver step (interpolant + analytical PK)",
    "danger_signal": "closed-form update of D' = src - kD for src linear per sub-step; "
                     "AUC, released amount and cumulative killing exact for that src",
    "cytokines": "LSODA rtol 1e-6 atol 1e-9 max_step 0.1 h; its DS state checked against the closed form",
    "fT_above_MIC": "analytical PK, crossings by Brent root finding (1e-12 h)",
    "fAUC_fCmax": "analytical PK: Gauss-Legendre between dosing breakpoints; peak at infusion ends",
    "dose_solving": "PK-only %fT>MIC, Brent to 1e-6 mg",
}


def refined_times(t: np.ndarray, m: int = REFINE) -> np.ndarray:
    """The solver grid with ``m - 1`` equally spaced points inside each step."""
    t = np.asarray(t, dtype=float)
    if m <= 1:
        return t.copy()
    f = np.arange(m) / m
    inner = (t[:-1, None] + np.diff(t)[:, None] * f[None, :]).ravel()
    return np.append(inner, t[-1])


def refined_trajectory(result, pd_model, m: int = REFINE) -> Dict[str, np.ndarray]:
    """Replicating burden, free effect-site concentration and drug kill rate,
    evaluated exactly at the refined times of a ``run_simulation`` result.

    ``result`` must have been solved with ``dense_output=True``. At the
    solver's own grid points the values are the solver's; between them they
    come from its interpolant (B) and the analytical PK (C).
    """
    if getattr(result, "dense", None) is None:
        raise ValueError("refined_trajectory needs run_simulation(..., dense_output=True)")
    t = refined_times(result.t, m)
    B_rep = np.asarray(result.dense(t)[result.state_names.index("B_rep")], dtype=float)
    # Grid points take the stored solution exactly (the interpolant agrees to
    # rounding; this keeps the end values bit-identical to the solve).
    B_rep[::m] = result.y[:, result.state_names.index("B_rep")]
    C = np.asarray(result.free_concentration_at(t), dtype=float)
    kill = np.asarray(pd_model.cidal_kill_rate(C), dtype=float)
    return {"t": t, "B_rep": B_rep, "C_eff": C, "kill": kill}


def driver_from_result(result, pd_model, mic: Optional[float] = None,
                       m: int = REFINE) -> Dict[str, object]:
    """Everything the danger-signal layer needs from one regimen.

    The refined trajectory (``t``, ``B_rep``, ``C_eff``, ``kill``), the net
    48-h log10 change of the total burden, and, when ``mic`` is given, the
    exact %fT>MIC over the simulated span. ``result`` must come from
    ``run_simulation(..., dense_output=True)``.
    """
    d = refined_trajectory(result, pd_model, m)
    _, B_total = result.get_bacterial_burden()
    d["net_log10"] = float(np.log10(max(B_total[-1], 1e-6) / B_total[0]))
    if mic is not None:
        t0, t1 = float(result.t[0]), float(result.t[-1])
        d["fT"] = 100.0 * time_above(result.free_concentration_at, mic, t0, t1,
                                     dosing_breakpoints(result.params["_regimen"], t1)) / (t1 - t0)
    return d


def linear_source_ds(t, src, k: float, D0: float = 0.0
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact solution of D' = src - k D for src linear on each step of ``t``.

    Over a step of length h with src = a + b s (0 <= s <= h):

        D1   = D0 e^{-kh} + a g1 + b g2
        int D = D0 g1 + a g2 + b g3

    with g1 = (1 - e^{-kh})/k, g2 = h/k - g1/k, g3 = h^2/(2k) - h/k^2 + g1/k^2.
    Returns (D on t, cumulative integral of D on t, cumulative integral of
    src on t); the last is the trapezoid sum, exact for a piecewise-linear
    rate. For small kh the closed forms lose digits to cancellation, so g2
    and g3 switch to their Taylor series there.
    """
    t = np.asarray(t, dtype=float)
    src = np.asarray(src, dtype=float)
    h = np.diff(t)
    a = src[:-1]
    b = np.diff(src) / h
    x = k * h
    e = np.exp(-x)
    g1 = -np.expm1(-x) / k
    small = x < 1e-3
    g2 = np.where(small, h * h * (0.5 - x / 6 + x * x / 24),
                  h / k - g1 / k)
    g3 = np.where(small, h ** 3 * (1 / 6 - x / 24 + x * x / 120),
                  h * h / (2 * k) - h / k ** 2 + g1 / k ** 2)
    # The recurrence is sequential; plain floats keep it fast.
    D = np.empty(len(t))
    area = np.empty(len(t))
    d, ar = float(D0), 0.0
    D[0], area[0] = d, 0.0
    ee, c1, c2, c3 = e.tolist(), g1.tolist(), g2.tolist(), g3.tolist()
    aa, bb = a.tolist(), b.tolist()
    for i in range(len(h)):
        ar += d * c1[i] + aa[i] * c2[i] + bb[i] * c3[i]
        d = d * ee[i] + aa[i] * c1[i] + bb[i] * c2[i]
        D[i + 1], area[i + 1] = d, ar
    released = np.concatenate([[0.0], np.cumsum(h * (src[1:] + src[:-1]) / 2.0)])
    return D, area, released


def gauss_release_reference(result, pd_model, unit_filament_rate, k: float,
                            nodes: int = 8) -> Dict[str, float]:
    """Reference integrals by Gauss-Legendre quadrature on the interpolant.

    An independent check of the refined-grid numbers: on every solver step
    the true release rate (no linear interpolation) is integrated with
    ``nodes`` Gauss points, and the endotoxin compartment is advanced with
    its exact exponential kernel. ``unit_filament_rate(C)`` is the filament
    activity phi(C). Returns cumulative killing (CFU/mL), the integrated
    filament driver, and the 48-h AUC of D per unit lysis and per unit
    filament rate (same units as linear_source_ds with src = kill*B or phi*B).
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    t = np.asarray(result.t, dtype=float)
    h = np.diff(t)
    tau = (t[:-1, None] + (x[None, :] + 1.0) * h[:, None] / 2.0)
    flat = tau.ravel()
    B = np.asarray(result.dense(flat)[result.state_names.index("B_rep")], float).reshape(tau.shape)
    C = np.asarray(result.free_concentration_at(flat), float).reshape(tau.shape)
    s_l = np.asarray(pd_model.cidal_kill_rate(C), float) * B
    s_f = np.asarray(unit_filament_rate(C), float) * B
    wh = w[None, :] * h[:, None] / 2.0
    kern = np.exp(-k * (t[1:, None] - tau))
    out = {"cum_killed": float(np.sum(wh * s_l)), "filament_driver": float(np.sum(wh * s_f))}
    for name, s in (("lys", s_l), ("fil", s_f)):
        S = np.sum(wh * s, axis=1)            # release per step
        Q = np.sum(wh * kern * s, axis=1)     # its contribution to D at step end
        D, area = 0.0, 0.0
        for e, si, qi in zip(np.exp(-k * h).tolist(), S.tolist(), Q.tolist()):
            Dn = D * e + qi
            area += (si - (Dn - D)) / k
            D = Dn
        out[f"auc_{name}"] = float(area)
    return out


def mass_balance_residual(D: np.ndarray, area: np.ndarray, released: np.ndarray,
                          k: float, D0: float = 0.0) -> float:
    """|released - k * AUC - (D(T) - D0)| / released at the end time."""
    rel = float(released[-1])
    return abs(rel - k * float(area[-1]) - (float(D[-1]) - D0)) / max(abs(rel), 1e-300)


def time_above(conc, threshold: float, t0: float, t1: float,
               breakpoints: Iterable[float] = (), samples: int = 200) -> float:
    """Hours in [t0, t1] with ``conc(t) >= threshold``, crossings by root finding.

    ``conc`` is vectorized. Between consecutive ``breakpoints`` (dose starts
    and infusion ends, where the concentration's derivative jumps) it is
    sampled on ``samples`` sub-intervals and each sign change is located to
    1e-12 h with Brent's method, so the result does not depend on any solver
    grid. A crossing pair closer than one sub-interval would be missed; the
    concentration here is monotone between breakpoints, so there is none.
    """
    from scipy.optimize import brentq

    edges = sorted({float(t0), float(t1),
                    *[float(b) for b in breakpoints if t0 < b < t1]})
    above = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        g = np.linspace(a, b, samples + 1)
        f = np.asarray(conc(g), dtype=float) - threshold
        pts = [a]
        for i in np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]:
            pts.append(brentq(lambda s: float(conc(np.array([s]))[0]) - threshold,
                              g[i], g[i + 1], xtol=1e-12, rtol=4 * np.finfo(float).eps))
        pts.append(b)
        for lo, hi in zip(pts[:-1], pts[1:]):
            if hi > lo and float(conc(np.array([(lo + hi) / 2]))[0]) >= threshold:
                above += hi - lo
    return float(above)


def concentration_exposure(conc, t0: float, t1: float, breakpoints: Iterable[float] = (),
                           nodes: int = 24) -> Tuple[float, float]:
    """(AUC, Cmax) of an analytical concentration over [t0, t1].

    The concentration is smooth between dosing breakpoints, so Gauss-Legendre
    quadrature on each segment integrates it to rounding, and its maximum
    lies at a breakpoint or an end (an infusion end, for these regimens), or
    at an interior turning point, which the node values bound from below.
    """
    edges = sorted({float(t0), float(t1), *[float(b) for b in breakpoints if t0 < b < t1]})
    x, w = np.polynomial.legendre.leggauss(nodes)
    auc, cmax = 0.0, float(np.max(conc(np.array(edges))))
    for a, b in zip(edges[:-1], edges[1:]):
        tau = a + (x + 1.0) * (b - a) / 2.0
        c = np.asarray(conc(tau), dtype=float)
        auc += float(np.dot(w, c)) * (b - a) / 2.0
        cmax = max(cmax, float(np.max(c)))
    return auc, cmax


def dosing_breakpoints(regimen, t_end: float) -> list:
    """Dose starts and infusion ends before ``t_end``."""
    inf_h = regimen.infusion_duration_min / 60.0
    out = []
    for td in regimen.get_dose_times():
        out += [td, td + inf_h]
    return [x for x in out if 0.0 < x < t_end]
