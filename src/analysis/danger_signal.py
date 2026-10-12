"""Danger-signal layer (Phase 1 pilot).

Implements the dual-pathway danger-signal model of
``docs/danger_signal_study/DESIGN.md``:

    DS release rate = Y_lysis * (kill rate x B_rep)            [lysis-coupled]
                    + Y_filament * phi(C) * B_rep              [kill-independent]

``phi(C)`` is the sub-MIC filamentation window: PBP-3-specific agents
(ceftazidime) induce filamentation and lipopolysaccharide (LPS) release at
concentrations that kill little, which Jackson & Kropp (1992) observed directly
and Yamaguchi (1999) reproduced (at 1/4 x MIC, piperacillin and ceftazidime did
not reduce colony counts yet released *more* free LPS). PBP-2-specific agents
(imipenem, meropenem) and ribosomal agents have ``Y_filament = 0``.

Released danger signal enters a plasma compartment with first-order clearance
(LPS is cleared within minutes by lipoproteins and the reticuloendothelial
system), and plasma DS drives cytokine production through a Hill function whose
operating point is anchored on human experimental endotoxemia.

**Why this module is written as a post-processor.** In the existing PD model
nothing downstream of the bacteria feeds back on them: immune recruitment is a
function of bacterial burden only, and cytokines do not modulate killing. The
DS/cytokine layer is therefore mathematically decoupled from the bacterial
subsystem, and integrating it on an already-solved bacterial trajectory is
*exactly* equivalent to integrating it simultaneously -- while leaving the
validated PK/PD code path untouched. If a feedback term is ever added (e.g. DS
-> immune recruitment), this must move into the coupled right-hand side.

**Design constraint being enforced here.** The previous host-damage layer failed
because its inflammation term ran ~1200x past saturation, which made it a binary
detector: any nonzero release produced the same ceiling, so all agents
collapsed together. Every function below reports its *saturation fraction*, and
``check_operating_point`` exists so that the acceptance test can be run before
anything is built on top.

Units are stated explicitly on every parameter because this layer spans ten
orders of magnitude (femtograms per cell to micrograms per litre of plasma) and
that is precisely where the earlier model went wrong.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# Literature anchors (see docs/danger_signal_study/PHASE1_PILOT.md for the
# derivation of each number and the source it comes from).
# --------------------------------------------------------------------------

#: Danger-signal yield per CFU killed, fg LPS/CFU. Derived from Yamaguchi 1999
#: (free LPS at 1 x MIC, 4-6 h, P. aeruginosa PAO1) divided by the nominal
#: 1e8 CFU/mL inoculum of that assay. Confirm the inoculum on digitization.
YIELD_ANCHORS_FG_PER_CFU = {
    "piperacillin": 5.15,   # 515 ng/mL at 1 x MIC, 4 h
    "ceftazidime": 4.93,    # 493 ng/mL at 1 x MIC, 4 h
    "isepamicin": 0.28,     # 28 +/- 2 ng/mL at 1 x MIC, 6 h
    # Imipenem: Yamaguchi reports "less LPS" without a figure in the abstract;
    # Jackson & Kropp put ceftazidime 10-40x above imipenem, which brackets
    # imipenem at 0.12-0.49 fg/CFU. Mid-range value used; swept in the pilot.
    "imipenem": 0.25,
    # Meropenem is also PBP-2-specific and a low releaser (Trautmann 1998);
    # assigned imipenem's value pending digitization.
    "meropenem": 0.25,
}
# NOTE: the carbapenem entries above are the PHASE 1 anchors, giving a 19.7x
# agent ratio. The pilot and Phase 2 results were produced with them and are
# kept on that basis. They are NOT the final model -- see FINAL_YIELDS below.

#: Nominal ceftazidime:carbapenem yield ratio of the FINAL model. Phase 5b
#: revised the agent effect to 3.5-7x from three independent measurements
#: (Trautmann ELISA 6.9x, Prins whole-blood TNF 3.5x, Simpson in-patient
#: 4.97x), retiring the Jackson & Kropp Limulus 10-40x as an assay outlier.
#: The midpoint of that interval is used for every nominal analysis; the
#: interval itself is swept wherever an uncertainty range is reported.
FINAL_YIELD_RATIO = 5.0

#: The single authoritative yield configuration for the danger-signal study.
#: Every reported analysis and figure must take its yields from here, so that
#: the parameter table and the numerical results describe the same model.
FINAL_YIELDS_FG_PER_CFU = {
    "piperacillin": YIELD_ANCHORS_FG_PER_CFU["piperacillin"],
    "ceftazidime": YIELD_ANCHORS_FG_PER_CFU["ceftazidime"],
    "isepamicin": YIELD_ANCHORS_FG_PER_CFU["isepamicin"],
    "imipenem": YIELD_ANCHORS_FG_PER_CFU["ceftazidime"] / FINAL_YIELD_RATIO,
    "meropenem": YIELD_ANCHORS_FG_PER_CFU["ceftazidime"] / FINAL_YIELD_RATIO,
}

#: Plasma volume of a 70 kg adult, L.
PLASMA_VOLUME_L = 3.0

#: First-order plasma clearance of free LPS, per hour. Human and primate
#: endotoxemia studies put the initial disappearance half-life at a few
#: minutes; 3 min -> 13.9/h. Swept over 2-30/h in the acceptance test.
K_DS_CLEAR_PER_H = 13.9

#: Plasma DS concentration at half-maximal receptor engagement, pg/mL.
#: CALIBRATED (Phase 2) against human experimental endotoxemia -- see
#: src/analysis/cytokine_calibration.py. Fitting the IL-6 peak magnitude at two
#: intravenous LPS doses AND the observed 2 h time-to-peak selects
#: DS50 ~ 300 pg/mL; 300-10000 pg/mL all fit acceptably and are carried as the
#: Layer-2 uncertainty range, while 50-100 pg/mL is rejected because it forces
#: a time-to-peak of ~3 h. The Phase 1 pilot ASSUMED 100 pg/mL.
DS50_PG_PER_ML = 300.0

#: Hill coefficient of the DS-receptor response. CALIBRATED: the human IL-6
#: response has a local log-log slope of 1.69 between the 1 and 2 ng/kg doses,
#: which the fit renders as hill ~1.71. The pilot assumed 1.0, i.e. a
#: proportional response. Because the layer runs well below saturation, a slope
#: above 1 AMPLIFIES differences in per-kill danger-signal yield rather than
#: transmitting them one-for-one.
HILL_DS = 1.714

#: Decay rate of the immune-activation signal, per hour. CALIBRATED jointly
#: with the above; this is what places the IL-6 peak at ~2 h.
K_SIGNAL_OFF = 0.930

#: IL-6 production rate at full pathway activation, pg/mL/h. CALIBRATED to the
#: absolute human IL-6 peak. Large because the operating occupancy is small.
K_IL6_MAX = 176148.0

#: Recovery rate from endotoxin tolerance, per hour. From Dillingh 2014, which
#: found ex-vivo hyporesponsiveness after an in-vivo LPS challenge maximal at
#: 6 h and resolved within 12 h: a decay half-life of ~2 h.
K_TOL_OFF = 0.35

#: Rate of tolerance induction per unit receptor occupancy, per hour.
#: CALIBRATED (Phase 5) so that a patient with established endotoxemia at the
#: level Simpson 2000 measured before treatment shows the suppressed
#: therapy-attributable IL-6 rise that the same study observed. See
#: cytokine_calibration.calibrate_tolerance.
#:
#: Why this parameter is needed at all: the cytokine layer was calibrated on
#: healthy volunteers given a single LPS bolus, and it over-predicts the
#: attributable cytokine rise in septic patients by ~7-fold. Septic patients
#: are endotoxin-TOLERANT -- Prins 1995 found septic serum suppressing TNF
#: production concentration-dependently, Dillingh 2014 documented
#: hyporesponsiveness after an in-vivo challenge, and Simpson 2000 found peak
#: cytokines uncorrelated with peak endotoxin in 68 septic patients. A model
#: calibrated on naive subjects cannot be applied to tolerant ones unchanged.
#:
#: Note the structure does the work, not the parameter: because tolerance
#: integrates occupancy over time, a brief bolus in a naive subject induces
#: almost none of it (plasma LPS clears in minutes), while sustained
#: endotoxemia induces a lot. The healthy-volunteer calibration therefore
#: survives the addition nearly untouched -- which is the check that this term
#: is a mechanism rather than a fudge factor.
K_TOL_ON = 32.7


@dataclass
class DangerSignalParameters:
    """Parameters of the danger-signal and cytokine layers."""

    # --- Layer 1: generation -------------------------------------------------
    Y_lysis: float = 1.0
    """Danger signal liberated per CFU killed, fg/CFU."""

    Y_filament: float = 0.0
    """Kill-independent release rate per surviving CFU in the sub-MIC
    filamentation window, fg/CFU/h. Zero for non-filamenting agents."""

    filament_window: Tuple[float, float] = (0.0625, 1.0)
    """Sub-MIC window as (low, high) multiples of the MIC over which
    filamentation release operates. Jackson & Kropp and Yamaguchi both saw it
    at 1/4 x MIC; the lower bound reflects that very low concentrations engage
    too little PBP-3 to matter."""

    V_focus_mL: float = 10.0
    """Volume of the infection focus, mL. Converts the CFU/mL of the PD model
    into a total body burden. 10 mL is a mid-range soft-tissue/pulmonary focus;
    swept in the acceptance test."""

    # --- Danger-signal disposition ------------------------------------------
    V_ds_L: float = PLASMA_VOLUME_L
    k_ds_clear: float = K_DS_CLEAR_PER_H
    """Plasma distribution volume (L) and first-order clearance (per hour)."""

    translocation_fraction: float = 1.0
    """Fraction of locally released danger signal that reaches the systemic
    compartment. 1.0 is the conservative (worst-case) assumption used for the
    saturation test; a bacteraemic focus is the realistic upper bound."""

    # --- Layer 2: cytokine response -----------------------------------------
    DS50: float = DS50_PG_PER_ML
    """Plasma DS at half-maximal receptor engagement, pg/mL."""

    hill_ds: float = HILL_DS
    """Hill coefficient of the DS-receptor response."""

    k_signal_off: float = K_SIGNAL_OFF
    """Decay rate of the immune-activation signal S, per hour.

    S is the transduction state between danger-signal *detection* and cytokine
    *appearance*: receptor engagement drives transcription and secretion, which
    continue after the ligand has gone. Without it the model's IL-6 peaks
    within minutes of an intravenous LPS bolus, whereas human endotoxemia peaks
    at ~2 h -- a tenfold kinetic error that calibration exposed and that no
    choice of DS50 or Hill coefficient can repair, because plasma LPS itself is
    cleared within minutes. Calibrated in cytokine_calibration.py.
    """

    k_IL6_max: float = K_IL6_MAX
    """IL-6 production rate at full activation, pg/mL/h."""

    # --- Endotoxin tolerance -------------------------------------------------
    k_tol_on: float = 0.0
    """Rate of tolerance induction per unit receptor occupancy, per hour.

    DEFAULTS TO ZERO, i.e. tolerance is OFF. The mechanism is implemented and
    was tested against the clinical data in Phase 5, in both of its usual
    forms -- suppression of cytokine output and a right-shift of the
    dose-response -- and NEITHER could reconcile the healthy-volunteer
    calibration with the septic clinical observations. It is left in the code,
    disabled, because the attempt and its failure are part of the result; see
    scripts/ds_phase5_tolerance.py and PHASE5_TOLERANCE.md. Enable it
    explicitly if you want to reproduce that analysis."""

    k_tol_off: float = K_TOL_OFF
    """Recovery rate from tolerance, per hour."""

    tol_max: float = 0.0
    """Fractional suppression of cytokine OUTPUT at full tolerance.

    Defaults to zero: tolerance is modelled as a right-shift of the
    dose-response (``tol_shift``), not as a gain reduction. Output scaling is
    retained as an option because it is the obvious first thing to try, but it
    does not work here -- see ``tol_shift``."""

    tol_shift: float = 0.0
    """Fold right-shift of DS50 at full tolerance. Zero = tolerance inert.

    Endotoxin tolerance is classically a shift of the concentration-response,
    not a reduction of its ceiling, and here the distinction decides whether
    the model can be made consistent at all. Scaling the output cannot
    reconcile the two clinical settings: the Hill slope fitted to volunteers is
    supralinear (1.71), so an increment of danger signal on top of a septic
    endotoxin background should produce a LARGER response than the same
    increment in a naive subject, while the clinical data show a smaller one.
    Suppressing the output fights that amplification head-on and needs a
    tolerance so extreme (0.999) that it destroys the volunteer calibration. A
    right-shift instead moves the septic operating point back down the curve,
    which damps the amplification rather than cancelling it."""

    T0: float = 0.0
    """Initial tolerance. Zero for a naive subject; a septic patient with
    established endotoxemia starts partially tolerant."""

    DS_baseline: float = 0.0
    """Pre-treatment plasma danger signal, pg/mL, sustained by the untreated
    infection itself.

    Zero for a healthy volunteer. For a septic patient it is NOT zero --
    Simpson 2000 measured a median of 65.5 pg/mL before the first dose, and
    96% of patients had detectable endotoxin. Omitting it was a real error:
    therapy-attributable release was being added to an empty compartment, so
    the model both mis-stated the relative size of the therapeutic
    perturbation and could not sustain the tolerance that the prior exposure
    induces. Implemented as a constant source that holds DS at this level in
    the absence of therapy."""

    k_IL6_clear: float = 0.2
    """IL-6 elimination rate, per hour (t1/2 ~3.5 h)."""

    IL6_baseline: float = 5.0
    """Healthy baseline IL-6, pg/mL."""

    TNF_IL6_ratio: float = 0.3
    k_TNF_clear: float = 0.3
    """TNF produced as a fraction of IL-6 production, and its clearance."""

    def describe(self) -> Dict:
        return {
            "Y_lysis_fg_per_CFU": self.Y_lysis,
            "Y_filament_fg_per_CFU_per_h": self.Y_filament,
            "filament_window_xMIC": list(self.filament_window),
            "V_focus_mL": self.V_focus_mL,
            "V_ds_L": self.V_ds_L,
            "k_ds_clear_per_h": self.k_ds_clear,
            "translocation_fraction": self.translocation_fraction,
            "DS50_pg_per_mL": self.DS50,
            "hill_ds": self.hill_ds,
            "k_IL6_max_pg_per_mL_per_h": self.k_IL6_max,
        }


def filamentation_activity(C: np.ndarray, mic: float,
                           window: Tuple[float, float]) -> np.ndarray:
    """Sub-MIC filamentation activity phi(C), in [0, 1].

    A smooth bump: rises over the lower edge of the window and falls away above
    the MIC, where the agent kills fast enough that lysis-coupled release
    dominates. Deliberately smooth -- a hard threshold here would reintroduce
    the step-function pathology the design document forbids.
    """
    C = np.asarray(C, dtype=float)
    lo, hi = window
    x = np.maximum(C, 0.0) / float(mic)
    # Logistic on-ramp at the lower edge, logistic roll-off above the window.
    on = 1.0 / (1.0 + np.exp(-(np.log10(np.maximum(x, 1e-12)) - np.log10(lo)) / 0.15))
    off = 1.0 / (1.0 + np.exp((np.log10(np.maximum(x, 1e-12)) - np.log10(hi)) / 0.15))
    return on * off


def simulate_danger_signal(
    t: np.ndarray,
    B_rep: np.ndarray,
    kill_rate: np.ndarray,
    C_effect: np.ndarray,
    mic: float,
    params: Optional[DangerSignalParameters] = None,
) -> Dict[str, np.ndarray]:
    """Integrate the DS and cytokine layers on a solved bacterial trajectory.

    Parameters
    ----------
    t : hours, strictly increasing.
    B_rep : replicating burden at each time, CFU/mL.
    kill_rate : instantaneous drug kill rate at each time, per hour (the Hill
        rate from the PD model, *not* the net rate -- immune and growth terms
        are excluded because they are not drug-induced lysis).
    C_effect : effect-site concentration at each time, mg/L.
    mic : isolate MIC, mg/L.

    Returns
    -------
    dict with ``t``, ``DS`` (pg/mL plasma), ``IL6`` and ``TNF`` (pg/mL),
    ``saturation`` (the Hill occupancy DS^h/(DS50^h+DS^h), dimensionless),
    ``release_lysis`` and ``release_filament`` (ng/h into plasma).
    """
    p = params or DangerSignalParameters()
    t = np.asarray(t, dtype=float)

    phi = filamentation_activity(C_effect, mic, p.filament_window)

    # Release rates in fg/h over the whole focus:
    #   fg/CFU * (1/h) * CFU/mL * mL  ->  fg/h
    lysis_fg_per_h = p.Y_lysis * kill_rate * B_rep * p.V_focus_mL
    fila_fg_per_h = p.Y_filament * phi * B_rep * p.V_focus_mL

    # Convert to the plasma-concentration scale. 1 fg/h into V_ds litres is
    # 1e-15 g / 1e3 mL = 1e-18 g/mL = 1e-6 pg/mL, so:
    #   d(pg/mL)/dh = fg_per_h * 1e-6 / V_ds_L
    to_conc = 1e-6 / p.V_ds_L * p.translocation_fraction
    src = (lysis_fg_per_h + fila_fg_per_h) * to_conc

    DS, IL6, TNF, occ, tol = integrate_ds_cytokine(t, src, p)

    return {
        "tolerance": tol,
        "t": t,
        "DS": DS,
        "IL6": np.maximum(IL6, 0.0),
        # The DS-attributable IL-6, i.e. the rise over the healthy baseline.
        # This -- not total IL-6 -- is the inflammatory cost attributable to
        # therapy, and it is the quantity the acceptance tests must be run on:
        # a constant baseline integrated over a multi-day window otherwise
        # dominates the AUC at low yields and masks the dose-response.
        "IL6_excess": np.maximum(IL6 - p.IL6_baseline, 0.0),
        "TNF": np.maximum(TNF, 0.0),
        "saturation": occ,
        "phi": phi,
        "release_lysis": lysis_fg_per_h * 1e-6,   # ng/h
        "release_filament": fila_fg_per_h * 1e-6,  # ng/h
    }


def _occupancy(DS, ds50: float, hill: float):
    """Hill occupancy at an explicit half-maximal concentration."""
    DS = np.maximum(np.asarray(DS, dtype=float), 0.0)
    return DS ** hill / (ds50 ** hill + DS ** hill)


def hill_occupancy(DS, p: "DangerSignalParameters"):
    """Fractional engagement of the danger-signal receptor pathway, in [0, 1).

    Reported everywhere, because the operating point of this function is the
    single thing that decided whether the previous model worked.
    """
    DS = np.maximum(np.asarray(DS, dtype=float), 0.0)
    return DS ** p.hill_ds / (p.DS50 ** p.hill_ds + DS ** p.hill_ds)


def integrate_ds_cytokine(
    t: np.ndarray,
    src: np.ndarray,
    p: "DangerSignalParameters",
    DS0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate plasma danger signal and the cytokine response.

    ``src`` is the danger-signal appearance rate in pg/mL/h of plasma, and
    ``DS0`` the initial plasma concentration (nonzero for an intravenous LPS
    bolus). This is the single cytokine ODE in the project: both the infection
    simulations and the human-endotoxemia calibration go through it, so a layer
    calibrated on one cannot silently differ from the layer used in the other.

    The state vector is (DS, T, S, IL6, TNF), where S is the immune-activation
    signal carrying the transduction delay between danger-signal detection and
    cytokine appearance, and T is the endotoxin-tolerance state that suppresses
    cytokine production in a subject with sustained prior exposure.

    Returns (DS, IL6, TNF, occupancy, T), each on ``t``.
    """
    t = np.asarray(t, dtype=float)
    src_interp = _interp(t, np.asarray(src, dtype=float))
    # Constant input sustaining the untreated infection's own danger-signal
    # level, so that therapy-attributable release adds ON TOP of it.
    src_base = p.k_ds_clear * p.DS_baseline
    if DS0 is None:
        DS0 = p.DS_baseline

    def rhs(tt, y):
        DS = max(y[0], 0.0)
        T = min(max(y[1], 0.0), 1.0)
        S = max(y[2], 0.0)
        IL6 = max(y[3], 0.0)
        TNF = max(y[4], 0.0)
        # Tolerance right-shifts the dose-response: the pathway becomes less
        # SENSITIVE, rather than capped at a lower ceiling.
        ds50_eff = p.DS50 * (1.0 + p.tol_shift * T)
        occ = float(_occupancy(DS, ds50_eff, p.hill_ds))
        il6_prod = p.k_IL6_max * S * (1.0 - p.tol_max * T)
        return [
            src_interp(tt) + src_base - p.k_ds_clear * DS,
            p.k_tol_on * occ * (1.0 - T) - p.k_tol_off * T,
            p.k_signal_off * (occ - S),
            il6_prod - p.k_IL6_clear * (IL6 - p.IL6_baseline),
            p.TNF_IL6_ratio * il6_prod - p.k_TNF_clear * TNF,
        ]

    # max_step bounds the solver so it cannot step over the interpolated
    # release forcing, which starts at exactly zero (no drug at t=0) and would
    # otherwise permit one huge first step straight past the first dose.
    sol = solve_ivp(rhs, (t[0], t[-1]), [DS0, p.T0, 0.0, p.IL6_baseline, 0.0],
                    t_eval=t, method="LSODA", max_step=0.1,
                    rtol=1e-6, atol=1e-9)
    # Checked BEFORE the clipping below, which would otherwise hide an
    # undershoot. Row 0 is plasma danger signal, the reported endpoint.
    from src.core.simulation import check_integration
    check_integration(sol, t_end=t[-1], where="danger-signal layer",
                      nonnegative_rows=[0])
    DS = np.maximum(sol.y[0], 0.0)
    return (DS, sol.y[3], sol.y[4], hill_occupancy(DS, p),
            np.clip(sol.y[1], 0.0, 1.0))


def _interp(t: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """Linear interpolant with flat extrapolation, for use inside an ODE."""
    def f(tt: float) -> float:
        return float(np.interp(tt, t, y, left=y[0], right=y[-1]))
    return f


def check_release_mass_balance(
    out: Dict[str, np.ndarray],
    params: Optional["DangerSignalParameters"] = None,
) -> Dict[str, float]:
    """Verify the danger-signal compartment's mass balance on a simulation.

    With ``D = DS - DS_baseline`` the therapy-attributable signal, the
    compartment obeys ``D' = src(t) - k_ds * D``, so over ``[0, T]``

        k_ds * integral(D dt) = integral(src dt) - [D(T) - D(0)]
        \\_____cleared_____/    \\___released___/   \\_boundary_term_/

    and ``integral(src dt)`` is the yield-weighted cumulative killed biomass
    plus the filamentation term, times a common conversion factor. The
    near-perfect rank correlation between the proposed index and the modelled
    endpoint is this identity, not an empirical finding; reporting the
    residual keeps that explicit and catches an integration failure.

    Returns the three terms, their residual, and the residual relative to the
    released quantity.
    """
    from scipy.integrate import trapezoid

    p = params or DangerSignalParameters()
    t = np.asarray(out["t"], dtype=float)
    D = np.asarray(out["DS"], dtype=float) - p.DS_baseline

    # release_lysis/release_filament are ng/h over the focus; undo the ng
    # scaling and apply the same conversion the integrator used.
    to_conc = 1e-6 / p.V_ds_L * p.translocation_fraction
    src = (np.asarray(out["release_lysis"], dtype=float)
           + np.asarray(out["release_filament"], dtype=float)) * 1e6 * to_conc

    cleared = float(p.k_ds_clear * trapezoid(D, t))
    released = float(trapezoid(src, t))
    boundary = float(D[-1] - D[0])
    residual = released - boundary - cleared
    scale = max(abs(released), np.finfo(float).tiny)
    return {
        "cleared": cleared,
        "released": released,
        "boundary_term": boundary,
        "residual": float(residual),
        "relative_residual": float(residual / scale),
    }


def summarize(out: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Scalar summaries of a danger-signal simulation."""
    from scipy.integrate import trapezoid
    t = out["t"]
    return {
        "DS_peak_pg_mL": float(np.max(out["DS"])),
        "AUC_DS_pg_h_mL": float(trapezoid(out["DS"], t)),
        # Therapy-ATTRIBUTABLE danger signal: the rise over the level the
        # untreated infection already sustains. With a septic baseline load the
        # total AUC is dominated by that shared background and stops
        # discriminating agents at all, exactly as total IL-6 did in Phase 2.
        "AUC_DS_excess_pg_h_mL": float(trapezoid(
            np.maximum(out["DS"] - out["DS"][0], 0.0), t)),
        "DS_excess_peak_pg_mL": float(np.max(out["DS"] - out["DS"][0])),
        "IL6_peak_pg_mL": float(np.max(out["IL6"])),
        "AUC_IL6_pg_h_mL": float(trapezoid(out["IL6"], t)),
        "AUC_IL6_excess_pg_h_mL": float(trapezoid(out["IL6_excess"], t)),
        "TNF_peak_pg_mL": float(np.max(out["TNF"])),
        "saturation_peak": float(np.max(out["saturation"])),
        "saturation_mean": float(np.mean(out["saturation"])),
        "filament_fraction_of_release": _filament_fraction(out),
    }


def _filament_fraction(out: Dict[str, np.ndarray]) -> float:
    from scipy.integrate import trapezoid
    t = out["t"]
    lys = trapezoid(out["release_lysis"], t)
    fil = trapezoid(out["release_filament"], t)
    total = lys + fil
    return float(fil / total) if total > 0 else 0.0


def check_operating_point(saturation_peak: float,
                          upper: float = 0.80,
                          lower: float = 1e-6) -> Dict[str, object]:
    """Acceptance test 1: is the cytokine layer operating below saturation?

    The binding requirement is the UPPER bound: an occupancy near 1 is the
    ceiling on which the previous host-damage layer failed, where every agent
    collapses to the same output.

    The lower bound only guards against a genuinely dead pathway. It is
    deliberately very small, because low occupancy does *not* imply poor
    discrimination here: below DS50 the response follows DS^hill, and with the
    calibrated hill of ~1.7 that regime discriminates agents better than the
    mid-range does, not worse (the Phase 2 dynamic range is ~1000-fold). The
    calibration data settle the point -- human experimental endotoxemia at
    2 ng/kg sits at an occupancy of about 0.04 while producing fever and an
    IL-6 peak near 200 pg/mL, so a few percent occupancy is simply where real
    endotoxemia operates. An earlier 0.01 floor would have rejected the very
    data the layer is fitted to. Discrimination is measured directly, and
    separately, by ``check_smoothness``.
    """
    return {
        "saturation_peak": float(saturation_peak),
        "below_saturation": bool(saturation_peak < upper),
        "engaged": bool(saturation_peak > lower),
        "pass": bool(lower < saturation_peak < upper),
        "band": [lower, upper],
    }


def check_smoothness(y_grid: np.ndarray, response: np.ndarray,
                     min_dynamic_range: float = 3.0,
                     max_local_jump: float = 0.35) -> Dict[str, object]:
    """Acceptance test 2: does the output vary smoothly with ``Y_lysis``?

    Three conditions, all pre-specified in the design document:

    1. **monotone** -- the response increases with yield (no inversions);
    2. **dynamic range** -- across the plausible two-decade yield range the
       response spans at least ``min_dynamic_range``-fold, i.e. the model can
       actually tell a high-yield agent from a low-yield one;
    3. **no step** -- the largest single-step relative increment on a log-spaced
       yield grid stays below ``max_local_jump``. A binary detector shows up
       here as one near-1.0 jump with flats either side.
    """
    y_grid = np.asarray(y_grid, dtype=float)
    r = np.asarray(response, dtype=float)
    diffs = np.diff(r)
    monotone = bool(np.all(diffs >= -1e-12))
    rng = float(r[-1] / r[0]) if r[0] > 0 else float("inf")
    # Relative increment per step, normalised by the total span.
    span = r[-1] - r[0]
    local = np.abs(diffs) / span if span > 0 else np.zeros_like(diffs)
    max_jump = float(np.max(local)) if local.size else 0.0
    return {
        "monotone": monotone,
        "dynamic_range": rng,
        "dynamic_range_ok": bool(rng >= min_dynamic_range),
        "max_step_fraction_of_span": max_jump,
        "no_step_function": bool(max_jump <= max_local_jump),
        "pass": bool(monotone and rng >= min_dynamic_range
                     and max_jump <= max_local_jump),
        "thresholds": {"min_dynamic_range": min_dynamic_range,
                       "max_local_jump": max_local_jump},
    }
