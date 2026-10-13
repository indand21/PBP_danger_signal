"""
Parameter definitions and default values for QSP model
"""

import numpy as np
import dataclasses
from dataclasses import dataclass
from typing import Dict

@dataclass
class BacterialParameters:
    """Bacterial growth and population dynamics parameters"""
    k_growth: float = 0.5  # per hour, intrinsic growth rate
    B_max: float = 1e9  # CFU/mL, carrying capacity (log-scale typically 10^8-10^10)
    k_pers: float = 0.01  # per hour, transition to persisters
    mu_mut: float = 1e-6  # mutation rate per cell per generation (SCV emergence under static) - increased from 1e-7 for clinically relevant heteroresistance
    k_repair: float = 0.3  # per hour, damage repair rate (t1/2 ≈ 2.3h, persists between doses)
    MIC_baseline: float = 1.0  # mg/L, baseline MIC for susceptible strain
    # Smooth SCV emergence switch (replaces hard H_static < 0.3 cutoff)
    scv_switch_midpoint: float = 0.3  # static-inhibition level at half-max SCV emergence
    scv_switch_width: float = 0.05    # sigmoid width; smaller = sharper
    # Saturating (time-dependent) bactericidal kill: rate plateaus at k_kill_max,
    # half-maximal at kill_C50 (effect-site mg/L). Replaces the earlier
    # over-potent, concentration-dependent kill.
    #
    # REFITTED 2026-10-13 (review R7) by scripts/ds_bacterial_refit.py, which
    # also records what the data can and cannot identify. Previously
    # k_kill_max = 3.0, kill_C50 = 0.6, neither fitted, with a comment
    # claiming kill_C50 "sets the effective MIC ~1 mg/L"; it set it to 0.40,
    # and that sentence is what made an assumed parameter read as a
    # calibrated one.
    #
    #   k_kill_max  the value the two digitized datasets return (meropenem
    #               time-kill n=3, neutropenic-thigh dose-response n=47). The
    #               objective is flat between about 1.1 and 2.2, so the digits
    #               past the second are not meaningful; the exact grid value
    #               is kept only so the fit is reproducible.
    #   kill_C50    NOT identifiable from the kill data, for two independent
    #               reasons: the kill rate depends on concentration only
    #               through C/kill_C50, and the model's effective MIC is
    #               exactly proportional to kill_C50, so MIC-normalised data
    #               cannot see it; and at the 4xMIC exposure of the time-kill
    #               a steep Hill saturates, so even absolute-unit data are
    #               flat in it from 0.1 to about 5 mg/L. It is therefore
    #               pinned by the calibration isolate's MEASURED MIC of
    #               1 mg/L (P. aeruginosa ATCC-27853): the model's broth
    #               stasis concentration is that same quantity, and since
    #               stasis is proportional to kill_C50 the requirement is a
    #               closed form, not a search. It costs 0.001 log10 RMSE.
    #   kill_hill   kept at 4. The formal optimum is 8, but the data cannot
    #               separate the two (0.05 log10 RMSE), and the residual kill
    #               difference between agents depends on this exponent -- 0.10
    #               log10 at hill 8 against 0.23 at hill 4 -- so the choice is
    #               stated rather than fitted. See the Limitations.
    k_kill_max: float = 1.5874010519681996  # per hour, max bactericidal kill
    kill_C50: float = 0.936   # mg/L effect-site, concentration at half-max kill
    kill_hill: float = 4.0    # Hill coefficient (steep onset near MIC -> time-dependence)
    # Path B: maximal direct bacteriostatic-associated kill rate (per hour), scaled
    # by target engagement (1 - H_static). 0 = original growth-inhibition-only
    # bacteriostatic mechanism (immune clearance as the sole sink).
    static_kill_max: float = 0.0
    # Fraction of the per-kill PAMP/DAMP yield released by bacteriostatic killing
    # relative to bactericidal (lytic) killing. 0 = non-lytic idealization;
    # 1 = no mechanistic difference in danger-signal release. The true value is a
    # measurable quantity (endotoxin / bacterial-DNA release per CFU killed by
    # agent class) and is the pivot of the cidal-vs-static host-damage comparison.
    static_pamp_fraction: float = 0.0

@dataclass
class ImmuneParameters:
    """Immune effector dynamics.

    The pool is homeostatic and burden-recruited:

        dN/dt = k_deg_immune * (target(B) - N)
        target(B) = N_eff_0 * (1 + (recruit_fold_max - 1) * B/(EC50_immune + B))

    so production is ``k_deg_immune * target`` in cells/mL/h -- the effector
    state's own units -- and the fixed points are N_eff_0 with no infection
    and ``recruit_fold_max * N_eff_0`` at saturating burden. Setting
    N_eff_0 = 0 makes the host neutropenic exactly: the baseline and the
    recruited increment both vanish, so N stays at zero for all time. That is
    how the neutropenic-thigh calibration condition is imposed (see
    ``src/analysis/external_validation._neutropenic_params``).

    REPAIRED 2026-10-13 (review R10). The previous recruitment term was

        recruit = k_prod * B/(EC50_immune + B)        # k_prod = 0.5

    which is a rate constant multiplying a dimensionless saturation, so it
    carried units of 1/h where the equation needs cells/mL/h. Against a loss
    of k_deg_immune * N_eff_0 = 5e5 cells/mL/h it was six orders of magnitude
    too small -- the largest pool it could sustain was 10 cells/mL -- so the
    layer was a decaying initial pool, not a recruited response: effectors
    fell to 9% of baseline by 48 h and the immune kill rate with them, from
    0.100 to 0.009 /h, independently of bacterial burden. ``k_prod`` is gone
    rather than rescaled, because the fold-rise is the quantity with a
    meaning: tying it to N_eff_0 keeps a reduced-baseline host correctly
    scaled instead of recruiting every host to the same absolute plateau.

    recruit_fold_max is NOT identified by any data in this repository. Both
    calibration datasets are immune-free by construction (a broth time-kill
    and a *neutropenic* thigh model), so nothing here constrains it, and the
    whole layer is illustrative -- see the entries in
    ``src/analysis/provenance.py``. Its admissible band is bounded above by
    the study's own premise rather than by data: at a plateau above about
    5 x N_eff_0 the immune system alone controls an untreated infection
    (72 h burden 5.09 against 8.90 log10 CFU at fold 7), which would
    contradict modelling infections that need antibiotics and would stop the
    failing-therapy arm failing. The default sits mid-band, and
    ``scripts/ds_structural_sensitivity.py`` runs the band's ends and the old
    decaying pool.
    """
    N_eff_0: float = 1e7  # cells/mL, homeostatic effector level (0 = neutropenic)
    recruit_fold_max: float = 3.0  # x N_eff_0, plateau at saturating burden
    EC50_immune: float = 1e5  # CFU/mL, half-max for immune recruitment
    k_deg_immune: float = 0.05  # per hour, effector turnover (also the response rate)
    k_kill_base: float = 1e-8  # per mL per hour, baseline immune killing capacity

@dataclass
class CytokineParameters:
    """Inflammatory mediators"""
    # FIXED: Reduced from 10.0 to 4.0 to match literature IL-6 levels
    # At B=10^8 CFU/mL: produces ~54,000 pg/mL (literature severe pneumonia: 10,000-100,000 pg/mL)
    k_IL6_prod: float = 4.0  # pg/mL/hour per 10^6 CFU, baseline production
    alpha_static: float = 1.0  # relative cytokine production by static drugs
    alpha_cidal: float = 3.0  # relative cytokine production by cidal drugs (TLR9-mediated)
    k_IL6_clear: float = 0.2  # per hour, IL-6 degradation
    TNF_IL6_ratio: float = 0.3  # TNF produced as fraction of IL-6

@dataclass
class HostDamageParameters:
    """Host-damage (damage-response framework) parameters.

    D_host accrues injury from pathogen burden and from normalised
    inflammatory intensity (fold-change over healthy baseline), and recovers.
    Values are illustrative and swept in the sensitivity analysis.
    """
    k_path: float = 0.08   # per hour, max pathogen-driven injury rate
    B50: float = 1e7       # CFU/mL, burden at half-max pathogen injury
    k_infl: float = 0.03   # per hour, inflammation-driven injury scale
    k_heal: float = 0.10   # per hour, host recovery rate
    w_TNF: float = 0.5     # weight of TNF fold-change relative to IL-6
    IL6_ref: float = 10.0  # pg/mL, healthy baseline reference for IL-6
    TNF_ref: float = 5.0   # pg/mL, healthy baseline reference for TNF
    I50: float = 5.0       # inflammatory intensity (fold-change) at half-max injury rate

@dataclass
class PKParameters:
    """Pharmacokinetic parameters (to be populated per drug)

    Note: CL and Q are total population values in L/h (NOT per-kg).
    Vc and Vp are per-kg (L/kg) and are scaled by patient weight in the simulation.
    This convention follows the analytical PK computation which uses:
        k_elim = CL / (Vc * weight_kg)  # per hour
    """
    CL: float  # L/h, total systemic clearance (population mean for 70 kg adult)
    Vc: float  # L/kg, central volume of distribution (scaled by weight in simulation)
    Vp: float  # L/kg, peripheral volume of distribution (scaled by weight in simulation)
    Q: float   # L/h, inter-compartmental clearance (population mean for 70 kg adult)
    Ka: float  # per hour, absorption rate constant (if oral)
    Kp: float  # tissue penetration coefficient (effect-site/plasma AUC ratio)


@dataclass
class AntibioticPDParameters:
    """Drug- and isolate-specific PK/PD descriptor for PK/PD-index reporting.

    ``drug_class`` says whether the agent inhibits growth or kills; ``effect_mode``
    names the exposure pattern that drives that action (the PK/PD-index taxonomy:
    time-dependent / concentration-dependent / exposure(AUC)-dependent / static).
    Keeping the two concepts separate stops every bactericidal antibiotic from
    being assumed beta-lactam-like.

    INTEGRATION NOTE (this repository): the calibrated within-host kill lives in
    ``BacterialParameters`` (a steep-Hill, MIC~1 mg/L cidal kill that already
    behaves time-dependently) and is NOT re-parameterised by this descriptor. Of
    the fields below, only ``mic`` and ``fraction_unbound`` feed the reported
    unbound Craig indices (``SimulationResult.get_pkpd_indices``); the potency /
    signal / static fields are carried for provenance and forward compatibility
    (e.g. adding AUC-driven comparators) and are inert in the default dynamics.

    ``mic`` is deliberately a scenario parameter: the profiles from
    :func:`get_drug_pd_parameters` encode the *type* of exposure response, not a
    validated organism-specific potency, so supply the measured isolate MIC.
    """

    drug_class: str = "cidal"            # "cidal" or "static"
    effect_mode: str = "time_dependent"  # time/concentration/exposure/static
    mic: float = 1.0                     # mg/L, isolate MIC at the effect site
    fraction_unbound: float = 1.0        # unbound fraction of effect-site drug
    # Descriptive target-engagement fields (reserved; see INTEGRATION NOTE).
    k_kill_max: float = 8.0
    ec50_mic_ratio: float = 0.6
    hill: float = 4.0
    growth_dependence: float = 1.0
    signal_on: float = 1.0
    signal_decay: float = 0.3
    signal50: float = 3.0
    static_ec50_mic_ratio: float = 0.1
    static_hill: float = 1.2

    def validate(self) -> None:
        """Raise ``ValueError`` for non-physical or unsupported parameters."""
        valid_modes = {
            "time_dependent", "concentration_dependent",
            "exposure_dependent", "static",
        }
        if self.drug_class not in {"cidal", "static"}:
            raise ValueError("drug_class must be 'cidal' or 'static'")
        if self.effect_mode not in valid_modes:
            raise ValueError(f"Unsupported effect_mode: {self.effect_mode}")
        if self.mic <= 0:
            raise ValueError("mic must be > 0 mg/L")
        if not 0 < self.fraction_unbound <= 1:
            raise ValueError("fraction_unbound must be in (0, 1]")
        if self.hill <= 0 or self.static_hill <= 0:
            raise ValueError("Hill coefficients must be > 0")


def get_drug_pd_parameters(drug_name: str, mic: float = 1.0) -> AntibioticPDParameters:
    """Return a parsimonious PK/PD descriptor for a supported antibiotic.

    These profiles encode the *type* of exposure response (the effect-mode
    taxonomy), not a validated organism-specific potency estimate, so supply the
    isolate/assay ``mic`` whenever it is known. In this repository the descriptor
    drives PK/PD-index reporting and framing; see ``AntibioticPDParameters``.
    """
    name = drug_name.lower()
    if name == 'meropenem':
        # Carbapenem: rapid saturation above MIC; efficacy driven by fT>MIC.
        profile = AntibioticPDParameters(
            drug_class='cidal', effect_mode='time_dependent', mic=mic,
            fraction_unbound=0.98, k_kill_max=3.0, ec50_mic_ratio=1.0,
            hill=4.0, growth_dependence=1.0, signal_decay=0.3,
        )
    elif name == 'ciprofloxacin':
        # Fluoroquinolone: exposure/AUC-driven killing.
        profile = AntibioticPDParameters(
            drug_class='cidal', effect_mode='exposure_dependent', mic=mic,
            fraction_unbound=0.70, k_kill_max=3.0, hill=2.0,
            growth_dependence=0.25, signal_on=1.0, signal_decay=0.15, signal50=3.0,
        )
    elif name == 'vancomycin':
        # Glycopeptide: conventionally an AUC/MIC driver.
        profile = AntibioticPDParameters(
            drug_class='cidal', effect_mode='exposure_dependent', mic=mic,
            fraction_unbound=0.55, k_kill_max=1.0, hill=1.5,
            growth_dependence=0.75, signal_on=0.5, signal_decay=0.10, signal50=3.0,
        )
    elif name == 'ceftazidime':
        # PBP-3-specific cephalosporin: time-dependent killing like other
        # beta-lactams, but mechanistically distinct downstream -- PBP-3
        # inhibition produces filaments rather than rapid lysis, which is why it
        # is the high-danger-signal-yield comparator in the danger-signal study
        # (see docs/danger_signal_study/DESIGN.md).
        profile = AntibioticPDParameters(
            drug_class='cidal', effect_mode='time_dependent', mic=mic,
            fraction_unbound=0.90, k_kill_max=3.0, ec50_mic_ratio=1.0,
            hill=4.0, growth_dependence=1.0, signal_decay=0.3,
        )
    elif name in {'doxycycline', 'linezolid', 'tigecycline'}:
        # Bacteriostatic: growth inhibition with immune-dependent clearance.
        profile = AntibioticPDParameters(
            drug_class='static', effect_mode='static', mic=mic,
            fraction_unbound=1.0, static_ec50_mic_ratio=1.0,
            static_hill=1.2, growth_dependence=0.0,
        )
    else:
        raise ValueError(f"Unknown drug: {drug_name}")
    profile.validate()
    return profile


def get_default_parameters() -> Dict:
    """
    Returns complete default parameter set
    """
    return {
        'bacteria': BacterialParameters(),
        'immune': ImmuneParameters(),
        'cytokine': CytokineParameters(),
        'damage': HostDamageParameters(),
        # Drug-specific PK will be loaded separately
    }

def immune_free_parameters(params: Dict = None) -> Dict:
    """Parameters with the effector pool switched off (in-vitro or neutropenic).

    Sets ``N_eff_0 = 0``, which zeroes the homeostatic target and the
    recruited increment together, so the pool decays to exactly zero and
    stays there. Callers should also start ``N_eff`` at zero; the two are not
    interchangeable, and which one is missing decides how it fails:

      * the set-point alone does not empty a pool that starts full;
      * the initial condition alone is refilled by homeostasis, which is what
        would silently re-immunise the broth time-kill and the neutropenic
        thigh model the bacterial layer is calibrated against.

    Before the R10 repair neither mattered, because recruitment was six
    orders of magnitude too weak to refill anything and the pool merely
    decayed. Every site that wants no immune contribution must now say so.

    Returns a new dict with a new ImmuneParameters; the caller's objects are
    left alone, because several callers pass in a parameter set they go on to
    use for immunocompetent runs as well.
    """
    src = get_default_parameters() if params is None else params
    p = dict(src)
    p['immune'] = dataclasses.replace(src['immune'], N_eff_0=0.0)
    return p

def get_drug_pk_parameters(drug_name: str) -> PKParameters:
    """
    Return population mean PK parameters for selected drugs.

    CL and Q are total values in L/h (population mean for 70 kg adult).
    Vc and Vp are per-kg in L/kg (scaled by patient weight in simulation).

    References:
        Meropenem: Mouton & van den Anker 1995, Clin Pharmacokinet 28:275-286
            (Vss ~21 L = 0.30 L/kg at 70 kg, t1/2 ~1 h, Cmax ~30 mg/L after
            1 g IV; CL 9-15 L/h). NOTE: Vc + Vp below is 0.40 L/kg, above the
            cited 0.30 -- see S2.1 and the PK validation result in S3.
        Ceftazidime: Leroy 1984, AAC 25:638-642 (PMID 6375562; normal
            subjects n=5: t1/2 1.57 h, central volume 0.127 L/kg, apparent
            Vd 0.230 L/kg, protein binding 17%). fraction_unbound = 0.90 below
            follows the product label, NOT Leroy, whose 17% implies 0.83.
            "Nix 1991" was cited here for disposition in error: that paper
            (AAC 35:1947 and 1953) is on tissue penetration and belongs to
            Kp, where it is now cited.
        Doxycycline: Agwuh & MacGowan 2006 (CL 3-8 L/h, Vd 0.5-1.0 L/kg)
        Linezolid: Blevins 2003 (CL 3-6 L/h, Vd 0.6-0.8 L/kg)
        Ciprofloxacin: Lettieri 1992 (CL 15-30 L/h, Vd 2.0 L/kg)
    """
    if drug_name.lower() == 'doxycycline':
        # Oral doxycycline: 100-200 mg q12-24h
        # Literature (Agwuh & MacGowan 2006): 200 mg gives Cmax ~3-5 mg/L at
        #   tmax ~2-3 h, t1/2 ~16-18 h, AUC ~90-113 mg.h/L. The cited AUC fixes the
        #   apparent oral clearance: CL/F = Dose/AUC = 200/100 ~= 2.0 L/h (an earlier
        #   code comment quoting "3-8 L/h" is inconsistent with that AUC and the
        #   18 h half-life). CL set to 1.8 L/h, which places t1/2 (16.2 h at 70 kg,
        #   = 0.693 * Vc*weight / CL), AUC0-inf (~111 mg.h/L) and Cmax (~4.3 mg/L)
        #   all within the published ranges. The prior CL=5.0 gave only ~5.8 h, an
        #   under-prediction of the observed 24 h tail. Ka raised 0.5 -> 1.3 /h to
        #   reproduce the observed ~2.7 h tmax.
        return PKParameters(
            CL=1.8,        # L/h apparent CL/F (consistent with AUC ~90-113 and t1/2 ~16-18 h)
            Vc=0.6,        # L/kg (literature: 0.5-1.0 L/kg)
            Vp=1.2,        # L/kg
            Q=5.0,         # L/h (estimated)
            Ka=1.3,        # per hour (oral absorption; tmax ~2.7 h)
            Kp=0.7         # lung penetration ratio
        )
    elif drug_name.lower() == 'meropenem':
        # IV meropenem: 500-2000 mg q6-8h
        # Literature: CL 9-15 L/h, Vd ~0.3 L/kg, t1/2 ~1 h, lung penetration moderate
        return PKParameters(
            CL=15.0,       # L/h (total; literature: 9-15 L/h for normal renal function)
            Vc=0.25,       # L/kg (literature: 0.2-0.3 L/kg)
            Vp=0.15,       # L/kg
            Q=8.0,         # L/h (estimated)
            Ka=0.0,        # IV only
            Kp=0.4         # lung penetration ratio (literature: 0.3-0.5)
        )
    elif drug_name.lower() == 'ceftazidime':
        # IV ceftazidime: 1-2 g q8h
        # Literature (Leroy 1984; FDA label): CL 6-8 L/h with normal renal
        # function, Vd(ss) ~0.21-0.28 L/kg, t1/2 ~1.6-2.0 h. 1 g IV 30 min
        # infusion gives Cmax ~65-80 mg/L. Protein binding ~10% is the
        # LABEL value; Leroy measured 17%. Kp below is where Nix 1991
        # (tissue penetration) belongs.
        # Required for the danger-signal study as the PBP-3 (high-yield)
        # comparator against meropenem/imipenem (PBP-2).
        return PKParameters(
            CL=7.0,        # L/h (total; literature 6-8 L/h)
            Vc=0.17,       # L/kg (central; Leroy 1984 measured 0.127 L/kg)
            Vp=0.09,       # L/kg (Vc + Vp = 0.26, inside Leroy's 0.230 +/- range)
            Q=5.0,         # L/h (estimated)
            Ka=0.0,        # IV only
            Kp=0.5         # tissue penetration ratio (Nix 1991, AAC 35:1947/1953)
        )
    elif drug_name.lower() == 'linezolid':
        # IV/PO linezolid: 600 mg q12h
        # Literature: half-life 4-5h, Vd ~0.6-0.8 L/kg, lung penetration good
        return PKParameters(
            CL=5.0,        # L/h (total; literature: 3-6 L/h)
            Vc=0.65,       # L/kg (literature: 0.6-0.8 L/kg)
            Vp=0.35,       # L/kg
            Q=4.0,         # L/h (estimated)
            Ka=0.4,        # per hour (if oral)
            Kp=0.75        # lung penetration ratio (literature: 0.6-0.8)
        )
    elif drug_name.lower() == 'ciprofloxacin':
        # Fluoroquinolone: 400 mg IV or 500-750 mg PO q12h
        # Literature: CL 15-30 L/h, Vd ~2.0 L/kg
        return PKParameters(
            CL=22.0,       # L/h (total; literature: 15-30 L/h)
            Vc=2.0,        # L/kg (literature: 1.5-2.5 L/kg)
            Vp=1.0,        # L/kg
            Q=10.0,        # L/h (estimated)
            Ka=0.8,        # per hour (oral)
            Kp=0.6         # moderate lung penetration
        )
    else:
        raise ValueError(f"Unknown drug: {drug_name}")

def normalize_pk_parameters(params: PKParameters, weight_kg: float) -> Dict:
    """
    Scale per-kg PK parameters (Vc, Vp) by patient weight.
    CL and Q are already total values (L/h) and are not scaled.

    Returns:
        Dict with keys 'CL' (L/h), 'Vc' (L), 'Vp' (L), 'Q' (L/h), 'Ka', 'Kp'
    """
    return {
        'CL': params.CL,          # L/h (already total, not scaled)
        'Vc': params.Vc * weight_kg,  # L (scaled from L/kg)
        'Vp': params.Vp * weight_kg,  # L (scaled from L/kg)
        'Q': params.Q,            # L/h (already total, not scaled)
        'Ka': params.Ka,
        'Kp': params.Kp,
    }
