"""Held-out validation of the danger-signal model (Phase 3).

The design document pre-registered that Layer-3 claims would be validated on
**direction, not magnitude**, against independent clinical and ex-vivo studies.
This module encodes those studies as explicit, checkable constraints so that a
validation cannot quietly become a fit.

Each constraint records what the source actually reports, what the model must
predict to pass, and -- importantly -- what the source does *not* establish.
Three of the four constraints are ordinal. The fourth is a magnitude *bound*,
and it is the one that matters most for Phase 3, because it is the only
independent check on the amplification factor that the calibrated Layer 2
produces.

None of these sources was used to fit any parameter.

A note on what is deliberately absent. The pre-registered Layer-1 held-out test
(predicting Jackson & Kropp's 10-40x ceftazidime:imipenem free-LPS ratio from
yields calibrated on Yamaguchi) cannot be run yet and is NOT run here. The only
imipenem yield currently available comes from Jackson & Kropp itself, so
testing against that paper would be circular. It becomes a real test once
Yamaguchi's imipenem curve is digitized. Saying so is better than running a
test that cannot fail.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class OrdinalConstraint:
    """A direction that the model must reproduce, with its provenance."""

    name: str
    source: str
    doi: str
    design: str
    n: Optional[int]
    reports: str
    #: Arms in the order the source puts them, least to most inflammatory.
    expected_order: Sequence[str]
    quantities: Sequence[str]
    caveat: str = ""

    def evaluate(self, scores: Dict[str, float]) -> Dict[str, object]:
        """Check the model's ranking against the reported direction.

        ``scores`` maps arm label to the model's predicted inflammatory
        quantity (higher = more inflammatory).
        """
        missing = [a for a in self.expected_order if a not in scores]
        if missing:
            return {"name": self.name, "evaluated": False,
                    "reason": f"no model prediction for {missing}"}
        predicted = sorted(self.expected_order, key=lambda a: scores[a])
        return {
            "name": self.name,
            "evaluated": True,
            "source": self.source,
            "doi": self.doi,
            "n": self.n,
            "expected_order": list(self.expected_order),
            "predicted_order": list(predicted),
            "scores": {a: scores[a] for a in self.expected_order},
            "pass": bool(list(predicted) == list(self.expected_order)),
            "caveat": self.caveat,
        }


#: Prins JM et al. J Infect Dis 1995;172:886-91. The pre-registered Layer-3
#: clinical held-out set.
PRINS_1995_CLINICAL = OrdinalConstraint(
    name="prins1995_urosepsis",
    source="Prins JM, van Agtmael MA, Kuijper EJ, van Deventer SJ, Speelman P. "
           "J Infect Dis 1995;172(3):886-91",
    doi="10.1093/infdis/172.3.886",
    design="double-blind randomized, gram-negative urosepsis, imipenem vs ceftazidime",
    n=30,
    reports="Slower defervescence with ceftazidime. After 4 h, blood endotoxin "
            "fell in all 3 endotoxemic imipenem patients but rose in 2 of 4 "
            "endotoxemic ceftazidime patients; urine endotoxin fell less with "
            "ceftazidime. Serum and urine cytokines rose 10-40% at 4 h with "
            "ceftazidime versus no rise with imipenem.",
    expected_order=("imipenem", "ceftazidime"),
    quantities=("blood endotoxin", "urine endotoxin", "serum/urine cytokines",
                "time to defervescence"),
    caveat="The cytokine difference was NOT statistically significant (P > .05) "
           "at n=30, so this supports direction only. The 10-40% figure is the "
           "magnitude bound used separately.",
)

#: Simpson AJ et al. J Infect Dis 2000;181:1014-9. Not in the original design;
#: found during Phase 3 and a better ordinal test than Toky 2005 -- larger,
#: randomized, and explicitly framed on PBP specificity.
SIMPSON_2000_MELIOIDOSIS = OrdinalConstraint(
    name="simpson2000_melioidosis",
    source="Simpson AJ, Opal SM, Angus BJ, Prins JM, Palardy JE, Parejo NA, "
           "Chaowagul W, White NJ. J Infect Dis 2000;181(3):1014-9",
    doi="10.1086/315306",
    design="prospective randomized, severe melioidosis (Burkholderia pseudomallei), "
           "ceftazidime (PBP-3) vs imipenem (PBP-2)",
    n=68,
    reports="Despite similar clinical, microbiological, endotoxin and cytokine "
            "measures at entry, ceftazidime-treated patients had significantly "
            "greater systemic endotoxin (P < .001) after the first dose. No "
            "difference in mortality (35% in both arms).",
    expected_order=("imipenem", "ceftazidime"),
    quantities=("systemic endotoxin",),
    caveat="ENDOTOXIN ONLY. Reading the paper's tables (see "
           "SIMPSON_2000_TABLES) shows the cytokine arm does not support the "
           "model and numerically trends the other way: ceftazidime's median "
           "peak IL-6 was LOWER than imipenem's (291.9 vs 594.8 pg/mL; "
           "P = .36), and peak cytokines did not correlate with peak endotoxin "
           "(IL-6 r = .23, P = .18). Earlier phases of this project cited this "
           "study as supporting the direction in 'endotoxin and cytokines'; "
           "that was wrong and is corrected here. The endotoxin difference "
           "also did NOT translate into a survival difference, so the model "
           "must never be read as predicting mortality.",
)

#: Simpson 2000, tables 2, 3 and 5, read from the published article. Medians
#: with interquartile ranges, pg/mL. These are the only in-patient
#: measurements of BOTH endotoxin and cytokines by agent that this study has,
#: and they are what Phase 5 calibrates the tolerance term against.
SIMPSON_2000_TABLES = {
    "source": SIMPSON_2000_MELIOIDOSIS.source,
    "doi": SIMPSON_2000_MELIOIDOSIS.doi,
    "n_per_arm": 34,
    "units": "pg/mL, median (interquartile range)",
    "assay_detection_limits": {"endotoxin": 2.0, "TNF": 3.0, "IL6": 5.0,
                               "IL10": 4.0},
    # Table 3 -- baseline (pre-treatment), by drug
    "baseline_by_drug": {
        "ceftazidime": {"TNF": (0.0, 0.0, 19.7), "IL6": (211.6, 71.6, 730.1),
                        "IL10": (39.5, 12.6, 143.7),
                        "endotoxin": (65.5, 28.4, 130.8)},
        "imipenem": {"TNF": (3.7, 0.0, 14.3), "IL6": (223.9, 99.2, 1021.6),
                     "IL10": (38.1, 14.0, 151.5),
                     "endotoxin": (28.2, 5.8, 146.0)},
        "p_values": {"TNF": 0.59, "IL6": 0.33, "IL10": 0.62, "endotoxin": 0.12},
    },
    # Table 5 -- peak after the first dose, by drug
    "peak_by_drug": {
        "ceftazidime": {"TNF": (10.0, 0.0, 37.6), "IL6": (291.9, 139.0, 1366.0),
                        "IL10": (36.9, 10.8, 301.6),
                        "endotoxin": (725.8, 231.8, 2477.2)},
        "imipenem": {"TNF": (13.7, 3.1, 30.9), "IL6": (594.8, 70.8, 4305.0),
                     "IL10": (39.6, 12.8, 348.1),
                     "endotoxin": (146.0, 54.2, 895.0)},
        "p_values": {"TNF": 0.62, "IL6": 0.36, "IL10": 0.74, "endotoxin": 0.008},
    },
    # Table 2 -- baseline by outcome. Note what it says: every cytokine
    # predicts death, and endotoxin does not.
    "baseline_by_outcome": {
        "survivors_n44": {"TNF": (0.0, 0.0, 3.7), "IL6": (116.5, 43.4, 245.2),
                          "IL10": (18.9, 9.1, 41.1),
                          "endotoxin": (58.4, 13.8, 170.3)},
        "nonsurvivors_n24": {"TNF": (14.4, 5.2, 54.0), "IL6": (825.2, 228.8, 2829.0),
                             "IL10": (191.9, 65.6, 526.6),
                             "endotoxin": (34.0, 10.8, 146.0)},
        "p_values": {"TNF": 0.001, "IL6": 0.001, "IL10": 0.001, "endotoxin": 0.77},
        "note": "The printed TNF interquartile range for nonsurvivors reads "
                "'5.2-5.4', which cannot bracket a median of 14.4; recorded "
                "here as 5.2-54 on the assumption of a typesetting error. "
                "Nothing in this study depends on it.",
    },
    "derived": {
        "peak_endotoxin_ratio_caz_over_ipm": 725.8 / 146.0,
        "caz_attributable_IL6_rise_pg_mL": 291.9 - 211.6,
        "caz_attributable_IL6_rise_pct": 100.0 * (291.9 - 211.6) / 211.6,
        "ipm_attributable_IL6_rise_pct": 100.0 * (594.8 - 223.9) / 223.9,
    },
    "what_it_supports": "Endotoxin ONLY. Ceftazidime produced ~5x the peak "
                        "endotoxin of imipenem (P = .008), confirming the "
                        "direction and giving the first in-patient estimate of "
                        "the size of the agent effect.",
    "what_it_contradicts":
        "The cytokine arm of the model. Despite ~5x the endotoxin, "
        "ceftazidime's median peak IL-6 was LOWER than imipenem's (291.9 vs "
        "594.8; P = .36), and peak cytokines did not correlate with peak "
        "endotoxin in any of the three cytokines (IL-6 r = .23, P = .18). The "
        "cytokine comparison is noise-dominated -- the interquartile ranges "
        "span more than a decade -- but it certainly does not support the "
        "model's direction, and it must not be cited as if it did.",
}

#: Prins JM et al. Infect Immun 1995;63:2236-42. Ex vivo whole blood: couples
#: Layer 1 to Layer 2 directly, because it measures endotoxin AND the cytokines
#: it induces, from the same cultures, for four agents at three concentrations.
PRINS_1995_WHOLE_BLOOD = OrdinalConstraint(
    name="prins1995_whole_blood",
    source="Prins JM, Kuijper EJ, Mevissen ML, Speelman P, van Deventer SJ. "
           "Infect Immun 1995;63(6):2236-42",
    doi="10.1128/iai.63.6.2236-2242.1995",
    design="E. coli killed ex vivo in human whole blood at 0.5, 5 and 50x MIC; "
           "endotoxin, TNF-alpha and IL-6 measured at 4 and 8 h",
    n=None,
    reports="At 50x MIC, ceftazidime and ciprofloxacin produced significantly "
            "higher endotoxin, TNF-alpha and IL-6 than imipenem and gentamicin "
            "(P < 0.001), with the same ordering at 8 h. Polymyxin B abolished "
            "TNF release, confirming the cytokine differences were endotoxin-"
            "mediated.",
    expected_order=("imipenem", "ceftazidime"),
    quantities=("free endotoxin", "TNF-alpha", "IL-6"),
    caveat="Closed ex-vivo system with no danger-signal clearance, so only the "
           "ordering transfers, not the absolute concentrations.",
)

#: The sub-MIC observation from the same paper, which independently bears on
#: the Phase 1 negative result about kill-independent release.
PRINS_SUB_MIC = {
    "source": PRINS_1995_WHOLE_BLOOD.source,
    "doi": PRINS_1995_WHOLE_BLOOD.doi,
    "reports": "At 0.5x MIC the differences between antibiotics in endotoxin "
               "and cytokine levels were small, with levels comparable to "
               "untreated cultures.",
    "model_must_predict": "sub-MIC exposure produces little cytokine induction "
                          "relative to an untreated control, and little "
                          "separation between agents",
    "bears_on": "Phase 1 concluded that the sub-MIC filamentation pathway "
                "contributes <=1% of danger-signal release at conventional "
                "exposures. This is independent corroboration from a different "
                "readout (cytokine induction rather than release kinetics) and "
                "a different laboratory.",
}

#: The magnitude bound. This is the only independent constraint on the
#: amplification factor, and it is in tension with the model, which is exactly
#: why it is recorded explicitly rather than folded into a narrative.
PRINS_MAGNITUDE_BOUND = {
    "source": PRINS_1995_CLINICAL.source,
    "doi": PRINS_1995_CLINICAL.doi,
    "observed_cytokine_rise_pct": (10.0, 40.0),
    "window_h": 4.0,
    "comparator": "no rise in the imipenem arm",
    "significance": "P > .05 (not significant at n=30)",
    "interpretation": "The ceftazidime-attributable cytokine rise in treated "
                      "gram-negative urosepsis is a modest fraction of the "
                      "patient's pre-treatment level, not a multi-fold "
                      "increase.",
}

#: Plasma IL-6 in gram-negative sepsis, for converting a relative rise into an
#: absolute one. Deliberately wide, and the conclusion is reported across the
#: whole interval rather than at a point.
SEPSIS_IL6_RANGE_PG_ML = (100.0, 2000.0)


def implied_pretreatment_il6(attributable_peak_pg_ml: float,
                             rise_pct_range=(10.0, 40.0)) -> Dict[str, float]:
    """Invert Prins' relative rise into the pre-treatment IL-6 it implies.

    If the model says therapy adds ``attributable_peak_pg_ml`` of IL-6, and the
    clinical observation is that ceftazidime adds 10-40% to the existing level,
    then the patient's pre-treatment IL-6 must have been
    ``attributable / (rise/100)``. Comparing that implied value against the
    observed range for gram-negative sepsis is a consistency test on the
    model's absolute magnitude that needs no new data.
    """
    hi_pct, lo_pct = max(rise_pct_range), min(rise_pct_range)
    return {
        "implied_pretreatment_IL6_low_pg_mL": attributable_peak_pg_ml / (hi_pct / 100.0),
        "implied_pretreatment_IL6_high_pg_mL": attributable_peak_pg_ml / (lo_pct / 100.0),
    }


def check_magnitude_consistency(attributable_peak_pg_ml: float,
                                sepsis_range=SEPSIS_IL6_RANGE_PG_ML
                                ) -> Dict[str, object]:
    """Is the model's absolute attributable IL-6 compatible with Prins?

    Passing means the implied pre-treatment IL-6 overlaps the observed range
    for gram-negative sepsis. Failing means the model's absolute magnitude is
    too large (or too small) to be reconciled with the clinical observation,
    and the structural scale parameters -- focus volume and translocation
    fraction, neither of which is calibrated -- are the terms that must absorb
    the difference.
    """
    imp = implied_pretreatment_il6(attributable_peak_pg_ml)
    lo, hi = imp["implied_pretreatment_IL6_low_pg_mL"], imp["implied_pretreatment_IL6_high_pg_mL"]
    s_lo, s_hi = sepsis_range
    overlap_lo, overlap_hi = max(lo, s_lo), min(hi, s_hi)
    overlap = overlap_hi > overlap_lo
    # How much of the implied interval is actually compatible? A bare sliver of
    # overlap is not the same as agreement, and reporting it as a pass would
    # hide the fact that the model's magnitude sits at the very edge of what
    # the clinical observation can accommodate.
    frac = (max(overlap_hi - overlap_lo, 0.0) / (hi - lo)) if hi > lo else 0.0
    return {
        "attributable_IL6_peak_pg_mL": attributable_peak_pg_ml,
        **imp,
        "sepsis_IL6_range_pg_mL": list(sepsis_range),
        "overlaps_observed_sepsis_range": bool(overlap),
        "overlap_fraction_of_implied_interval": float(frac),
        "marginal": bool(overlap and frac < 0.30),
        "verdict": ("over-predicts" if lo > s_hi else
                    "under-predicts" if hi < s_lo else
                    "marginally consistent, biased high" if frac < 0.30 and hi > s_hi else
                    "marginally consistent, biased low" if frac < 0.30 else
                    "consistent"),
        "over_predicts": bool(lo > s_hi),
        "under_predicts": bool(hi < s_lo),
        "required_scale_correction": (float(s_hi / lo) if lo > s_hi else
                                      (float(s_lo / hi) if hi < s_lo else 1.0)),
    }


#: Trautmann M et al. J Antimicrob Chemother 1998;41:163-9, tables II and III,
#: read from the article. Free and cell-associated LPS in mg/L, mean +/- SEM
#: of 3-5 experiments, E. coli strain Bort, LPS capture ELISA (monoclonal,
#: NOT Limulus). This is the single most useful Layer-1 dataset found: one
#: study, one assay, one organism, five agents, two exposure levels, with a
#: measured inoculum.
TRAUTMANN_1998 = {
    "source": "Trautmann M, Heinemann M, Zick R, Moricke A, Seidelmann M, "
              "Berger D. J Antimicrob Chemother 1998;41(2):163-9",
    "doi": "10.1093/jac/41.2.163",
    "organism": "E. coli Bort (O18)",
    "assay": "LPS capture ELISA, monoclonal anti-O18; detection limit 2 ng/mL; "
             "verified free of antibiotic interference (their table I)",
    "inoculum_cfu_per_mL": 5e5,
    "broth_volume_mL": 20.0,
    "mic_mg_L": {"ceftazidime": 0.125, "meropenem": 0.015, "imipenem": 0.125,
                 "tobramycin": 1.0, "ciprofloxacin": 0.0078},
    "units": "mg/L LPS",
    # Free LPS, by time (h), at 2x and 50x MIC.
    "free_LPS_2xMIC": {
        "time_h": [0, 2, 4, 6, 8],
        "no_antibiotic": [0.01, 0.11, 0.5, 4.9, 5.3],
        "meropenem": [None, 0.03, 0.08, 0.15, 0.17],
        "imipenem": [None, 0.05, 0.13, 0.29, 0.29],
        "ceftazidime": [None, 0.45, 0.49, 0.53, 0.71],
        "ciprofloxacin": [None, 0.03, 0.06, 0.24, 0.31],
        "tobramycin": [None, 0.03, 0.03, 0.04, 0.05],
    },
    "free_LPS_50xMIC": {
        "time_h": [0, 2, 4, 6, 8],
        "no_antibiotic": [0.04, 0.07, 0.74, 4.7, 5.9],
        "meropenem": [None, 0.05, 0.06, 0.10, 0.13],
        "imipenem": [None, 0.02, 0.03, 0.08, 0.09],
        "ceftazidime": [None, 0.15, 0.39, 0.38, 0.62],
        "ciprofloxacin": [None, 0.07, 0.06, 0.18, 0.12],
    },
    "cell_associated_LPS_50xMIC": {
        "time_h": [0, 2, 4, 6, 8],
        "no_antibiotic": [0.6, 4.5, 9.0, 13.9, 14.8],
        "meropenem": [None, 1.0, 0.7, 1.1, 1.2],
        "imipenem": [None, 0.7, 0.6, 0.5, 0.5],
        "ceftazidime": [None, 3.7, 2.5, 3.1, 2.8],
        "ciprofloxacin": [None, 0.6, 0.6, 0.7, 0.8],
    },
    "derived": {
        "caz_over_imipenem_free_8h_50x": 0.62 / 0.09,
        "caz_over_meropenem_free_8h_50x": 0.62 / 0.13,
        "caz_over_imipenem_free_8h_2x": 0.71 / 0.29,
    },
    "key_findings": [
        "Ceftazidime released ~5-7x the free LPS of the carbapenems at 50x "
        "MIC -- NOT the 10-40x that Jackson & Kropp reported by Limulus.",
        "Ciprofloxacin induced FILAMENTS yet released as little free LPS as "
        "the carbapenems, with killing curves nearly identical to the "
        "beta-lactams. Morphology therefore does not predict release, and the "
        "mechanism must be PBP-3/cell-wall-specific rather than filamentation "
        "as such.",
        "Meropenem behaved like imipenem despite binding PBP-3 as well as "
        "PBP-2 -- low release either way.",
        "THE UNTREATED CONTROL RELEASED MORE FREE LPS THAN ANY TREATED ARM "
        "(5.9 mg/L at 8 h, 50x MIC, against 0.09-0.62 for the antibiotics). "
        "Release tracks biomass, so failing to control the infection costs "
        "more danger signal than any agent choice -- direct experimental "
        "support for the Phase 4 finding that under-dosing is bad for both "
        "objectives at once.",
        "The authors attribute the higher free-LPS values of Limulus-based "
        "studies to known LAL hypersensitivity for free LPS, citing "
        "Mattsby-Baltzer. That is the most likely explanation for Jackson & "
        "Kropp's 10-40x.",
    ],
}

#: Prins JM et al. Infect Immun 1995, results read from the article. Whole
#: blood ex vivo, E. coli O4:K2, 4 h, Limulus endotoxin and ELISA cytokines.
PRINS_1995_WHOLE_BLOOD_DATA = {
    "source": PRINS_1995_WHOLE_BLOOD.source,
    "doi": PRINS_1995_WHOLE_BLOOD.doi,
    "organism": "E. coli O4:K2",
    "bacteriology": "untreated grew 120-fold in 4 h to 1.2e5 CFU/mL; at 5x and "
                    "50x MIC counts fell below 10 CFU/mL by 4 h; at 0.5x MIC "
                    "4- to 12-fold growth occurred. The antibiotics did NOT "
                    "differ in their ability to inhibit growth -- i.e. killing "
                    "was matched, as this study's model assumes.",
    "detection_limits": {"endotoxin_EU_per_mL": 0.036, "TNF_pg_per_mL": 50.0,
                         "IL6_pg_per_mL": 3.0},
    "TNF_4h_50xMIC_pg_per_mL": {
        "no_antibiotic": (4941.0, 1044.0),
        "ceftazidime": (7678.0, 1093.0),
        "imipenem": (2203.0, 446.0),
        "gentamicin": (3173.0, 248.0),
        "ciprofloxacin": (7267.0, 644.0),
        "p_value": "<0.001",
        "format": "(mean, SEM)",
    },
    "derived": {"caz_over_imipenem_TNF_4h_50x": 7678.0 / 2203.0},
    "key_findings": [
        "Ceftazidime induced ~3.5x the TNF of imipenem at matched killing.",
        "THE UNTREATED CONTROL (4941 pg/mL) INDUCED MORE TNF THAN IMIPENEM "
        "(2203) OR GENTAMICIN (3173). Only ceftazidime and ciprofloxacin "
        "exceeded it. Effective low-yield therapy is anti-inflammatory "
        "relative to no therapy -- the same conclusion as Trautmann's control.",
        "At 0.5x MIC, endotoxin and TNF did not differ significantly between "
        "agents or from untreated cultures, which is the sub-MIC observation "
        "that conflicts with Jackson & Kropp.",
        "Ciprofloxacin induced nearly as much TNF as ceftazidime here, which "
        "CONTRADICTS Trautmann's ELISA finding that ciprofloxacin releases "
        "little free LPS. The discrepancy is assay-linked: this study used "
        "Limulus, Trautmann used a monoclonal ELISA and argues that LAL "
        "overestimates free LPS.",
    ],
}

#: The agent effect size, from every source that measures it. The in-vitro
#: ELISA, the ex-vivo whole-blood cytokine readout and the in-patient plasma
#: measurement agree on roughly 3.5-7x. Jackson & Kropp's 10-40x -- the
#: interval this project has been carrying since Phase 1 -- is the outlier,
#: and is the one most likely to be assay-inflated.
AGENT_EFFECT_ESTIMATES = {
    "trautmann1998_free_LPS_elisa_invitro": 6.9,
    "prins1995_TNF_wholeblood_exvivo": 3.5,
    "simpson2000_plasma_endotoxin_patients": 4.97,
    "jackson1992_free_LPS_limulus_invitro": (10.0, 40.0),
    "recommended_interval": (3.5, 7.0),
    "note": "Three independent estimates by three methods in three settings "
            "cluster at 3.5-7x. The 10-40x interval should be retired as the "
            "primary uncertainty range and reported as an outlier with its "
            "likely assay explanation.",
}


ORDINAL_CONSTRAINTS: List[OrdinalConstraint] = [
    PRINS_1995_CLINICAL, SIMPSON_2000_MELIOIDOSIS, PRINS_1995_WHOLE_BLOOD,
]


def evaluate_all(scores: Dict[str, float]) -> Dict[str, object]:
    """Evaluate every ordinal constraint against one set of model scores."""
    results = [c.evaluate(scores) for c in ORDINAL_CONSTRAINTS]
    done = [r for r in results if r.get("evaluated")]
    return {
        "results": results,
        "n_evaluated": len(done),
        "n_passed": sum(1 for r in done if r["pass"]),
        "all_pass": bool(done and all(r["pass"] for r in done)),
    }
