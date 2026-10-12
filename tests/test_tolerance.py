"""
Tests for the pre-treatment danger-signal baseline and the endotoxin-tolerance
term (Phase 5), and for the Simpson 2000 table data they are calibrated on.

The tolerance term FAILED its acceptance test and ships disabled. These tests
therefore guard two different things: that it stays disabled by default so
earlier phases are unaffected, and that it still behaves correctly when
switched on, so the failed analysis remains reproducible.
"""
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.cytokine_calibration import (
    simulate_endotoxemia, steady_state_tolerance,
)
from src.analysis.danger_signal import (
    DangerSignalParameters, simulate_danger_signal, summarize,
)
from src.analysis.held_out_validation import (
    SIMPSON_2000_MELIOIDOSIS, SIMPSON_2000_TABLES,
)


def _driver(n=600, t_end=48.0, kill_rate=1.0, B=1e7, C=4.0):
    t = np.linspace(0, t_end, n)
    return dict(t=t, B_rep=np.full(n, B), kill=np.full(n, kill_rate),
                C_eff=np.full(n, C))


class TestToleranceDisabledByDefault:
    def test_default_parameters_make_tolerance_inert(self):
        """Phases 1-4 were run without tolerance. It must stay off unless a
        caller asks for it, or those results silently change."""
        p = DangerSignalParameters()
        assert p.k_tol_on == 0.0
        assert p.tol_max == 0.0
        assert p.tol_shift == 0.0

    def test_tolerance_state_stays_zero_by_default(self):
        d = _driver()
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     1.0, DangerSignalParameters(Y_lysis=1.0))
        assert float(np.max(out["tolerance"])) == pytest.approx(0.0, abs=1e-12)

    def test_default_baseline_is_zero(self):
        assert DangerSignalParameters().DS_baseline == 0.0


class TestPretreatmentBaseline:
    def test_baseline_is_held_without_therapy(self):
        """With no killing and no filamentation, plasma danger signal must sit
        at the pre-treatment level the untreated infection sustains."""
        d = _driver(kill_rate=0.0)
        p = DangerSignalParameters(Y_lysis=0.0, Y_filament=0.0,
                                   DS_baseline=65.5)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                     d["C_eff"], 1.0, p)
        assert out["DS"][0] == pytest.approx(65.5, rel=1e-6)
        assert out["DS"][-1] == pytest.approx(65.5, rel=1e-3)

    def test_therapy_adds_on_top_of_the_baseline(self):
        d = _driver(kill_rate=1.0)
        p = DangerSignalParameters(Y_lysis=5.0, DS_baseline=65.5)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                     d["C_eff"], 1.0, p)
        assert float(np.max(out["DS"])) > 65.5

    def test_total_auc_stops_discriminating_but_excess_does_not(self):
        """The reason AUC_DS_excess exists. A large shared background swamps
        the total, exactly as the constant IL-6 baseline did in Phase 2."""
        d = _driver(kill_rate=1.0)
        tot, exc = [], []
        for y in (0.25, 4.93):
            p = DangerSignalParameters(Y_lysis=y, DS_baseline=65.5)
            s = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                                 d["C_eff"], 1.0, p))
            tot.append(s["AUC_DS_pg_h_mL"])
            exc.append(s["AUC_DS_excess_pg_h_mL"])
        assert tot[1] / tot[0] < 1.5, "total AUC should be swamped"
        assert exc[1] / exc[0] > 15.0, "attributable AUC must still separate"

    def test_excess_is_zero_without_release(self):
        d = _driver(kill_rate=0.0)
        p = DangerSignalParameters(Y_lysis=0.0, Y_filament=0.0,
                                   DS_baseline=65.5)
        s = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                             d["C_eff"], 1.0, p))
        assert s["AUC_DS_excess_pg_h_mL"] == pytest.approx(0.0, abs=1e-6)


class TestToleranceWhenEnabled:
    def test_steady_state_formula(self):
        p = DangerSignalParameters(k_tol_on=34.89, k_tol_off=0.35)
        T = steady_state_tolerance(65.5, p)
        assert 0.0 < T < 1.0
        # T = k_on*occ / (k_on*occ + k_off)
        from src.analysis.danger_signal import hill_occupancy
        occ = float(hill_occupancy(65.5, p))
        assert T == pytest.approx(34.89 * occ / (34.89 * occ + 0.35), rel=1e-9)

    def test_zero_induction_gives_zero_tolerance(self):
        p = DangerSignalParameters(k_tol_on=0.0)
        assert steady_state_tolerance(1e4, p) == pytest.approx(0.0)

    def test_output_scaling_suppresses_the_response(self):
        d = _driver()
        base = DangerSignalParameters(Y_lysis=4.93)
        tol = replace(base, k_tol_on=34.89, tol_max=1.0)
        a = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                             d["C_eff"], 1.0, base))
        b = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                             d["C_eff"], 1.0, tol))
        assert b["AUC_IL6_excess_pg_h_mL"] < a["AUC_IL6_excess_pg_h_mL"]

    def test_dose_shift_suppresses_the_response(self):
        d = _driver()
        base = DangerSignalParameters(Y_lysis=4.93)
        tol = replace(base, k_tol_on=34.89, tol_shift=50.0)
        a = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                             d["C_eff"], 1.0, base))
        b = summarize(simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                             d["C_eff"], 1.0, tol))
        assert b["AUC_IL6_excess_pg_h_mL"] < a["AUC_IL6_excess_pg_h_mL"]

    def test_a_brief_bolus_induces_little_tolerance(self):
        """The asymmetry the mechanism rests on: plasma LPS clears in minutes,
        so a bolus integrates almost no tolerance. This is true -- it is just
        not enough to reconcile the two settings, which is why the term is
        disabled."""
        p = replace(DangerSignalParameters(), k_tol_on=34.89)
        r = simulate_endotoxemia(2.0, p)
        assert float(np.max(r["tolerance"])) < 0.10

    def test_sustained_exposure_induces_much_more(self):
        p = DangerSignalParameters(Y_lysis=0.0, Y_filament=0.0,
                                   DS_baseline=65.5, k_tol_on=34.89)
        d = _driver(kill_rate=0.0)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                     d["C_eff"], 1.0, p)
        assert float(out["tolerance"][-1]) > 0.5


class TestSimpsonTables:
    def test_peak_endotoxin_ratio(self):
        d = SIMPSON_2000_TABLES["derived"]
        assert d["peak_endotoxin_ratio_caz_over_ipm"] == pytest.approx(
            725.8 / 146.0, rel=1e-9)
        assert 4.0 < d["peak_endotoxin_ratio_caz_over_ipm"] < 6.0

    def test_attributable_rise_matches_the_tables(self):
        tb = SIMPSON_2000_TABLES
        caz_base = tb["baseline_by_drug"]["ceftazidime"]["IL6"][0]
        caz_peak = tb["peak_by_drug"]["ceftazidime"]["IL6"][0]
        assert tb["derived"]["caz_attributable_IL6_rise_pg_mL"] == pytest.approx(
            caz_peak - caz_base, rel=1e-9)

    def test_endotoxin_difference_is_significant_and_cytokines_are_not(self):
        """The asymmetry that forced the Phase 3 correction."""
        pv = SIMPSON_2000_TABLES["peak_by_drug"]["p_values"]
        assert pv["endotoxin"] < 0.05
        for c in ("TNF", "IL6", "IL10"):
            assert pv[c] > 0.05

    def test_ceftazidime_had_lower_median_peak_il6(self):
        """Numerically the cytokines trend AGAINST the model. Recorded as a
        test so it cannot quietly disappear from the narrative."""
        tb = SIMPSON_2000_TABLES["peak_by_drug"]
        assert tb["ceftazidime"]["IL6"][0] < tb["imipenem"]["IL6"][0]

    def test_endotoxin_did_not_predict_death(self):
        out = SIMPSON_2000_TABLES["baseline_by_outcome"]
        assert out["p_values"]["endotoxin"] > 0.5
        for c in ("TNF", "IL6", "IL10"):
            assert out["p_values"][c] <= 0.001

    def test_constraint_is_scoped_to_endotoxin_only(self):
        assert SIMPSON_2000_MELIOIDOSIS.quantities == ("systemic endotoxin",)
        assert "ENDOTOXIN ONLY" in SIMPSON_2000_MELIOIDOSIS.caveat


class TestTrautmannAndPrinsData:
    """The two PDFs obtained in Phase 5b. Between them they replace the
    agent-effect interval the project has carried since Phase 1."""

    def test_trautmann_ratios_match_the_tables(self):
        from src.analysis.held_out_validation import TRAUTMANN_1998 as T
        f = T["free_LPS_50xMIC"]
        assert T["derived"]["caz_over_imipenem_free_8h_50x"] == pytest.approx(
            f["ceftazidime"][-1] / f["imipenem"][-1], rel=1e-9)
        assert T["derived"]["caz_over_meropenem_free_8h_50x"] == pytest.approx(
            f["ceftazidime"][-1] / f["meropenem"][-1], rel=1e-9)

    def test_untreated_released_more_than_every_treated_arm(self):
        """Trautmann's control is direct experimental support for the Phase 4
        result that under-dosing costs more danger signal than agent choice."""
        from src.analysis.held_out_validation import TRAUTMANN_1998 as T
        f = T["free_LPS_50xMIC"]
        control = f["no_antibiotic"][-1]
        for agent in ("ceftazidime", "meropenem", "imipenem", "ciprofloxacin"):
            assert control > f[agent][-1], agent

    def test_prins_untreated_induced_more_tnf_than_the_carbapenem(self):
        from src.analysis.held_out_validation import (
            PRINS_1995_WHOLE_BLOOD_DATA as P)
        d = P["TNF_4h_50xMIC_pg_per_mL"]
        assert d["no_antibiotic"][0] > d["imipenem"][0]
        assert d["no_antibiotic"][0] > d["gentamicin"][0]
        assert d["ceftazidime"][0] > d["no_antibiotic"][0]

    def test_three_independent_estimates_agree(self):
        """In vitro ELISA, ex vivo whole blood and in-patient plasma all land
        between 3 and 8x."""
        from src.analysis.held_out_validation import AGENT_EFFECT_ESTIMATES as A
        for key in ("trautmann1998_free_LPS_elisa_invitro",
                    "prins1995_TNF_wholeblood_exvivo",
                    "simpson2000_plasma_endotoxin_patients"):
            assert 3.0 < A[key] < 8.0, key

    def test_jackson_kropp_is_the_outlier(self):
        from src.analysis.held_out_validation import AGENT_EFFECT_ESTIMATES as A
        jk_lo = A["jackson1992_free_LPS_limulus_invitro"][0]
        others = [A["trautmann1998_free_LPS_elisa_invitro"],
                  A["prins1995_TNF_wholeblood_exvivo"],
                  A["simpson2000_plasma_endotoxin_patients"]]
        assert jk_lo > max(others), "10-40x should sit above every other estimate"

    def test_recommended_interval_brackets_the_three_estimates(self):
        from src.analysis.held_out_validation import AGENT_EFFECT_ESTIMATES as A
        lo, hi = A["recommended_interval"]
        for key in ("trautmann1998_free_LPS_elisa_invitro",
                    "prins1995_TNF_wholeblood_exvivo",
                    "simpson2000_plasma_endotoxin_patients"):
            assert lo <= A[key] <= hi, key

    def test_ciprofloxacin_conflict_is_recorded(self):
        """Trautmann (ELISA) says ciprofloxacin releases little; Prins (LAL)
        says it induces nearly as much TNF as ceftazidime. Both are recorded
        so the mechanism claim cannot quietly pick one."""
        from src.analysis.held_out_validation import (
            PRINS_1995_WHOLE_BLOOD_DATA as P, TRAUTMANN_1998 as T)
        cip_low = T["free_LPS_50xMIC"]["ciprofloxacin"][-1]
        caz = T["free_LPS_50xMIC"]["ceftazidime"][-1]
        assert cip_low < caz / 3.0, "Trautmann: ciprofloxacin is a low releaser"
        d = P["TNF_4h_50xMIC_pg_per_mL"]
        assert d["ciprofloxacin"][0] > 0.8 * d["ceftazidime"][0], (
            "Prins: ciprofloxacin is a high inducer")
        assert any("CONTRADICTS" in k for k in P["key_findings"])
