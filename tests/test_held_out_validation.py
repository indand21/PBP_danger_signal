"""
Tests for the Phase 3 held-out validation layer, the TNF calibration, and the
tie handling in the rank-invariance check.

Two of these guard defects found while building Phase 3: a rank-invariance
check that turned ties into spurious disagreements, and a magnitude test that
reported a 12% sliver of interval overlap as a clean pass.
"""
import numpy as np
import pytest

from src.analysis.cytokine_calibration import (
    HIJMA_2020, calibrate_family, calibrate_tnf, simulate_endotoxemia,
)
from src.analysis.held_out_validation import (
    ORDINAL_CONSTRAINTS, PRINS_1995_CLINICAL, SIMPSON_2000_MELIOIDOSIS,
    SEPSIS_IL6_RANGE_PG_ML, check_magnitude_consistency, evaluate_all,
    implied_pretreatment_il6,
)
from src.analysis.injury_layer import check_rank_invariance


class TestOrdinalConstraints:
    def test_every_constraint_expects_ceftazidime_to_be_worse(self):
        """All three held-out studies compare a PBP-3 agent against a PBP-2
        agent and report the PBP-3 agent as the more inflammatory one."""
        for c in ORDINAL_CONSTRAINTS:
            assert c.expected_order[-1] == "ceftazidime"
            assert "imipenem" in c.expected_order

    def test_constraints_carry_provenance_and_caveats(self):
        for c in ORDINAL_CONSTRAINTS:
            assert c.doi and c.source and c.reports
            assert c.caveat, f"{c.name} must record what it does NOT establish"

    def test_passes_when_model_ranks_correctly(self):
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 10.0, "ceftazidime": 500.0})
        assert res["pass"]
        assert res["predicted_order"] == ["imipenem", "ceftazidime"]

    def test_fails_when_model_ranks_backwards(self):
        """The validation must be able to fail; otherwise it is decoration."""
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 500.0, "ceftazidime": 10.0})
        assert not res["pass"]

    def test_reports_when_a_prediction_is_missing(self):
        res = SIMPSON_2000_MELIOIDOSIS.evaluate({"ceftazidime": 1.0})
        assert not res["evaluated"]
        assert "imipenem" in res["reason"]

    def test_evaluate_all_aggregates(self):
        ev = evaluate_all({"imipenem": 1.0, "ceftazidime": 2.0})
        assert ev["n_evaluated"] == len(ORDINAL_CONSTRAINTS)
        assert ev["n_passed"] == ev["n_evaluated"]
        assert ev["all_pass"]

    def test_evaluate_all_detects_a_wrong_direction(self):
        ev = evaluate_all({"imipenem": 2.0, "ceftazidime": 1.0})
        assert ev["n_passed"] == 0
        assert not ev["all_pass"]


class TestTiesAreNotPasses:
    """The external review (R5) found that identical predictions scored 3/3:
    Python's sort is stable, so a tie preserved the expected input order and
    was read as a correct prediction. A criterion that cannot distinguish
    'right' from 'said nothing' does not validate anything."""

    def test_a_tie_does_not_pass(self):
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 1.0, "ceftazidime": 1.0})
        assert not res["pass"]

    def test_a_tie_is_reported_as_inconclusive(self):
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 1.0, "ceftazidime": 1.0})
        assert res["verdict"] == "inconclusive"

    def test_a_separation_below_tolerance_is_inconclusive(self):
        """Floating-point noise is not a prediction."""
        res = PRINS_1995_CLINICAL.evaluate(
            {"imipenem": 1.0, "ceftazidime": 1.0 + 1e-12})
        assert res["verdict"] == "inconclusive"

    def test_a_clear_separation_passes(self):
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 1.0, "ceftazidime": 5.0})
        assert res["verdict"] == "pass" and res["pass"]

    def test_a_reversal_fails_rather_than_being_inconclusive(self):
        res = PRINS_1995_CLINICAL.evaluate({"imipenem": 5.0, "ceftazidime": 1.0})
        assert res["verdict"] == "fail" and not res["pass"]

    def test_evaluate_all_scores_no_passes_for_identical_predictions(self):
        """The exact case the review reported."""
        ev = evaluate_all({"imipenem": 1.0, "ceftazidime": 1.0})
        assert ev["n_passed"] == 0
        assert not ev["all_pass"]
        assert ev["n_inconclusive"] == ev["n_evaluated"]


class TestSourceUseLedger:
    """R5: three studies were described as 'entirely held out' while two of
    them had been used to choose the 3.5-7x yield interval. Selecting a
    parameter from a study is model development, whatever the mechanism."""

    def test_ledger_covers_every_ordinal_constraint(self):
        from src.analysis.held_out_validation import SOURCE_USE_LEDGER
        keys = {e["doi"] for e in SOURCE_USE_LEDGER}
        for c in ORDINAL_CONSTRAINTS:
            assert c.doi in keys, f"{c.name} missing from the ledger"

    def test_every_entry_declares_recognised_roles(self):
        from src.analysis.held_out_validation import (
            SOURCE_ROLES, SOURCE_USE_LEDGER,
        )
        for e in SOURCE_USE_LEDGER:
            assert e["roles"], f"{e['doi']} declares no role"
            for r in e["roles"]:
                assert r in SOURCE_ROLES, f"unknown role {r}"

    def test_a_source_used_for_parameter_selection_is_not_independent(self):
        """The property that was previously violated."""
        from src.analysis.held_out_validation import SOURCE_USE_LEDGER
        for e in SOURCE_USE_LEDGER:
            if "parameter_selection" in e["roles"]:
                assert "untouched_evaluation" not in e["roles"], (
                    f"{e['doi']} cannot both set a parameter and be an "
                    "untouched test of that parameter")

    def test_the_yield_interval_sources_are_marked_as_used(self):
        from src.analysis.held_out_validation import SOURCE_USE_LEDGER
        used = {e["doi"] for e in SOURCE_USE_LEDGER
                if "parameter_selection" in e["roles"]}
        # Prins whole blood and Simpson both informed the 3.5-7x interval.
        assert "10.1128/iai.63.6.2236-2242.1995" in used
        assert "10.1086/315306" in used

    def test_no_source_is_claimed_as_untouched(self):
        """Nothing in this repository is a prospective holdout; the ledger
        must not imply one exists."""
        from src.analysis.held_out_validation import SOURCE_USE_LEDGER
        assert not [e for e in SOURCE_USE_LEDGER
                    if "untouched_evaluation" in e["roles"]]


class TestConstraintEndpointScoping:
    """R5 also found Phase 3 feeding modelled IL-6 to every constraint,
    including Simpson, whose reported quantity is endotoxin alone."""

    def test_every_constraint_declares_the_model_endpoint_it_tests(self):
        for c in ORDINAL_CONSTRAINTS:
            assert c.model_endpoint, f"{c.name} must name its model endpoint"

    def test_simpson_is_scored_on_danger_signal_not_cytokine(self):
        assert SIMPSON_2000_MELIOIDOSIS.model_endpoint == "AUC_DS_pg_h_mL"

    def test_scoring_on_the_wrong_endpoint_is_not_evaluated(self):
        res = SIMPSON_2000_MELIOIDOSIS.evaluate(
            {"imipenem": 1.0, "ceftazidime": 5.0},
            endpoint="AUC_IL6_excess_pg_h_mL")
        assert not res["evaluated"]
        assert "endpoint" in res["reason"]

    def test_scoring_on_the_declared_endpoint_is_evaluated(self):
        res = SIMPSON_2000_MELIOIDOSIS.evaluate(
            {"imipenem": 1.0, "ceftazidime": 5.0},
            endpoint="AUC_DS_pg_h_mL")
        assert res["evaluated"] and res["pass"]


class TestMagnitudeBound:
    def test_implied_baseline_inverts_the_percentage(self):
        """If therapy adds 100 pg/mL and that is a 10-40% rise, the
        pre-treatment level was 250-1000 pg/mL."""
        imp = implied_pretreatment_il6(100.0, (10.0, 40.0))
        assert imp["implied_pretreatment_IL6_low_pg_mL"] == pytest.approx(250.0)
        assert imp["implied_pretreatment_IL6_high_pg_mL"] == pytest.approx(1000.0)

    def test_consistent_when_implied_sits_inside_the_range(self):
        """An attributable peak of 100 pg/mL implies 250-1000 pg/mL, which lies
        well inside the septic range."""
        r = check_magnitude_consistency(100.0, SEPSIS_IL6_RANGE_PG_ML)
        assert r["overlaps_observed_sepsis_range"]
        assert not r["over_predicts"]
        assert r["verdict"] == "consistent"

    def test_over_prediction_is_detected(self):
        r = check_magnitude_consistency(1e5, SEPSIS_IL6_RANGE_PG_ML)
        assert r["over_predicts"]
        assert r["verdict"] == "over-predicts"
        assert r["required_scale_correction"] < 1.0

    def test_under_prediction_is_detected(self):
        r = check_magnitude_consistency(1e-3, SEPSIS_IL6_RANGE_PG_ML)
        assert r["under_predicts"]
        assert r["verdict"] == "under-predicts"

    def test_sliver_of_overlap_is_flagged_as_marginal(self):
        """Regression guard: a bare overlap must not report as agreement. An
        attributable peak of ~589 pg/mL implies 1473-5891 pg/mL against an
        observed 100-2000, which overlaps over only ~12% of the interval."""
        r = check_magnitude_consistency(589.0, SEPSIS_IL6_RANGE_PG_ML)
        assert r["overlaps_observed_sepsis_range"]
        assert r["marginal"]
        assert r["overlap_fraction_of_implied_interval"] < 0.30
        assert "biased high" in r["verdict"]


class TestTnfCalibration:
    @pytest.fixture(scope="class")
    def fitted(self):
        fam = [e for e in calibrate_family() if e["feasible"]]
        return calibrate_tnf(min(fam, key=lambda e: e["cost"]))

    def test_reproduces_observed_tnf_peaks(self, fitted):
        for pred, obs in zip(fitted["pred_TNF_peak"], fitted["obs_TNF_peak"]):
            assert pred == pytest.approx(obs, rel=0.15)

    def test_tnf_peaks_near_one_hour(self, fitted):
        assert 0.6 < fitted["pred_t_peak_TNF_h"][0] < 1.6

    def test_tnf_peaks_before_il6(self, fitted):
        """Observed ordering: TNF at 1 h, IL-6 at 2 h. Both are driven by the
        same activation signal, so the ordering comes from clearance alone and
        is a genuine prediction rather than a fitted offset."""
        r = simulate_endotoxemia(2.0, fitted["params"])
        t_tnf = float(r["t"][int(np.argmax(r["TNF"]))])
        t_il6 = float(r["t"][int(np.argmax(r["IL6"]))])
        assert t_tnf < t_il6

    def test_observed_data_has_tnf_earlier_than_il6(self):
        assert HIJMA_2020["t_peak_TNF_h"] < HIJMA_2020["t_peak_IL6_h"]


class TestRankInvarianceTieHandling:
    @staticmethod
    def _arm(level, n=600):
        t = np.linspace(0.0, 48.0, n)
        return {"t": t, "IL6_excess": np.full_like(t, float(level))}

    def test_identical_arms_are_a_tie_not_a_conflict_and_not_a_ranking(self):
        """Regression guard. Two arms with identical exposure are tied under
        every map. Sorting is stable, so the old check read insertion order as
        a definite ranking; a tie is an absence of evidence, not disagreement.

        Note what is asserted: no conflict, but also NO claimed ranking. Every
        map is degenerate here, so there is nothing to be invariant about, and
        reporting invariance would assert an ordering the scores do not
        support."""
        arms = {"a": self._arm(100.0), "b": self._arm(100.0)}
        res = check_rank_invariance(arms)
        assert res["conflicting_pairs"] == []
        assert res["informative_maps"] == []
        assert res["consensus_ranking"] is None
        assert not res["invariant"]

    def test_arms_too_small_to_separate_do_not_fail_invariance(self):
        """At a very small focus volume every arm's injury is near zero and
        thresholded maps tie. That must not read as a broken ranking -- it is
        what produced a spurious failure in the Phase 3 scale sweep."""
        arms = {"tiny_lo": self._arm(1e-6), "tiny_hi": self._arm(2e-6)}
        res = check_rank_invariance(arms)
        assert res["invariant"], res["conflicting_pairs"]

    def test_degenerate_maps_are_reported(self):
        """A map that gives every arm the same score cannot discriminate, and
        must be named rather than silently counted as agreeing."""
        arms = {"a": self._arm(1e-9), "b": self._arm(2e-9)}
        res = check_rank_invariance(arms, maps=["soft_threshold", "linear"])
        assert "soft_threshold" in res["degenerate_maps"]
        assert "linear" in res["informative_maps"]

    def test_genuine_conflict_still_detected(self):
        """The tie tolerance must not blunt the check: a real map-dependent
        ordering still has to be caught."""
        t = np.linspace(0.0, 48.0, 2400)
        spike = np.zeros_like(t)
        spike[(t > 1.0) & (t < 3.0)] = 1200.0
        sustained = np.full_like(t, 1200.0 * 2.0 / 48.0)
        arms = {"spike": {"t": t, "IL6_excess": spike},
                "sustained": {"t": t, "IL6_excess": sustained}}
        res = check_rank_invariance(arms, metric="auc",
                                    maps=["quadratic", "hill1"])
        assert res["conflicting_pairs"]
        assert not res["invariant"]

    def test_clear_ordering_is_still_invariant(self):
        arms = {"low": self._arm(5.0), "mid": self._arm(80.0),
                "high": self._arm(600.0)}
        res = check_rank_invariance(arms)
        assert res["invariant"]
        assert res["consensus_ranking"] == ["low", "mid", "high"]
        assert res["degenerate_maps"] == []
