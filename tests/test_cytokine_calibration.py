"""
Tests for the Layer-2 cytokine calibration and the ordinal injury layer
(Phase 2 of the danger-signal study).

The tests that earn their keep here are the ones that would have caught the two
real defects found while building this: an instantaneous cytokine response that
peaked ten times too early, and a thresholded injury map that integrated to
zero because the solver stepped over the forcing.
"""
import numpy as np
import pytest

from src.analysis.cytokine_calibration import (
    DILLINGH_2014, HIJMA_2020, bolus_plasma_ds0, calibrate_family,
    check_lower_bound_consistency, observed_log_slope, predict_peaks,
    simulate_endotoxemia,
)
from src.analysis.danger_signal import DangerSignalParameters
from src.analysis.injury_layer import (
    INJURY_MAPS, InjuryParameters, check_rank_invariance, injury_trajectory,
    rank_by_injury, saturation_diagnostics,
)


class TestDoseConversion:
    def test_bolus_concentration(self):
        """1 ng/kg into a 70 kg adult with 3 L of plasma is ~23 pg/mL."""
        p = DangerSignalParameters()
        assert bolus_plasma_ds0(1.0, p) == pytest.approx(70.0 / 3.0, rel=1e-9)

    def test_scales_linearly_with_dose(self):
        p = DangerSignalParameters()
        assert bolus_plasma_ds0(2.0, p) == pytest.approx(
            2 * bolus_plasma_ds0(1.0, p), rel=1e-12)


class TestObservedSlope:
    def test_slope_is_supralinear(self):
        """The human IL-6 response rises faster than proportionally with dose:
        a 2-fold dose step gives a ~3.2-fold rise. The Phase 1 pilot assumed
        proportional (slope 1), which understated agent differences."""
        s = observed_log_slope()
        assert 1.5 < s < 1.9

    def test_matches_hand_computation(self):
        o1, o2 = HIJMA_2020["IL6_peak_pg_mL"]
        base = DangerSignalParameters()
        expect = np.log((o2 - base.IL6_baseline) / (o1 - base.IL6_baseline)) / np.log(2.0)
        assert observed_log_slope() == pytest.approx(float(expect), rel=1e-9)


class TestTransductionDelay:
    def test_il6_peak_is_delayed_by_hours(self):
        """With the calibrated transduction rate the IL-6 peak lands near the
        observed 2 h, not within minutes of the bolus."""
        p = DangerSignalParameters()
        r = simulate_endotoxemia(2.0, p)
        t_peak = float(r["t"][int(np.argmax(r["IL6"]))])
        assert 1.0 < t_peak < 3.5

    def test_instantaneous_limit_peaks_early(self):
        """As k_signal_off -> infinity the layer collapses to the Phase 1
        prototype, whose IL-6 peaked within minutes. This is the regression
        guard: if the cascade is ever dropped, this is what comes back."""
        from dataclasses import replace
        p = replace(DangerSignalParameters(), k_signal_off=1e6)
        r = simulate_endotoxemia(2.0, p)
        t_peak = float(r["t"][int(np.argmax(r["IL6"]))])
        assert t_peak < 0.5

    def test_danger_signal_itself_is_cleared_fast(self):
        """Plasma LPS has a half-life of minutes; only the cellular response is
        slow. If DS lingered for hours the delay would be modelled in the wrong
        place."""
        p = DangerSignalParameters()
        r = simulate_endotoxemia(2.0, p, t_end=2.0, n=4001)
        half = bolus_plasma_ds0(2.0, p) / 2.0
        i = int(np.argmax(r["DS"] <= half))
        assert r["t"][i] < 0.2


class TestCalibration:
    @pytest.fixture(scope="class")
    def family(self):
        return calibrate_family()

    def test_some_parameterization_fits(self, family):
        assert any(e["feasible"] for e in family)

    def test_feasible_entries_reproduce_observed_peaks(self, family):
        for e in family:
            if not e["feasible"]:
                continue
            for pred, obs in zip(e["pred_IL6_peak"], e["obs_IL6_peak"]):
                assert pred == pytest.approx(obs, rel=0.10)

    def test_low_ds50_rejected_on_timing(self, family):
        """DS50 of 50-100 pg/mL can match the peak magnitudes but forces the
        peak ~3 h out. The timing observable is what breaks the otherwise
        total degeneracy between DS50 and the Hill coefficient."""
        low = [e for e in family if e["DS50"] <= 100.0]
        assert low, "grid should include low DS50 values to exercise this"
        assert all(not e["feasible"] for e in low)

    def test_hill_coefficient_is_constrained(self, family):
        """Whatever DS50 is assumed, the fitted Hill coefficient stays in a
        narrow supralinear band -- so the amplification it implies is a
        property of the data, not of the assumed DS50."""
        hills = [e["hill"] for e in family if e["feasible"]]
        assert min(hills) > 1.2
        assert max(hills) < 2.2

    def test_operates_below_saturation_at_calibration_doses(self, family):
        for e in family:
            if e["feasible"]:
                assert e["saturation_peak_at_2ng_kg"] < 0.5

    def test_held_out_foot_constraints(self, family):
        """Dillingh (different laboratory, not used in the fit): detectable at
        0.5 ng/kg, quiet at 0.2 ng/kg."""
        for e in family:
            if not e["feasible"]:
                continue
            c = check_lower_bound_consistency(e)
            assert c["detectable_at_0.5"], (e["DS50"], c)
            assert c["quiet_at_0.2"], (e["DS50"], c)
            assert c["monotone_foot"]

    def test_predicted_peaks_increase_with_dose(self, family):
        e = next(x for x in family if x["feasible"])
        pk = predict_peaks(e["params"], [0.5, 1.0, 2.0])["IL6_peak"]
        assert pk[0] < pk[1] < pk[2]


class TestInjuryMaps:
    @pytest.mark.parametrize("name", list(INJURY_MAPS))
    def test_maps_are_monotone_and_vanish_at_zero(self, name):
        g = INJURY_MAPS[name]
        x = np.linspace(0.0, 20.0, 4001)
        y = np.asarray(g(x), dtype=float)
        assert y[0] == pytest.approx(0.0, abs=1e-12)
        assert np.all(np.diff(y) >= -1e-12)

    def test_thresholded_map_still_integrates_a_late_pulse(self):
        """Regression test for a solver defect that silently produced a false
        result. A thresholded map makes the injury right-hand side exactly zero
        until the cytokine clears the threshold; with an unbounded step size the
        solver stepped over the pulse entirely and reported zero injury for the
        most exposed arm, which then looked like a failed rank-invariance test
        rather than a numerical bug."""
        t = np.linspace(0.0, 48.0, 600)
        cyto = np.zeros_like(t)
        cyto[(t > 20.0) & (t < 26.0)] = 600.0  # late, well above threshold
        r = injury_trajectory(t, cyto, "soft_threshold")
        assert r["peak"] > 1.0
        assert r["auc"] > 1.0

    def test_injury_recovers_after_exposure_ends(self):
        t = np.linspace(0.0, 96.0, 2000)
        cyto = np.where(t < 12.0, 400.0, 0.0)
        r = injury_trajectory(t, cyto, "linear")
        assert r["injury"][-1] < 0.25 * r["peak"]

    def test_more_exposure_gives_more_injury(self):
        t = np.linspace(0.0, 48.0, 1200)
        lo = injury_trajectory(t, np.full_like(t, 50.0), "linear")
        hi = injury_trajectory(t, np.full_like(t, 500.0), "linear")
        assert hi["peak"] > lo["peak"]
        assert hi["auc"] > lo["auc"]


class TestRankInvariance:
    @staticmethod
    def _arm(level, t_end=48.0, n=800):
        t = np.linspace(0.0, t_end, n)
        return {"t": t, "IL6_excess": np.full_like(t, float(level))}

    def test_clearly_ordered_arms_rank_invariantly(self):
        arms = {"low": self._arm(5.0), "mid": self._arm(80.0),
                "high": self._arm(600.0)}
        res = check_rank_invariance(arms)
        assert res["invariant"], res["rankings"]
        assert res["consensus_ranking"] == ["low", "mid", "high"]

    def test_invariance_holds_for_auc_as_well_as_peak(self):
        arms = {"low": self._arm(5.0), "high": self._arm(600.0)}
        for metric in ("peak", "auc"):
            assert check_rank_invariance(arms, metric=metric)["invariant"]

    def test_ranking_is_detected_as_map_dependent_when_it_is(self):
        """The invariance check must be able to FAIL. Two arms with the same
        total exposure delivered differently -- a brief intense spike versus a
        sustained low level -- are ordered one way by a supralinear map and the
        other way by a saturating one, and the check must say so rather than
        report a ranking."""
        t = np.linspace(0.0, 48.0, 2400)
        spike = np.zeros_like(t)
        spike[(t > 1.0) & (t < 3.0)] = 1200.0
        sustained = np.full_like(t, 1200.0 * 2.0 / 48.0)
        arms = {"spike": {"t": t, "IL6_excess": spike},
                "sustained": {"t": t, "IL6_excess": sustained}}
        res = check_rank_invariance(arms, metric="auc",
                                    maps=["quadratic", "hill1"])
        assert not res["invariant"]
        assert res["consensus_ranking"] is None

    def test_rank_by_injury_agrees_with_the_invariance_check(self):
        arms = {"low": self._arm(5.0), "high": self._arm(600.0)}
        assert rank_by_injury(arms, "linear") == ["low", "high"]

    def test_saturation_diagnostics_flag_a_pegged_map(self):
        arms = {"huge": self._arm(1e5)}
        d = saturation_diagnostics(arms)["huge"]
        assert d["hill1_peak_occupancy"] > 0.99
        assert d["hill2_peak_occupancy"] > 0.99

    def test_saturation_diagnostics_scale_with_reference(self):
        arms = {"a": self._arm(100.0)}
        d = saturation_diagnostics(arms, InjuryParameters(cyto_ref=100.0))["a"]
        assert d["max_x_over_ref"] == pytest.approx(1.0, rel=1e-9)
        assert d["hill1_peak_occupancy"] == pytest.approx(0.5, rel=1e-6)
