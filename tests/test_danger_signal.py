"""
Tests for the danger-signal layer (src/analysis/danger_signal.py).

The tests that matter most here are the ones guarding the failure mode that
sank the previous host-damage layer: a pathway with so much gain that it acts
as a binary detector. ``test_no_step_function`` and
``test_unsaturated_at_physiological_levels`` exist for exactly that, and are
the unit-test form of the pre-registered stop-gate in
docs/danger_signal_study/DESIGN.md.
"""
import numpy as np
import pytest

from src.analysis.danger_signal import (
    DangerSignalParameters, YIELD_ANCHORS_FG_PER_CFU, check_operating_point,
    check_smoothness, filamentation_activity, simulate_danger_signal, summarize,
)


def _driver(n=600, t_end=24.0, kill_rate=1.0, B=1e7, C=4.0):
    """A constant-exposure driver: steady kill of a steady burden."""
    t = np.linspace(0, t_end, n)
    return dict(t=t,
                B_rep=np.full(n, B),
                kill=np.full(n, kill_rate),
                C_eff=np.full(n, C))


class TestFilamentationActivity:
    def test_zero_without_drug(self):
        assert filamentation_activity(0.0, 1.0, (0.0625, 1.0)) == pytest.approx(0.0, abs=1e-6)

    def test_negligible_far_above_mic(self):
        """Above the MIC the agent kills; lysis-coupled release takes over.

        The roll-off is a logistic in log-concentration, deliberately rather
        than a hard cutoff, so it decays smoothly instead of vanishing
        exactly -- 1e-4 of peak activity at 50x MIC is the relevant tolerance.
        """
        assert filamentation_activity(50.0, 1.0, (0.0625, 1.0)) < 1e-4

    def test_active_inside_the_window(self):
        """The sub-MIC window is where Jackson & Kropp saw release without lysis."""
        assert filamentation_activity(0.25, 1.0, (0.0625, 1.0)) > 0.9

    def test_smooth_across_the_edges(self):
        """No hard threshold anywhere -- a step here would reintroduce exactly
        the pathology the design document forbids."""
        x = np.logspace(-3, 2, 2000)
        phi = filamentation_activity(x, 1.0, (0.0625, 1.0))
        assert np.max(np.abs(np.diff(phi))) < 0.05

    def test_scales_with_mic(self):
        """The window is defined in multiples of the MIC, not absolute mg/L."""
        a = filamentation_activity(0.25, 1.0, (0.0625, 1.0))
        b = filamentation_activity(2.5, 10.0, (0.0625, 1.0))
        assert a == pytest.approx(b, rel=1e-9)


class TestDangerSignalDynamics:
    def test_no_release_without_kill_or_filamentation(self):
        d = _driver(kill_rate=0.0)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     1.0, DangerSignalParameters(Y_filament=0.0))
        assert np.max(out["DS"]) == pytest.approx(0.0, abs=1e-12)
        assert np.max(out["IL6_excess"]) == pytest.approx(0.0, abs=1e-9)

    def test_release_proportional_to_yield_when_unsaturated(self):
        """Far below DS50 the plasma DS must be linear in Y_lysis. This is the
        property the previous model lacked."""
        d = _driver()
        peaks = []
        for y in (0.1, 1.0, 10.0):
            p = DangerSignalParameters(Y_lysis=y, Y_filament=0.0, DS50=1e9)
            out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                         d["C_eff"], 1.0, p)
            peaks.append(np.max(out["DS"]))
        assert peaks[1] / peaks[0] == pytest.approx(10.0, rel=1e-3)
        assert peaks[2] / peaks[1] == pytest.approx(10.0, rel=1e-3)

    def test_steady_state_matches_analytic_balance(self):
        """With constant release R and clearance k, DS -> R/k."""
        d = _driver(t_end=12.0)
        p = DangerSignalParameters(Y_lysis=1.0, Y_filament=0.0, DS50=1e9,
                                   k_ds_clear=10.0, V_focus_mL=10.0, V_ds_L=3.0)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     1.0, p)
        # fg/h -> pg/mL/h : Y * kill * B * V_focus * 1e-6 / V_ds_L
        R = 1.0 * 1.0 * 1e7 * 10.0 * 1e-6 / 3.0
        assert out["DS"][-1] == pytest.approx(R / 10.0, rel=1e-3)

    def test_faster_clearance_lowers_exposure(self):
        d = _driver()
        vals = []
        for k in (2.0, 20.0):
            p = DangerSignalParameters(Y_lysis=1.0, Y_filament=0.0, k_ds_clear=k)
            out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                         d["C_eff"], 1.0, p)
            vals.append(summarize(out)["AUC_DS_pg_h_mL"])
        assert vals[0] > vals[1]

    def test_filamentation_pathway_needs_sub_mic_exposure(self):
        """With no kill at all, release can still occur -- but only inside the
        sub-MIC window. This is the kill-independent pathway."""
        p = DangerSignalParameters(Y_lysis=0.0, Y_filament=1.5)
        sub = _driver(kill_rate=0.0, C=0.25)   # inside the window
        sup = _driver(kill_rate=0.0, C=50.0)   # far above it
        a = simulate_danger_signal(sub["t"], sub["B_rep"], sub["kill"],
                                   sub["C_eff"], 1.0, p)
        b = simulate_danger_signal(sup["t"], sup["B_rep"], sup["kill"],
                                   sup["C_eff"], 1.0, p)
        assert np.max(a["DS"]) > 0
        # Four orders of magnitude down, not identically zero: the window's
        # upper edge rolls off smoothly (see test_negligible_far_above_mic).
        assert np.max(b["DS"]) < 1e-4 * np.max(a["DS"])
        assert summarize(a)["filament_fraction_of_release"] == pytest.approx(1.0)

    def test_il6_excess_is_measured_from_baseline(self):
        d = _driver(kill_rate=0.0)
        p = DangerSignalParameters(Y_filament=0.0, IL6_baseline=7.0)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     1.0, p)
        assert out["IL6"][-1] == pytest.approx(7.0, rel=1e-6)
        assert summarize(out)["AUC_IL6_excess_pg_h_mL"] == pytest.approx(0.0, abs=1e-6)


class TestAcceptanceGates:
    def test_unsaturated_at_physiological_levels(self):
        """Stop-gate 1: with literature yields and a realistic focus, the
        cytokine layer must not be pegged at the ceiling -- the previous
        model's failure mode."""
        d = _driver(kill_rate=1.0, B=1e7)
        p = DangerSignalParameters(
            Y_lysis=YIELD_ANCHORS_FG_PER_CFU["ceftazidime"], Y_filament=0.0,
            V_focus_mL=10.0)
        out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"], d["C_eff"],
                                     1.0, p)
        chk = check_operating_point(summarize(out)["saturation_peak"])
        assert chk["pass"], chk
        assert chk["saturation_peak"] < 0.1, (
            "the calibrated layer should run far below saturation at "
            "physiological yields")

    def test_low_occupancy_still_discriminates(self):
        """The companion to the test above, and the reason its lower bound is
        permissive: at the calibrated operating point the occupancy is only a
        few percent, but because the response follows DS^hill there, the output
        still separates agents strongly. Low occupancy is not a dead zone."""
        d = _driver(kill_rate=1.0, B=1e7)
        vals = []
        for y in (YIELD_ANCHORS_FG_PER_CFU["meropenem"],
                  YIELD_ANCHORS_FG_PER_CFU["ceftazidime"]):
            p = DangerSignalParameters(Y_lysis=y, Y_filament=0.0,
                                       V_focus_mL=10.0)
            out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                         d["C_eff"], 1.0, p)
            s = summarize(out)
            assert s["saturation_peak"] < 0.1
            vals.append(s["AUC_IL6_excess_pg_h_mL"])
        yield_ratio = (YIELD_ANCHORS_FG_PER_CFU["ceftazidime"]
                       / YIELD_ANCHORS_FG_PER_CFU["meropenem"])
        # Supralinear transmission: the output ratio must EXCEED the input
        # yield ratio, which is what the measured Hill coefficient implies.
        assert vals[1] / vals[0] > yield_ratio

    def test_no_step_function(self):
        """Stop-gate 2: the response must vary smoothly and monotonically with
        the per-kill yield over the whole plausible two-decade range."""
        d = _driver()
        y_grid = np.logspace(-1, 1, 31)
        auc = []
        for y in y_grid:
            p = DangerSignalParameters(Y_lysis=y, Y_filament=0.0)
            out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                         d["C_eff"], 1.0, p)
            auc.append(summarize(out)["AUC_IL6_excess_pg_h_mL"])
        chk = check_smoothness(y_grid, np.array(auc))
        assert chk["monotone"], chk
        assert chk["no_step_function"], chk
        assert chk["dynamic_range_ok"], chk

    def test_binary_detector_is_caught(self):
        """The gate must actually be able to fail. A saturated pathway (DS50
        far below the operating point) is the previous model's failure mode;
        check_smoothness must reject it."""
        d = _driver()
        y_grid = np.logspace(-1, 1, 31)
        auc = []
        for y in y_grid:
            p = DangerSignalParameters(Y_lysis=y, Y_filament=0.0, DS50=1e-6)
            out = simulate_danger_signal(d["t"], d["B_rep"], d["kill"],
                                         d["C_eff"], 1.0, p)
            auc.append(summarize(out)["AUC_IL6_excess_pg_h_mL"])
        chk = check_smoothness(y_grid, np.array(auc))
        assert not chk["pass"], (
            "a fully saturated pathway passed the smoothness gate -- the gate "
            "cannot detect the failure it exists to detect")

    def test_operating_point_check_rejects_both_extremes(self):
        assert not check_operating_point(0.999)["pass"]
        assert not check_operating_point(1e-6)["pass"]
        assert check_operating_point(0.2)["pass"]


class TestKillRateSharedWithPDModel:
    def test_matches_the_pd_model_rhs(self, pd_model):
        """The DS layer multiplies the PD model's own kill rate; if the helper
        ever diverged from the rate used inside rhs, release would silently
        decouple from killing."""
        C = 2.0
        y = np.array([1e6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        dydt = pd_model.rhs(0.0, y, C_effect=C, drug_class="cidal")
        growth = pd_model.p_bact.k_growth * (1.0 - 1e6 / pd_model.p_bact.B_max) * 1e6
        to_pers = pd_model.p_bact.k_pers * 1e6
        implied_kill = growth - to_pers - dydt[0]
        assert implied_kill == pytest.approx(
            float(pd_model.cidal_kill_rate(C)) * 1e6, rel=1e-9)
