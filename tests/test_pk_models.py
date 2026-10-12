"""
Tests for pharmacokinetic models.
"""
import pytest
import numpy as np
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen


class TestTwoCompartmentPKModel:
    """Tests for TwoCompartmentPKModel."""

    def test_initialization(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        assert model.CL == 10
        assert model.Vc == 1000
        assert model.effect_site_model is True

    def test_concentration_central(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        conc = model.concentration_central(1000)
        assert conc == 1.0

    def test_concentration_effect(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        conc = model.concentration_effect(1000)
        # C_effect = Kp * (A_c/Vc) * 1000 (mg/L conversion)
        assert conc == 400.0  # 0.4 * (1000/1000) * 1000

    def test_no_effect_site(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4,
            effect_site_model=False
        )
        conc = model.concentration_effect(1000)
        assert conc == 1000.0  # C_central * 1000 (mg/L conversion)

    def test_ode_rhs_shape(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        A = np.array([1000, 0, 0, 0])
        dA = model.ode_rhs(0, A)
        assert dA.shape == (4,)

    def test_ode_rhs_no_infusion(self):
        """Without infusion, drug should clear from central."""
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        A = np.array([1000, 0, 0, 0])
        dA = model.ode_rhs(0, A)
        assert dA[0] < 0  # Central compartment decreases

    def test_ode_rhs_with_infusion(self):
        """With infusion, drug should enter central."""
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        A = np.array([0, 0, 0, 0])
        dA = model.ode_rhs(0, A, infusion_rate=100)
        assert dA[0] > 0  # Central compartment increases

    @pytest.mark.xfail(strict=True, reason=(
        "Known defect (external review R3): ode_rhs does not conserve drug. "
        "The study no longer uses it -- see test_pk_representation.py -- but "
        "the legacy therapy modules still do. strict=True makes this test "
        "FAIL the suite the moment the defect is fixed, so the marker is "
        "removed rather than left to rot."))
    def test_ode_rhs_conserves_drug_without_input_or_clearance(self):
        """With CL = 0, no infusion and no absorption, drug only moves between
        compartments, so total central + peripheral drug must be constant.

        The previous version of this test checked only that the peripheral
        derivative reproduced its own formula, and its comment called the
        imbalance 'a known quirk'. It therefore passed on the defect."""
        model = TwoCompartmentPKModel(
            CL=0, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        A = np.array([1000, 500, 0, 0])
        dA = model.ode_rhs(0, A)
        assert abs(dA[0] + dA[1]) < 1e-9

    def test_get_ode_indices(self):
        model = TwoCompartmentPKModel(
            CL=10, Vc=1000, Vp=500, Q=2.0, Ka=0.0, Kp=0.4
        )
        indices = model.get_ode_indices()
        assert indices["A_central"] == 0
        assert indices["A_peripheral"] == 1
        assert indices["A_absorption"] == 2
        assert indices["A_effect"] == 3


class TestDosingRegimen:
    """Tests for DosingRegimen."""

    def test_initialization(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=0, n_doses=3
        )
        assert reg.dose_mg == 1000
        assert reg.interval_hours == 8
        assert reg.n_doses == 3

    def test_dose_times(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=0, n_doses=3
        )
        times = reg.get_dose_times()
        expected = np.array([0, 8, 16])
        np.testing.assert_array_equal(times, expected)

    def test_dose_times_with_start(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=2, n_doses=3
        )
        times = reg.get_dose_times()
        expected = np.array([2, 10, 18])
        np.testing.assert_array_equal(times, expected)

    def test_infusion_rate_during_dose(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=0, n_doses=3,
            infusion_duration_min=60
        )
        # During first dose (0-1h)
        rate = reg.get_infusion_rate(0.5)
        assert rate > 0

    def test_infusion_rate_between_doses(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=0, n_doses=3,
            infusion_duration_min=60
        )
        # Between doses (e.g., at 4h)
        rate = reg.get_infusion_rate(4)
        assert rate == 0

    def test_infusion_rate_after_last_dose(self):
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=8, start_time=0, n_doses=3,
            infusion_duration_min=60
        )
        # After last dose finishes (16 + 1 = 17h)
        rate = reg.get_infusion_rate(20)
        assert rate == 0

    def test_infusion_rate_zero_doses(self):
        reg = DosingRegimen(
            dose_mg=0, interval_hours=8, start_time=0, n_doses=0
        )
        rate = reg.get_infusion_rate(0)
        assert rate == 0

    def test_multiple_overlapping_doses(self):
        """If doses overlap, rates should add."""
        reg = DosingRegimen(
            dose_mg=1000, interval_hours=0.5, start_time=0, n_doses=10,
            infusion_duration_min=60
        )
        # With interval = 0.5h and duration = 1h, doses overlap
        rate = reg.get_infusion_rate(1.0)
        assert rate > reg.dose_mg / 1.0  # Should be at least double
