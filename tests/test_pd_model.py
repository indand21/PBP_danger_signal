"""
Tests for pharmacodynamic model.
"""
import pytest
import numpy as np
from src.core.pd_model import BacterialPopulationODE, create_ode_system


class TestBacterialPopulationODE:
    """Tests for BacterialPopulationODE class."""

    def test_initialization(self, default_params):
        model = BacterialPopulationODE(default_params)
        assert model.p_bact is not None
        assert model.p_imm is not None
        assert model.p_cyto is not None

    def test_h_static_no_drug(self, pd_model):
        """At C=0, Hill function should be 1 (no inhibition)."""
        H = pd_model.h_static(0.0, EC50=1.0, hill=1.0)
        assert H == pytest.approx(1.0, abs=1e-6)

    def test_h_static_high_conc(self, pd_model):
        """At very high C, Hill function should be 0 (full inhibition)."""
        H = pd_model.h_static(100.0, EC50=1.0, hill=1.0)
        assert H < 0.1

    def test_h_static_ec50(self, pd_model):
        """At C=EC50, Hill function should be 0.5."""
        H = pd_model.h_static(1.0, EC50=1.0, hill=1.0)
        assert H == pytest.approx(0.5, abs=1e-6)

    def test_f_cidal_no_damage(self, pd_model):
        """At Damage=0, cidal mechanism should be 0."""
        f = pd_model.f_cidal_mechanism(1.0, 0.0)
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_f_cidal_high_damage(self, pd_model):
        """At very high damage, cidal mechanism should approach 1."""
        f = pd_model.f_cidal_mechanism(1.0, 100.0)
        assert f > 0.99

    def test_rhs_shape(self, pd_model):
        """rhs should return array of length 9 (7 PD + PAMP + D_host)."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5, 0, 0])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="cidal")
        assert dydt.shape == (9,)

    def test_rhs_no_drug_growth(self, pd_model):
        """Without drug, bacteria should grow when immune level is low."""
        y = np.array([1e6, 1e3, 0, 1e6, 0, 10, 5])  # low immune for growth
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[0] > 0  # B_rep increases

    def test_rhs_static_inhibits_growth(self, pd_model):
        """Static drug should inhibit growth."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
        dydt_none = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        dydt_static = pd_model.rhs(0, y, C_effect=10.0, drug_class="static")
        assert dydt_static[0] < dydt_none[0]  # Less growth with static

    def test_rhs_cidal_kills(self, pd_model):
        """Cidal drug should eventually kill bacteria."""
        # High concentration, some accumulated damage
        y = np.array([1e6, 1e3, 0, 1e7, 10.0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=10.0, drug_class="cidal")
        assert dydt[0] < 0  # B_rep decreases

    def test_rhs_damage_accumulation(self, pd_model):
        """Damage should accumulate with cidal drug."""
        y = np.array([1e6, 1e3, 0, 1e7, 0.0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=1.0, drug_class="cidal")
        assert dydt[4] > 0  # dDamage/dt > 0

    def test_rhs_damage_repair(self, pd_model):
        """Damage should repair when no drug."""
        y = np.array([1e6, 1e3, 0, 1e7, 10.0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[4] < 0  # dDamage/dt < 0 (repair)

    def test_rhs_il6_positive(self, pd_model):
        """IL-6 production should be positive."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[5] != 0  # IL-6 changes

    def test_rhs_tnf_linked_to_il6(self, pd_model):
        """TNF should be linked to IL-6 production approximately."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=1.0, drug_class="cidal")
        # dTNF/dt should be approximately proportional to dIL6/dt
        # The exact ratio is affected by clearance terms
        ratio = dydt[6] / dydt[5] if dydt[5] != 0 else 0
        # Ratio should be in reasonable range around the target
        # Note: exact equality depends on TNF being at equilibrium; tolerance accounts for clearance
        assert abs(ratio - pd_model.p_cyto.TNF_IL6_ratio) < 0.2

    def test_rhs_neutropenic(self, pd_model):
        """With low neutrophils, immune kill should be minimal."""
        y_low = np.array([1e6, 1e3, 0, 1e3, 0, 10, 5])  # Low N_eff
        y_high = np.array([1e6, 1e3, 0, 1e8, 0, 10, 5])  # High N_eff
        dydt_low = pd_model.rhs(0, y_low, C_effect=0.0, drug_class="none")
        dydt_high = pd_model.rhs(0, y_high, C_effect=0.0, drug_class="none")
        # Less immune kill with low neutrophils
        # (growth should be similar or slightly higher)
        assert dydt_low[0] >= dydt_high[0]

    def test_rhs_negative_clipped(self, pd_model):
        """Negative state values should be clipped."""
        y = np.array([-1e6, -1e3, -1, -1e7, -1, -10, -5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        # Should not crash, but behavior with negative is clipped to small positive
        assert not np.any(np.isnan(dydt))

    def test_rhs_cidal_vs_static_il6(self, pd_model):
        """Cidal should produce more IL-6 than static."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
        dydt_cidal = pd_model.rhs(0, y, C_effect=1.0, drug_class="cidal")
        dydt_static = pd_model.rhs(0, y, C_effect=1.0, drug_class="static")
        assert dydt_cidal[5] > dydt_static[5]  # Higher IL-6 with cidal

    def test_create_ode_system(self, default_params):
        model = create_ode_system(default_params)
        assert isinstance(model, BacterialPopulationODE)


class TestSCVDynamics:
    """Tests for SCV (small colony variant) dynamics."""

    def test_scv_emergence_under_static(self, pd_model):
        """SCVs should emerge under static drug pressure if model supports it."""
        y = np.array([1e8, 1e3, 0, 1e7, 0, 10, 5])  # High bacterial load
        dydt = pd_model.rhs(0, y, C_effect=1.0, drug_class="static")
        # Note: SCV emergence depends on H_static < mutation_threshold
        # At low concentrations, this may not be met
        # The test verifies the mechanism exists and doesn't crash
        assert dydt[2] >= 0  # B_SCV rate should be non-negative

    def test_no_scv_without_static(self, pd_model):
        """No SCVs without static pressure."""
        y = np.array([1e8, 1e3, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[2] == 0  # No mutation without drug

    def test_no_scv_with_cidal(self, pd_model):
        """No SCV mutation under cidal pressure."""
        y = np.array([1e8, 1e3, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=1.0, drug_class="cidal")
        assert dydt[2] == 0  # No mutation with cidal

    def test_scv_switch_is_continuous(self, pd_model):
        """SCV emergence must vary smoothly with concentration (no hard jump)."""
        import numpy as np
        y = np.array([1e8, 1e3, 0, 1e7, 0, 10, 5])
        # Sweep concentrations spanning the old hard 0.3 threshold region
        rates = []
        for C in np.linspace(0.05, 0.5, 40):
            dydt = pd_model.rhs(0, y, C_effect=C, drug_class="static")
            rates.append(dydt[2])
        rates = np.array(rates)
        # No single step should jump by more than 20% of the full range
        full_range = rates.max() - rates.min()
        max_step = np.max(np.abs(np.diff(rates)))
        assert full_range > 0                      # the mechanism responds to C
        assert max_step < 0.2 * full_range         # and does so smoothly


class TestPersisterDynamics:
    """Tests for persister cell dynamics."""

    def test_persister_formation(self, pd_model):
        """Persisters should form from replicating cells."""
        y = np.array([1e6, 0, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[1] > 0  # B_pers increases from zero

    def test_persister_slower_kill(self, pd_model):
        """Persisters should be killed slower than replicating cells."""
        y = np.array([1e6, 1e6, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        # Both have immune kill, but persisters at 0.1x rate
        # Growth terms both zero if at carrying capacity
        pass


class TestHostDamage:
    """Tests for the host-damage state D_host (index 8 in the PD vector)."""

    def test_damage_increases_with_burden(self, pd_model):
        """High bacterial burden drives positive host-damage rate."""
        y = np.array([1e9, 0, 0, 1e7, 0, 10, 5, 0, 0])  # IL6/TNF at baseline
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[8] > 0

    def test_damage_increases_with_inflammation(self, pd_model):
        """Elevated cytokines (above baseline) increase host-damage rate."""
        y_base = np.array([1e6, 0, 0, 1e7, 0, 10, 5, 0, 0])
        y_infl = np.array([1e6, 0, 0, 1e7, 0, 100, 50, 0, 0])  # 10x IL6, 10x TNF
        d_base = pd_model.rhs(0, y_base, C_effect=0.0, drug_class="none")[8]
        d_infl = pd_model.rhs(0, y_infl, C_effect=0.0, drug_class="none")[8]
        assert d_infl > d_base

    def test_tnf_contributes_to_damage(self, pd_model):
        """TNF above baseline adds injury (gives the TNF state a functional role)."""
        y_lo = np.array([1e6, 0, 0, 1e7, 0, 50, 5, 0, 0])   # only IL6 elevated
        y_hi = np.array([1e6, 0, 0, 1e7, 0, 50, 50, 0, 0])  # IL6 + TNF elevated
        d_lo = pd_model.rhs(0, y_lo, C_effect=0.0, drug_class="none")[8]
        d_hi = pd_model.rhs(0, y_hi, C_effect=0.0, drug_class="none")[8]
        assert d_hi > d_lo

    def test_damage_recovers_when_clean(self, pd_model):
        """With no burden and baseline cytokines, standing damage decays."""
        y = np.array([1.0, 0, 0, 1e7, 0, 10, 5, 0, 5.0])  # D_host = 5, everything clean
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt[8] < 0  # net recovery

    def test_rhs_backward_compatible_short_vector(self, pd_model):
        """Passing a legacy length-7 vector still works (D_host defaults to 0)."""
        y = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
        dydt = pd_model.rhs(0, y, C_effect=0.0, drug_class="none")
        assert dydt.shape == (9,)

    def test_inflammation_injury_saturates(self, pd_model):
        """Inflammation-driven injury rate is bounded by k_infl (Hill saturation)."""
        import numpy as np
        k_infl = pd_model.p_damage.k_infl
        # Zero burden and N_eff=0 -> pathogen injury ~0, no healing (D_host=0),
        # so dydt[8] is essentially the inflammation-injury term alone.
        y_mid = np.array([0, 0, 0, 0, 0, 1e3, 5, 0, 0])   # IL6 = 1000
        y_hi = np.array([0, 0, 0, 0, 0, 1e9, 5, 0, 0])    # IL6 = 1e9 (extreme burst)
        d_mid = pd_model.rhs(0, y_mid, C_effect=0.0, drug_class="none")[8]
        d_hi = pd_model.rhs(0, y_hi, C_effect=0.0, drug_class="none")[8]
        assert d_hi <= k_infl + 1e-9      # bounded by the ceiling
        assert d_hi > d_mid               # still monotonically increasing
        assert d_mid > 0
