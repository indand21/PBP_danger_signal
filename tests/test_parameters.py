"""
Tests for parameter definitions and loading.
"""
import pytest
import numpy as np
from src.core.parameters import (
    get_default_parameters,
    get_drug_pk_parameters,
    normalize_pk_parameters,
    BacterialParameters,
    ImmuneParameters,
    CytokineParameters,
    PKParameters,
)


class TestBacterialParameters:
    """Tests for BacterialParameters dataclass."""

    def test_default_values(self):
        bp = BacterialParameters()
        assert bp.k_growth == 0.5
        assert bp.B_max == 1e9
        assert bp.k_pers == 0.01
        assert bp.mu_mut == 1e-6
        assert bp.k_repair == 0.3  # t1/2 ≈ 2.3h, persists between doses
        assert bp.MIC_baseline == 1.0

    def test_custom_values(self):
        bp = BacterialParameters(k_growth=0.8, B_max=5e8)
        assert bp.k_growth == 0.8
        assert bp.B_max == 5e8
        # Other defaults preserved
        assert bp.k_pers == 0.01

    def test_positive_constraints(self):
        """All rate parameters should be non-negative."""
        bp = BacterialParameters()
        assert bp.k_growth >= 0
        assert bp.k_pers >= 0
        assert bp.mu_mut >= 0
        assert bp.k_repair >= 0


class TestImmuneParameters:
    """Tests for ImmuneParameters dataclass."""

    def test_default_values(self):
        ip = ImmuneParameters()
        assert ip.N_eff_0 == 1e7
        assert ip.k_prod == 0.5  # back to original value
        assert ip.EC50_immune == 1e5
        assert ip.k_deg_immune == 0.05
        assert ip.k_kill_base == 1e-8  # back to original value

    def test_ec50_positive(self):
        ip = ImmuneParameters()
        assert ip.EC50_immune > 0


class TestCytokineParameters:
    """Tests for CytokineParameters dataclass."""

    def test_default_values(self):
        cp = CytokineParameters()
        assert cp.k_IL6_prod == 4.0
        assert cp.alpha_static == 1.0
        assert cp.alpha_cidal == 3.0
        assert cp.k_IL6_clear == 0.2
        assert cp.TNF_IL6_ratio == 0.3

    def test_cidal_greater_than_static(self):
        """Cidal should produce more cytokines than static."""
        cp = CytokineParameters()
        assert cp.alpha_cidal > cp.alpha_static

    def test_clearance_positive(self):
        cp = CytokineParameters()
        assert cp.k_IL6_clear > 0


class TestPKParameters:
    """Tests for PKParameters dataclass."""

    def test_creation(self):
        pk = PKParameters(CL=10, Vc=1.0, Vp=0.5, Q=2.0, Ka=0.0, Kp=0.4)
        assert pk.CL == 10
        assert pk.Vc == 1.0
        assert pk.Ka == 0.0

    def test_all_positive(self):
        pk = PKParameters(CL=10, Vc=1.0, Vp=0.5, Q=2.0, Ka=0.0, Kp=0.4)
        assert pk.CL > 0
        assert pk.Vc > 0
        assert pk.Vp >= 0
        assert pk.Q >= 0
        assert pk.Kp >= 0


class TestGetDefaultParameters:
    """Tests for get_default_parameters function."""

    def test_returns_dict(self):
        params = get_default_parameters()
        assert isinstance(params, dict)

    def test_has_required_keys(self):
        params = get_default_parameters()
        assert "bacteria" in params
        assert "immune" in params
        assert "cytokine" in params

    def test_correct_types(self):
        params = get_default_parameters()
        assert isinstance(params["bacteria"], BacterialParameters)
        assert isinstance(params["immune"], ImmuneParameters)
        assert isinstance(params["cytokine"], CytokineParameters)


class TestGetDrugPKParameters:
    """Tests for get_drug_pk_parameters function."""

    def test_doxycycline(self):
        pk = get_drug_pk_parameters("doxycycline")
        assert isinstance(pk, PKParameters)
        assert pk.Ka > 0  # Oral
        assert pk.Kp > 0

    def test_meropenem(self):
        pk = get_drug_pk_parameters("meropenem")
        assert isinstance(pk, PKParameters)
        assert pk.Ka == 0  # IV only

    def test_linezolid(self):
        pk = get_drug_pk_parameters("linezolid")
        assert isinstance(pk, PKParameters)

    def test_ciprofloxacin(self):
        pk = get_drug_pk_parameters("ciprofloxacin")
        assert isinstance(pk, PKParameters)

    def test_case_insensitive(self):
        pk1 = get_drug_pk_parameters("MEROPENEM")
        pk2 = get_drug_pk_parameters("meropenem")
        assert pk1.CL == pk2.CL

    def test_unknown_drug_raises(self):
        with pytest.raises(ValueError, match="Unknown drug"):
            get_drug_pk_parameters("unknown_drug")


class TestNormalizePKParameters:
    """Tests for normalize_pk_parameters function."""

    def test_returns_dict(self):
        pk = get_drug_pk_parameters("meropenem")
        normalized = normalize_pk_parameters(pk, weight_kg=70)
        assert isinstance(normalized, dict)

    def test_scales_with_weight(self):
        pk = get_drug_pk_parameters("meropenem")
        n1 = normalize_pk_parameters(pk, weight_kg=50)
        n2 = normalize_pk_parameters(pk, weight_kg=100)
        # Vc and Vp should double with weight; CL and Q should NOT change (already total)
        assert n2["Vc"] > n1["Vc"]
        assert n2["Vp"] > n1["Vp"]
        assert n1["CL"] == n2["CL"]  # CL is total, not per-kg
        assert n1["Q"] == n2["Q"]     # Q is total, not per-kg

    def test_all_keys_present(self):
        pk = get_drug_pk_parameters("meropenem")
        normalized = normalize_pk_parameters(pk, weight_kg=70)
        required_keys = ["CL", "Vc", "Vp", "Q", "Ka", "Kp"]
        for key in required_keys:
            assert key in normalized

    def test_positive_values(self):
        pk = get_drug_pk_parameters("meropenem")
        normalized = normalize_pk_parameters(pk, weight_kg=70)
        for key, val in normalized.items():
            assert val >= 0, f"{key} should be non-negative"
