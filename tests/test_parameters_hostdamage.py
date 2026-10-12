"""Tests for host-damage and SCV-switch parameters."""
from src.core.parameters import (
    get_default_parameters,
    HostDamageParameters,
    BacterialParameters,
)


def test_host_damage_parameters_defaults():
    p = HostDamageParameters()
    assert p.k_path > 0
    assert p.B50 > 0
    assert p.k_infl > 0
    assert p.k_heal > 0
    assert p.w_TNF >= 0
    assert p.IL6_ref > 0
    assert p.TNF_ref > 0


def test_default_parameters_has_damage_key():
    params = get_default_parameters()
    assert "damage" in params
    assert isinstance(params["damage"], HostDamageParameters)


def test_bacterial_params_have_scv_switch():
    b = BacterialParameters()
    assert 0.0 < b.scv_switch_midpoint < 1.0
    assert b.scv_switch_width > 0
