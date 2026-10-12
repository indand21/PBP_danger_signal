"""Host-damage outcomes from full PK/PD simulations."""
import numpy as np


def test_state_vector_includes_d_host(short_simulation_result):
    r = short_simulation_result
    assert "D_host" in r.state_names
    assert r.y.shape[1] == 13  # 4 PK + 9 PD


def test_get_host_damage_shape(short_simulation_result):
    t, d = short_simulation_result.get_host_damage()
    assert d.shape == t.shape


def test_peak_and_terminal_host_damage(short_simulation_result):
    r = short_simulation_result
    peak = r.peak_host_damage()
    terminal = r.terminal_host_damage()
    assert peak >= terminal        # peak is the max over the trajectory
    assert peak > 0                # an active infection accrues some damage
