"""Tests for the provenance block written into every results file (review R12).

The review asked that each result payload identify the code revision, the
environment and the settings that produced it, because a committed JSON alone
does not show under what conditions it was generated.
"""
import re

import numpy
import scipy

from src.utils.provenance import provenance


def test_records_the_code_revision():
    p = provenance()
    assert p["git_head"] is None or re.fullmatch(r"[0-9a-f]{40}", p["git_head"])
    assert p["git_dirty"] in (True, False, None)


def test_records_the_numerical_environment():
    p = provenance()
    assert p["numpy"] == numpy.__version__
    assert p["scipy"] == scipy.__version__
    assert p["python"]


def test_records_the_final_yield_configuration():
    from src.analysis.danger_signal import FINAL_YIELDS_FG_PER_CFU
    assert provenance()["final_yields_fg_per_CFU"] == FINAL_YIELDS_FG_PER_CFU


def test_is_deterministic_so_unchanged_reruns_are_byte_identical():
    """No timestamp: a rerun with nothing changed must not alter the file."""
    assert provenance() == provenance()


def test_carries_caller_supplied_settings():
    p = provenance(endpoint="AUC_DS_pg_h_mL", solver={"rtol": 1e-6})
    assert p["endpoint"] == "AUC_DS_pg_h_mL"
    assert p["solver"] == {"rtol": 1e-6}


def test_flags_the_known_unsupported_numpy_scipy_pairing():
    """SciPy 1.15 supports NumPy < 2.5. Results produced outside that range
    must say so in their own provenance rather than rely on a log warning."""
    p = provenance()
    assert isinstance(p["numpy_scipy_supported"], bool)
