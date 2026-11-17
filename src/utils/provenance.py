"""Provenance block written into every danger-signal results file.

Added after an external review (R12) noted that a committed results JSON does
not, by itself, show the code revision, environment or settings that produced
it. Each results file now carries this block.

The block is deliberately deterministic -- no timestamp -- so that rerunning
an analysis with nothing changed reproduces the file byte for byte, and a
diff shows only what actually moved.
"""
import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Dict, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _numpy_scipy_supported() -> Optional[bool]:
    """Whether the installed NumPy satisfies SciPy's own declared range.

    Read from SciPy's package metadata rather than hard-coded, so it stays
    correct when either library is upgraded.
    """
    try:
        from packaging.requirements import Requirement
        import numpy
        for req in importlib.metadata.requires("scipy") or []:
            r = Requirement(req)
            if r.name.lower() == "numpy" and r.marker is None:
                return numpy.__version__ in r.specifier
    except Exception:
        return None
    return None


_INPUT_DIRS = ("src", "scripts", "data", "config")
_INPUT_FILES = ("requirements.txt", "requirements-lock.txt", "pyproject.toml")


def _is_input(path: str) -> bool:
    """Whether a repository path can affect a computed result."""
    path = path.split(" -> ")[-1]          # renames report "old -> new"
    return path.split("/")[0] in _INPUT_DIRS or path in _INPUT_FILES


def provenance(**extra) -> Dict[str, object]:
    """Code revision, environment and the final yield configuration.

    ``extra`` is merged in for caller-specific settings, such as the reported
    endpoint and the solver tolerances of each layer.
    """
    import numpy
    import scipy
    from src.analysis.danger_signal import FINAL_YIELDS_FG_PER_CFU

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    # Only files that can change a result count: code, data and the
    # environment. Results are outputs (a pipeline run rewrites them one after
    # another, which would mark every later file dirty) and the documents are
    # not inputs.
    changed = None if status is None else [
        ln for ln in status.splitlines() if _is_input(ln[3:].strip().strip('"'))]
    block = {
        # The commit checked out when the file was written. If git_dirty is
        # true the file was produced from uncommitted changes to code, data or
        # the environment specification on top of it.
        "git_head": head,
        "git_dirty": None if changed is None else bool(changed),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numpy_scipy_supported": _numpy_scipy_supported(),
        "platform": sys.platform,
        "final_yields_fg_per_CFU": dict(FINAL_YIELDS_FG_PER_CFU),
        "default_parameters": _default_parameters(),
    }
    block.update(extra)
    return block


def _default_parameters() -> Dict[str, object]:
    """The parameter set every analysis starts from: danger-signal layer,
    bacterial PD and the two agents' PK and drug PD. An analysis that varies
    any of them records its own values beside its results."""
    from src.analysis.danger_signal import DangerSignalParameters
    from src.core.parameters import (get_default_parameters, get_drug_pd_parameters,
                                     get_drug_pk_parameters)
    agents = ("ceftazidime", "meropenem")
    return {
        "danger_signal": snapshot(DangerSignalParameters()),
        "bacterial_pd": snapshot(get_default_parameters()),
        "pk": {a: snapshot(get_drug_pk_parameters(a)) for a in agents},
        "drug_pd": {a: snapshot(get_drug_pd_parameters(a)) for a in agents},
    }


def snapshot(obj):
    """A JSON-safe copy of a parameter object: dataclasses, dicts, sequences,
    NumPy scalars and arrays, and plain objects (by their attributes).

    Used to record the complete parameter set a results file was produced
    with, so it can be checked without rerunning the code that built it.
    """
    import dataclasses

    import numpy as np

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: snapshot(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [snapshot(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: snapshot(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)
