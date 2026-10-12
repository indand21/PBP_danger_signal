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
    block = {
        # The commit checked out when the file was written. If git_dirty is
        # true the file was produced from uncommitted changes on top of it.
        "git_head": head,
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "numpy_scipy_supported": _numpy_scipy_supported(),
        "platform": sys.platform,
        "final_yields_fg_per_CFU": dict(FINAL_YIELDS_FG_PER_CFU),
    }
    block.update(extra)
    return block
