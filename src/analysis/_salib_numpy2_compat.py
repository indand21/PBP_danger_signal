"""NumPy 2.0 compatibility shim for SALib 1.4.x Sobol analysis.

SALib 1.4.8's Sobol estimators call ``ndarray.ptp()``, a method NumPy 2.0
removed (only the ``np.ptp(arr)`` free function survives). ``numpy.ndarray`` is
an immutable C type, so the method cannot simply be restored on it.

This shim re-executes SALib's own ``SALib.analyze.sobol`` module source into its
existing namespace with the four ``.ptp()`` method calls rewritten to the
equivalent ``np.ptp(...)`` free-function calls. The estimator formulas are
otherwise untouched, so the computed Sobol indices are numerically identical to
what SALib 1.4.8 produced under NumPy < 2.0 -- this restores reproducibility
without changing any result. Preferred over upgrading SALib, which could alter
the estimators.

The shim is a no-op when it is not needed (``ndarray`` still has ``ptp``, i.e.
NumPy < 2.0 or a future SALib that no longer uses the method) and is idempotent.
"""
import numpy as np


def apply():
    """Patch SALib's Sobol module in place if (and only if) NumPy 2 needs it."""
    if hasattr(np.ndarray, "ptp"):
        return  # NumPy < 2.0: SALib works unmodified.
    try:
        import SALib.analyze.sobol as _sobol
    except Exception:
        return  # SALib not installed / import error -- nothing to do.
    if getattr(_sobol, "_numpy2_ptp_patched", False):
        return  # Already patched this process.

    import inspect
    try:
        src = inspect.getsource(_sobol)
    except (OSError, TypeError):
        return  # Source unavailable (e.g. frozen) -- cannot patch safely.

    patched = (
        src
        .replace("np.r_[A[r], B[r]].ptp()", "np.ptp(np.r_[A[r], B[r]])")
        .replace("y.ptp()", "np.ptp(y)")
    )
    if patched == src:
        return  # Target text absent (unexpected SALib version); leave as-is.
    if ".ptp()" in patched:
        # An unhandled .ptp() method call remains: refuse a partial patch rather
        # than silently leaving a landmine. The caller sees the original error.
        raise RuntimeError(
            "SALib numpy-2 ptp shim is incomplete for this SALib version; "
            "residual '.ptp()' calls remain in SALib.analyze.sobol."
        )

    # Re-run the (fixed) module code in its own namespace. Its top-level imports
    # are re-executed against already-loaded modules, and every function --
    # analyze(), first_order(), total_order(), second_order() -- is redefined
    # with the ptp fix while sharing that namespace at call time.
    exec(compile(patched, _sobol.__file__, "exec"), _sobol.__dict__)
    _sobol.__dict__["_numpy2_ptp_patched"] = True


apply()
