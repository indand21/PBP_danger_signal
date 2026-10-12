"""Analysis module exports.

Sensitivity analysis depends on SALib. Keep that import optional so validation
and optimization modules remain usable in lightweight environments.
"""

try:
    from .sensitivity_analysis import *
except ModuleNotFoundError as exc:
    if exc.name != "SALib":
        raise

from .literature_validation import *
from .dosing_optimization import *
