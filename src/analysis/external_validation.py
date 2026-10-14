"""
External (curve-level) validation for the QSP antibiotic model.

This module is the quantitative complement to ``literature_validation.py``.
Where that module checks whether a single model output falls inside a published
*range* (face validity), this module compares full model *trajectories* against
digitized data points extracted from independent published figures, and reports
quantitative discrepancy metrics (RMSE, bias, MAE, R-squared, normalized RMSE)
plus a visual-predictive-check (VPC) style overlay.

Independence is enforced in code: every dataset declares a ``role`` of either
``calibration`` (a source whose data was used to tune model constants) or
``validation`` (a source held out entirely). Reported external-validation
metrics are computed over the ``validation`` set only, so "validation" means
what it says.

Three dataset kinds are supported:

  * ``pk``        - plasma concentration vs time. The model PK layer is driven by
                    the matching dose/regimen and the central (plasma) concentration
                    is reconstructed analytically and compared to the digitized
                    concentration-time profile.
  * ``timekill``  - log10 CFU/mL vs time under a *constant* drug concentration
                    (the standard in-vitro time-kill condition). The 8-state PD
                    ODE is integrated directly with a fixed effect-site
                    concentration and immune effectors switched off (broth, no
                    neutrophils), isolating the drug's intrinsic kill kinetics.
  * ``cytokine``  - IL-6 / TNF vs time. A standard infection scenario is run and
                    the cytokine trajectory shape (peak magnitude, peak timing) is
                    compared qualitatively.

Digitized data live as CSV files (columns ``time_h,value``) under
``data/external_validation/``, indexed by ``registry.json``. See that folder's
README for the digitization workflow.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import brentq

from src.core.parameters import (get_default_parameters,
                                 get_drug_pk_parameters,
                                 immune_free_parameters)
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "external_validation")
REGISTRY_PATH = os.path.join(DATA_DIR, "registry.json")
DEFAULT_FIG_DIR = os.path.join(_PROJECT_ROOT, "results", "figures", "external_validation")
DEFAULT_REPORT_PATH = os.path.join(
    _PROJECT_ROOT, "results", "trial_data", "external_validation_report.json"
)


# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------

@dataclass
class DigitizedDataset:
    """A single digitized published dataset and the scenario that reproduces it."""

    id: str
    kind: str            # 'pk' | 'timekill' | 'cytokine'
    drug: str
    drug_class: str      # 'cidal' | 'static'
    role: str            # 'calibration' | 'validation'
    csv: str             # filename relative to DATA_DIR
    y_unit: str
    y_space: str         # 'linear' or 'log10' -- the space metrics are computed in
    source: str          # full citation
    scenario: Dict       # kind-specific scenario configuration
    organism: Optional[str] = None
    notes: str = ""

    # populated after loading the CSV
    t_obs: Optional[np.ndarray] = field(default=None, repr=False)
    y_obs: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def csv_path(self) -> str:
        return os.path.join(DATA_DIR, self.csv)

    @property
    def has_data(self) -> bool:
        return (
            os.path.exists(self.csv_path)
            and self.t_obs is not None
            and len(self.t_obs) > 0
        )

    def load(self) -> "DigitizedDataset":
        """Load (time, value) pairs from the CSV, skipping blank/placeholder rows."""
        t, y = [], []
        if not os.path.exists(self.csv_path):
            return self
        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                cell = row[0].strip()
                if not cell or cell.startswith("#") or cell.lower() in ("time_h", "time"):
                    continue
                try:
                    ti = float(row[0])
                    yi = float(row[1])
                except (ValueError, IndexError):
                    continue
                t.append(ti)
                y.append(yi)
        if t:
            order = np.argsort(t)
            self.t_obs = np.asarray(t)[order]
            self.y_obs = np.asarray(y)[order]
        return self


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(path: str = REGISTRY_PATH) -> List[DigitizedDataset]:
    """Load all datasets declared in the registry and attach their CSV data."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Registry not found at {path}. Create it (see "
            f"data/external_validation/README.md)."
        )
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    datasets = []
    for entry in spec.get("datasets", []):
        ds = DigitizedDataset(
            id=entry["id"],
            kind=entry["kind"],
            drug=entry["drug"],
            drug_class=entry.get("drug_class", "cidal"),
            role=entry.get("role", "validation"),
            csv=entry["csv"],
            y_unit=entry.get("y_unit", ""),
            y_space=entry.get("y_space", "linear"),
            source=entry.get("source", ""),
            scenario=entry.get("scenario", {}),
            organism=entry.get("organism"),
            notes=entry.get("notes", ""),
        ).load()
        datasets.append(ds)
    return datasets


# ---------------------------------------------------------------------------
# Model prediction at observed time points
# ---------------------------------------------------------------------------

def predict_pk(ds: DigitizedDataset) -> np.ndarray:
    """Model plasma (central) concentration (mg/L) at the dataset's time points."""
    sc = ds.scenario
    pk = get_drug_pk_parameters(ds.drug)
    pk_model = TwoCompartmentPKModel(
        CL=pk.CL, Vc=pk.Vc, Vp=pk.Vp, Q=pk.Q, Ka=pk.Ka, Kp=pk.Kp,
        effect_site_model=True,
    )
    regimen = DosingRegimen(
        dose_mg=sc.get("dose_mg", 1000.0),
        interval_hours=sc.get("interval_hours", 8.0),
        start_time=0.0,
        n_doses=sc.get("n_doses", 1),
        infusion_duration_min=sc.get("infusion_min", 30.0),
    )
    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)
    ic = {"B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0.0, "N_eff": 1e7,
          "Damage": 0, "IL6": 10, "TNF": 5}
    t_end = float(max(ds.t_obs[-1], regimen.interval_hours * regimen.n_doses)) + 1.0
    res = run_simulation(
        pk_model=pk_model, regimen=regimen, pd_model=pd_model,
        initial_conditions=ic, t_span=(0, t_end),
        drug_class=ds.drug_class, weight_kg=sc.get("weight_kg", 70.0),
    )
    # get_effect_site_concentration returns Kp * C_plasma; divide out Kp for plasma.
    c_effect = res.get_effect_site_concentration()
    c_plasma = c_effect / max(pk.Kp, 1e-9)
    return np.interp(ds.t_obs, res.t, c_plasma)


def predict_timekill(ds: DigitizedDataset) -> np.ndarray:
    """
    Model log10 CFU/mL at the dataset's time points under a CONSTANT drug
    concentration (in-vitro time-kill condition), with immune effectors off.
    """
    sc = ds.scenario
    params = parameters_for_scenario(sc)
    # The effector pool is homeostatic (R10), so its set-point must match the
    # scenario's effector level or the pool would be repopulated towards the
    # default baseline. For broth that level is zero, and this is what keeps a
    # time-kill immune-free; it is set below, once `immune` is resolved.
    pd_model = BacterialPopulationODE(params)

    if "mic_multiple" in sc:
        # Experimental calibration uses its measured reference. Historical
        # model-stasis normalisation must be explicitly marked in the registry.
        c_const = float(sc["mic_multiple"]) * reference_mic(ds.drug_class, sc)
    else:
        c_const = float(sc.get("concentration_mgL", 0.0))
    immune = float(sc.get("immune", 0.0))  # broth: no neutrophils by default
    params["immune"].N_eff_0 = immune
    drug_class = ds.drug_class
    is_static = (drug_class == "static")

    # 9 PD states: B_rep, B_pers, B_SCV, N_eff, Damage, IL6, TNF, PAMP, D_host
    y0 = np.array([
        float(sc.get("initial_burden", 1e6)),
        float(sc.get("initial_persister", 1e2)),
        float(sc.get("initial_scv", 0.0)),
        immune,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ])

    def rhs(t, y):
        return pd_model.rhs(t, y, C_effect=c_const,
                            drug_class=drug_class, is_static=is_static)

    t_end = float(ds.t_obs[-1]) + 0.5
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                    max_step=0.1, rtol=1e-6, atol=1e-8, dense_output=True)
    y_at = sol.sol(ds.t_obs)
    B_total = np.maximum(y_at[0] + y_at[1] + y_at[2], 1e-10)
    return np.log10(B_total)


def predict_cytokine(ds: DigitizedDataset, marker: str = "IL6") -> np.ndarray:
    """Model cytokine concentration (pg/mL) at the dataset's time points."""
    sc = ds.scenario
    pk = get_drug_pk_parameters(ds.drug)
    pk_model = TwoCompartmentPKModel(
        CL=pk.CL, Vc=pk.Vc, Vp=pk.Vp, Q=pk.Q, Ka=pk.Ka, Kp=pk.Kp,
        effect_site_model=True,
    )
    regimen = DosingRegimen(
        dose_mg=sc.get("dose_mg", 1000.0),
        interval_hours=sc.get("interval_hours", 8.0),
        start_time=0.0,
        n_doses=sc.get("n_doses", 12),
        infusion_duration_min=sc.get("infusion_min", 30.0),
    )
    params = get_default_parameters()
    # Homeostatic set-point must match the scenario's effector level (R10).
    params["immune"].N_eff_0 = float(sc.get("immune", 1e7))
    pd_model = BacterialPopulationODE(params)
    ic = {"B_rep": sc.get("initial_burden", 1e6), "B_pers": 1e2, "B_SCV": 0.0,
          "N_eff": sc.get("immune", 1e7), "Damage": 0, "IL6": 10, "TNF": 5}
    t_end = float(ds.t_obs[-1]) + 1.0
    res = run_simulation(
        pk_model=pk_model, regimen=regimen, pd_model=pd_model,
        initial_conditions=ic, t_span=(0, t_end),
        drug_class=ds.drug_class, weight_kg=sc.get("weight_kg", 70.0),
    )
    t, il6, tnf = res.get_cytokines()
    series = il6 if marker.upper() == "IL6" else tnf
    return np.interp(ds.t_obs, t, series)


# ---------------------------------------------------------------------------
# Dose-response (neutropenic-thigh PK/PD index) validation
#
# Reproduces the Craig-Andes dose-fractionation paradigm: in a neutropenic host
# (immune effectors off), the 24 h Delta-log10 CFU endpoint is mapped against the
# PK/PD index (%fT>MIC for cidal drugs, fAUC/MIC for static drugs). Because the
# model's kill functions are absolute-concentration (not MIC-scaled), we first
# define an *effective MIC* as the constant concentration giving 24 h stasis in
# the neutropenic host, and express the index relative to it. This is the
# "index-driven exposure" approach: the drug's own PK is used, dose is varied to
# sweep the index, and no murine PK is required.
# ---------------------------------------------------------------------------

def parameters_for_scenario(scenario: Dict) -> Dict:
    """Apply explicit experimental bacterial parameters without changing defaults.

    Context growth, starting population and measured MIC are different pieces
    of information. A thigh-growth override must not silently redefine MIC.
    Reject misspelled overrides rather than adding inert dataclass attributes.
    """
    params = get_default_parameters()
    for name, value in scenario.get("bacterial_parameters", {}).items():
        if name not in params["bacteria"].__dataclass_fields__:
            raise ValueError(f"Unknown bacterial parameter: {name}")
        if not np.isfinite(value) or value < 0 or (name in {'kill_C50','kill_hill','B_max'} and value==0):
            raise ValueError(f"Invalid bacterial parameter {name}: {value}")
        setattr(params["bacteria"], name, float(value))
    return params


def _neutropenic_params(scenario=None):
    """Default params with the effector pool switched off (neutropenic host).

    N_eff_0 = 0 zeroes the homeostatic target and the recruited increment
    together, so an effector pool started at zero stays at exactly zero for
    all time. The callers below also start N_eff at 0; both are required,
    since the initial condition alone would otherwise be repopulated by
    homeostasis.

    This is deliberately not wrapped in a try/except. Before the R10 repair
    the line here set ``k_prod = 0``, which was a no-op -- recruitment was
    six orders of magnitude too small to matter either way -- so a silent
    failure was invisible. It is now the only thing that makes the host
    neutropenic, and the bacterial layer is calibrated in that condition, so
    an AttributeError must surface rather than be swallowed.

    ``get_default_parameters`` is resolved from THIS module's namespace on
    purpose: scripts/ds_bacterial_refit.py monkey-patches it here to sweep
    bacterial parameters, so calling ``immune_free_parameters()`` with no
    argument would quietly discard the swept values and fit the defaults.
    """
    return immune_free_parameters(parameters_for_scenario(scenario or {}))


def reference_mic(drug_class: str, scenario: Dict) -> float:
    """Resolve the exposure-index denominator independently of the PD response.

    Prefer a measured MIC in mg/L. ``mic_eff`` remains a compatibility alias
    for an explicitly fixed reference. Historical model-stasis-normalised
    analyses remain available, but their moving denominator must be requested
    explicitly; it is not a default for experimental calibration.
    """
    fixed = scenario.get("mic_reference_mgL", scenario.get("mic_eff"))
    if fixed is not None:
        if not np.isfinite(fixed) or fixed <= 0:
            raise ValueError("MIC reference must be finite and positive")
        return float(fixed)
    if scenario.get("mic_reference_mode") == "model_stasis":
        return effective_mic(drug_class, scenario)
    raise ValueError("Specify a fixed mic_reference_mgL or explicitly request "
                     "mic_reference_mode='model_stasis' for a synthetic analysis")


def _delta_log10_const_conc(pd_model, C, drug_class, initial_burden, t_end=24.0):
    """24 h Delta-log10 CFU under constant concentration C, no immune effectors."""
    is_static = (drug_class == "static")
    y0 = np.array([initial_burden, 1e2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def rhs(t, y):
        return pd_model.rhs(t, y, C_effect=C, drug_class=drug_class, is_static=is_static)

    sol = solve_ivp(rhs, (0.0, t_end), y0, method="LSODA",
                    max_step=0.2, rtol=1e-6, atol=1e-8)
    b0 = y0[0] + y0[1] + y0[2]
    bend = max(sol.y[0, -1] + sol.y[1, -1] + sol.y[2, -1], 1e-10)
    return float(np.log10(bend) - np.log10(max(b0, 1e-10)))


def effective_mic(drug_class: str, scenario: Dict) -> float:
    """
    Effective MIC = constant concentration giving 24 h stasis in the neutropenic
    host. For cidal drugs stasis is Delta-log10 = 0 (a true zero-crossing); for
    static drugs, which asymptote to but never cross stasis, it is the
    concentration where Delta-log10 falls to a small epsilon (default 0.1).

    The coarse grid below only BRACKETS the crossing; the value is then solved
    for continuously. Returning the grid's linear interpolation instead, as
    this function used to, carried a systematic error of about 4%: 25 points
    spread geometrically over 0.01-100 mg/L step by a factor of 1.47, and
    interpolating Delta-log10 linearly in log C across a step that wide is not
    accurate. That error went straight into the MIC-normalised exposures of
    the curve-level validation, and it made this function disagree with the
    two other effective-MIC calculations in the repository
    (``scripts/model_qualification.effective_mic`` and
    ``scripts/ds_phase4_analyses.effect_site_mic``), both of which already
    solved continuously. The three now agree to within the 0.006% their
    slightly different initial states imply.
    """
    params = _neutropenic_params(scenario)
    pd_model = BacterialPopulationODE(params)
    b0 = scenario.get("initial_burden", 1e6)
    target = 0.0 if drug_class == "cidal" else scenario.get("static_stasis_eps", 0.1)
    Cs = np.geomspace(scenario.get("mic_search_lo", 0.01),
                      scenario.get("mic_search_hi", 100.0),
                      scenario.get("mic_search_n", 25))
    deltas = np.array([_delta_log10_const_conc(pd_model, C, drug_class, b0) for C in Cs])

    def excess(log_c):
        return _delta_log10_const_conc(
            pd_model, float(np.exp(log_c)), drug_class, b0) - target

    # deltas decrease as C increases; find the crossing of `target`.
    for i in range(1, len(Cs)):
        if deltas[i - 1] >= target >= deltas[i]:
            log_c = brentq(excess, np.log(Cs[i - 1]), np.log(Cs[i]),
                           xtol=1e-8, rtol=1e-12)
            return float(np.exp(log_c))
    # fallback: closest point to target
    return float(Cs[int(np.argmin(np.abs(deltas - target)))])


def dose_response_curve(ds: DigitizedDataset):
    """
    Build the model's (PK/PD index, Delta-log10 CFU) curve by sweeping dose at a
    fixed interval in the neutropenic host. Returns (index_grid, delta_grid,
    mic_eff), sorted by index.
    """
    sc = ds.scenario
    mic = reference_mic(ds.drug_class, sc)
    pk = get_drug_pk_parameters(ds.drug)
    params = _neutropenic_params(sc)

    interval = sc.get("interval_hours", 6.0)
    t_end = sc.get("t_end", 24.0)
    n_doses = int(np.ceil(t_end / interval))
    weight = sc.get("weight_kg", 70.0)
    index_kind = sc.get("index", "ft_mic")

    # Reference dose whose peak effect-site concentration is ~ MIC, then sweep
    # from sub-MIC to high multiples to span the full index range.
    vc_total = pk.Vc * weight
    ref_dose = mic * vc_total / max(pk.Kp, 1e-9)
    doses = np.geomspace(sc.get("dose_lo_mult", 0.05) * ref_dose,
                         sc.get("dose_hi_mult", 200.0) * ref_dose,
                         sc.get("n_grid", 18))

    idx, delta = [], []
    tu = np.linspace(0.0, t_end, 1000)
    for dose in doses:
        pk_model = TwoCompartmentPKModel(
            CL=pk.CL, Vc=pk.Vc, Vp=pk.Vp, Q=pk.Q, Ka=pk.Ka, Kp=pk.Kp,
            effect_site_model=True)
        regimen = DosingRegimen(
            dose_mg=float(dose), interval_hours=interval, start_time=0.0,
            n_doses=n_doses, infusion_duration_min=sc.get("infusion_min", 30.0))
        pd_model = BacterialPopulationODE(params)
        ic = {"B_rep": sc.get("initial_burden", 1e6),
              "B_pers": sc.get("initial_persister", 1e2), "B_SCV": 0.0,
              "N_eff": 0.0, "Damage": 0, "IL6": 0, "TNF": 0, "PAMP": 0}
        res = run_simulation(pk_model, regimen, pd_model, ic, t_span=(0, t_end),
                             drug_class=ds.drug_class, weight_kg=weight)
        c_eff = np.interp(tu, res.t, res.get_effect_site_concentration())
        if index_kind == "ft_mic":
            # Resolve crossings continuously instead of counting 1000 samples.
            # This keeps the index from jumping by 0.1 percentage point when
            # dose or a context parameter changes slightly.
            from src.core.simulation import SimulationResult
            ix = 100.0 * float(SimulationResult._time_above_threshold(
                res.t, res.get_effect_site_concentration(), mic)) / t_end
        else:  # fauc_mic
            ix = float(trapezoid(c_eff, tu) / mic)          # fAUC/MIC over t_end
        _, B = res.get_bacterial_burden()
        d = np.log10(max(B[-1], 1e-10)) - np.log10(max(B[0], 1e-10))
        idx.append(ix)
        delta.append(d)

    order = np.argsort(idx)
    return np.asarray(idx)[order], np.asarray(delta)[order], mic


def stasis_index(index_grid, delta_grid, target: float = 0.0):
    """Index value where the model Delta-log10 curve crosses `target` (stasis)."""
    ig, dg = np.asarray(index_grid), np.asarray(delta_grid)
    for i in range(1, len(ig)):
        if dg[i - 1] >= target >= dg[i]:
            return float(np.interp(target, [dg[i], dg[i - 1]], [ig[i], ig[i - 1]]))
    return None  # never reaches stasis within the swept range


def predict_dose_response(ds: DigitizedDataset) -> np.ndarray:
    """Model Delta-log10 CFU at the dataset's observed index values."""
    ig, dg, _ = dose_response_curve(ds)
    return np.interp(ds.t_obs, ig, dg)


def predict(ds: DigitizedDataset) -> np.ndarray:
    """Dispatch to the appropriate predictor for the dataset kind."""
    if ds.kind == "pk":
        return predict_pk(ds)
    if ds.kind == "timekill":
        return predict_timekill(ds)
    if ds.kind == "cytokine":
        marker = ds.scenario.get("marker", "IL6")
        return predict_cytokine(ds, marker=marker)
    if ds.kind == "dose_response":
        return predict_dose_response(ds)
    raise ValueError(f"Unknown dataset kind: {ds.kind!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class DiscrepancyMetrics:
    n: int
    rmse: float
    mae: float
    bias: float          # mean signed error (pred - obs)
    r2: float            # coefficient of determination
    nrmse: float         # RMSE normalized by observed range
    space: str           # 'linear' or 'log10'

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(obs: np.ndarray, pred: np.ndarray, space: str) -> DiscrepancyMetrics:
    """
    Quantitative discrepancy between observed and predicted series.

    For ``space == 'log10'`` the inputs are already in log10 units (the natural
    space for CFU comparisons), so errors are in log10 CFU/mL. For ``'linear'``
    (e.g. PK in mg/L) errors are in the native unit.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    n = int(obs.size)
    if n == 0:
        return DiscrepancyMetrics(0, np.nan, np.nan, np.nan, np.nan, np.nan, space)

    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    rng = float(np.max(obs) - np.min(obs))
    nrmse = float(rmse / rng) if rng > 0 else np.nan

    return DiscrepancyMetrics(n, rmse, mae, bias, r2, nrmse, space)


# ---------------------------------------------------------------------------
# Per-dataset validation
# ---------------------------------------------------------------------------

@dataclass
class ExternalValidationResult:
    dataset_id: str
    kind: str
    drug: str
    role: str
    source: str
    metrics: Optional[DiscrepancyMetrics]
    status: str          # 'OK' | 'NO_DATA' | 'ERROR'
    message: str = ""
    t_obs: Optional[List[float]] = None
    y_obs: Optional[List[float]] = None
    y_pred: Optional[List[float]] = None
    extra: Optional[dict] = None   # kind-specific extras (e.g. stasis thresholds)

    def to_dict(self) -> dict:
        d = {
            "dataset_id": self.dataset_id,
            "kind": self.kind,
            "drug": self.drug,
            "role": self.role,
            "source": self.source,
            "status": self.status,
            "message": self.message,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


def validate_dataset(ds: DigitizedDataset) -> ExternalValidationResult:
    """Run the model for one dataset and compute discrepancy metrics."""
    if not ds.has_data:
        return ExternalValidationResult(
            ds.id, ds.kind, ds.drug, ds.role, ds.source, None,
            status="NO_DATA",
            message=f"No digitized points found in {ds.csv} (awaiting digitization).",
        )
    try:
        y_pred = predict(ds)
        metrics = compute_metrics(ds.y_obs, y_pred, ds.y_space)
        extra = None
        if ds.kind == "dose_response":
            ig, dg, mic = dose_response_curve(ds)
            model_stasis = stasis_index(ig, dg, target=0.0)
            obs_stasis = stasis_index(ds.t_obs, ds.y_obs, target=0.0)
            extra = {
                "index": ds.scenario.get("index", "ft_mic"),
                "effective_mic_mgL": mic,
                "model_stasis_index": model_stasis,
                "observed_stasis_index": obs_stasis,
                "stasis_index_error": (None if (model_stasis is None or obs_stasis is None)
                                       else float(model_stasis - obs_stasis)),
            }
        return ExternalValidationResult(
            ds.id, ds.kind, ds.drug, ds.role, ds.source, metrics,
            status="OK",
            t_obs=ds.t_obs.tolist(),
            y_obs=ds.y_obs.tolist(),
            y_pred=[float(v) for v in y_pred],
            extra=extra,
        )
    except Exception as e:  # noqa: BLE001 - surface any model/scenario failure
        return ExternalValidationResult(
            ds.id, ds.kind, ds.drug, ds.role, ds.source, None,
            status="ERROR", message=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# VPC-style overlay plotting
# ---------------------------------------------------------------------------

def plot_overlay(ds: DigitizedDataset, result: ExternalValidationResult,
                 save_path: Optional[str] = None) -> Optional[str]:
    """Overlay the model trajectory and the digitized points for one dataset."""
    if result.status != "OK":
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dose-response: x-axis is the PK/PD index, y is 24 h Delta-log10 CFU.
    if ds.kind == "dose_response":
        ig, dg, mic = dose_response_curve(ds)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ig, dg, "-", color="#4C72B0", lw=2, label="Model")
        ax.plot(ds.t_obs, ds.y_obs, "o", color="#C44E52", ms=7,
                label="Digitized (published)", zorder=5)
        ax.axhline(0.0, color="0.5", ls="--", lw=1, label="Stasis")
        ex = result.extra or {}
        ms, os_ = ex.get("model_stasis_index"), ex.get("observed_stasis_index")
        if ms is not None:
            ax.axvline(ms, color="#4C72B0", ls=":", lw=1)
        if os_ is not None:
            ax.axvline(os_, color="#C44E52", ls=":", lw=1)
        idx_label = "%fT>MIC" if ex.get("index", "ft_mic") == "ft_mic" else "fAUC/MIC"
        ax.set_xlabel(idx_label)
        ax.set_ylabel("24 h Δlog₁₀ CFU (vs inoculum)")
        ax.set_title(f"{ds.id}  [{ds.role.upper()}]\n{ds.drug} - neutropenic dose-response")
        txt = f"MIC_eff = {mic:.3g} mg/L"
        if ms is not None and os_ is not None:
            txt += f"\nstasis: model {ms:.1f} vs obs {os_:.1f}"
        if result.metrics and np.isfinite(result.metrics.rmse):
            txt += f"\nRMSE = {result.metrics.rmse:.2f} log₁₀"
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        if save_path is None:
            os.makedirs(DEFAULT_FIG_DIR, exist_ok=True)
            save_path = os.path.join(DEFAULT_FIG_DIR, f"{ds.id}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    # Dense model trajectory for the smooth line
    t_dense = np.linspace(0.0, float(ds.t_obs[-1]), 200)
    dense_ds = DigitizedDataset(
        id=ds.id, kind=ds.kind, drug=ds.drug, drug_class=ds.drug_class,
        role=ds.role, csv=ds.csv, y_unit=ds.y_unit, y_space=ds.y_space,
        source=ds.source, scenario=ds.scenario, organism=ds.organism,
    )
    dense_ds.t_obs = t_dense
    try:
        y_dense = predict(dense_ds)
    except Exception:
        y_dense = None

    fig, ax = plt.subplots(figsize=(7, 5))
    if y_dense is not None:
        ax.plot(t_dense, y_dense, "-", color="#4C72B0", lw=2,
                label="Model prediction")
    ax.plot(ds.t_obs, ds.y_obs, "o", color="#C44E52", ms=7,
            label="Digitized (published)", zorder=5)

    m = result.metrics
    unit = ds.y_unit
    if ds.y_space == "log10":
        ax.set_ylabel(f"log10 {unit}" if "log10" not in unit else unit)
    else:
        ax.set_ylabel(unit)
    ax.set_xlabel("Time (h)")
    role_tag = ds.role.upper()
    ax.set_title(f"{ds.id}  [{role_tag}]\n{ds.drug} - {ds.kind}")
    if m and np.isfinite(m.rmse):
        txt = f"RMSE = {m.rmse:.3g} {unit}\nbias = {m.bias:+.3g}\nR2 = {m.r2:.3f}  (n={m.n})"
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if save_path is None:
        os.makedirs(DEFAULT_FIG_DIR, exist_ok=True)
        save_path = os.path.join(DEFAULT_FIG_DIR, f"{ds.id}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_external_validation(
    registry_path: str = REGISTRY_PATH,
    roles: Tuple[str, ...] = ("validation",),
    make_plots: bool = True,
    report_path: str = DEFAULT_REPORT_PATH,
    verbose: bool = True,
) -> dict:
    """
    Run curve-level external validation over the registry.

    By default only ``validation``-role datasets contribute to the reported
    metrics (true held-out external validation). Pass ``roles=("calibration",
    "validation")`` to also report calibration-set fit for context.
    """
    datasets = load_registry(registry_path)
    selected = [d for d in datasets if d.role in roles]

    results: List[ExternalValidationResult] = []
    for ds in selected:
        res = validate_dataset(ds)
        results.append(res)
        if make_plots and res.status == "OK":
            plot_overlay(ds, res)
        if verbose:
            if res.status == "OK":
                m = res.metrics
                print(f"[OK ]  {ds.id:<34} {ds.role:<11} "
                      f"RMSE={m.rmse:.3g} bias={m.bias:+.3g} R2={m.r2:.3f} n={m.n}")
            else:
                print(f"[{res.status:<4}] {ds.id:<34} {ds.role:<11} {res.message}")

    ok = [r for r in results if r.status == "OK"]
    # Pooled RMSE per space, weighted by n
    def _pooled(predicate) -> Optional[float]:
        rs = [r for r in ok if r.metrics and predicate(r)]
        if not rs:
            return None
        num = sum((r.metrics.rmse ** 2) * r.metrics.n for r in rs)
        den = sum(r.metrics.n for r in rs)
        return float(np.sqrt(num / den)) if den else None

    summary = {
        "n_datasets": len(selected),
        "n_evaluated": len(ok),
        "n_no_data": sum(1 for r in results if r.status == "NO_DATA"),
        "n_error": sum(1 for r in results if r.status == "ERROR"),
        "roles_included": list(roles),
        "pooled_rmse_log10_cfu": _pooled(lambda r: r.metrics.space == "log10"),
        "pooled_rmse_linear": _pooled(lambda r: r.metrics.space == "linear"),
        "pooled_rmse_timekill": _pooled(lambda r: r.kind == "timekill"),
        "pooled_rmse_dose_response": _pooled(lambda r: r.kind == "dose_response"),
    }
    report = {"summary": summary, "results": [r.to_dict() for r in results]}

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print("EXTERNAL VALIDATION SUMMARY")
        print("=" * 70)
        print(f"  datasets (roles={list(roles)}): {summary['n_datasets']}")
        print(f"  evaluated: {summary['n_evaluated']}   "
              f"no-data: {summary['n_no_data']}   error: {summary['n_error']}")
        if summary["pooled_rmse_log10_cfu"] is not None:
            print(f"  pooled RMSE (log10 CFU/mL): {summary['pooled_rmse_log10_cfu']:.3f}")
        if summary["pooled_rmse_linear"] is not None:
            print(f"  pooled RMSE (linear units): {summary['pooled_rmse_linear']:.3f}")
        print(f"  report: {report_path}")
        print("=" * 70)

    return report


if __name__ == "__main__":
    run_external_validation()
