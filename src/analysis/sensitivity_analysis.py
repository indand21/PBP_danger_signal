"""
Global Sensitivity Analysis (GSA) for the QSP Antibiotic Model.

Uses SALib (Sensitivity Analysis Library) to perform variance-based
sensitivity analysis using the Sobol method.

Supports:
- Sobol first-order (S1) and total-order (ST) indices
- Parallel evaluation of parameter samples
- Customizable output metrics (e.g., AUC of bacterial burden, peak IL-6)
- Visualization of results
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
from typing import Dict, List, Tuple, Optional, Callable
import warnings

from SALib.sample import saltelli
from SALib.analyze import sobol

# Restore SALib 1.4.x Sobol under NumPy 2.0 (ndarray.ptp removal). No-op
# otherwise; keeps the Sobol estimators numerically identical. See the module
# docstring for details.
from src.analysis import _salib_numpy2_compat  # noqa: F401

from src.core.parameters import (
    BacterialParameters, ImmuneParameters, CytokineParameters,
    get_drug_pk_parameters, get_default_parameters
)
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation


# ---------------------------------------------------------------------------
# Default parameter bounds for sensitivity analysis
# Each entry: (name, [lower, upper])  — units match the parameter definition
# ---------------------------------------------------------------------------

DEFAULT_SA_BOUNDS: Dict[str, List[float]] = {
    # Bacterial parameters
    "k_growth":   [0.1, 1.0],       # per hour
    "B_max":      [1e8, 1e10],      # CFU/mL
    "k_pers":     [0.001, 0.05],    # per hour
    "mu_mut":     [1e-7, 1e-5],     # per cell per generation
    "k_repair":   [0.01, 0.5],      # per hour

    # Immune parameters
    "k_prod":          [0.1, 2.0],       # per hour
    "EC50_immune":     [1e4, 1e6],       # CFU/mL
    "k_deg_immune":    [0.01, 0.2],      # per hour
    "k_kill_base":     [1e-9, 1e-7],     # per mL per hour

    # Cytokine parameters
    "k_IL6_prod":      [0.5, 10.0],      # pg/mL/hour per 10^6 CFU
    "alpha_cidal":     [1.0, 8.0],       # relative production
    "alpha_static":    [0.2, 3.0],       # relative production
    "k_IL6_clear":     [0.05, 0.5],      # per hour
    "TNF_IL6_ratio":   [0.1, 0.6],       # dimensionless

    # Host-damage parameters
    "k_infl":     [0.005, 0.10],    # per hour, inflammation-driven injury scale
    "k_path":     [0.02, 0.20],     # per hour, pathogen-driven injury scale
    "k_heal":     [0.02, 0.30],     # per hour, host recovery rate

    # PK parameters (per drug — will be scaled by weight)
    "CL":  [5.0, 30.0],   # mL/min/kg
    "Vc":  [0.1, 1.0],    # L/kg
    "Vp":  [0.3, 2.0],    # L/kg
    "Q":   [0.5, 5.0],    # mL/min/kg
    "Kp":  [0.1, 1.0],    # dimensionless (tissue penetration)
}

# Mapping from SA parameter name to (dataclass, attribute)
_PARAM_MAP = {
    "k_growth":        ("bacteria", "k_growth"),
    "B_max":           ("bacteria", "B_max"),
    "k_pers":          ("bacteria", "k_pers"),
    "mu_mut":          ("bacteria", "mu_mut"),
    "k_repair":        ("bacteria", "k_repair"),
    "k_prod":          ("immune", "k_prod"),
    "EC50_immune":     ("immune", "EC50_immune"),
    "k_deg_immune":    ("immune", "k_deg_immune"),
    "k_kill_base":     ("immune", "k_kill_base"),
    "k_IL6_prod":      ("cytokine", "k_IL6_prod"),
    "alpha_cidal":     ("cytokine", "alpha_cidal"),
    "alpha_static":    ("cytokine", "alpha_static"),
    "k_IL6_clear":     ("cytokine", "k_IL6_clear"),
    "TNF_IL6_ratio":   ("cytokine", "TNF_IL6_ratio"),
    "k_infl":          ("damage", "k_infl"),
    "k_path":          ("damage", "k_path"),
    "k_heal":          ("damage", "k_heal"),
    "CL":              ("pk", "CL"),
    "Vc":              ("pk", "Vc"),
    "Vp":              ("pk", "Vp"),
    "Q":               ("pk", "Q"),
    "Kp":              ("pk", "Kp"),
}


# ---------------------------------------------------------------------------
# Output metrics — scalar summaries of a simulation trajectory
# ---------------------------------------------------------------------------

def metric_auc_bacterial_burden(result) -> float:
    """Area-under-curve of log10(total bacterial burden) over time."""
    t, B = result.get_bacterial_burden()
    B_safe = np.maximum(B, 1e-10)
    log_B = np.log10(B_safe)
    return float(trapezoid(log_B, t))


def metric_peak_bacterial_burden(result) -> float:
    """Peak (max) log10(total bacterial burden)."""
    _, B = result.get_bacterial_burden()
    return float(np.log10(max(B.max(), 1e-10)))


def metric_final_bacterial_burden(result) -> float:
    """log10(total bacterial burden) at the final time point."""
    _, B = result.get_bacterial_burden()
    return float(np.log10(max(B[-1], 1e-10)))


def metric_auc_il6(result) -> float:
    """Area-under-curve of IL-6 over time."""
    t, il6, _ = result.get_cytokines()  # get_cytokines -> (t, IL6, TNF)
    return float(trapezoid(il6, t))


def metric_peak_il6(result) -> float:
    """Peak IL-6 concentration."""
    _, il6, _ = result.get_cytokines()  # get_cytokines -> (t, IL6, TNF)
    return float(il6.max())


def metric_peak_resistance_fraction(result) -> float:
    """Peak fraction of SCV population."""
    _, frac = result.get_resistance_fraction()
    return float(frac.max())


def metric_peak_host_damage(result) -> float:
    """Peak host damage (D_host) over the simulation."""
    return float(result.peak_host_damage())


METRICS: Dict[str, Callable] = {
    "auc_burden":         metric_auc_bacterial_burden,
    "peak_burden":        metric_peak_bacterial_burden,
    "final_burden":       metric_final_bacterial_burden,
    "auc_il6":            metric_auc_il6,
    "peak_il6":           metric_peak_il6,
    "peak_resistance":    metric_peak_resistance_fraction,
    "peak_host_damage":   metric_peak_host_damage,
}


# ---------------------------------------------------------------------------
# Core SA engine
# ---------------------------------------------------------------------------

def build_sa_problem(
    param_names: Optional[List[str]] = None,
    bounds: Optional[Dict[str, List[float]]] = None,
) -> dict:
    """
    Build a SALib Problem dict from parameter names and bounds.

    Parameters:
        param_names: list of parameter names to include. If None, uses all
                     keys in `bounds` (or DEFAULT_SA_BOUNDS).
        bounds: dict of {name: [lower, upper]}. If None, uses DEFAULT_SA_BOUNDS.

    Returns:
        SALib-compatible problem dict with keys: num_vars, names, bounds.
    """
    if bounds is None:
        bounds = DEFAULT_SA_BOUNDS

    if param_names is None:
        param_names = list(bounds.keys())

    # Filter to requested params that exist in bounds
    valid_names = [n for n in param_names if n in bounds]
    if not valid_names:
        raise ValueError("No valid parameter names found in bounds dictionary.")

    return {
        "num_vars": len(valid_names),
        "names": valid_names,
        "bounds": [bounds[n] for n in valid_names],
    }


def _apply_sample_to_params(
    sample: np.ndarray,
    problem: dict,
    base_params: Dict,
    drug_name: str,
    weight_kg: float,
) -> Tuple[BacterialPopulationODE, TwoCompartmentPKModel, DosingRegimen]:
    """
    Create model instances with parameters set from a single SA sample vector.

    Parameters:
        sample: 1-D array of length problem['num_vars']
        problem: SALib problem dict
        base_params: base parameter dict (from get_default_parameters())
        drug_name: drug name for PK lookup
        weight_kg: patient weight

    Returns:
        (pd_model, pk_model, regimen)
    """
    import copy

    # Deep-copy base params so we don't mutate across evaluations
    params = copy.deepcopy(base_params)

    # Also get base PK params
    pk_params = get_drug_pk_parameters(drug_name)

    for i, name in enumerate(problem["names"]):
        val = sample[i]
        group, attr = _PARAM_MAP[name]
        if group == "pk":
            setattr(pk_params, attr, val)
        elif group in ("bacteria", "immune", "cytokine", "damage"):
            setattr(params[group], attr, val)
        else:
            raise ValueError(f"Unknown parameter group: {group}")

    pd_model = BacterialPopulationODE(params)
    pk_model = TwoCompartmentPKModel(
        CL=pk_params.CL,
        Vc=pk_params.Vc,
        Vp=pk_params.Vp,
        Q=pk_params.Q,
        Ka=pk_params.Ka,
        Kp=pk_params.Kp,
        effect_site_model=True,
    )
    regimen = DosingRegimen(
        dose_mg=1000.0,
        interval_hours=8.0,
        start_time=0.0,
        n_doses=12,
        infusion_duration_min=60,
    )
    return pd_model, pk_model, regimen


def _run_single_sample(
    sample: np.ndarray,
    problem: dict,
    base_params: Dict,
    drug_name: str,
    drug_class: str,
    weight_kg: float,
    initial_conditions: Dict,
    t_span: Tuple[float, float],
    metric_fn: Callable,
) -> float:
    """Run a single simulation for one SA sample and return the metric value."""
    pd_model, pk_model, regimen = _apply_sample_to_params(
        sample, problem, base_params, drug_name, weight_kg
    )
    try:
        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=t_span,
            drug_class=drug_class,
            weight_kg=weight_kg,
        )
        return metric_fn(result)
    except Exception as e:
        warnings.warn(f"Simulation failed for sample: {e}")
        return np.nan


def run_sensitivity_analysis(
    param_names: Optional[List[str]] = None,
    bounds: Optional[Dict[str, List[float]]] = None,
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    weight_kg: float = 70.0,
    initial_conditions: Optional[Dict] = None,
    t_span: Tuple[float, float] = (0, 96),
    metric: str = "auc_burden",
    metric_fn: Optional[Callable] = None,
    n_samples: int = 256,
    calc_second_order: bool = True,
    seed: int = 42,
    print_progress: bool = True,
) -> dict:
    """
    Run a full Sobol sensitivity analysis.

    Parameters:
        param_names: list of parameter names to include. None = all defaults.
        bounds: dict of {name: [lower, upper]}. None = DEFAULT_SA_BOUNDS.
        drug_name: drug to simulate (e.g. "meropenem", "doxycycline").
        drug_class: "cidal" or "static".
        weight_kg: patient weight in kg.
        initial_conditions: dict of initial state values. None = defaults.
        t_span: (t_start, t_end) in hours.
        metric: name of output metric (key in METRICS dict).
        metric_fn: custom metric function accepting a SimulationResult. Overrides `metric`.
        n_samples: base sample count (actual evaluations = n_samples*(2D+2) for Sobol).
        calc_second_order: whether to compute second-order indices.
        seed: random seed for reproducibility.
        print_progress: whether to print progress info.

    Returns:
        dict with keys:
            - "Si": SALib SobolResult object
            - "problem": SALib problem dict
            - "Y": array of metric values
            - "metric_name": name of the metric used
    """
    if initial_conditions is None:
        initial_conditions = {
            "B_rep": 1e5,
            "B_pers": 1e2,
            "B_SCV": 0,
            "N_eff": 1e7,
            "Damage": 0,
            "IL6": 10,
            "TNF": 5,
        }

    if metric_fn is None:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(METRICS.keys())}")
        metric_fn = METRICS[metric]

    problem = build_sa_problem(param_names, bounds)
    base_params = get_default_parameters()

    # Generate Saltelli samples
    # Total evaluations = n_samples * (2D + 2) if second_order, else n_samples * (D + 2)
    X = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    n_total = X.shape[0]

    if print_progress:
        print(f"Sobol sensitivity analysis: {n_total} evaluations")
        print(f"  Parameters: {problem['names']}")
        print(f"  Metric: {metric}")
        print(f"  Drug: {drug_name} ({drug_class})")

    # Evaluate model at each sample point
    Y = np.zeros(n_total)
    for i in range(n_total):
        Y[i] = _run_single_sample(
            X[i], problem, base_params,
            drug_name, drug_class, weight_kg,
            initial_conditions, t_span, metric_fn,
        )
        if print_progress and (i + 1) % max(1, n_total // 10) == 0:
            print(f"  Progress: {i+1}/{n_total} ({100*(i+1)/n_total:.0f}%)")

    # Handle NaN values
    nan_count = np.isnan(Y).sum()
    if nan_count > 0:
        warnings.warn(f"{nan_count}/{n_total} evaluations produced NaN. Replacing with median.")
        Y[np.isnan(Y)] = np.nanmedian(Y)

    # Analyze
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)

    if print_progress:
        print("\nSobol First-Order Indices (S1):")
        for name, s1, s1_conf in zip(problem["names"], Si["S1"], Si["S1_conf"]):
            print(f"  {name:20s}: S1 = {s1:.4f} ± {s1_conf:.4f}")
        print("\nSobol Total-Order Indices (ST):")
        for name, st, st_conf in zip(problem["names"], Si["ST"], Si["ST_conf"]):
            print(f"  {name:20s}: ST = {st:.4f} ± {st_conf:.4f}")

    return {
        "Si": Si,
        "problem": problem,
        "Y": Y,
        "metric_name": metric if metric_fn == METRICS.get(metric) else "custom",
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sobol_indices(
    sa_result: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
) -> None:
    """
    Plot Sobol first-order and total-order sensitivity indices.

    Parameters:
        sa_result: dict returned by run_sensitivity_analysis().
        save_path: if provided, save figure to this path.
        figsize: figure size (width, height) in inches.
        show: whether to call plt.show().
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Si = sa_result["Si"]
    names = sa_result["problem"]["names"]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    s1 = Si["S1"]
    st = Si["ST"]
    s1_conf = Si["S1_conf"]
    st_conf = Si["ST_conf"]

    # Clamp negative indices to zero for display
    s1 = np.maximum(s1, 0)
    st = np.maximum(st, 0)

    bars1 = ax.bar(x - width / 2, s1, width, label="First-order (S1)",
                   yerr=s1_conf, capsize=3, color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, st, width, label="Total-order (ST)",
                   yerr=st_conf, capsize=3, color="#DD8452", alpha=0.85)

    ax.set_ylabel("Sensitivity Index")
    ax.set_title(f"Sobol Sensitivity Analysis — {sa_result['metric_name']}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.close(fig)
    else:
        plt.close(fig)


def plot_convergence(
    sa_results: List[dict],
    sample_counts: List[int],
    metric_name: str = "auc_burden",
    param_name: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Plot convergence of Sobol indices as sample size increases.

    Parameters:
        sa_results: list of SA result dicts from different sample sizes.
        sample_counts: corresponding sample sizes.
        metric_name: metric label for the title.
        param_name: if provided, show only this parameter; otherwise show all.
        save_path: if provided, save figure.
        figsize: figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    names = sa_results[0]["problem"]["names"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    indices_to_show = range(len(names)) if param_name is None else [names.index(param_name)]

    for idx in indices_to_show:
        s1_vals = [r["Si"]["S1"][idx] for r in sa_results]
        st_vals = [r["Si"]["ST"][idx] for r in sa_results]
        label = names[idx]

        axes[0].plot(sample_counts, s1_vals, "o-", label=label, color=colors[idx % len(colors)])
        axes[1].plot(sample_counts, st_vals, "s-", label=label, color=colors[idx % len(colors)])

    axes[0].set_title("First-order (S1) Convergence")
    axes[0].set_xlabel("Base sample size (N)")
    axes[0].set_ylabel("S1")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_title("Total-order (ST) Convergence")
    axes[1].set_xlabel("Base sample size (N)")
    axes[1].set_ylabel("ST")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle(f"Sobol Index Convergence — {metric_name}", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: quick single-run SA with default settings
# ---------------------------------------------------------------------------

def quick_sa(
    n_samples: int = 128,
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    metric: str = "auc_burden",
    param_subset: Optional[List[str]] = None,
) -> dict:
    """
    Run a quick sensitivity analysis with default settings.

    Parameters:
        n_samples: base sample count.
        drug_name: drug to simulate.
        drug_class: "cidal" or "static".
        metric: output metric name.
        param_subset: optional list of parameter names to analyze (default: all).

    Returns:
        SA result dict.
    """
    return run_sensitivity_analysis(
        param_names=param_subset,
        drug_name=drug_name,
        drug_class=drug_class,
        metric=metric,
        n_samples=n_samples,
        calc_second_order=True,
        seed=42,
        print_progress=True,
    )


if __name__ == "__main__":
    # Example: quick SA on bacterial parameters only
    bacterial_params = ["k_growth", "k_pers", "mu_mut", "k_repair", "B_max"]
    result = quick_sa(
        n_samples=64,
        drug_name="meropenem",
        drug_class="cidal",
        metric="auc_burden",
        param_subset=bacterial_params,
    )
    plot_sobol_indices(result, save_path="sensitivity_sobol_indices.png")
    print("\nSaved plot: sensitivity_sobol_indices.png")
