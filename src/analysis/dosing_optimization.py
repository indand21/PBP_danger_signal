"""
Dosing Optimization Module for the QSP Antibiotic Model.

Solves the inverse problem: given desired outcomes, find optimal dosing regimens.

Optimization objectives:
1. Minimize bacterial burden (maximize killing)
2. Minimize resistance development (minimize MIC increase)
3. Minimize toxicity (minimize peak drug concentration)
4. Minimize treatment duration
5. Multi-objective optimization (Pareto front)

Optimization methods:
- Grid search over dose/interval space
- SciPy minimize with constraints
- Pareto front for multi-objective trade-offs

Key PK/PD targets:
- Cidal drugs: maximize %fT>MIC (target: 40-100%)
- Static drugs: maximize AUC/MIC ratio (target: >25)
- Minimize mutant selection window exposure
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import copy

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation, SimulationResult


# ---------------------------------------------------------------------------
# Optimization parameters
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConstraints:
    """
    Constraints for dosing optimization.

    Attributes:
        dose_min: minimum dose in mg
        dose_max: maximum dose in mg
        interval_min: minimum dosing interval in hours
        interval_max: maximum dosing interval in hours
        n_doses_min: minimum number of doses
        n_doses_max: maximum number of doses
        max_peak_concentration: maximum allowed peak concentration (mg/L)
        min_ft_mic: minimum %fT>MIC target (for cidal drugs)
        min_auc_mic: minimum AUC/MIC target (for static drugs)
    """
    dose_min: float = 100.0
    dose_max: float = 4000.0
    interval_min: float = 4.0
    interval_max: float = 24.0
    n_doses_min: int = 3
    n_doses_max: int = 24
    max_peak_concentration: float = 100.0  # mg/L
    min_ft_mic: float = 40.0  # %fT>MIC
    min_auc_mic: float = 25.0  # AUC/MIC ratio


@dataclass
class OptimizationResult:
    """
    Result of dosing optimization.

    Attributes:
        optimal_dose: optimal dose in mg
        optimal_interval: optimal dosing interval in hours
        optimal_n_doses: optimal number of doses
        objective_value: value of the objective function at optimum
        simulation_result: SimulationResult at optimal parameters
        success: whether optimization converged
        message: convergence message
        all_results: list of all evaluated points (for Pareto analysis)
    """
    optimal_dose: float = 0.0
    optimal_interval: float = 0.0
    optimal_n_doses: int = 0
    objective_value: float = np.inf
    simulation_result: Optional[SimulationResult] = None
    success: bool = False
    message: str = ""
    all_results: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def objective_minimize_burden(
    params: np.ndarray,
    drug_name: str,
    drug_class: str,
    initial_conditions: Dict,
    weight_kg: float = 70.0,
) -> float:
    """
    Objective function: minimize final bacterial burden.

    Parameters:
        params: array [dose_mg, interval_hours, n_doses]
        drug_name: drug name
        drug_class: "cidal" or "static"
        initial_conditions: initial state values
        weight_kg: patient weight

    Returns:
        Objective value (lower is better)
    """
    dose_mg = params[0]
    interval_hours = params[1]
    n_doses = max(1, int(params[2]))

    try:
        # Build models
        params_dict = get_default_parameters()
        pd_model = BacterialPopulationODE(params_dict)

        pk_params = get_drug_pk_parameters(drug_name)
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
            dose_mg=dose_mg,
            interval_hours=interval_hours,
            start_time=0.0,
            n_doses=n_doses,
            infusion_duration_min=60.0,
        )

        t_span = (0, n_doses * interval_hours + 24)

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=t_span,
            drug_class=drug_class,
            weight_kg=weight_kg,
        )

        _, B = result.get_bacterial_burden()
        final_burden = max(B[-1], 1e-10)
        return np.log10(final_burden)

    except Exception:
        return 1e6  # Return large value on failure


def objective_minimize_resistance(
    params: np.ndarray,
    drug_name: str,
    drug_class: str,
    initial_conditions: Dict,
    weight_kg: float = 70.0,
) -> float:
    """
    Objective function: minimize resistance development.

    Parameters:
        params: array [dose_mg, interval_hours, n_doses]
        drug_name: drug name
        drug_class: "cidal" or "static"
        initial_conditions: initial state values
        weight_kg: patient weight

    Returns:
        Objective value (lower is better)
    """
    dose_mg = params[0]
    interval_hours = params[1]
    n_doses = max(1, int(params[2]))

    try:
        params_dict = get_default_parameters()
        pd_model = BacterialPopulationODE(params_dict)

        pk_params = get_drug_pk_parameters(drug_name)
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
            dose_mg=dose_mg,
            interval_hours=interval_hours,
            start_time=0.0,
            n_doses=n_doses,
            infusion_duration_min=60.0,
        )

        t_span = (0, n_doses * interval_hours + 24)

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=t_span,
            drug_class=drug_class,
            weight_kg=weight_kg,
        )

        _, frac_scv = result.get_resistance_fraction()
        return float(frac_scv.max())

    except Exception:
        return 1e6


def objective_multi(
    params: np.ndarray,
    drug_name: str,
    drug_class: str,
    initial_conditions: Dict,
    weight_kg: float = 70.0,
    w_burden: float = 1.0,
    w_resistance: float = 0.5,
    w_toxicity: float = 0.1,
) -> float:
    """
    Multi-objective function combining burden, resistance, and toxicity.

    Parameters:
        params: array [dose_mg, interval_hours, n_doses]
        drug_name: drug name
        drug_class: "cidal" or "static"
        initial_conditions: initial state values
        weight_kg: patient weight
        w_burden: weight for burden objective
        w_resistance: weight for resistance objective
        w_toxicity: weight for toxicity objective

    Returns:
        Weighted objective value (lower is better)
    """
    dose_mg = params[0]
    interval_hours = params[1]
    n_doses = max(1, int(params[2]))

    try:
        params_dict = get_default_parameters()
        pd_model = BacterialPopulationODE(params_dict)

        pk_params = get_drug_pk_parameters(drug_name)
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
            dose_mg=dose_mg,
            interval_hours=interval_hours,
            start_time=0.0,
            n_doses=n_doses,
            infusion_duration_min=60.0,
        )

        t_span = (0, n_doses * interval_hours + 24)

        result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=t_span,
            drug_class=drug_class,
            weight_kg=weight_kg,
        )

        # Burden objective
        _, B = result.get_bacterial_burden()
        final_burden = max(B[-1], 1e-10)
        burden_score = np.log10(final_burden)

        # Resistance objective
        _, frac_scv = result.get_resistance_fraction()
        resistance_score = float(frac_scv.max()) * 100  # Scale to 0-100

        # Toxicity objective (peak drug concentration)
        A_central = result.y[:, 0]
        peak_conc = max(float(A_central.max()) / pk_params.Vc, 0)
        toxicity_score = peak_conc / 10.0  # Normalize

        return (w_burden * burden_score +
                w_resistance * resistance_score +
                w_toxicity * toxicity_score)

    except Exception:
        return 1e6


# ---------------------------------------------------------------------------
# PK/PD target computation
# ---------------------------------------------------------------------------

def compute_ft_mic(
    result: SimulationResult,
    MIC: float = 1.0,
) -> float:
    """
    Compute %fT>MIC (fraction of time drug concentration exceeds MIC).

    Parameters:
        result: SimulationResult
        MIC: minimum inhibitory concentration

    Returns:
        %fT>MIC (0-100)
    """
    t = result.t
    A_central = result.y[:, 0]

    # Get PK parameters from result
    pk_params = get_drug_pk_parameters("meropenem")  # Default
    Vc = pk_params.Vc

    C = A_central / Vc
    above_mic = np.sum(C > MIC)
    total_points = len(C)

    if total_points == 0:
        return 0.0

    return 100.0 * above_mic / total_points


def compute_auc_mic(
    result: SimulationResult,
    MIC: float = 1.0,
) -> float:
    """
    Compute AUC/MIC ratio.

    Parameters:
        result: SimulationResult
        MIC: minimum inhibitory concentration

    Returns:
        AUC/MIC ratio
    """
    from scipy.integrate import trapezoid

    t = result.t
    A_central = result.y[:, 0]

    pk_params = get_drug_pk_parameters("meropenem")
    Vc = pk_params.Vc

    C = A_central / Vc
    auc = trapezoid(C, t)

    if MIC <= 0:
        return 0.0

    return auc / MIC


# ---------------------------------------------------------------------------
# Optimization engines
# ---------------------------------------------------------------------------

def optimize_dosing_grid(
    drug_name: str,
    drug_class: str = "cidal",
    objective: str = "burden",
    initial_conditions: Optional[Dict] = None,
    weight_kg: float = 70.0,
    constraints: Optional[OptimizationConstraints] = None,
    n_dose_points: int = 5,
    n_interval_points: int = 5,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize dosing using grid search.

    Parameters:
        drug_name: drug to optimize
        drug_class: "cidal" or "static"
        objective: "burden", "resistance", or "multi"
        initial_conditions: initial state values
        weight_kg: patient weight
        constraints: OptimizationConstraints
        n_dose_points: number of dose levels to test
        n_interval_points: number of interval levels to test
        verbose: whether to print progress

    Returns:
        OptimizationResult
    """
    if initial_conditions is None:
        initial_conditions = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

    if constraints is None:
        constraints = OptimizationConstraints()

    # Select objective function
    obj_fn = {
        "burden": objective_minimize_burden,
        "resistance": objective_minimize_resistance,
        "multi": objective_multi,
    }[objective]

    # Grid search
    doses = np.linspace(constraints.dose_min, constraints.dose_max, n_dose_points)
    intervals = np.linspace(constraints.interval_min, constraints.interval_max, n_interval_points)

    best_value = np.inf
    best_params = None
    all_results = []

    for dose in doses:
        for interval in intervals:
            n_doses = max(constraints.n_doses_min,
                         min(constraints.n_doses_max,
                             int(168 / interval)))  # ~1 week of treatment

            params = np.array([dose, interval, n_doses])
            value = obj_fn(params, drug_name, drug_class, initial_conditions, weight_kg)

            all_results.append({
                "dose": dose,
                "interval": interval,
                "n_doses": n_doses,
                "objective": value,
            })

            if value < best_value:
                best_value = value
                best_params = params.copy()

            if verbose:
                print(f"  Dose={dose:.0f}mg, Interval={interval:.1f}h, "
                      f"N={n_doses}: obj={value:.4f}")

    # Run final simulation at optimal parameters
    if best_params is not None:
        try:
            params_dict = get_default_parameters()
            pd_model = BacterialPopulationODE(params_dict)

            pk_params = get_drug_pk_parameters(drug_name)
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
                dose_mg=best_params[0],
                interval_hours=best_params[1],
                start_time=0.0,
                n_doses=max(1, int(best_params[2])),
                infusion_duration_min=60.0,
            )

            t_span = (0, regimen.n_doses * regimen.interval_hours + 24)

            sim_result = run_simulation(
                pk_model=pk_model,
                regimen=regimen,
                pd_model=pd_model,
                initial_conditions=initial_conditions,
                t_span=t_span,
                drug_class=drug_class,
                weight_kg=weight_kg,
            )

            return OptimizationResult(
                optimal_dose=best_params[0],
                optimal_interval=best_params[1],
                optimal_n_doses=max(1, int(best_params[2])),
                objective_value=best_value,
                simulation_result=sim_result,
                success=True,
                message="Grid search converged",
                all_results=all_results,
            )
        except Exception as e:
            return OptimizationResult(
                optimal_dose=best_params[0],
                optimal_interval=best_params[1],
                optimal_n_doses=max(1, int(best_params[2])),
                objective_value=best_value,
                success=False,
                message=f"Final simulation failed: {e}",
                all_results=all_results,
            )

    return OptimizationResult(
        success=False,
        message="No valid parameters found",
        all_results=all_results,
    )


def optimize_dosing_scipy(
    drug_name: str,
    drug_class: str = "cidal",
    objective: str = "burden",
    initial_conditions: Optional[Dict] = None,
    weight_kg: float = 70.0,
    constraints: Optional[OptimizationConstraints] = None,
    method: str = "Nelder-Mead",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Optimize dosing using SciPy minimize.

    Parameters:
        drug_name: drug to optimize
        drug_class: "cidal" or "static"
        objective: "burden", "resistance", or "multi"
        initial_conditions: initial state values
        weight_kg: patient weight
        constraints: OptimizationConstraints
        method: SciPy optimization method
        verbose: whether to print progress

    Returns:
        OptimizationResult
    """
    if initial_conditions is None:
        initial_conditions = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

    if constraints is None:
        constraints = OptimizationConstraints()

    obj_fn = {
        "burden": objective_minimize_burden,
        "resistance": objective_minimize_resistance,
        "multi": objective_multi,
    }[objective]

    # Initial guess: midpoint of constraints
    x0 = np.array([
        (constraints.dose_min + constraints.dose_max) / 2,
        (constraints.interval_min + constraints.interval_max) / 2,
        12.0,  # n_doses
    ])

    # Bounds
    bounds = [
        (constraints.dose_min, constraints.dose_max),
        (constraints.interval_min, constraints.interval_max),
        (constraints.n_doses_min, constraints.n_doses_max),
    ]

    def wrapped_obj(x):
        return obj_fn(x, drug_name, drug_class, initial_conditions, weight_kg)

    try:
        result = minimize(
            wrapped_obj,
            x0,
            method=method,
            bounds=bounds,
            options={"maxiter": 100, "disp": verbose},
        )

        # Run final simulation at optimal parameters
        opt_params = result.x
        params_dict = get_default_parameters()
        pd_model = BacterialPopulationODE(params_dict)

        pk_params = get_drug_pk_parameters(drug_name)
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
            dose_mg=opt_params[0],
            interval_hours=opt_params[1],
            start_time=0.0,
            n_doses=max(1, int(opt_params[2])),
            infusion_duration_min=60.0,
        )

        t_span = (0, regimen.n_doses * regimen.interval_hours + 24)

        sim_result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=initial_conditions,
            t_span=t_span,
            drug_class=drug_class,
            weight_kg=weight_kg,
        )

        return OptimizationResult(
            optimal_dose=opt_params[0],
            optimal_interval=opt_params[1],
            optimal_n_doses=max(1, int(opt_params[2])),
            objective_value=result.fun,
            simulation_result=sim_result,
            success=result.success,
            message=result.message,
        )

    except Exception as e:
        return OptimizationResult(
            success=False,
            message=f"Optimization failed: {e}",
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_optimize(
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    objective: str = "multi",
    method: str = "grid",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Run a quick dosing optimization.

    Parameters:
        drug_name: drug to optimize
        drug_class: "cidal" or "static"
        objective: "burden", "resistance", or "multi"
        method: "grid" or "scipy"
        verbose: whether to print progress

    Returns:
        OptimizationResult
    """
    if method == "grid":
        return optimize_dosing_grid(
            drug_name=drug_name,
            drug_class=drug_class,
            objective=objective,
            verbose=verbose,
        )
    else:
        return optimize_dosing_scipy(
            drug_name=drug_name,
            drug_class=drug_class,
            objective=objective,
            verbose=verbose,
        )


def compare_dosing_strategies(
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    strategies: Optional[List[Dict]] = None,
    initial_conditions: Optional[Dict] = None,
    weight_kg: float = 70.0,
) -> Dict[str, SimulationResult]:
    """
    Compare multiple dosing strategies.

    Parameters:
        drug_name: drug name
        drug_class: "cidal" or "static"
        strategies: list of dicts with "dose", "interval", "n_doses"
        initial_conditions: initial state values
        weight_kg: patient weight

    Returns:
        dict mapping strategy name to SimulationResult
    """
    if strategies is None:
        strategies = [
            {"dose": 500, "interval": 8, "n_doses": 12, "name": "Standard (500mg q8h)"},
            {"dose": 1000, "interval": 8, "n_doses": 12, "name": "High-dose (1000mg q8h)"},
            {"dose": 2000, "interval": 8, "n_doses": 12, "name": "Very high-dose (2000mg q8h)"},
            {"dose": 1000, "interval": 6, "n_doses": 16, "name": "Frequent (1000mg q6h)"},
            {"dose": 1000, "interval": 12, "n_doses": 8, "name": "Extended interval (1000mg q12h)"},
        ]

    if initial_conditions is None:
        initial_conditions = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

    results = {}

    for strategy in strategies:
        name = strategy.get("name", f"{strategy['dose']}mg q{strategy['interval']}h")

        try:
            params_dict = get_default_parameters()
            pd_model = BacterialPopulationODE(params_dict)

            pk_params = get_drug_pk_parameters(drug_name)
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
                dose_mg=strategy["dose"],
                interval_hours=strategy["interval"],
                start_time=0.0,
                n_doses=strategy["n_doses"],
                infusion_duration_min=60.0,
            )

            t_span = (0, regimen.n_doses * regimen.interval_hours + 24)

            results[name] = run_simulation(
                pk_model=pk_model,
                regimen=regimen,
                pd_model=pd_model,
                initial_conditions=initial_conditions,
                t_span=t_span,
                drug_class=drug_class,
                weight_kg=weight_kg,
            )
        except Exception as e:
            results[name] = None

    return results
