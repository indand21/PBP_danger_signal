"""
Parallel simulation engine for the QSP Antibiotic Model.

Provides parallel execution of multiple simulations using multiprocessing,
useful for:
- Virtual patient cohorts with varying parameters
- Parameter sweeps
- Monte Carlo uncertainty quantification
- Batch scenario comparisons

Supports:
- ProcessPoolExecutor-based parallelism with configurable worker count
- Progress tracking via callbacks
- Error handling per simulation (graceful degradation)
- Result aggregation utilities
"""

import numpy as np
from scipy.integrate import trapezoid  # np.trapz was removed in NumPy 2.0
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import multiprocessing
import time
import warnings

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation, SimulationResult


# ---------------------------------------------------------------------------
# Virtual patient definition
# ---------------------------------------------------------------------------

@dataclass
class VirtualPatient:
    """
    A virtual patient with potentially varied parameters.

    Attributes:
        patient_id: unique identifier
        weight_kg: body weight
        immune_level: neutrophil/macrophage count (1e7 = normal, 1e3 = neutropenic)
        initial_burden: initial bacterial burden (CFU/mL)
        drug_name: antibiotic to use
        drug_class: "cidal" or "static"
        param_overrides: dict of parameter overrides (e.g., {"bacteria": {"k_growth": 0.8}})
        dose_mg: dose amount in mg
        interval_hours: dosing interval
        n_doses: number of doses
        infusion_min: infusion duration in minutes
    """
    patient_id: int = 0
    weight_kg: float = 70.0
    immune_level: float = 1e7
    initial_burden: float = 1e5
    drug_name: str = "meropenem"
    drug_class: str = "cidal"
    param_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    dose_mg: float = 1000.0
    interval_hours: float = 8.0
    n_doses: int = 12
    infusion_min: float = 60.0


@dataclass
class PatientResult:
    """
    Result container for a single virtual patient simulation.

    Attributes:
        patient_id: matching VirtualPatient.patient_id
        success: whether simulation completed without error
        sim_result: SimulationResult object (None if failed)
        error_message: error description (None if successful)
        metrics: dict of extracted scalar metrics
    """
    patient_id: int
    success: bool
    sim_result: Optional[SimulationResult] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default metric extractors
# ---------------------------------------------------------------------------

def default_metrics(result: SimulationResult) -> Dict[str, float]:
    """Extract a standard set of scalar metrics from a SimulationResult."""
    t, B = result.get_bacterial_burden()
    _, _, il6 = result.get_cytokines()
    _, frac_scv = result.get_resistance_fraction()

    B_safe = np.maximum(B, 1e-10)
    log_B = np.log10(B_safe)

    return {
        "final_burden_log10": float(log_B[-1]),
        "peak_burden_log10": float(log_B.max()),
        "min_burden_log10": float(log_B.min()),
        "peak_il6": float(il6.max()),
        "final_il6": float(il6[-1]),
        "peak_scv_fraction": float(frac_scv.max()),
        "auc_burden_log10": float(trapezoid(log_B, t)),
    }


# ---------------------------------------------------------------------------
# Single-patient simulation (top-level for pickling)
# ---------------------------------------------------------------------------

def _simulate_patient(patient: VirtualPatient) -> PatientResult:
    """
    Simulate a single virtual patient. Designed to be called in a worker process.

    Parameters:
        patient: VirtualPatient specification.

    Returns:
        PatientResult with simulation output or error info.
    """
    try:
        # Build parameters with overrides
        params = get_default_parameters()
        for group, overrides in patient.param_overrides.items():
            if group in params:
                for attr, val in overrides.items():
                    if hasattr(params[group], attr):
                        setattr(params[group], attr, val)

        pd_model = BacterialPopulationODE(params)

        # Build PK model
        pk_params = get_drug_pk_parameters(patient.drug_name)
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
            dose_mg=patient.dose_mg,
            interval_hours=patient.interval_hours,
            start_time=0.0,
            n_doses=patient.n_doses,
            infusion_duration_min=patient.infusion_min,
        )

        ic = {
            "B_rep": patient.initial_burden,
            "B_pers": patient.initial_burden * 0.001,
            "B_SCV": 0,
            "N_eff": patient.immune_level,
            "Damage": 0,
            "IL6": 10,
            "TNF": 5,
        }

        t_span = (0, patient.n_doses * patient.interval_hours + 24)

        sim_result = run_simulation(
            pk_model=pk_model,
            regimen=regimen,
            pd_model=pd_model,
            initial_conditions=ic,
            t_span=t_span,
            drug_class=patient.drug_class,
            weight_kg=patient.weight_kg,
        )

        metrics = default_metrics(sim_result)

        return PatientResult(
            patient_id=patient.patient_id,
            success=True,
            sim_result=sim_result,
            metrics=metrics,
        )

    except Exception as e:
        return PatientResult(
            patient_id=patient.patient_id,
            success=False,
            error_message=str(e),
        )


# ---------------------------------------------------------------------------
# Cohort generation
# ---------------------------------------------------------------------------

def generate_cohort(
    n_patients: int = 100,
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    weight_range: Tuple[float, float] = (50.0, 100.0),
    burden_range: Tuple[float, float] = (1e4, 1e7),
    immune_range: Tuple[float, float] = (1e5, 1e8),
    dose_mg: float = 1000.0,
    interval_hours: float = 8.0,
    n_doses: int = 12,
    seed: int = 42,
    include_neutropenic: bool = False,
    neutropenic_fraction: float = 0.2,
) -> List[VirtualPatient]:
    """
    Generate a cohort of virtual patients with randomized parameters.

    Parameters:
        n_patients: number of patients to generate.
        drug_name: antibiotic name.
        drug_class: "cidal" or "static".
        weight_range: (min, max) body weight in kg.
        burden_range: (min, max) initial bacterial burden (CFU/mL).
        immune_range: (min, max) immune effector count.
        dose_mg: dose amount.
        interval_hours: dosing interval.
        n_doses: number of doses.
        seed: random seed for reproducibility.
        include_neutropenic: if True, some patients are neutropenic.
        neutropenic_fraction: fraction of patients that are neutropenic.

    Returns:
        List of VirtualPatient objects.
    """
    rng = np.random.default_rng(seed)
    patients = []

    for i in range(n_patients):
        weight = rng.uniform(*weight_range)
        burden = 10 ** rng.uniform(np.log10(burden_range[0]), np.log10(burden_range[1]))

        if include_neutropenic and rng.random() < neutropenic_fraction:
            immune = rng.uniform(1e2, 1e4)  # neutropenic
        else:
            immune = 10 ** rng.uniform(np.log10(immune_range[0]), np.log10(immune_range[1]))

        patients.append(VirtualPatient(
            patient_id=i,
            weight_kg=weight,
            immune_level=immune,
            initial_burden=burden,
            drug_name=drug_name,
            drug_class=drug_class,
            dose_mg=dose_mg,
            interval_hours=interval_hours,
            n_doses=n_doses,
        ))

    return patients


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def run_cohort_parallel(
    patients: List[VirtualPatient],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[PatientResult]:
    """
    Run a cohort of virtual patients in parallel.

    Parameters:
        patients: list of VirtualPatient objects.
        max_workers: number of worker processes. None = cpu_count().
        progress_callback: optional fn(completed, total) called after each patient.

    Returns:
        List of PatientResult objects, ordered by patient_id.
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(patients))

    results: List[PatientResult] = [None] * len(patients)  # type: ignore
    completed = 0

    # Use ProcessPoolExecutor for true parallelism
    # Note: On Windows, must use if __name__ == "__main__" guard
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_simulate_patient, p): i
            for i, p in enumerate(patients)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = PatientResult(
                    patient_id=patients[idx].patient_id,
                    success=False,
                    error_message=f"Worker exception: {e}",
                )
            completed += 1
            if progress_callback:
                progress_callback(completed, len(patients))

    return results


def run_cohort_sequential(
    patients: List[VirtualPatient],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[PatientResult]:
    """
    Run a cohort sequentially (useful for debugging or small cohorts).

    Parameters:
        patients: list of VirtualPatient objects.
        progress_callback: optional fn(completed, total).

    Returns:
        List of PatientResult objects.
    """
    results = []
    for i, patient in enumerate(patients):
        result = _simulate_patient(patient)
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, len(patients))
    return results


def run_cohort(
    patients: List[VirtualPatient],
    parallel: bool = True,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[PatientResult]:
    """
    Run a cohort of virtual patients, choosing parallel or sequential execution.

    Parameters:
        patients: list of VirtualPatient objects.
        parallel: if True, use multiprocessing; otherwise run sequentially.
        max_workers: number of worker processes (parallel mode only).
        progress_callback: optional fn(completed, total).

    Returns:
        List of PatientResult objects.
    """
    if parallel and len(patients) > 1:
        return run_cohort_parallel(patients, max_workers, progress_callback)
    else:
        return run_cohort_sequential(patients, progress_callback)


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: List[PatientResult]) -> Dict[str, Any]:
    """
    Aggregate metrics from a cohort of PatientResults.

    Returns:
        dict with keys:
            - "n_total": total patients
            - "n_success": successful simulations
            - "n_failed": failed simulations
            - "metrics_mean": dict of mean metric values
            - "metrics_std": dict of std metric values
            - "metrics_min": dict of min metric values
            - "metrics_max": dict of max metric values
            - "success_rate": fraction of successful simulations
    """
    successful = [r for r in results if r.success]
    n_total = len(results)
    n_success = len(successful)
    n_failed = n_total - n_success

    if n_success == 0:
        return {
            "n_total": n_total,
            "n_success": 0,
            "n_failed": n_failed,
            "success_rate": 0.0,
            "metrics_mean": {},
            "metrics_std": {},
            "metrics_min": {},
            "metrics_max": {},
        }

    # Collect all metric keys
    all_keys = set()
    for r in successful:
        all_keys.update(r.metrics.keys())

    # Build arrays for each metric
    metric_arrays: Dict[str, List[float]] = {k: [] for k in all_keys}
    for r in successful:
        for k in all_keys:
            metric_arrays[k].append(r.metrics.get(k, np.nan))

    metrics_mean = {}
    metrics_std = {}
    metrics_min = {}
    metrics_max = {}

    for k in all_keys:
        arr = np.array(metric_arrays[k])
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            metrics_mean[k] = float(np.mean(arr))
            metrics_std[k] = float(np.std(arr))
            metrics_min[k] = float(np.min(arr))
            metrics_max[k] = float(np.max(arr))
        else:
            metrics_mean[k] = np.nan
            metrics_std[k] = np.nan
            metrics_min[k] = np.nan
            metrics_max[k] = np.nan

    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": n_success / n_total,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "metrics_min": metrics_min,
        "metrics_max": metrics_max,
    }


def results_to_dataframe(results: List[PatientResult]) -> "pd.DataFrame":
    """
    Convert a list of PatientResults to a pandas DataFrame.

    Each row is one patient. Columns include patient attributes and metrics.
    """
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "patient_id": r.patient_id,
            "success": r.success,
            "error": r.error_message,
        }
        row.update(r.metrics)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cohort_summary(
    results: List[PatientResult],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> None:
    """
    Create a multi-panel summary plot of cohort simulation results.

    Panels:
    1. Final burden distribution (histogram)
    2. Peak IL-6 distribution
    3. AUC burden distribution
    4. Success rate pie chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [r for r in results if r.success]
    if not successful:
        warnings.warn("No successful simulations to plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Final burden
    burdens = [r.metrics.get("final_burden_log10", np.nan) for r in successful]
    burdens = [b for b in burdens if np.isfinite(b)]
    axes[0, 0].hist(burdens, bins=20, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0, 0].axvline(x=5.0, color="red", linestyle="--", label="VAP threshold (5.0)")
    axes[0, 0].axvline(x=2.0, color="orange", linestyle="--", label="Bloodstream (2.0)")
    axes[0, 0].set_xlabel("Final burden (log10 CFU/mL)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Final Bacterial Burden Distribution")
    axes[0, 0].legend(fontsize=8)

    # Panel 2: Peak IL-6
    il6_vals = [r.metrics.get("peak_il6", np.nan) for r in successful]
    il6_vals = [v for v in il6_vals if np.isfinite(v) and v > 0]
    if il6_vals:
        axes[0, 1].hist(np.log10(il6_vals), bins=20, color="#DD8452", alpha=0.8, edgecolor="white")
        axes[0, 1].set_xlabel("Peak IL-6 (log10 pg/mL)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Peak IL-6 Distribution")

    # Panel 3: AUC burden
    auc_vals = [r.metrics.get("auc_burden_log10", np.nan) for r in successful]
    auc_vals = [v for v in auc_vals if np.isfinite(v)]
    if auc_vals:
        axes[1, 0].hist(auc_vals, bins=20, color="#55A868", alpha=0.8, edgecolor="white")
        axes[1, 0].set_xlabel("AUC burden (log10 CFU/mL × h)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("AUC Bacterial Burden Distribution")

    # Panel 4: Success rate
    n_success = len(successful)
    n_failed = len(results) - n_success
    axes[1, 1].pie(
        [n_success, n_failed],
        labels=["Success", "Failed"],
        colors=["#55A868", "#C44E52"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1, 1].set_title(f"Simulation Success Rate (n={len(results)})")

    fig.suptitle("Virtual Patient Cohort Summary", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kinetics_overlay(
    results: List[PatientResult],
    max_traces: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Overlay bacterial burden kinetics from multiple patients.

    Parameters:
        results: list of PatientResult objects.
        max_traces: maximum number of trajectories to plot.
        save_path: if provided, save figure.
        figsize: figure size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [r for r in results if r.success and r.sim_result is not None]
    if not successful:
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subsample if too many
    if len(successful) > max_traces:
        indices = np.linspace(0, len(successful) - 1, max_traces, dtype=int)
        plot_results = [successful[i] for i in indices]
    else:
        plot_results = successful

    for r in plot_results:
        t, B = r.sim_result.get_bacterial_burden()
        log_B = np.log10(np.maximum(B, 1e-10))
        axes[0].plot(t, log_B, alpha=0.3, linewidth=0.8, color="#4C72B0")

    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Bacterial burden (log10 CFU/mL)")
    axes[0].set_title(f"Bacterial Kinetics (n={len(plot_results)})")
    axes[0].axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="VAP threshold")
    axes[0].legend(fontsize=8)

    # IL-6 overlay
    for r in plot_results:
        t, _, il6 = r.sim_result.get_cytokines()
        axes[1].plot(t, il6, alpha=0.3, linewidth=0.8, color="#DD8452")

    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("IL-6 (pg/mL)")
    axes[1].set_title(f"IL-6 Kinetics (n={len(plot_results)})")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience: quick cohort run
# ---------------------------------------------------------------------------

def quick_cohort(
    n_patients: int = 50,
    drug_name: str = "meropenem",
    drug_class: str = "cidal",
    parallel: bool = True,
    max_workers: Optional[int] = None,
    save_plot: bool = False,
) -> Dict[str, Any]:
    """
    Generate and run a virtual patient cohort with default settings.

    Parameters:
        n_patients: number of virtual patients.
        drug_name: antibiotic to simulate.
        drug_class: "cidal" or "static".
        parallel: use multiprocessing.
        max_workers: number of workers.
        save_plot: if True, save summary plots.

    Returns:
        Aggregated results dict.
    """
    patients = generate_cohort(
        n_patients=n_patients,
        drug_name=drug_name,
        drug_class=drug_class,
    )

    def progress(done, total):
        if done % max(1, total // 5) == 0 or done == total:
            print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    print(f"Running cohort of {n_patients} virtual patients...")
    results = run_cohort(patients, parallel=parallel, max_workers=max_workers,
                         progress_callback=progress)

    agg = aggregate_results(results)
    print(f"\nCohort Summary:")
    print(f"  Success rate: {agg['success_rate']:.1%}")
    if agg["metrics_mean"]:
        print(f"  Mean final burden: {agg['metrics_mean'].get('final_burden_log10', 'N/A'):.2f} log10 CFU/mL")
        print(f"  Mean peak IL-6: {agg['metrics_mean'].get('peak_il6', 'N/A'):.0f} pg/mL")

    if save_plot:
        plot_cohort_summary(results, save_path="cohort_summary.png")
        plot_kinetics_overlay(results, save_path="cohort_kinetics.png")
        print("  Saved plots: cohort_summary.png, cohort_kinetics.png")

    return agg


if __name__ == "__main__":
    quick_cohort(n_patients=20, save_plot=True)
