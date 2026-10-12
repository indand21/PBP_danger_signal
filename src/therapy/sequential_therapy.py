"""
Sequential Therapy Module for the QSP Antibiotic Model.

Models treatment strategies where antibiotics are changed over time:
1. IV-to-oral step-down therapy
2. Antibiotic cycling (rotating drugs to reduce resistance)
3. De-escalation (broad-spectrum → narrow-spectrum)
4. Escalation (narrow-spectrum → broad-spectrum)
5. Custom multi-phase protocols

Key concepts:
- Treatment is divided into phases, each with its own drug, dose, and duration
- Transitions between phases can be time-based or condition-based
- The model tracks bacterial dynamics across phase transitions
- Resistance evolution is influenced by sequential drug exposure
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from scipy.integrate import solve_ivp
import copy

from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import BacterialPopulationODE
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation, SimulationResult


# ---------------------------------------------------------------------------
# Treatment phase definition
# ---------------------------------------------------------------------------

@dataclass
class TreatmentPhase:
    """
    Represents a single phase of sequential therapy.

    Attributes:
        phase_id: unique identifier for this phase
        drug_name: antibiotic name
        drug_class: "cidal" or "static"
        dose_mg: dose amount in mg
        interval_hours: dosing interval in hours
        duration_hours: duration of this phase in hours
        infusion_min: infusion duration in minutes
        description: human-readable description
    """
    phase_id: int = 0
    drug_name: str = "meropenem"
    drug_class: str = "cidal"
    dose_mg: float = 1000.0
    interval_hours: float = 8.0
    duration_hours: float = 72.0
    infusion_min: float = 60.0
    description: str = ""

    @property
    def n_doses(self) -> int:
        """Number of doses in this phase."""
        return max(1, int(self.duration_hours / self.interval_hours))


@dataclass
class SequentialProtocol:
    """
    Defines a complete sequential therapy protocol.

    Attributes:
        name: protocol name
        phases: list of TreatmentPhase objects
        description: human-readable description
    """
    name: str = "sequential"
    phases: List[TreatmentPhase] = field(default_factory=list)
    description: str = ""

    @property
    def total_duration(self) -> float:
        """Total duration of all phases in hours."""
        return sum(p.duration_hours for p in self.phases)

    def get_phase_at_time(self, t: float) -> Optional[TreatmentPhase]:
        """Get the active phase at a given time point."""
        cumulative = 0.0
        for phase in self.phases:
            if cumulative <= t < cumulative + phase.duration_hours:
                return phase
            cumulative += phase.duration_hours
        return None


# ---------------------------------------------------------------------------
# Pre-defined protocols
# ---------------------------------------------------------------------------

def create_stepdown_protocol(
    iv_drug: str = "meropenem",
    oral_drug: str = "doxycycline",
    iv_duration: float = 72.0,
    oral_duration: float = 96.0,
    iv_dose: float = 1000.0,
    oral_dose: float = 200.0,
    iv_interval: float = 8.0,
    oral_interval: float = 12.0,
) -> SequentialProtocol:
    """
    Create an IV-to-oral step-down therapy protocol.

    Parameters:
        iv_drug: IV antibiotic name
        oral_drug: oral antibiotic name
        iv_duration: IV phase duration in hours
        oral_duration: oral phase duration in hours
        iv_dose: IV dose in mg
        oral_dose: oral dose in mg
        iv_interval: IV dosing interval in hours
        oral_interval: oral dosing interval in hours

    Returns:
        SequentialProtocol object
    """
    phases = [
        TreatmentPhase(
            phase_id=0,
            drug_name=iv_drug,
            drug_class="cidal",
            dose_mg=iv_dose,
            interval_hours=iv_interval,
            duration_hours=iv_duration,
            infusion_min=60.0,
            description=f"IV {iv_drug} {iv_dose}mg q{iv_interval}h",
        ),
        TreatmentPhase(
            phase_id=1,
            drug_name=oral_drug,
            drug_class="static",
            dose_mg=oral_dose,
            interval_hours=oral_interval,
            duration_hours=oral_duration,
            infusion_min=0.0,  # oral
            description=f"Oral {oral_drug} {oral_dose}mg q{oral_interval}h",
        ),
    ]

    return SequentialProtocol(
        name="iv_to_oral_stepdown",
        phases=phases,
        description=f"IV {iv_drug} → oral {oral_drug} step-down therapy",
    )


def create_cycling_protocol(
    drugs: List[Dict[str, str]],
    cycle_duration: float = 168.0,
    n_cycles: int = 2,
    dose_mg: float = 1000.0,
    interval_hours: float = 8.0,
) -> SequentialProtocol:
    """
    Create an antibiotic cycling protocol.

    Parameters:
        drugs: list of dicts with "drug_name" and "drug_class"
        cycle_duration: duration of each cycle in hours
        n_cycles: number of complete cycles
        dose_mg: dose for each drug
        interval_hours: dosing interval

    Returns:
        SequentialProtocol object
    """
    phases = []
    phase_id = 0

    for cycle in range(n_cycles):
        for drug_info in drugs:
            phases.append(TreatmentPhase(
                phase_id=phase_id,
                drug_name=drug_info["drug_name"],
                drug_class=drug_info.get("drug_class", "cidal"),
                dose_mg=dose_mg,
                interval_hours=interval_hours,
                duration_hours=cycle_duration / len(drugs),
                infusion_min=60.0,
                description=f"Cycle {cycle+1}: {drug_info['drug_name']}",
            ))
            phase_id += 1

    return SequentialProtocol(
        name="antibiotic_cycling",
        phases=phases,
        description=f"Antibiotic cycling: {', '.join(d['drug_name'] for d in drugs)} × {n_cycles} cycles",
    )


def create_deescalation_protocol(
    broad_drug: str = "meropenem",
    narrow_drug: str = "doxycycline",
    broad_duration: float = 48.0,
    narrow_duration: float = 120.0,
    broad_dose: float = 2000.0,
    narrow_dose: float = 200.0,
) -> SequentialProtocol:
    """
    Create a de-escalation protocol (broad → narrow spectrum).

    Parameters:
        broad_drug: broad-spectrum drug name
        narrow_drug: narrow-spectrum drug name
        broad_duration: broad-spectrum phase duration
        narrow_duration: narrow-spectrum phase duration
        broad_dose: broad-spectrum dose
        narrow_dose: narrow-spectrum dose

    Returns:
        SequentialProtocol object
    """
    phases = [
        TreatmentPhase(
            phase_id=0,
            drug_name=broad_drug,
            drug_class="cidal",
            dose_mg=broad_dose,
            interval_hours=8.0,
            duration_hours=broad_duration,
            infusion_min=60.0,
            description=f"De-escalation: broad {broad_drug}",
        ),
        TreatmentPhase(
            phase_id=1,
            drug_name=narrow_drug,
            drug_class="static",
            dose_mg=narrow_dose,
            interval_hours=12.0,
            duration_hours=narrow_duration,
            infusion_min=0.0,
            description=f"De-escalation: narrow {narrow_drug}",
        ),
    ]

    return SequentialProtocol(
        name="de_escalation",
        phases=phases,
        description=f"De-escalation: {broad_drug} → {narrow_drug}",
    )


# ---------------------------------------------------------------------------
# Sequential therapy simulation
# ---------------------------------------------------------------------------

def run_sequential_simulation(
    protocol: SequentialProtocol,
    initial_conditions: Optional[Dict] = None,
    weight_kg: float = 70.0,
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a simulation with sequential therapy protocol.

    Parameters:
        protocol: SequentialProtocol object
        initial_conditions: dict of initial state values
        weight_kg: patient weight
        verbose: whether to print progress info

    Returns:
        SimulationResult with trajectories across all phases
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

    # Build initial state vector (13 states: 4 PK + 9 PD incl. PAMP and D_host)
    y0 = np.zeros(13)
    y0[4] = initial_conditions.get("B_rep", 1e5)
    y0[5] = initial_conditions.get("B_pers", 1e2)
    y0[6] = initial_conditions.get("B_SCV", 0)
    y0[7] = initial_conditions.get("N_eff", 1e7)
    y0[8] = initial_conditions.get("Damage", 0)
    y0[9] = initial_conditions.get("IL6", 10)
    y0[10] = initial_conditions.get("TNF", 5)
    y0[11] = initial_conditions.get("PAMP", 0)
    y0[12] = initial_conditions.get("D_host", 0)

    # Run each phase sequentially, carrying state forward
    all_t = []
    all_y = []
    current_y = y0.copy()
    current_t_start = 0.0

    for phase_idx, phase in enumerate(protocol.phases):
        if verbose:
            print(f"Phase {phase_idx}: {phase.description} "
                  f"({current_t_start:.1f} - {current_t_start + phase.duration_hours:.1f}h)")

        # Build PK model for this phase
        pk_params = get_drug_pk_parameters(phase.drug_name)
        pk_model = TwoCompartmentPKModel(
            CL=pk_params.CL,
            Vc=pk_params.Vc,
            Vp=pk_params.Vp,
            Q=pk_params.Q,
            Ka=pk_params.Ka,
            Kp=pk_params.Kp,
            effect_site_model=True,
        )

        # Build regimen for this phase
        regimen = DosingRegimen(
            dose_mg=phase.dose_mg,
            interval_hours=phase.interval_hours,
            start_time=0.0,  # Relative to phase start
            n_doses=phase.n_doses,
            infusion_duration_min=phase.infusion_min,
        )

        # Build PD model
        params = get_default_parameters()
        pd_model = BacterialPopulationODE(params)

        # Define ODE for this phase
        def phase_rhs(t, y, _pk_model=pk_model, _regimen=regimen,
                      _pd_model=pd_model, _drug_class=phase.drug_class):
            """ODE function for this treatment phase."""
            inf_rate = _regimen.get_infusion_rate(t)

            # PK dynamics
            pk_state = y[0:4]
            pk_dydt = _pk_model.ode_rhs(t, pk_state, infusion_rate=inf_rate)

            # Drug concentration
            A_central = y[0]
            C_effect = _pk_model.concentration_effect(A_central)

            # PD dynamics (9 PD states incl. PAMP and D_host)
            pd_state = y[4:13]
            pd_dydt = _pd_model.rhs(t, pd_state, C_effect=C_effect,
                                    drug_class=_drug_class)

            return np.concatenate([pk_dydt, pd_dydt])

        # Solve ODE for this phase
        t_span = (0, phase.duration_hours)

        try:
            sol = solve_ivp(phase_rhs, t_span, current_y, method='RK45',
                            max_step=0.1, rtol=1e-6, atol=1e-8)

            if not sol.success and verbose:
                print(f"  Warning: ODE solver issue: {sol.message}")
        except Exception as e:
            if verbose:
                print(f"  Error in phase {phase_idx}: {e}")
            raise

        # Store results
        t_offset = current_t_start
        phase_t = sol.t + t_offset
        phase_y = sol.y.T

        all_t.append(phase_t)
        all_y.append(phase_y)

        # Update state for next phase
        current_y = phase_y[-1].copy()
        current_t_start += phase.duration_hours

    # Concatenate all phases
    t_combined = np.concatenate(all_t)
    y_combined = np.vstack(all_y)

    state_names = [
        'A_central', 'A_peripheral', 'A_absorption', 'A_effect',
        'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF', 'PAMP',
        'D_host',
    ]

    return SimulationResult(t_combined, y_combined, state_names,
                           {'protocol': protocol.name, 'weight': weight_kg})


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_sequential_vs_monotherapy(
    protocol: SequentialProtocol,
    monotherapy_drug: str = "meropenem",
    monotherapy_class: str = "cidal",
    monotherapy_dose: float = 1000.0,
    monotherapy_interval: float = 8.0,
    initial_conditions: Optional[Dict] = None,
    weight_kg: float = 70.0,
) -> Dict[str, SimulationResult]:
    """
    Compare sequential therapy vs continuous monotherapy.

    Parameters:
        protocol: SequentialProtocol to compare
        monotherapy_drug: drug for monotherapy comparison
        monotherapy_class: drug class for monotherapy
        monotherapy_dose: dose for monotherapy
        monotherapy_interval: dosing interval for monotherapy
        initial_conditions: initial state values
        weight_kg: patient weight

    Returns:
        dict with keys "sequential" and "monotherapy"
    """
    results = {}

    # Sequential therapy
    results["sequential"] = run_sequential_simulation(
        protocol, initial_conditions, weight_kg,
    )

    # Monotherapy
    if initial_conditions is None:
        initial_conditions = {
            "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
            "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
        }

    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)

    pk_params = get_drug_pk_parameters(monotherapy_drug)
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
        dose_mg=monotherapy_dose,
        interval_hours=monotherapy_interval,
        start_time=0.0,
        n_doses=int(protocol.total_duration / monotherapy_interval),
        infusion_duration_min=60.0,
    )

    results["monotherapy"] = run_simulation(
        pk_model=pk_model,
        regimen=regimen,
        pd_model=pd_model,
        initial_conditions=initial_conditions,
        t_span=(0, protocol.total_duration),
        drug_class=monotherapy_class,
        weight_kg=weight_kg,
    )

    return results


def compute_sequential_benefit(
    results: Dict[str, SimulationResult],
) -> Dict[str, float]:
    """
    Compute the benefit of sequential therapy vs monotherapy.

    Parameters:
        results: dict from compare_sequential_vs_monotherapy()

    Returns:
        dict with benefit metrics
    """
    metrics = {}

    _, B_seq = results["sequential"].get_bacterial_burden()
    _, B_mono = results["monotherapy"].get_bacterial_burden()

    final_seq = max(B_seq[-1], 1e-10)
    final_mono = max(B_mono[-1], 1e-10)

    metrics["burden_reduction"] = np.log10(final_mono) - np.log10(final_seq)
    metrics["final_burden_sequential"] = np.log10(final_seq)
    metrics["final_burden_monotherapy"] = np.log10(final_mono)

    # IL-6 comparison
    _, _, il6_seq = results["sequential"].get_cytokines()
    _, _, il6_mono = results["monotherapy"].get_cytokines()

    metrics["peak_il6_sequential"] = float(il6_seq.max())
    metrics["peak_il6_monotherapy"] = float(il6_mono.max())

    return metrics


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_stepdown(
    iv_drug: str = "meropenem",
    oral_drug: str = "doxycycline",
    iv_duration: float = 72.0,
    oral_duration: float = 96.0,
) -> Dict[str, float]:
    """
    Run a quick step-down therapy comparison.

    Parameters:
        iv_drug: IV drug name
        oral_drug: oral drug name
        iv_duration: IV phase duration
        oral_duration: oral phase duration

    Returns:
        dict with benefit metrics
    """
    protocol = create_stepdown_protocol(
        iv_drug=iv_drug,
        oral_drug=oral_drug,
        iv_duration=iv_duration,
        oral_duration=oral_duration,
    )

    results = compare_sequential_vs_monotherapy(protocol)
    return compute_sequential_benefit(results)
