"""
Combination Therapy Module for the QSP Antibiotic Model.

Models the effects of using multiple antibiotics simultaneously:
1. Synergistic interactions (combined effect > sum of individual effects)
2. Antagonistic interactions (combined effect < sum of individual effects)
3. Additive interactions (combined effect = sum of individual effects)
4. PK/PD interactions (drug-drug interactions affecting clearance/volume)

Key concepts:
- Each drug has its own PK model and contributes to the total effect-site concentration
- The Bliss Independence model is used for synergism/antagonism
- Fractional Inhibitory Concentration (FIC) index quantifies interaction type
- Drug combinations can be optimized for maximal bacterial kill

Interaction types:
- Synergy: FIC < 0.5
- Additive: FIC = 0.5-1.0
- Indifference: FIC = 1.0-2.0
- Antagonism: FIC > 2.0
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
# Combination therapy parameters
# ---------------------------------------------------------------------------

@dataclass
class DrugInCombination:
    """
    Represents a single drug within a combination therapy regimen.

    Attributes:
        drug_name: name of the drug (e.g., "meropenem", "doxycycline")
        drug_class: "cidal" or "static"
        dose_mg: dose amount in mg
        interval_hours: dosing interval in hours
        n_doses: number of doses
        infusion_min: infusion duration in minutes
        weight_kg: patient weight for PK scaling
    """
    drug_name: str
    drug_class: str = "cidal"
    dose_mg: float = 1000.0
    interval_hours: float = 8.0
    n_doses: int = 12
    infusion_min: float = 60.0
    weight_kg: float = 70.0


@dataclass
class InteractionParameters:
    """
    Parameters describing drug-drug interactions.

    Attributes:
        synergy_factor: factor for synergistic interaction (0-1, lower = more synergistic)
        antagonism_factor: factor for antagonistic interaction (>1 = antagonistic)
        FIC_index: Fractional Inhibitory Concentration index
            FIC < 0.5: synergy
            FIC 0.5-1.0: additivity
            FIC 1.0-2.0: indifference
            FIC > 2.0: antagonism
        interaction_type: "synergy", "additive", "indifference", or "antagonism"
    """
    synergy_factor: float = 1.0
    antagonism_factor: float = 1.0
    FIC_index: float = 1.0
    interaction_type: str = "additive"

    @classmethod
    def from_FIC(cls, FIC: float) -> "InteractionParameters":
        """Create InteractionParameters from FIC index."""
        if FIC < 0.5:
            interaction_type = "synergy"
            synergy_factor = FIC / 0.5  # 0 to 1
            antagonism_factor = 1.0
        elif FIC <= 1.0:
            interaction_type = "additive"
            synergy_factor = 1.0
            antagonism_factor = 1.0
        elif FIC <= 2.0:
            interaction_type = "indifference"
            synergy_factor = 1.0
            antagonism_factor = FIC
        else:
            interaction_type = "antagonism"
            synergy_factor = 1.0
            antagonism_factor = FIC

        return cls(
            synergy_factor=synergy_factor,
            antagonism_factor=antagonism_factor,
            FIC_index=FIC,
            interaction_type=interaction_type,
        )


# ---------------------------------------------------------------------------
# Combination therapy model
# ---------------------------------------------------------------------------

class CombinationTherapyModel:
    """
    Models the combined effect of multiple antibiotics.

    Uses the Bliss Independence model for drug interactions:
        E_combined = 1 - ∏(1 - E_i)

    Where E_i is the effect of drug i alone.

    For synergistic interactions, the combined effect is amplified.
    For antagonistic interactions, the combined effect is reduced.

    Parameters:
        drugs: list of DrugInCombination objects
        interaction: InteractionParameters for drug-drug interactions
    """

    def __init__(
        self,
        drugs: List[DrugInCombination],
        interaction: Optional[InteractionParameters] = None,
    ):
        self.drugs = drugs
        self.interaction = interaction or InteractionParameters()

        if len(drugs) < 2:
            raise ValueError("Combination therapy requires at least 2 drugs")

    def compute_combined_effect(
        self,
        individual_effects: List[float],
    ) -> float:
        """
        Compute the combined drug effect using Bliss Independence.

        Parameters:
            individual_effects: list of effect values for each drug (0 to 1)

        Returns:
            Combined effect (0 to 1)
        """
        # Bliss Independence: E = 1 - ∏(1 - E_i)
        survival_product = 1.0
        for effect in individual_effects:
            survival_product *= (1.0 - max(0.0, min(1.0, effect)))

        combined_effect = 1.0 - survival_product

        # Apply interaction modifier
        if self.interaction.interaction_type == "synergy":
            # Synergy amplifies the combined effect
            combined_effect *= (1.0 + (1.0 - self.interaction.synergy_factor))
        elif self.interaction.interaction_type == "antagonism":
            # Antagonism reduces the combined effect
            combined_effect /= self.interaction.antagonism_factor

        return np.clip(combined_effect, 0.0, 1.0)

    def compute_individual_effects(
        self,
        concentrations: List[float],
        MICs: List[float],
    ) -> List[float]:
        """
        Compute the individual effect of each drug based on concentration and MIC.

        Uses a Hill-type function: E = C^n / (MIC^n + C^n)

        Parameters:
            concentrations: list of drug concentrations at effect site
            MICs: list of MIC values for each drug

        Returns:
            List of individual effect values (0 to 1)
        """
        effects = []
        for C, MIC in zip(concentrations, MICs):
            if MIC <= 0:
                effect = 1.0
            else:
                # Hill function with n=1
                cmic_ratio = C / MIC
                effect = cmic_ratio / (1.0 + cmic_ratio)
            effects.append(effect)
        return effects

    def compute_FIC_index(
        self,
        concentrations: List[float],
        MICs: List[float],
    ) -> float:
        """
        Compute the Fractional Inhibitory Concentration (FIC) index.

        FIC = Σ(C_i / MIC_i)

        Parameters:
            concentrations: list of drug concentrations
            MICs: list of MIC values

        Returns:
            FIC index value
        """
        FIC = 0.0
        for C, MIC in zip(concentrations, MICs):
            if MIC > 0:
                FIC += C / MIC
        return FIC


# ---------------------------------------------------------------------------
# Combination therapy simulation
# ---------------------------------------------------------------------------

def run_combination_simulation(
    drugs: List[DrugInCombination],
    interaction: Optional[InteractionParameters] = None,
    initial_conditions: Optional[Dict] = None,
    t_span: Tuple[float, float] = (0, 96),
    verbose: bool = False,
) -> SimulationResult:
    """
    Run a simulation with combination therapy.

    This function runs separate PK models for each drug and combines their
    effects at the PD level using the Bliss Independence model.

    Parameters:
        drugs: list of DrugInCombination objects
        interaction: InteractionParameters for drug-drug interactions
        initial_conditions: dict of initial state values
        t_span: (t_start, t_end) in hours
        verbose: whether to print progress info

    Returns:
        SimulationResult with combined effect trajectories
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

    # Build PK models for each drug
    pk_models = []
    regimens = []
    for drug in drugs:
        pk_params = get_drug_pk_parameters(drug.drug_name)
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
            dose_mg=drug.dose_mg,
            interval_hours=drug.interval_hours,
            start_time=0.0,
            n_doses=drug.n_doses,
            infusion_duration_min=drug.infusion_min,
        )
        pk_models.append(pk_model)
        regimens.append(regimen)

    # Build PD model
    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)

    # Initialize state vector (4 PK placeholders + 9 PD states incl. PAMP and D_host)
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

    # Create combination therapy model
    combo = CombinationTherapyModel(drugs, interaction)

    def combined_rhs(t, y):
        """ODE function for combination therapy."""
        # Compute PK for each drug
        total_inf_rate = 0.0
        C_effects = []

        for i, (pk_model, regimen) in enumerate(zip(pk_models, regimens)):
            inf_rate = regimen.get_infusion_rate(t)
            total_inf_rate += inf_rate

            # Each drug has its own PK state
            # For simplicity, we track all PK states in the combined vector
            pk_start = i * 4
            pk_state = y[pk_start:pk_start + 4]
            pk_dydt = pk_model.ode_rhs(t, pk_state, infusion_rate=inf_rate)

            A_central = y[pk_start]
            C_effect = pk_model.concentration_effect(A_central)
            C_effects.append(C_effect)

            # Store PK derivatives
            if i == 0:
                dydt_pk = pk_dydt
            else:
                dydt_pk = np.concatenate([dydt_pk, pk_dydt])

        # Compute combined drug effect
        MICs = [1.0] * len(drugs)  # Default MIC for each drug
        individual_effects = combo.compute_individual_effects(C_effects, MICs)
        combined_effect = combo.compute_combined_effect(individual_effects)

        # PD dynamics with combined effect (9 PD states incl. PAMP and D_host)
        pd_state = y[len(drugs) * 4:len(drugs) * 4 + 9]

        # Use the first drug's class for PD dynamics
        # The combined effect modifies the concentration
        primary_drug_class = drugs[0].drug_class
        C_combined = combined_effect * 10.0  # Scale to effective concentration

        pd_dydt = pd_model.rhs(t, pd_state, C_effect=C_combined,
                               drug_class=primary_drug_class)

        return np.concatenate([dydt_pk, pd_dydt])

    # Initial conditions for all PK states + PD states (9 PD incl. PAMP and D_host)
    y0_full = np.zeros(len(drugs) * 4 + 9)
    y0_full[len(drugs) * 4:] = y0[4:]

    # Solve ODE
    try:
        sol = solve_ivp(combined_rhs, t_span, y0_full, method='RK45',
                        max_step=0.1, rtol=1e-6, atol=1e-8)

        if not sol.success:
            if verbose:
                print(f"Warning: ODE solver did not converge: {sol.message}")
    except Exception as e:
        if verbose:
            print(f"Error in ODE integration: {e}")
        raise

    # Extract results
    t = sol.t
    y = sol.y.T

    # Build state names
    state_names = []
    for i, drug in enumerate(drugs):
        state_names.extend([
            f"A_central_{drug.drug_name}",
            f"A_peripheral_{drug.drug_name}",
            f"A_absorption_{drug.drug_name}",
            f"A_effect_{drug.drug_name}",
        ])
    state_names.extend([
        "B_rep", "B_pers", "B_SCV", "N_eff", "Damage", "IL6", "TNF", "PAMP",
        "D_host",
    ])

    return SimulationResult(t, y, state_names,
                           {"drug_class": drugs[0].drug_class, "weight": drugs[0].weight_kg})


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_monotherapy_vs_combination(
    drug1: DrugInCombination,
    drug2: DrugInCombination,
    interaction: Optional[InteractionParameters] = None,
    initial_conditions: Optional[Dict] = None,
    t_span: Tuple[float, float] = (0, 96),
) -> Dict[str, SimulationResult]:
    """
    Compare monotherapy vs combination therapy.

    Parameters:
        drug1: first drug
        drug2: second drug
        interaction: interaction parameters for combination
        initial_conditions: initial state values
        t_span: time span

    Returns:
        dict with keys "drug1_alone", "drug2_alone", "combination"
    """
    results = {}

    # Drug 1 alone
    try:
        results["drug1_alone"] = run_simulation(
            pk_model=TwoCompartmentPKModel(
                **{k: getattr(get_drug_pk_parameters(drug1.drug_name), k)
                   for k in ['CL', 'Vc', 'Vp', 'Q', 'Ka', 'Kp']},
                effect_site_model=True,
            ),
            regimen=DosingRegimen(
                dose_mg=drug1.dose_mg,
                interval_hours=drug1.interval_hours,
                n_doses=drug1.n_doses,
                infusion_duration_min=drug1.infusion_min,
            ),
            pd_model=BacterialPopulationODE(get_default_parameters()),
            initial_conditions=initial_conditions or {
                "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
                "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
            },
            t_span=t_span,
            drug_class=drug1.drug_class,
            weight_kg=drug1.weight_kg,
        )
    except Exception as e:
        results["drug1_alone"] = None

    # Drug 2 alone
    try:
        results["drug2_alone"] = run_simulation(
            pk_model=TwoCompartmentPKModel(
                **{k: getattr(get_drug_pk_parameters(drug2.drug_name), k)
                   for k in ['CL', 'Vc', 'Vp', 'Q', 'Ka', 'Kp']},
                effect_site_model=True,
            ),
            regimen=DosingRegimen(
                dose_mg=drug2.dose_mg,
                interval_hours=drug2.interval_hours,
                n_doses=drug2.n_doses,
                infusion_duration_min=drug2.infusion_min,
            ),
            pd_model=BacterialPopulationODE(get_default_parameters()),
            initial_conditions=initial_conditions or {
                "B_rep": 1e5, "B_pers": 1e2, "B_SCV": 0,
                "N_eff": 1e7, "Damage": 0, "IL6": 10, "TNF": 5,
            },
            t_span=t_span,
            drug_class=drug2.drug_class,
            weight_kg=drug2.weight_kg,
        )
    except Exception as e:
        results["drug2_alone"] = None

    # Combination
    try:
        results["combination"] = run_combination_simulation(
            drugs=[drug1, drug2],
            interaction=interaction,
            initial_conditions=initial_conditions,
            t_span=t_span,
        )
    except Exception as e:
        results["combination"] = None

    return results


def compute_combination_benefit(
    results: Dict[str, SimulationResult],
) -> Dict[str, float]:
    """
    Compute the benefit of combination therapy vs monotherapy.

    Parameters:
        results: dict from compare_monotherapy_vs_combination()

    Returns:
        dict with benefit metrics:
            - "burden_reduction_vs_drug1": log10 reduction in final burden vs drug1 alone
            - "burden_reduction_vs_drug2": log10 reduction in final burden vs drug2 alone
            - "synergy_score": measure of synergistic benefit (positive = synergy)
    """
    metrics = {}

    _, B_combo = results["combination"].get_bacterial_burden()
    final_combo = max(B_combo[-1], 1e-10)

    if results["drug1_alone"] is not None:
        _, B1 = results["drug1_alone"].get_bacterial_burden()
        final1 = max(B1[-1], 1e-10)
        metrics["burden_reduction_vs_drug1"] = np.log10(final1) - np.log10(final_combo)
    else:
        metrics["burden_reduction_vs_drug1"] = np.nan

    if results["drug2_alone"] is not None:
        _, B2 = results["drug2_alone"].get_bacterial_burden()
        final2 = max(B2[-1], 1e-10)
        metrics["burden_reduction_vs_drug2"] = np.log10(final2) - np.log10(final_combo)
    else:
        metrics["burden_reduction_vs_drug2"] = np.nan

    # Synergy score: positive if combination is better than best monotherapy
    best_mono = min(
        metrics.get("burden_reduction_vs_drug1", 0),
        metrics.get("burden_reduction_vs_drug2", 0),
    )
    metrics["synergy_score"] = -best_mono  # Positive = combination is better

    return metrics


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_combination(
    drug1_name: str = "meropenem",
    drug2_name: str = "doxycycline",
    drug1_class: str = "cidal",
    drug2_class: str = "static",
    FIC: float = 0.75,
    t_span: Tuple[float, float] = (0, 96),
) -> Dict[str, float]:
    """
    Run a quick combination therapy comparison.

    Parameters:
        drug1_name: first drug name
        drug2_name: second drug name
        drug1_class: first drug class
        drug2_class: second drug class
        FIC: Fractional Inhibitory Concentration index
        t_span: time span

    Returns:
        dict with benefit metrics
    """
    drug1 = DrugInCombination(drug_name=drug1_name, drug_class=drug1_class)
    drug2 = DrugInCombination(drug_name=drug2_name, drug_class=drug2_class)
    interaction = InteractionParameters.from_FIC(FIC)

    results = compare_monotherapy_vs_combination(
        drug1, drug2, interaction, t_span=t_span,
    )

    return compute_combination_benefit(results)
