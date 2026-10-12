"""
Resistance Evolution Module for the QSP Antibiotic Model.

Models dynamic MIC increases during antibiotic treatment through:
1. Spontaneous mutations (stochastic MIC step increases)
2. Selection pressure (sub-MIC concentrations favor resistant subpopulations)
3. Fitness costs (resistant strains grow slower)
4. Resistant subpopulation dynamics (separate ODE states for resistance levels)

Key concepts:
- MIC is modeled as a continuous variable that can increase stepwise
- Resistant subpopulations emerge when drug exposure creates selective pressure
- Each resistance level carries a fitness cost (reduced growth rate)
- Resistance can be transferred between subpopulations via mutation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import copy


# ---------------------------------------------------------------------------
# Resistance parameters
# ---------------------------------------------------------------------------

@dataclass
class ResistanceParameters:
    """
    Parameters controlling resistance evolution dynamics.

    Attributes:
        MIC_baseline: initial MIC (mg/L) for susceptible strain
        MIC_max: maximum achievable MIC (mg/L)
        mu_resistance: mutation rate per cell per generation for resistance acquisition
        n_resistance_levels: number of discrete resistance levels to track
        MIC_fold_change: fold-increase per resistance step (e.g., 2.0 for 2-fold)
        selection_window_low: lower bound of C/MIC ratio for selection (fraction)
        selection_window_high: upper bound of C/MIC ratio for selection (fraction)
        fitness_cost: growth rate reduction per resistance level (fraction)
        reversion_rate: rate of resistance loss without selective pressure (per hour)
        resistance_transfer_rate: rate of resistance transfer between subpopulations
    """
    MIC_baseline: float = 1.0        # mg/L
    MIC_max: float = 256.0           # mg/L (max achievable)
    mu_resistance: float = 1e-7      # per cell per generation
    n_resistance_levels: int = 4     # 0=susceptible, 1=intermediate, 2=resistant, 3=highly resistant
    MIC_fold_change: float = 2.0     # 2-fold increase per step
    selection_window_low: float = 0.25  # C/MIC ratio lower bound
    selection_window_high: float = 1.0  # C/MIC ratio upper bound
    fitness_cost: float = 0.1        # 10% growth reduction per resistance level
    reversion_rate: float = 1e-4     # per hour
    resistance_transfer_rate: float = 1e-6  # per cell per hour


# ---------------------------------------------------------------------------
# Resistance state tracker
# ---------------------------------------------------------------------------

@dataclass
class ResistanceState:
    """
    Tracks the current resistance state of a bacterial population.

    Attributes:
        MIC_current: current effective MIC (mg/L)
        resistance_level: discrete resistance level (0 to n_levels)
        resistant_fraction: fraction of population that is resistant
        mutation_accumulator: accumulated mutation pressure (drives step increases)
    """
    MIC_current: float = 1.0
    resistance_level: int = 0
    resistant_fraction: float = 0.0
    mutation_accumulator: float = 0.0

    def to_dict(self) -> dict:
        return {
            "MIC_current": self.MIC_current,
            "resistance_level": self.resistance_level,
            "resistant_fraction": self.resistant_fraction,
            "mutation_accumulator": self.mutation_accumulator,
        }


# ---------------------------------------------------------------------------
# Resistance evolution model
# ---------------------------------------------------------------------------

class ResistanceEvolutionModel:
    """
    Models dynamic resistance evolution during antibiotic treatment.

    This model tracks:
    - Effective MIC as a function of drug exposure and mutation pressure
    - Resistant subpopulation dynamics
    - Selection pressure from sub-MIC drug concentrations
    - Fitness costs associated with resistance

    The model can operate in two modes:
    1. Continuous MIC: MIC evolves as a continuous variable
    2. Stepwise MIC: MIC increases in discrete steps (2-fold, 4-fold, etc.)

    Parameters:
        params: ResistanceParameters instance
    """

    def __init__(self, params: Optional[ResistanceParameters] = None):
        self.params = params or ResistanceParameters()
        self._validate_params()

    def _validate_params(self):
        """Validate parameter consistency."""
        p = self.params
        assert p.MIC_baseline > 0, "MIC_baseline must be positive"
        assert p.MIC_max >= p.MIC_baseline, "MIC_max must be >= MIC_baseline"
        assert p.mu_resistance > 0, "mu_resistance must be positive"
        assert p.n_resistance_levels >= 1, "n_resistance_levels must be >= 1"
        assert p.MIC_fold_change > 1.0, "MIC_fold_change must be > 1.0"
        assert 0 <= p.selection_window_low < p.selection_window_high, \
            "selection_window_low must be < selection_window_high"
        assert 0 <= p.fitness_cost < 1.0, "fitness_cost must be in [0, 1)"

    def get_mic_for_level(self, resistance_level: int) -> float:
        """
        Get the MIC corresponding to a given resistance level.

        Parameters:
            resistance_level: integer level (0 = susceptible, 1+ = increasingly resistant)

        Returns:
            MIC value in mg/L
        """
        return self.params.MIC_baseline * (self.params.MIC_fold_change ** resistance_level)

    def get_max_resistance_level(self) -> int:
        """Get the maximum resistance level achievable."""
        level = 0
        mic = self.params.MIC_baseline
        while mic * self.params.MIC_fold_change <= self.params.MIC_max:
            level += 1
            mic *= self.params.MIC_fold_change
        return level

    def compute_selection_pressure(
        self,
        C_effect: float,
        MIC_current: float,
    ) -> float:
        """
        Compute selection pressure based on drug concentration relative to MIC.

        Selection is strongest when C/MIC is in the "mutant selection window"
        (between selection_window_low and selection_window_high).

        Parameters:
            C_effect: current drug concentration at effect site (mg/L)
            MIC_current: current effective MIC (mg/L)

        Returns:
            Selection pressure coefficient (0 to 1)
        """
        if MIC_current <= 0:
            return 0.0

        cmic_ratio = C_effect / MIC_current

        if cmic_ratio < self.params.selection_window_low:
            # Below selection window — no selective pressure
            return 0.0
        elif cmic_ratio > self.params.selection_window_high:
            # Above selection window — drug kills everything, no selection for resistance
            return 0.0
        else:
            # Inside selection window — maximum selection at midpoint
            mid = (self.params.selection_window_low + self.params.selection_window_high) / 2.0
            width = (self.params.selection_window_high - self.params.selection_window_low) / 2.0
            # Gaussian-like selection pressure peaking at midpoint
            return np.exp(-0.5 * ((cmic_ratio - mid) / (width / 2)) ** 2)

    def compute_mutation_rate(
        self,
        B_total: float,
        selection_pressure: float,
        drug_class: str = "cidal",
    ) -> float:
        """
        Compute the rate of resistance mutations.

        Parameters:
            B_total: total bacterial burden (CFU/mL)
            selection_pressure: selection pressure coefficient (0 to 1)
            drug_class: "cidal" or "static"

        Returns:
            Mutation rate (mutations per hour per mL)
        """
        # Base mutation rate scaled by population size
        base_rate = self.params.mu_resistance * B_total

        # Cidal drugs increase mutation rate via DNA damage
        if drug_class == "cidal":
            cidal_boost = 2.0
        else:
            cidal_boost = 1.0

        # Selection pressure amplifies mutation accumulation
        return base_rate * cidal_boost * (1.0 + selection_pressure)

    def compute_fitness_modifier(self, resistance_level: int) -> float:
        """
        Compute the growth rate modifier for a given resistance level.

        Higher resistance levels carry greater fitness costs.

        Parameters:
            resistance_level: integer resistance level

        Returns:
            Growth rate multiplier (1.0 = no cost, 0.0 = lethal)
        """
        return max(0.0, 1.0 - self.params.fitness_cost * resistance_level)

    def update_mic_continuous(
        self,
        MIC_current: float,
        mutation_rate: float,
        selection_pressure: float,
        dt: float,
    ) -> float:
        """
        Update MIC continuously based on mutation and selection pressure.

        Parameters:
            MIC_current: current MIC value
            mutation_rate: rate of resistance mutations
            selection_pressure: selection pressure coefficient
            dt: time step (hours)

        Returns:
            Updated MIC value
        """
        # MIC increase proportional to mutation rate and selection pressure
        mic_increase_rate = mutation_rate * selection_pressure * 0.001
        new_mic = MIC_current + mic_increase_rate * dt

        # Clamp to maximum
        return min(new_mic, self.params.MIC_max)

    def update_mic_stepwise(
        self,
        MIC_current: float,
        mutation_accumulator: float,
        selection_pressure: float,
        dt: float,
        B_total: float,
    ) -> Tuple[float, float]:
        """
        Update MIC in discrete steps based on accumulated mutation pressure.

        Parameters:
            MIC_current: current MIC value
            mutation_accumulator: accumulated mutation pressure
            selection_pressure: selection pressure coefficient
            dt: time step (hours)
            B_total: total bacterial burden

        Returns:
            Tuple of (new_MIC, new_mutation_accumulator)
        """
        # Accumulate mutation pressure
        # Scaled by population size and selection pressure
        acc_rate = self.params.mu_resistance * B_total * selection_pressure * dt
        new_accumulator = mutation_accumulator + acc_rate

        # Check if accumulated pressure exceeds threshold for step increase
        # Threshold is population-size dependent (more bacteria = faster evolution)
        threshold = max(1.0, B_total * self.params.mu_resistance * 100)

        new_mic = MIC_current
        if new_accumulator >= threshold:
            # Step increase
            new_mic = min(
                MIC_current * self.params.MIC_fold_change,
                self.params.MIC_max,
            )
            new_accumulator = 0.0  # Reset accumulator

        # Reversion: if no selection pressure, slowly lose resistance
        if selection_pressure < 0.01:
            new_accumulator *= (1.0 - self.params.reversion_rate * dt)
            # Slow MIC reversion toward baseline
            if MIC_current > self.params.MIC_baseline:
                new_mic = max(
                    new_mic - self.params.reversion_rate * dt * (new_mic - self.params.MIC_baseline),
                    self.params.MIC_baseline,
                )

        return new_mic, new_accumulator

    def compute_resistant_fraction(
        self,
        resistant_fraction: float,
        selection_pressure: float,
        fitness_modifier: float,
        mutation_rate: float,
        dt: float,
    ) -> float:
        """
        Update the fraction of the population that is resistant.

        Parameters:
            resistant_fraction: current resistant fraction (0 to 1)
            selection_pressure: selection pressure coefficient
            fitness_modifier: growth rate modifier for resistant cells
            mutation_rate: rate of new resistance mutations
            dt: time step (hours)

        Returns:
            Updated resistant fraction
        """
        # New resistant cells from mutation
        mutation_gain = mutation_rate * dt * (1.0 - resistant_fraction)

        # Selection: resistant cells grow faster under drug pressure
        if selection_pressure > 0:
            selection_gain = selection_pressure * resistant_fraction * (1.0 - resistant_fraction) * 0.1 * dt
        else:
            selection_gain = 0.0

        # Fitness cost: resistant cells grow slower without drug
        if selection_pressure < 0.01:
            fitness_loss = (1.0 - fitness_modifier) * resistant_fraction * 0.01 * dt
        else:
            fitness_loss = 0.0

        # Reversion: resistant cells can lose resistance
        reversion_loss = self.params.reversion_rate * resistant_fraction * dt

        new_fraction = resistant_fraction + mutation_gain + selection_gain - fitness_loss - reversion_loss
        return np.clip(new_fraction, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Multi-level resistance ODE system
# ---------------------------------------------------------------------------

class ResistanceODESystem:
    """
    ODE system for tracking multiple resistance levels as separate subpopulations.

    Extends the basic bacterial ODE to track:
    - B_susceptible: susceptible population (MIC = MIC_baseline)
    - B_intermediate: intermediate resistance (MIC = MIC_baseline * fold_change)
    - B_resistant: resistant population (MIC = MIC_baseline * fold_change^2)
    - B_highly_resistant: highly resistant (MIC = MIC_baseline * fold_change^3)

    Each subpopulation has different:
    - Growth rates (fitness cost)
    - Drug sensitivity (MIC)
    - Mutation rates to/from other levels

    Parameters:
        resistance_params: ResistanceParameters instance
        n_levels: number of resistance levels to track (default: from params)
    """

    def __init__(
        self,
        resistance_params: Optional[ResistanceParameters] = None,
        n_levels: Optional[int] = None,
    ):
        self.params = resistance_params or ResistanceParameters()
        self.n_levels = n_levels or self.params.n_resistance_levels
        self.resistance_model = ResistanceEvolutionModel(self.params)

    def get_state_indices(self) -> Dict[str, int]:
        """
        Get the ODE state indices for resistance subpopulations.

        Returns:
            dict mapping state names to indices in the extended state vector.
        """
        indices = {}
        for i in range(self.n_levels):
            if i == 0:
                indices[f"B_resist_{i}"] = 0  # susceptible
            else:
                indices[f"B_resist_{i}"] = i
        return indices

    def get_n_states(self) -> int:
        """Get the number of additional ODE states for resistance tracking."""
        return self.n_levels  # One state per resistance level

    def rhs(
        self,
        t: float,
        B_resist: np.ndarray,
        B_pers: float,
        B_SCV: float,
        N_eff: float,
        C_effect: float,
        k_growth: float,
        B_max: float,
        k_kill_base: float,
        drug_class: str = "cidal",
    ) -> np.ndarray:
        """
        Right-hand side of the resistance level ODE system.

        Parameters:
            t: time (hours)
            B_resist: array of bacterial counts at each resistance level
            B_pers: persister population (not tracked for resistance)
            B_SCV: SCV population (not tracked for resistance)
            N_eff: immune effector count
            C_effect: drug concentration at effect site
            k_growth: base growth rate
            B_max: carrying capacity
            k_kill_base: base immune killing rate
            drug_class: "cidal" or "static"

        Returns:
            dB_resist/dt array
        """
        dB = np.zeros(self.n_levels)
        B_total = np.sum(B_resist) + B_pers + B_SCV

        for i in range(self.n_levels):
            B_i = max(B_resist[i], 1e-10)
            mic_i = self.resistance_model.get_mic_for_level(i)
            fitness = self.resistance_model.compute_fitness_modifier(i)

            # Growth (with fitness cost and carrying capacity)
            growth = k_growth * fitness * (1.0 - B_total / B_max) * B_i

            # Drug effect depends on C/MIC ratio for this level
            cmic_ratio = C_effect / mic_i if mic_i > 0 else 0.0

            if drug_class == "cidal":
                # Cidal killing proportional to C/MIC ratio
                if cmic_ratio > 0:
                    kill_rate = 2.0 * (1.0 + C_effect) * (cmic_ratio / (1.0 + cmic_ratio))
                else:
                    kill_rate = 0.0
                drug_kill = kill_rate * B_i
            else:
                # Static: growth inhibition proportional to C/MIC
                growth *= 1.0 / (1.0 + cmic_ratio)
                drug_kill = 0.0

            # Immune killing
            immune_kill = k_kill_base * N_eff * B_i

            # Mutation to higher resistance level
            mutation_up = 0.0
            if i < self.n_levels - 1:
                selection = self.resistance_model.compute_selection_pressure(C_effect, mic_i)
                mutation_up = self.params.mu_resistance * B_i * (1.0 + selection)

            # Reversion to lower resistance level
            mutation_down = 0.0
            if i > 0:
                mutation_down = self.params.reversion_rate * B_i

            # Net change
            dB[i] = growth - immune_kill - drug_kill - mutation_up + mutation_down

            # Add incoming mutations from lower level
            if i > 0:
                mic_lower = self.resistance_model.get_mic_for_level(i - 1)
                selection_lower = self.resistance_model.compute_selection_pressure(C_effect, mic_lower)
                dB[i] += self.params.mu_resistance * max(B_resist[i - 1], 0) * (1.0 + selection_lower)

            # Remove outgoing mutations (already subtracted as mutation_up)

        return dB


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def create_resistance_model(
    MIC_baseline: float = 1.0,
    n_levels: int = 4,
    MIC_fold_change: float = 2.0,
    fitness_cost: float = 0.1,
    mu_resistance: float = 1e-7,
) -> ResistanceEvolutionModel:
    """
    Create a ResistanceEvolutionModel with specified parameters.

    Parameters:
        MIC_baseline: initial MIC (mg/L)
        n_levels: number of resistance levels
        MIC_fold_change: fold-increase per resistance step
        fitness_cost: growth rate reduction per level
        mu_resistance: mutation rate

    Returns:
        ResistanceEvolutionModel instance
    """
    params = ResistanceParameters(
        MIC_baseline=MIC_baseline,
        n_resistance_levels=n_levels,
        MIC_fold_change=MIC_fold_change,
        fitness_cost=fitness_cost,
        mu_resistance=mu_resistance,
    )
    return ResistanceEvolutionModel(params)


def create_resistance_ode(
    n_levels: int = 4,
    MIC_baseline: float = 1.0,
    MIC_fold_change: float = 2.0,
    fitness_cost: float = 0.1,
) -> ResistanceODESystem:
    """
    Create a ResistanceODESystem with specified parameters.

    Parameters:
        n_levels: number of resistance levels
        MIC_baseline: initial MIC
        MIC_fold_change: fold-increase per step
        fitness_cost: growth rate reduction per level

    Returns:
        ResistanceODESystem instance
    """
    params = ResistanceParameters(
        MIC_baseline=MIC_baseline,
        n_resistance_levels=n_levels,
        MIC_fold_change=MIC_fold_change,
        fitness_cost=fitness_cost,
    )
    return ResistanceODESystem(params, n_levels)


def simulate_resistance_evolution(
    drug_concentration: float,
    MIC_baseline: float = 1.0,
    initial_burden: float = 1e6,
    immune_level: float = 1e7,
    drug_class: str = "cidal",
    duration_hours: float = 168.0,
    dt: float = 0.1,
    n_levels: int = 4,
    fitness_cost: float = 0.1,
    mu_resistance: float = 1e-7,
) -> Dict[str, np.ndarray]:
    """
    Simulate resistance evolution over time using the multi-level ODE system.

    Parameters:
        drug_concentration: constant drug concentration (mg/L)
        MIC_baseline: initial MIC
        initial_burden: initial total bacterial burden
        immune_level: immune effector count
        drug_class: "cidal" or "static"
        duration_hours: simulation duration
        dt: time step
        n_levels: number of resistance levels
        fitness_cost: fitness cost per level
        mu_resistance: mutation rate

    Returns:
        dict with keys:
            - "t": time array
            - "B_resist": (n_times, n_levels) array of bacterial counts
            - "B_total": total bacterial burden over time
            - "MIC_effective": effective MIC over time
            - "resistant_fraction": fraction of resistant population over time
    """
    params = ResistanceParameters(
        MIC_baseline=MIC_baseline,
        n_resistance_levels=n_levels,
        fitness_cost=fitness_cost,
        mu_resistance=mu_resistance,
    )
    ode_sys = ResistanceODESystem(params, n_levels)

    # Initial conditions: all bacteria susceptible
    B_resist = np.zeros(n_levels)
    B_resist[0] = initial_burden

    n_steps = int(duration_hours / dt)
    t_arr = np.zeros(n_steps + 1)
    B_arr = np.zeros((n_steps + 1, n_levels))
    B_arr[0] = B_resist.copy()

    B_pers = initial_burden * 0.001
    B_SCV = 0.0

    k_growth = 0.5
    B_max = 1e9
    k_kill_base = 1e-8

    for step in range(n_steps):
        t = step * dt

        # Simple Euler integration
        dB = ode_sys.rhs(
            t, B_resist, B_pers, B_SCV, immune_level,
            drug_concentration, k_growth, B_max, k_kill_base,
            drug_class=drug_class,
        )

        B_resist = np.maximum(B_resist + dB * dt, 0.0)
        t_arr[step + 1] = t + dt
        B_arr[step + 1] = B_resist.copy()

    # Compute derived quantities
    B_total = np.sum(B_arr, axis=1)
    MIC_effective = np.zeros(n_steps + 1)
    resistant_fraction = np.zeros(n_steps + 1)

    res_model = ResistanceEvolutionModel(params)
    for i in range(n_steps + 1):
        if B_total[i] > 0:
            # Weighted average MIC
            mic_sum = 0.0
            for j in range(n_levels):
                mic_sum += B_arr[i, j] * res_model.get_mic_for_level(j)
            MIC_effective[i] = mic_sum / B_total[i]

            # Resistant fraction (level >= 1)
            resistant_fraction[i] = np.sum(B_arr[i, 1:]) / B_total[i]
        else:
            MIC_effective[i] = MIC_baseline
            resistant_fraction[i] = 0.0

    return {
        "t": t_arr,
        "B_resist": B_arr,
        "B_total": B_total,
        "MIC_effective": MIC_effective,
        "resistant_fraction": resistant_fraction,
    }


# ===========================================================================
# Two-stage pipeline: Core simulation → Resistance analysis
# ===========================================================================

def simulate_resistance_from_profile(
    concentration_profile: "np.ndarray",
    time_array: "np.ndarray",
    MIC_baseline: float = 1.0,
    initial_burden: float = 1e5,
    drug_class: str = "cidal",
    dt: float = 0.5,
    n_levels: int = 4,
    fitness_cost: float = 0.1,
    mu_resistance: float = 1e-7,
) -> Dict[str, "np.ndarray"]:
    """
    Simulate resistance evolution using a drug concentration profile
    exported from a core PK/PD simulation (two-stage pipeline).

    This is the recommended way to analyze resistance: first run the core
    simulation (PK + PD + immune + cytokines), then feed the resulting
    concentration trajectory into this function to assess resistance
    selection pressure.

    Parameters:
        concentration_profile: effect-site concentration (mg/L) at each time point,
            as returned by SimulationResult.get_effect_site_concentration()
        time_array: time points (hours) corresponding to the concentration profile
        MIC_baseline: baseline MIC for susceptible strain (mg/L)
        initial_burden: initial bacterial burden (CFU/mL)
        drug_class: "cidal" or "static"
        dt: internal integration time step (hours)
        n_levels: number of discrete resistance levels
        fitness_cost: fitness cost per resistance level (fraction of growth rate)
        mu_resistance: mutation rate per cell per generation

    Returns:
        dict with keys:
            - "t": time array (hours)
            - "B_resist": (n_times, n_levels) bacterial counts per level
            - "B_total": total bacterial burden over time
            - "MIC_effective": effective MIC trajectory
            - "resistant_fraction": fraction of resistant population

    Example:
        >>> result = run_simulation(pk_model, regimen, pd_model, ic, t_span=(0, 72))
        >>> C = result.get_effect_site_concentration(pk_model)
        >>> res = simulate_resistance_from_profile(C, result.t, MIC_baseline=1.0)
        >>> print(f"Final MIC: {res['MIC_effective'][-1]:.2f}x baseline")
    """
    import numpy as np

    # Interpolate concentration profile onto internal time grid
    t_resistance = np.arange(0, time_array[-1] + dt, dt)
    C_interp = np.interp(t_resistance, time_array, concentration_profile)

    params = ResistanceParameters(
        MIC_baseline=MIC_baseline,
        n_resistance_levels=n_levels,
        fitness_cost=fitness_cost,
        mu_resistance=mu_resistance,
    )

    res_model = ResistanceEvolutionModel(params)

    # Initial state
    B_resist = np.zeros(n_levels)
    B_resist[0] = initial_burden

    n_steps = len(t_resistance)
    B_arr = np.zeros((n_steps, n_levels))
    B_arr[0] = B_resist.copy()

    for step in range(n_steps - 1):
        C = C_interp[step]
        B_total = np.sum(B_resist)

        for j in range(n_levels):
            if B_resist[j] <= 0:
                continue

            mic_j = res_model.get_mic_for_level(j)
            fitness_j = res_model.compute_fitness_modifier(j)

            # Selection (returns float 0-1)
            selection = res_model.compute_selection_pressure(C, mic_j)

            # Net growth (reduced by fitness cost)
            k_net = 0.5 * (1 - selection) * (1 - fitness_j)

            # Mutation forward (j -> j+1)
            if j < n_levels - 1:
                mu_forward = mu_resistance * B_resist[j] * max(k_net, 0)
            else:
                mu_forward = 0

            # Mutation backward (j -> j-1)
            if j > 0:
                mu_backward = mu_resistance * 0.1 * B_resist[j] * max(k_net, 0)
            else:
                mu_backward = 0

            # Drug kill
            if drug_class == "cidal":
                damage_rate = 2.0 * C * B_resist[j]
                repair_rate = 0.1 * max(0, 6.0 - C)  # simplified
                kill = max(0, damage_rate - repair_rate) * 2.0
            else:
                h_static = 5.0 * C**1.2 / (MIC_baseline**1.2 + C**1.2)
                kill = h_static * B_resist[j]

            # Immune killing (simplified)
            immune_kill = 1e-8 * 1e7 * B_resist[j]

            # Net change
            dB = (k_net * B_resist[j] - kill - immune_kill
                  - mu_forward + mu_backward)

            if j < n_levels - 1:
                # Receive mutations from level j-1
                if j > 0:
                    prev_mic = res_model.get_mic_for_level(j - 1)
                    prev_sel = res_model.compute_selection_pressure(C, prev_mic)
                    prev_fit = res_model.compute_fitness_modifier(j - 1)
                    mu_from_prev = mu_resistance * B_resist[j - 1] * max(
                        0.5 * (1 - prev_sel) * (1 - prev_fit), 0)
                    dB += mu_from_prev

            B_resist[j] = max(B_resist[j] + dB * dt, 0)

        B_arr[step + 1] = B_resist.copy()

    # Compute derived quantities
    B_total = np.sum(B_arr, axis=1)
    MIC_effective = np.zeros(n_steps)
    resistant_fraction = np.zeros(n_steps)

    res_model2 = ResistanceEvolutionModel(params)
    for i in range(n_steps):
        if B_total[i] > 0:
            mic_sum = 0.0
            for j in range(n_levels):
                mic_sum += B_arr[i, j] * res_model2.get_mic_for_level(j)
            MIC_effective[i] = mic_sum / B_total[i]
            resistant_fraction[i] = np.sum(B_arr[i, 1:]) / B_total[i]
        else:
            MIC_effective[i] = MIC_baseline
            resistant_fraction[i] = 0.0

    return {
        "t": t_resistance,
        "B_resist": B_arr,
        "B_total": B_total,
        "MIC_effective": MIC_effective,
        "resistant_fraction": resistant_fraction,
    }


def run_simulation_with_resistance(
    pk_model,
    regimen,
    pd_model,
    initial_conditions: Dict,
    t_span: Tuple[float, float] = (0, 72),
    drug_class: str = "cidal",
    MIC_baseline: float = 1.0,
    fitness_cost: float = 0.1,
    mu_resistance: float = 1e-7,
    **sim_kwargs
) -> Tuple["SimulationResult", Dict[str, "np.ndarray"]]:
    """
    Two-stage pipeline: run core PK/PD simulation, then analyze
    resistance evolution from the resulting concentration profile.

    Parameters:
        pk_model: TwoCompartmentPKModel instance
        regimen: DosingRegimen instance
        pd_model: BacterialPopulationODE instance
        initial_conditions: dict for PD initial state
        t_span: (t_start, t_end) in hours
        drug_class: "cidal" or "static"
        MIC_baseline: baseline MIC for resistance model
        fitness_cost: fitness cost per resistance level
        mu_resistance: mutation rate
        **sim_kwargs: additional keyword arguments passed to run_simulation()

    Returns:
        Tuple of (SimulationResult, resistance_dict) where:
            - SimulationResult: the core simulation output
            - resistance_dict: resistance evolution results with keys
              "t", "B_resist", "B_total", "MIC_effective", "resistant_fraction"

    Example:
        >>> sim_res, res = run_simulation_with_resistance(
        ...     pk_model, regimen, pd_model, ic,
        ...     t_span=(0, 72), drug_class="cidal"
        ... )
        >>> _, burden = sim_res.get_bacterial_burden()
        >>> print(f"Final MIC: {res['MIC_effective'][-1]:.2f}x baseline")
    """
    from src.core.simulation import run_simulation

    # Stage 1: Core simulation
    sim_result = run_simulation(
        pk_model, regimen, pd_model, initial_conditions,
        t_span=t_span, drug_class=drug_class, **sim_kwargs
    )

    # Stage 2: Resistance analysis from concentration profile
    C_profile = sim_result.get_effect_site_concentration(pk_model)
    resistance_result = simulate_resistance_from_profile(
        concentration_profile=C_profile,
        time_array=sim_result.t,
        MIC_baseline=MIC_baseline,
        initial_burden=initial_conditions.get("B_rep", 1e5) +
                       initial_conditions.get("B_pers", 1e2) +
                       initial_conditions.get("B_SCV", 0),
        drug_class=drug_class,
        fitness_cost=fitness_cost,
        mu_resistance=mu_resistance,
    )

    return sim_result, resistance_result
