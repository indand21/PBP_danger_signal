"""
Main simulation engine: couples PK + PD + ODE integration
"""

import numpy as np
from scipy.integrate import solve_ivp, odeint, trapezoid
from typing import Dict, Tuple, Optional
import pandas as pd

class SimulationResult:
    """Container for simulation outputs"""

    def __init__(self,
                 t: np.ndarray,
                 y: np.ndarray,
                 state_names: list,
                 params: Dict):
        self.t = t
        self.y = y
        self.state_names = state_names
        self.params = params

        # Create DataFrame for convenience
        self.df = pd.DataFrame(y, columns=state_names)
        self.df['time'] = t

    def get_bacterial_burden(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return time and total bacterial burden"""
        B_rep = self.y[:, 4]
        B_pers = self.y[:, 5]
        B_SCV = self.y[:, 6]
        B_total = B_rep + B_pers + B_SCV
        return self.t, B_total

    def get_resistance_fraction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return fraction of SCV/resistant population"""
        B_total = self.y[:, 4] + self.y[:, 5] + self.y[:, 6]
        B_total = np.maximum(B_total, 1e-6)
        frac_scv = self.y[:, 6] / B_total
        return self.t, frac_scv

    def get_cytokines(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return time, IL-6, TNF"""
        return self.t, self.y[:, 9], self.y[:, 10]

    def get_host_damage(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return time and host-damage (D_host) trajectory."""
        idx = self.state_names.index('D_host')
        return self.t, self.y[:, idx]

    def peak_host_damage(self) -> float:
        """Maximum host damage over the trajectory (severity-of-illness proxy)."""
        idx = self.state_names.index('D_host')
        return float(np.max(self.y[:, idx]))

    def terminal_host_damage(self) -> float:
        """Host damage at the final time point (residual injury)."""
        idx = self.state_names.index('D_host')
        return float(self.y[-1, idx])

    def get_effect_site_concentration(self, pk_model=None, weight_kg: float = 70.0) -> np.ndarray:
        """
        Reconstruct effect-site concentration trajectory using the analytical
        PK formula (consistent with the ODE integration).

        Parameters:
            pk_model: the TwoCompartmentPKModel instance (unused; kept for API compat)
            weight_kg: patient weight for Vc scaling (default 70 kg)

        Returns:
            Array of effect-site concentrations (mg/L) at each time point
        """
        # Use the same analytical formula as the ODE integration
        CL_val = self.params.get('_CL_val', 15.0)
        Vc_val = self.params.get('_Vc_val', 17.5)
        Kp_val = self.params.get('_Kp_val', 0.4)
        Ka_val = self.params.get('_Ka_val', 0.0)
        regimen = self.params.get('_regimen')

        if regimen is None:
            # Fallback: compute from PK state
            A_central = self.y[:, 0]  # mg
            Vc_total = Vc_val  # L
            C_central = A_central / Vc_total  # mg/L
            return Kp_val * C_central

        k_elim = CL_val / Vc_val  # per hour
        inf_duration_h = regimen.infusion_duration_min / 60.0
        dose_times = regimen.get_dose_times()
        is_oral = Ka_val > 0

        C_effects = np.zeros(len(self.t))
        for i, t in enumerate(self.t):
            C_total = 0.0
            for dt in dose_times:
                if t <= dt:
                    continue
                tau = t - dt
                if is_oral:
                    if abs(Ka_val - k_elim) > 0.001:
                        C = regimen.dose_mg / Vc_val * Ka_val / (Ka_val - k_elim) * \
                            (np.exp(-k_elim * tau) - np.exp(-Ka_val * tau))
                    else:
                        C = regimen.dose_mg / Vc_val * Ka_val * tau * np.exp(-Ka_val * tau)
                else:
                    if tau <= inf_duration_h and inf_duration_h > 0.001:
                        R0 = regimen.dose_mg / inf_duration_h
                        C = R0 / (k_elim * Vc_val) * (1 - np.exp(-k_elim * tau))
                    else:
                        if inf_duration_h > 0.001:
                            R0 = regimen.dose_mg / inf_duration_h
                            C_end = R0 / (k_elim * Vc_val) * (1 - np.exp(-k_elim * inf_duration_h))
                        else:
                            C_end = regimen.dose_mg / Vc_val
                        C = C_end * np.exp(-k_elim * (tau - max(inf_duration_h, 0)))
                C_total += C
            C_effects[i] = Kp_val * C_total

        return C_effects

    @staticmethod
    def _time_above_threshold(t: np.ndarray, values: np.ndarray, threshold: float) -> float:
        """Integrate time above a threshold with linear crossing interpolation.

        Correct on the (non-uniform) solver grid: it interpolates the exact
        crossing time within each interval, so it does NOT over-count densely
        sampled peaks the way ``np.mean(values > threshold)`` would.
        """
        total = 0.0
        for t0, t1, v0, v1 in zip(t[:-1], t[1:], values[:-1], values[1:]):
            dt = t1 - t0
            above0 = v0 >= threshold
            above1 = v1 >= threshold
            if above0 and above1:
                total += dt
            elif above0 != above1 and v1 != v0:
                crossing_fraction = (threshold - v0) / (v1 - v0)
                crossing_fraction = float(np.clip(crossing_fraction, 0.0, 1.0))
                # crossing_fraction is the position within [t0,t1] where the value
                # crosses the threshold. On a RISING crossing (v1 above) the time
                # ABOVE is the part AFTER the crossing, (1 - crossing_fraction)*dt;
                # on a FALLING crossing (v0 above) it is the part BEFORE, i.e.
                # crossing_fraction*dt.
                total += dt * (1.0 - crossing_fraction if above1 else crossing_fraction)
        return float(total)

    def get_pkpd_indices(
        self,
        mic: Optional[float] = None,
        fraction_unbound: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute the three canonical unbound PK/PD exposure indices.

        Returns ``fT_above_MIC_pct`` (percent of the interval the unbound
        effect-site concentration exceeds the MIC), ``fAUC_MIC`` and
        ``fCmax_MIC`` over the simulated interval. The isolate-specific MIC and
        the unbound fraction may be supplied to ``run_simulation`` (stored as
        ``_mic`` / ``_fraction_unbound``) or passed explicitly here.
        """
        mic = self.params.get('_mic') if mic is None else mic
        if fraction_unbound is None:
            fraction_unbound = self.params.get('_fraction_unbound', 1.0)
        if mic is None or mic <= 0:
            raise ValueError("A positive isolate-specific MIC is required")
        if not 0 < fraction_unbound <= 1:
            raise ValueError("fraction_unbound must be in (0, 1]")

        concentration = fraction_unbound * self.get_effect_site_concentration()
        duration = float(self.t[-1] - self.t[0])
        time_above = self._time_above_threshold(self.t, concentration, mic)
        auc = float(trapezoid(concentration, self.t))
        return {
            'fT_above_MIC_pct': 100.0 * time_above / duration if duration > 0 else 0.0,
            'fAUC_MIC': auc / mic,
            'fCmax_MIC': float(np.max(concentration)) / mic,
        }


def run_simulation(
    pk_model,
    regimen,
    pd_model,
    initial_conditions: Dict,
    t_span: Tuple[float, float] = (0, 96),  # hours
    drug_class: str = 'cidal',
    weight_kg: float = 70.0,
    events=None,
    dense_output=False,
    method='RK45',
    pd_parameters=None,
) -> SimulationResult:
    """
    Run a complete QSP simulation (PK coupled to bacterial PD)

    Parameters:
        pk_model: TwoCompartmentPKModel instance
        regimen: DosingRegimen instance
        pd_model: BacterialPopulationODE instance
        initial_conditions: dict with keys 'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF'
        t_span: (t_start, t_end) in hours
        drug_class: 'cidal' or 'static'
        weight_kg: patient weight for PK scaling
        events: optional event handling (e.g., dose administration)
        dense_output: if True, return interpolation function
        method: scipy.integrate.solve_ivp method

    Returns:
        SimulationResult object with trajectories
    """

    # Initialize state vector: PK (4 compartments) + PD (8 states including PAMP)
    # PK: A_central, A_peripheral, A_absorption, A_effect
    # PD: B_rep, B_pers, B_SCV, N_eff, Damage, IL6, TNF, PAMP

    y0 = np.zeros(13)

    # PK initial: all zero, will load doses via regimen
    y0[0:4] = 0

    # PD initial
    y0[4] = initial_conditions.get('B_rep', 1e5)  # CFU/mL
    y0[5] = initial_conditions.get('B_pers', 1e2)
    y0[6] = initial_conditions.get('B_SCV', 0)
    y0[7] = initial_conditions.get('N_eff', 1e7)
    y0[8] = initial_conditions.get('Damage', 0)
    y0[9] = initial_conditions.get('IL6', 0)
    y0[10] = initial_conditions.get('TNF', 0)
    y0[11] = initial_conditions.get('PAMP', 0)  # PAMPs start at zero
    y0[12] = initial_conditions.get('D_host', 0)  # host damage starts at zero

    # Pre-compute PK parameter values for analytical concentration.
    # CONVENTION: the PK model must be built from the RAW per-kg PK parameters
    # (get_drug_pk_parameters); Vc/Vp are per-kg (L/kg) and are scaled by weight
    # here. Do NOT pass normalize_pk_parameters() output (already weight-scaled) --
    # that double-scales Vc, collapsing k_elim = CL/Vc to ~0.012/h (t1/2 ~57 h) so
    # the concentration barely decays (spurious AUC/concentration weighting).
    CL_val = pk_model.CL             # L/h (total, not per-kg)
    Vc_val = pk_model.Vc * weight_kg  # L (L/kg * kg)
    Vp_val = pk_model.Vp * weight_kg  # L
    Q_val = pk_model.Q               # L/h (total)
    Ka_val = pk_model.Ka             # per hour
    Kp_val = pk_model.Kp             # tissue penetration ratio
    is_oral = Ka_val > 0

    # Elimination rate constant: CL (L/h) / Vc (L) = per hour
    k_elim = CL_val / Vc_val  # per hour

    def get_C_effect_analytical(t):
        """Compute effect-site concentration analytically from dosing history."""
        C_total = 0.0
        dose_times = regimen.get_dose_times()
        inf_duration_h = regimen.infusion_duration_min / 60.0

        # Elimination rate constant: CL / Vc per hour
        k_elim = CL_val / Vc_val  # per hour

        for dt in dose_times:
            if t <= dt:
                continue
            tau = t - dt

            if is_oral:
                # First-order oral absorption
                # C(tau) = D * Ka / Vc / (Ka - k_elim) * (exp(-k_elim*tau) - exp(-Ka*tau))
                if abs(Ka_val - k_elim) > 0.001:
                    C = regimen.dose_mg / Vc_val * Ka_val / (Ka_val - k_elim) * \
                        (np.exp(-k_elim * tau) - np.exp(-Ka_val * tau))
                else:
                    C = regimen.dose_mg / Vc_val * Ka_val * tau * np.exp(-Ka_val * tau)
            else:
                # IV infusion
                if tau <= inf_duration_h and inf_duration_h > 0.001:
                    # During infusion
                    R0 = regimen.dose_mg / inf_duration_h  # mg/h
                    C = R0 / (k_elim * Vc_val) * (1 - np.exp(-k_elim * tau))
                else:
                    # After infusion
                    if inf_duration_h > 0.001:
                        R0 = regimen.dose_mg / inf_duration_h
                        C_end = R0 / (k_elim * Vc_val) * (1 - np.exp(-k_elim * inf_duration_h))
                    else:
                        # Instant bolus
                        C_end = regimen.dose_mg / Vc_val
                    C = C_end * np.exp(-k_elim * (tau - max(inf_duration_h, 0)))

            C_total += C

        return Kp_val * C_total  # already in mg/L if Vc is in L

    def combined_rhs(t, y):
        """ODE function for PK + PD"""

        # Get current infusion rate from regimen (only for IV infusion PK state)
        inf_rate = regimen.get_infusion_rate(t)

        # PK dynamics (update PK state for completeness, but use analytical C)
        pk_state = y[0:4]
        pk_dydt = pk_model.ode_rhs(t, pk_state, infusion_rate=inf_rate)

        # Use analytical concentration (robust for both IV and oral)
        C_effect = get_C_effect_analytical(t)

        # PD dynamics
        pd_state = y[4:13]  # 9 PD states (incl. PAMP and D_host)
        pd_dydt = pd_model.rhs(t, pd_state, C_effect=C_effect,
                               drug_class=drug_class, is_static=(drug_class=='static'))

        dydt = np.concatenate([pk_dydt, pd_dydt])
        return dydt

    # Solve ODE with error handling and adaptive stepping
    try:
        sol = solve_ivp(combined_rhs, t_span, y0, method=method,
                        dense_output=dense_output, max_step=0.1,
                        rtol=1e-6, atol=1e-8)

        if not sol.success:
            print(f"Warning: ODE solver did not converge successfully. Message: {sol.message}")
    except Exception as e:
        print(f"Error in ODE integration: {e}")
        raise

    # Extract results
    t = sol.t
    y = sol.y.T  # transpose to (n_times, n_states)

    state_names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                   'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF', 'PAMP',
                   'D_host']

    # Optional PK/PD descriptor: store the isolate MIC and unbound fraction so
    # SimulationResult.get_pkpd_indices() can report unbound Craig indices without
    # re-passing them. This is metadata only -- it does NOT affect the dynamics
    # above (the calibrated kill lives in the pd_model / BacterialParameters).
    result_params = {'drug_class': drug_class, 'weight': weight_kg,
                     '_CL_val': CL_val, '_Vc_val': Vc_val, '_Kp_val': Kp_val,
                     '_Ka_val': Ka_val, '_regimen': regimen}
    if pd_parameters is not None:
        result_params['_mic'] = getattr(pd_parameters, 'mic', None)
        result_params['_fraction_unbound'] = getattr(pd_parameters, 'fraction_unbound', 1.0)

    return SimulationResult(t, y, state_names, result_params)
