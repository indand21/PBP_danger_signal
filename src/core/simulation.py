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

    def get_free_effect_site_concentration(self, pk_model=None,
                                           weight_kg: float = 70.0) -> np.ndarray:
        """Unbound effect-site concentration, mg/L.

        ``fu`` times the total. This is the concentration that drove killing
        and the one the reported free-drug indices are computed from.
        """
        fu = self.params.get('_fraction_unbound', 1.0)
        return fu * self.get_effect_site_concentration(pk_model=pk_model,
                                                       weight_kg=weight_kg)

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


def check_integration(sol, t_end: float, where: str,
                      nonnegative_rows=None, neg_rel_tol: float = 1e-9,
                      neg_abs_tol: float = 1e-3) -> None:
    """Refuse an integration that did not complete or is not physical.

    Raises RuntimeError, naming the layer, if the solver reported failure,
    stopped before ``t_end``, produced non-finite states, or drove any row in
    ``nonnegative_rows`` (population states) materially below zero.

    "Materially" means below ``-max(neg_abs_tol, neg_rel_tol * scale)``.
    Clipping in the right-hand side keeps states near zero rather than
    exactly at it, so small undershoot is integration noise: rounding level
    on a large state, and of order 1e-5 on a state that is itself zero (a
    zero inoculum gave -3e-5 CFU/mL). The absolute floor of 1e-3 state units
    -- a thousandth of an organism per mL, or of a pg/mL -- sits well above
    that noise and far below anything physical.

    Added after an external review (R11) found failed integrations being
    returned with only a printed warning, and the danger-signal layer reading
    solver output without checking success at all.
    """
    if not sol.success:
        raise RuntimeError(f"{where}: integration did not succeed "
                           f"({getattr(sol, 'message', 'no message')})")
    t = np.asarray(sol.t, dtype=float)
    if t.size == 0 or abs(t[-1] - t_end) > 1e-9 * max(1.0, abs(t_end)):
        reached = t[-1] if t.size else float("nan")
        raise RuntimeError(f"{where}: integration stopped at t = {reached:g}, "
                           f"before the requested end t = {t_end:g}")
    y = np.asarray(sol.y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"{where}: non-finite state values in the solution")
    for row in (nonnegative_rows or []):
        lo, scale = float(np.min(y[row])), float(np.max(np.abs(y[row])))
        if lo < -max(neg_abs_tol, neg_rel_tol * scale):
            raise RuntimeError(f"{where}: state row {row} went negative "
                               f"(min {lo:.3g} on a scale of {scale:.3g})")


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
    fraction_unbound: Optional[float] = None,
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

    # PK REPRESENTATION. This is a ONE-compartment model: elimination at
    # k = CL/Vc, solved analytically from the dosing history, with the
    # effect-site concentration taken as Kp times the central concentration.
    # Vp and Q are accepted by the PK model but do not enter. The drug states
    # in the result are filled from this same solution after integration
    # (see below), so the concentration that drives killing and the amounts
    # the result exports cannot disagree. Previously four "two-compartment"
    # ODE states were integrated alongside from pk_model.ode_rhs, which does
    # not conserve drug; they were exported but never drove killing.

    # BINDING CONVENTION. Kp is a TOTAL-concentration partition, so only the
    # unbound fraction of the effect-site concentration is pharmacologically
    # active. fu is applied in exactly one place, to the concentration that
    # drives killing; get_effect_site_concentration() still returns the total,
    # so get_pkpd_indices() multiplying by fu is not a second application.
    # Default 1.0 reproduces the behaviour of callers predating this.
    if fraction_unbound is not None:
        fu_val = float(fraction_unbound)
    elif pd_parameters is not None:
        fu_val = float(getattr(pd_parameters, 'fraction_unbound', 1.0))
    else:
        fu_val = 1.0
    if not 0 < fu_val <= 1:
        raise ValueError("fraction_unbound must be in (0, 1]")

    def absorption_amount_analytical(t):
        """Drug remaining in the oral absorption depot, mg (zero for IV)."""
        if not is_oral:
            return 0.0
        return float(sum(regimen.dose_mg * np.exp(-Ka_val * (t - td))
                         for td in regimen.get_dose_times() if t > td))

    def get_C_effect_analytical(t):
        """Effect-site concentration, mg/L: Kp times the central value."""
        return Kp_val * central_conc_analytical(t)

    def central_conc_analytical(t):
        """Central concentration, mg/L, from the one-compartment solution."""
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

        return C_total  # mg/L, since Vc is in L

    def combined_rhs(t, y):
        """ODE function for PK + PD"""

        # The drug states carry no dynamics here: PK is solved analytically,
        # and they are filled from that solution after integration. Keeping
        # their derivatives at zero also keeps them out of the adaptive
        # solver's error control, which the old non-conserving states shared
        # with the bacterial states.
        pk_dydt = np.zeros(4)

        # Only unbound drug acts. Kp is a TOTAL-concentration partition, so
        # the free effect-site concentration is fu times it, and this is the
        # single place binding is applied; get_effect_site_concentration()
        # deliberately still returns the total.
        C_effect = fu_val * get_C_effect_analytical(t)

        # PD dynamics
        pd_state = y[4:13]  # 9 PD states (incl. PAMP and D_host)
        pd_dydt = pd_model.rhs(t, pd_state, C_effect=C_effect,
                               drug_class=drug_class, is_static=(drug_class=='static'))

        dydt = np.concatenate([pk_dydt, pd_dydt])
        # A non-finite derivative must stop the run here. Left to the solver,
        # NaN derivatives at the first step give RK45 a NaN step size, every
        # exit test then compares false, and solve_ivp never returns.
        if not np.all(np.isfinite(dydt)):
            raise RuntimeError(f"bacterial PD (run_simulation): non-finite "
                               f"derivative at t = {t:g}")
        return dydt

    # Solve ODE with error handling and adaptive stepping
    sol = solve_ivp(combined_rhs, t_span, y0, method=method,
                    dense_output=dense_output, max_step=0.1,
                    rtol=1e-6, atol=1e-8)
    # Fail loudly rather than return a partial or non-finite trajectory. Rows
    # 4-7 are the bacterial subpopulations and immune effectors.
    check_integration(sol, t_end=t_span[1], where="bacterial PD (run_simulation)",
                      nonnegative_rows=[4, 5, 6, 7])

    # Extract results
    t = sol.t
    y = np.array(sol.y.T)  # (n_times, n_states); a copy, so sol is untouched

    # Drug states, from the one-compartment solution that drove killing.
    # There is no peripheral compartment and no effect-site AMOUNT (the
    # effect site is an instantaneous partition, Kp * C_central), so those
    # two states are identically zero; get_effect_site_concentration()
    # returns the effect-site concentration.
    y[:, 0] = [Vc_val * central_conc_analytical(tt) for tt in t]
    y[:, 1] = 0.0
    y[:, 2] = [absorption_amount_analytical(tt) for tt in t]
    y[:, 3] = 0.0

    state_names = ['A_central', 'A_peripheral', 'A_absorption', 'A_effect',
                   'B_rep', 'B_pers', 'B_SCV', 'N_eff', 'Damage', 'IL6', 'TNF', 'PAMP',
                   'D_host']

    # Optional PK/PD descriptor: store the isolate MIC and unbound fraction so
    # SimulationResult.get_pkpd_indices() can report unbound Craig indices without
    # re-passing them. This is metadata only -- it does NOT affect the dynamics
    # above (the calibrated kill lives in the pd_model / BacterialParameters).
    result_params = {'drug_class': drug_class, 'weight': weight_kg,
                     '_CL_val': CL_val, '_Vc_val': Vc_val, '_Kp_val': Kp_val,
                     '_Ka_val': Ka_val, '_regimen': regimen,
                     '_fraction_unbound': fu_val}
    if pd_parameters is not None:
        result_params['_mic'] = getattr(pd_parameters, 'mic', None)

    return SimulationResult(t, y, state_names, result_params)
