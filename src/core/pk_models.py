"""
Pharmacokinetic models for individual drugs
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Callable, Dict, Tuple

class TwoCompartmentPKModel:
    """
    Two-compartment linear PK model with optional effect-site compartment

    Dynamics:
        dA_c/dt = -CL/Vc * A_c - Q/Vc * (A_c - A_p) + Input
        dA_p/dt = Q/Vc * A_c - Q/Vp * A_p
        dA_e/dt = Kp * (C_c - C_e)  [if effect_site_model=True]
    """

    def __init__(self, 
                 CL: float, 
                 Vc: float, 
                 Vp: float, 
                 Q: float,
                 Ka: float = 0.0,
                 Kp: float = 1.0,
                 effect_site_model: bool = True):
        """
        Parameters:
            CL: clearance (mL/min)
            Vc: central volume (mL)
            Vp: peripheral volume (mL)
            Q: inter-compartmental clearance (mL/min)
            Ka: absorption rate (per hour, 0 if IV)
            Kp: effect-site penetration coefficient (Effect-site AUC / Central AUC)
            effect_site_model: whether to model lung/tissue compartment
        """
        self.CL = CL
        self.Vc = Vc
        self.Vp = Vp
        self.Q = Q
        self.Ka = Ka
        self.Kp = Kp
        self.effect_site_model = effect_site_model

    def get_ode_indices(self) -> Dict[str, int]:
        """
        Return indices in the state vector where PK compartments live
        Used to couple PK to bacterial ODE
        """
        return {
            'A_central': 0,
            'A_peripheral': 1,
            'A_absorption': 2,
            'A_effect': 3 if self.effect_site_model else None,
        }

    def concentration_central(self, A_central: float) -> float:
        """Central compartment concentration (mass/volume)"""
        return A_central / self.Vc

    def concentration_effect(self, A_central: float) -> float:
        """Effective site (tissue) concentration in mg/L"""
        if self.effect_site_model:
            # Convert mg/mL to mg/L: multiply by 1000
            return self.Kp * self.concentration_central(A_central) * 1000
        else:
            return self.concentration_central(A_central) * 1000

    def ode_rhs(self, t: float, A: np.ndarray, infusion_rate: float = 0.0) -> np.ndarray:
        """
        Right-hand side of PK ODE for state vector A = [A_c, A_p, A_a, A_e]

        Parameters:
            t: time (hour)
            A: state array [A_central, A_peripheral, A_absorption, A_effect]
            infusion_rate: IV infusion rate (mg/hour)

        Returns:
            dA/dt
        """
        dA = np.zeros(4)
        A_c = A[0]
        A_p = A[1]
        A_a = A[2]

        # Central compartment
        dA[0] = (self.Ka * A_a + infusion_rate - 
                 (self.CL / 60.0 + self.Q / 60.0) * A_c + 
                 (self.Q / 60.0) * A_p)

        # Peripheral compartment
        dA[1] = (self.Q / 60.0) * (A_c - A_p) / self.Vp

        # Absorption (if oral)
        dA[2] = -self.Ka * A_a

        # Effect-site (simple equilibration to central)
        if self.effect_site_model:
            C_c = self.concentration_central(A_c)
            C_e = A[3] / 1000.0  # assume A_e is in micrograms
            dA[3] = 0.1 * (C_c - C_e)  # slow equilibration

        return dA


class DosingRegimen:
    """
    Dosing schedule: times and amounts
    """
    def __init__(self, 
                 dose_mg: float,
                 interval_hours: float,
                 start_time: float = 0.0,
                 n_doses: int = 10,
                 infusion_duration_min: float = 60):
        """
        Parameters:
            dose_mg: dose amount
            interval_hours: inter-dose interval
            start_time: time of first dose
            n_doses: number of doses to administer
            infusion_duration_min: for IV infusion, duration in minutes
        """
        self.dose_mg = dose_mg
        self.interval_hours = interval_hours
        self.start_time = start_time
        self.n_doses = n_doses
        self.infusion_duration_min = infusion_duration_min

    def get_dose_times(self) -> np.ndarray:
        """Return times of each dose"""
        return np.array([self.start_time + i * self.interval_hours 
                        for i in range(self.n_doses)])

    def get_infusion_rate(self, t: float) -> float:
        """
        Return infusion rate at time t (mg/hour)
        Assumes each dose is infused over infusion_duration_min
        """
        dose_times = self.get_dose_times()
        infusion_duration_hours = self.infusion_duration_min / 60.0

        rate = 0.0
        for dose_time in dose_times:
            if dose_time <= t < dose_time + infusion_duration_hours:
                rate += self.dose_mg / infusion_duration_hours

        return rate


def create_pk_model_with_regimen(pk_params: Dict, 
                                 regimen: DosingRegimen) -> Tuple[TwoCompartmentPKModel, DosingRegimen]:
    """
    Factory function to instantiate a PK model and regimen
    """
    model = TwoCompartmentPKModel(
        CL=pk_params['CL'],
        Vc=pk_params['Vc'],
        Vp=pk_params['Vp'],
        Q=pk_params['Q'],
        Ka=pk_params['Ka'],
        Kp=pk_params['Kp'],
        effect_site_model=True
    )
    return model, regimen
