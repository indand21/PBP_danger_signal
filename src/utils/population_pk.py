"""
Population pharmacokinetics module
Implements between-subject variability and covariate models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

from src.core.parameters import PKParameters, get_drug_pk_parameters


@dataclass
class PopulationPKVariability:
    """
    Between-subject variability (BSV) parameters
    Omega values represent %CV on log-scale
    """
    omega_CL: float = 0.30  # 30% CV for clearance
    omega_Vc: float = 0.25  # 25% CV for central volume
    omega_Vp: float = 0.35  # 35% CV for peripheral volume
    omega_Q: float = 0.40   # 40% CV for inter-compartmental clearance
    omega_Ka: float = 0.50  # 50% CV for absorption (oral only)
    omega_Kp: float = 0.20  # 20% CV for tissue penetration

    # Covariance terms (simplified - assume independent for now)
    corr_CL_Vc: float = 0.3  # Correlation between CL and Vc


@dataclass
class PatientCovariates:
    """
    Patient-specific covariates for PK modeling
    """
    age: float = 60  # years
    weight: float = 70  # kg
    height: float = 170  # cm
    sex: str = 'M'  # 'M' or 'F'
    creatinine: float = 1.0  # mg/dL serum creatinine
    albumin: float = 4.0  # g/dL
    liver_function: str = 'normal'  # 'normal', 'mild', 'moderate', 'severe'

    def creatinine_clearance_cockcroft_gault(self) -> float:
        """
        Calculate creatinine clearance using Cockcroft-Gault equation
        CrCl = [(140-age) × weight × (0.85 if female)] / (72 × Scr)
        """
        factor = 0.85 if self.sex == 'F' else 1.0
        crcl = ((140 - self.age) * self.weight * factor) / (72 * self.creatinine)
        return crcl  # mL/min

    def body_surface_area_dubois(self) -> float:
        """
        Calculate BSA using DuBois formula
        BSA = 0.007184 × height^0.725 × weight^0.425
        """
        bsa = 0.007184 * (self.height ** 0.725) * (self.weight ** 0.425)
        return bsa  # m^2

    def ideal_body_weight(self) -> float:
        """
        Calculate ideal body weight
        """
        if self.sex == 'M':
            ibw = 50 + 0.91 * (self.height - 152.4)
        else:
            ibw = 45.5 + 0.91 * (self.height - 152.4)
        return ibw  # kg


class PopulationPKModel:
    """
    Population PK model with covariate effects and between-subject variability
    """

    def __init__(self, drug_name: str,
                 variability: Optional[PopulationPKVariability] = None):
        """
        Parameters:
            drug_name: Name of drug (doxycycline, meropenem, linezolid, ciprofloxacin)
            variability: BSV parameter object
        """
        self.drug_name = drug_name
        self.typical_params = get_drug_pk_parameters(drug_name)
        self.variability = variability or PopulationPKVariability()

        # Drug-specific covariate models
        self.covariate_model = self._get_covariate_model(drug_name)

    def _get_covariate_model(self, drug_name: str) -> Dict:
        """
        Define drug-specific covariate effects on PK parameters
        """
        if drug_name.lower() == 'meropenem':
            # Renal clearance drug
            return {
                'CL_crcl_power': 0.75,  # CL scales with CrCl^0.75
                'CL_weight_power': 0.75,
                'Vc_weight_power': 1.0
            }
        elif drug_name.lower() == 'doxycycline':
            # Hepatic clearance
            return {
                'CL_liver_reduction': 0.5,  # 50% reduction in liver impairment
                'CL_weight_power': 0.75,
                'Vc_weight_power': 1.0
            }
        elif drug_name.lower() == 'linezolid':
            # Mixed clearance
            return {
                'CL_crcl_power': 0.4,  # Partial renal
                'CL_weight_power': 0.75,
                'Vc_weight_power': 1.0
            }
        else:
            # Default model
            return {
                'CL_weight_power': 0.75,
                'Vc_weight_power': 1.0
            }

    def apply_covariate_effects(self, typical_pk: PKParameters,
                                covariates: PatientCovariates) -> PKParameters:
        """
        Apply covariate effects to typical population parameters
        """
        # Start with typical values
        CL = typical_pk.CL
        Vc = typical_pk.Vc
        Vp = typical_pk.Vp
        Q = typical_pk.Q
        Ka = typical_pk.Ka
        Kp = typical_pk.Kp

        # Weight effects (allometric scaling)
        weight_ref = 70.0  # kg reference weight
        CL = CL * (covariates.weight / weight_ref) ** self.covariate_model.get('CL_weight_power', 0.75)
        Vc = Vc * (covariates.weight / weight_ref) ** self.covariate_model.get('Vc_weight_power', 1.0)
        Vp = Vp * (covariates.weight / weight_ref) ** self.covariate_model.get('Vc_weight_power', 1.0)
        Q = Q * (covariates.weight / weight_ref) ** 0.75

        # Renal function effects (if applicable)
        if 'CL_crcl_power' in self.covariate_model:
            crcl = covariates.creatinine_clearance_cockcroft_gault()
            crcl_ref = 100.0  # mL/min reference
            CL = CL * (crcl / crcl_ref) ** self.covariate_model['CL_crcl_power']

        # Liver function effects (if applicable)
        if 'CL_liver_reduction' in self.covariate_model:
            if covariates.liver_function == 'moderate':
                CL = CL * 0.7
            elif covariates.liver_function == 'severe':
                CL = CL * self.covariate_model['CL_liver_reduction']

        # Age effects on Kp (elderly have reduced tissue perfusion)
        if covariates.age > 65:
            age_factor = 1.0 - 0.002 * (covariates.age - 65)  # 0.2% reduction per year over 65
            Kp = Kp * max(age_factor, 0.7)  # Cap at 70%

        return PKParameters(CL=CL, Vc=Vc, Vp=Vp, Q=Q, Ka=Ka, Kp=Kp)

    def sample_individual_params(self, covariates: PatientCovariates,
                                 seed: Optional[int] = None) -> PKParameters:
        """
        Sample individual PK parameters with BSV
        """
        if seed is not None:
            np.random.seed(seed)

        # Apply covariate effects first
        covariate_adjusted = self.apply_covariate_effects(self.typical_params, covariates)

        # Sample random effects (log-normal distribution)
        eta_CL = np.random.normal(0, self.variability.omega_CL)
        eta_Vc = np.random.normal(0, self.variability.omega_Vc)
        eta_Vp = np.random.normal(0, self.variability.omega_Vp)
        eta_Q = np.random.normal(0, self.variability.omega_Q)
        eta_Ka = np.random.normal(0, self.variability.omega_Ka)
        eta_Kp = np.random.normal(0, self.variability.omega_Kp)

        # Apply BSV (log-normal)
        CL_ind = covariate_adjusted.CL * np.exp(eta_CL)
        Vc_ind = covariate_adjusted.Vc * np.exp(eta_Vc)
        Vp_ind = covariate_adjusted.Vp * np.exp(eta_Vp)
        Q_ind = covariate_adjusted.Q * np.exp(eta_Q)
        Ka_ind = covariate_adjusted.Ka * np.exp(eta_Ka) if covariate_adjusted.Ka > 0 else 0
        Kp_ind = covariate_adjusted.Kp * np.exp(eta_Kp)

        # Ensure positive values
        CL_ind = max(CL_ind, 0.1)
        Vc_ind = max(Vc_ind, 0.01)
        Kp_ind = np.clip(Kp_ind, 0.1, 2.0)

        return PKParameters(
            CL=CL_ind,
            Vc=Vc_ind,
            Vp=Vp_ind,
            Q=Q_ind,
            Ka=Ka_ind,
            Kp=Kp_ind
        )

    def sample_cohort(self, n_patients: int,
                     covariate_distributions: Optional[Dict] = None) -> pd.DataFrame:
        """
        Sample a cohort of virtual patients with PK parameters
        """
        if covariate_distributions is None:
            # Default distributions
            covariate_distributions = {
                'age': {'mean': 60, 'std': 15},
                'weight': {'mean': 75, 'std': 15},
                'height': {'mean': 170, 'std': 10},
                'sex': {'prob_male': 0.5},
                'creatinine': {'mean': 1.0, 'std': 0.3}
            }

        cohort_data = []

        for i in range(n_patients):
            # Sample covariates
            age = max(18, np.random.normal(
                covariate_distributions['age']['mean'],
                covariate_distributions['age']['std']
            ))
            weight = max(40, np.random.normal(
                covariate_distributions['weight']['mean'],
                covariate_distributions['weight']['std']
            ))
            height = max(150, np.random.normal(
                covariate_distributions['height']['mean'],
                covariate_distributions['height']['std']
            ))
            sex = 'M' if np.random.rand() < covariate_distributions['sex']['prob_male'] else 'F'
            creatinine = max(0.5, np.random.normal(
                covariate_distributions['creatinine']['mean'],
                covariate_distributions['creatinine']['std']
            ))

            covariates = PatientCovariates(
                age=age,
                weight=weight,
                height=height,
                sex=sex,
                creatinine=creatinine
            )

            # Sample PK parameters
            pk_params = self.sample_individual_params(covariates, seed=None)

            # Store data
            patient_data = {
                'patient_id': i,
                'age': age,
                'weight': weight,
                'height': height,
                'sex': sex,
                'creatinine': creatinine,
                'crcl': covariates.creatinine_clearance_cockcroft_gault(),
                'bsa': covariates.body_surface_area_dubois(),
                'CL': pk_params.CL,
                'Vc': pk_params.Vc,
                'Vp': pk_params.Vp,
                'Q': pk_params.Q,
                'Ka': pk_params.Ka,
                'Kp': pk_params.Kp
            }
            cohort_data.append(patient_data)

        return pd.DataFrame(cohort_data)


def visualize_population_pk_variability(cohort_df: pd.DataFrame,
                                       drug_name: str,
                                       output_file: str = None):
    """
    Visualize PK parameter distributions in virtual cohort
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Plot distributions
    params_to_plot = ['CL', 'Vc', 'Kp', 'weight', 'crcl', 'age']
    titles = ['Clearance (mL/min)', 'Central Volume (L)', 'Tissue Penetration',
              'Body Weight (kg)', 'CrCl (mL/min)', 'Age (years)']

    for idx, (param, title) in enumerate(zip(params_to_plot, titles)):
        ax = axes.flatten()[idx]
        values = cohort_df[param].values

        # Histogram
        ax.hist(values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {values.mean():.2f}')
        ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(values):.2f}')

        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Population PK Variability: {drug_name.capitalize()}',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def run_population_pk_stage():
    """
    Execute Stage 2: Population PK implementation
    """
    print('='*80)
    print('STAGE 2: POPULATION PK WITH BETWEEN-SUBJECT VARIABILITY')
    print('='*80)

    # Test for both drugs
    drugs = ['doxycycline', 'meropenem']

    for drug in drugs:
        print(f'\n[Step 2.{drugs.index(drug)+1}] Generating population PK for {drug.upper()}...')

        # Create population PK model
        pop_pk_model = PopulationPKModel(
            drug_name=drug,
            variability=PopulationPKVariability(
                omega_CL=0.30,
                omega_Vc=0.25,
                omega_Vp=0.35,
                omega_Kp=0.20
            )
        )

        # Define covariate distributions (clinical trial-like population)
        covariate_dist = {
            'age': {'mean': 62, 'std': 14},
            'weight': {'mean': 78, 'std': 18},
            'height': {'mean': 170, 'std': 12},
            'sex': {'prob_male': 0.52},
            'creatinine': {'mean': 1.1, 'std': 0.4}
        }

        # Sample virtual cohort
        print(f'  > Sampling n=1000 virtual patients...')
        cohort = pop_pk_model.sample_cohort(
            n_patients=1000,
            covariate_distributions=covariate_dist
        )

        # Summary statistics
        print(f'\n  Population Summary ({drug}):')
        print(f'    Age: {cohort["age"].mean():.1f} ± {cohort["age"].std():.1f} years')
        print(f'    Weight: {cohort["weight"].mean():.1f} ± {cohort["weight"].std():.1f} kg')
        print(f'    CrCl: {cohort["crcl"].mean():.1f} ± {cohort["crcl"].std():.1f} mL/min')
        print(f'    CL: {cohort["CL"].mean():.1f} ± {cohort["CL"].std():.1f} mL/min (CV: {100*cohort["CL"].std()/cohort["CL"].mean():.1f}%)')
        print(f'    Vc: {cohort["Vc"].mean():.3f} ± {cohort["Vc"].std():.3f} L (CV: {100*cohort["Vc"].std()/cohort["Vc"].mean():.1f}%)')
        print(f'    Kp: {cohort["Kp"].mean():.3f} ± {cohort["Kp"].std():.3f} (CV: {100*cohort["Kp"].std()/cohort["Kp"].mean():.1f}%)')

        # Save cohort
        cohort_file = f'virtual_cohort_{drug}.csv'
        cohort.to_csv(cohort_file, index=False)
        print(f'  Cohort saved: {cohort_file}')

        # Visualize
        fig_file = f'population_pk_variability_{drug}.png'
        visualize_population_pk_variability(cohort, drug, fig_file)
        print(f'  Visualization saved: {fig_file}')

    # Demonstrate covariate effects
    print('\n[Step 2.3] Demonstrating covariate effects...')
    print('\n  Example: Meropenem PK in different patient scenarios')

    scenarios = [
        {'name': 'Healthy adult', 'age': 30, 'weight': 70, 'creatinine': 0.9},
        {'name': 'Elderly', 'age': 75, 'weight': 65, 'creatinine': 1.5},
        {'name': 'Obese', 'age': 50, 'weight': 120, 'creatinine': 1.0},
        {'name': 'Renal impairment', 'age': 60, 'weight': 70, 'creatinine': 2.5}
    ]

    pop_pk = PopulationPKModel('meropenem')

    print('\n  Patient Scenario Comparisons:')
    print('  ' + '-'*76)
    print(f'  {"Scenario":<20} {"CL (mL/min)":<15} {"Vc (L)":<12} {"CrCl (mL/min)":<15}')
    print('  ' + '-'*76)

    for scenario in scenarios:
        covariates = PatientCovariates(
            age=scenario['age'],
            weight=scenario['weight'],
            creatinine=scenario['creatinine']
        )
        pk = pop_pk.apply_covariate_effects(pop_pk.typical_params, covariates)
        crcl = covariates.creatinine_clearance_cockcroft_gault()

        print(f'  {scenario["name"]:<20} {pk.CL:<15.1f} {pk.Vc:<12.3f} {crcl:<15.1f}')

    print('  ' + '-'*76)

    print('\n' + '='*80)
    print('STAGE 2 COMPLETE: POPULATION PK')
    print('='*80)
    print('\nKey Features Implemented:')
    print('  - Log-normal BSV on CL, Vc, Vp, Q, Kp')
    print('  - Allometric scaling with body weight')
    print('  - Renal function covariate model (CrCl effects on CL)')
    print('  - Age effects on tissue penetration')
    print('  - Virtual cohort generation (n=1000 per drug)')
    print('='*80)


if __name__ == '__main__':
    run_population_pk_stage()
