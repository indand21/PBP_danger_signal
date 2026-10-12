"""
Virtual patient cohort generator
Integrates PK variability, immune status, and pathogen characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from enum import Enum
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from src.utils.population_pk import PopulationPKModel, PatientCovariates, PopulationPKVariability
from src.core.parameters import get_default_parameters


class ImmuneStatus(Enum):
    """Immune phenotypes for patient stratification"""
    IMMUNOCOMPETENT = "immunocompetent"
    NEUTROPENIC = "neutropenic"
    HYPERINFLAMMATORY = "hyperinflammatory"  # e.g., sepsis, COVID-19
    IMMUNOSUPPRESSED = "immunosuppressed"  # e.g., transplant, steroids


class InfectionSite(Enum):
    """Infection site categories"""
    PNEUMONIA = "pneumonia"
    BLOODSTREAM = "bloodstream"
    URINARY_TRACT = "urinary_tract"
    INTRA_ABDOMINAL = "intra_abdominal"
    SKIN_SOFT_TISSUE = "skin_soft_tissue"


class VirtualPatient:
    """
    Complete virtual patient profile
    """

    def __init__(self,
                 patient_id: int,
                 demographics: PatientCovariates,
                 pk_params: Dict,
                 immune_status: ImmuneStatus,
                 infection_site: InfectionSite,
                 pathogen_params: Dict):
        self.patient_id = patient_id
        self.demographics = demographics
        self.pk_params = pk_params
        self.immune_status = immune_status
        self.infection_site = infection_site
        self.pathogen_params = pathogen_params

    def get_initial_conditions(self) -> Dict:
        """
        Generate initial conditions for ODE simulation based on patient characteristics
        """
        # Bacterial burden depends on infection site and immune status
        if self.infection_site == InfectionSite.BLOODSTREAM:
            B_rep_0 = self.pathogen_params['initial_burden'] * 0.5  # Lower in blood
        else:
            B_rep_0 = self.pathogen_params['initial_burden']

        # Immune effector baseline depends on immune status
        if self.immune_status == ImmuneStatus.NEUTROPENIC:
            N_eff_0 = 1e5  # Very low neutrophils
        elif self.immune_status == ImmuneStatus.HYPERINFLAMMATORY:
            N_eff_0 = 5e7  # Elevated baseline
        elif self.immune_status == ImmuneStatus.IMMUNOSUPPRESSED:
            N_eff_0 = 5e6  # Moderately reduced
        else:  # IMMUNOCOMPETENT
            N_eff_0 = 1e7  # Normal

        # Baseline cytokines
        if self.immune_status == ImmuneStatus.HYPERINFLAMMATORY:
            IL6_0 = 100  # Elevated baseline
            TNF_0 = 50
        else:
            IL6_0 = 10
            TNF_0 = 5

        return {
            'B_rep': B_rep_0,
            'B_pers': B_rep_0 * 0.001,  # 0.1% persisters
            'B_SCV': 0,
            'N_eff': N_eff_0,
            'Damage': 0,
            'IL6': IL6_0,
            'TNF': TNF_0
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame storage"""
        return {
            'patient_id': self.patient_id,
            'age': self.demographics.age,
            'weight': self.demographics.weight,
            'sex': self.demographics.sex,
            'creatinine': self.demographics.creatinine,
            'crcl': self.demographics.creatinine_clearance_cockcroft_gault(),
            'immune_status': self.immune_status.value,
            'infection_site': self.infection_site.value,
            'initial_burden': self.pathogen_params['initial_burden'],
            'MIC': self.pathogen_params['MIC'],
            **{f'pk_{k}': v for k, v in self.pk_params.items()}
        }


class VirtualCohortGenerator:
    """
    Generate comprehensive virtual patient cohorts
    """

    def __init__(self, drug_name: str):
        self.drug_name = drug_name
        self.pop_pk_model = PopulationPKModel(
            drug_name=drug_name,
            variability=PopulationPKVariability()
        )

    def sample_immune_status(self, immune_distribution: Optional[Dict] = None) -> ImmuneStatus:
        """
        Sample immune status according to specified distribution
        """
        if immune_distribution is None:
            # Default: mostly immunocompetent
            immune_distribution = {
                ImmuneStatus.IMMUNOCOMPETENT: 0.70,
                ImmuneStatus.NEUTROPENIC: 0.10,
                ImmuneStatus.HYPERINFLAMMATORY: 0.10,
                ImmuneStatus.IMMUNOSUPPRESSED: 0.10
            }

        statuses = list(immune_distribution.keys())
        probabilities = [immune_distribution[s] for s in statuses]
        return np.random.choice(statuses, p=probabilities)

    def sample_infection_site(self, site_distribution: Optional[Dict] = None) -> InfectionSite:
        """
        Sample infection site
        """
        if site_distribution is None:
            # Default distribution (pneumonia-focused)
            site_distribution = {
                InfectionSite.PNEUMONIA: 0.50,
                InfectionSite.BLOODSTREAM: 0.20,
                InfectionSite.URINARY_TRACT: 0.15,
                InfectionSite.INTRA_ABDOMINAL: 0.10,
                InfectionSite.SKIN_SOFT_TISSUE: 0.05
            }

        sites = list(site_distribution.keys())
        probabilities = [site_distribution[s] for s in sites]
        return np.random.choice(sites, p=probabilities)

    def sample_pathogen_params(self, MIC_distribution: str = 'log_normal') -> Dict:
        """
        Sample pathogen characteristics (initial burden, MIC)
        """
        # Initial bacterial burden (CFU/mL)
        # Log-normal: mean 1e6, std 1 log10 unit
        log_burden_mean = 6.0  # log10(1e6)
        log_burden_std = 1.0
        initial_burden = 10 ** np.random.normal(log_burden_mean, log_burden_std)
        initial_burden = np.clip(initial_burden, 1e3, 1e8)

        # MIC distribution (resistance surveillance data)
        if MIC_distribution == 'log_normal':
            # Log-normal: geometric mean 1.0 mg/L
            log_MIC_mean = 0.0  # log10(1.0)
            log_MIC_std = 0.5
            MIC = 10 ** np.random.normal(log_MIC_mean, log_MIC_std)
            MIC = np.clip(MIC, 0.125, 16.0)
        elif MIC_distribution == 'resistant_enriched':
            # Bimodal: 70% susceptible, 30% resistant
            if np.random.rand() < 0.7:
                MIC = 10 ** np.random.normal(0, 0.3)  # Susceptible
            else:
                MIC = 10 ** np.random.normal(1.5, 0.5)  # Resistant
            MIC = np.clip(MIC, 0.125, 64.0)
        else:
            MIC = 1.0  # Fixed

        return {
            'initial_burden': initial_burden,
            'MIC': MIC,
            'growth_rate_factor': np.random.uniform(0.8, 1.2)  # Strain variability
        }

    def generate_cohort(self,
                       n_patients: int,
                       immune_distribution: Optional[Dict] = None,
                       site_distribution: Optional[Dict] = None,
                       MIC_distribution: str = 'log_normal',
                       covariate_distributions: Optional[Dict] = None) -> List[VirtualPatient]:
        """
        Generate complete virtual patient cohort
        """
        print(f'\n  Generating n={n_patients} virtual patients for {self.drug_name}...')

        # Use default covariate distributions if not provided
        if covariate_distributions is None:
            covariate_distributions = {
                'age': {'mean': 62, 'std': 14},
                'weight': {'mean': 78, 'std': 18},
                'height': {'mean': 170, 'std': 12},
                'sex': {'prob_male': 0.52},
                'creatinine': {'mean': 1.1, 'std': 0.4}
            }

        cohort = []

        for i in range(n_patients):
            # Sample demographics
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

            demographics = PatientCovariates(
                age=age, weight=weight, height=height, sex=sex, creatinine=creatinine
            )

            # Sample PK parameters
            pk_params_obj = self.pop_pk_model.sample_individual_params(demographics)
            pk_params = {
                'CL': pk_params_obj.CL,
                'Vc': pk_params_obj.Vc,
                'Vp': pk_params_obj.Vp,
                'Q': pk_params_obj.Q,
                'Ka': pk_params_obj.Ka,
                'Kp': pk_params_obj.Kp
            }

            # Sample immune status
            immune_status = self.sample_immune_status(immune_distribution)

            # Sample infection site
            infection_site = self.sample_infection_site(site_distribution)

            # Sample pathogen parameters
            pathogen_params = self.sample_pathogen_params(MIC_distribution)

            # Create virtual patient
            patient = VirtualPatient(
                patient_id=i,
                demographics=demographics,
                pk_params=pk_params,
                immune_status=immune_status,
                infection_site=infection_site,
                pathogen_params=pathogen_params
            )

            cohort.append(patient)

        return cohort

    def cohort_to_dataframe(self, cohort: List[VirtualPatient]) -> pd.DataFrame:
        """Convert cohort to DataFrame"""
        return pd.DataFrame([p.to_dict() for p in cohort])


def visualize_cohort_characteristics(cohort_df: pd.DataFrame, output_file: str = None):
    """
    Visualize virtual cohort characteristics
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Age distribution
    ax = axes[0, 0]
    ax.hist(cohort_df['age'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Age (years)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Age Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Weight distribution
    ax = axes[0, 1]
    ax.hist(cohort_df['weight'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Weight (kg)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Weight Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Immune status
    ax = axes[0, 2]
    immune_counts = cohort_df['immune_status'].value_counts()
    ax.bar(range(len(immune_counts)), immune_counts.values, color='coral', edgecolor='black')
    ax.set_xticks(range(len(immune_counts)))
    ax.set_xticklabels(immune_counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Immune Status Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Infection site
    ax = axes[1, 0]
    site_counts = cohort_df['infection_site'].value_counts()
    ax.bar(range(len(site_counts)), site_counts.values, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(site_counts)))
    ax.set_xticklabels(site_counts.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Infection Site Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # MIC distribution
    ax = axes[1, 1]
    ax.hist(np.log10(cohort_df['MIC']), bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Log10(MIC) [mg/L]', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('MIC Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Initial bacterial burden
    ax = axes[1, 2]
    ax.hist(np.log10(cohort_df['initial_burden']), bins=30, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Log10(Initial Burden) [CFU/mL]', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Initial Bacterial Burden', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Virtual Patient Cohort Characteristics', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def run_virtual_cohort_stage():
    """
    Execute Stage 3: Virtual patient cohort generation
    """
    print('='*80)
    print('STAGE 3: VIRTUAL PATIENT COHORT GENERATOR')
    print('='*80)

    # Generate cohorts for both drugs
    drugs = ['doxycycline', 'meropenem']

    for drug in drugs:
        print(f'\n[Step 3.{drugs.index(drug)+1}] Generating comprehensive cohort for {drug.upper()}')

        generator = VirtualCohortGenerator(drug_name=drug)

        # Define immune distribution (enriched for heterogeneity)
        immune_dist = {
            ImmuneStatus.IMMUNOCOMPETENT: 0.60,
            ImmuneStatus.NEUTROPENIC: 0.15,
            ImmuneStatus.HYPERINFLAMMATORY: 0.15,
            ImmuneStatus.IMMUNOSUPPRESSED: 0.10
        }

        # Define infection site distribution
        site_dist = {
            InfectionSite.PNEUMONIA: 0.60,
            InfectionSite.BLOODSTREAM: 0.25,
            InfectionSite.INTRA_ABDOMINAL: 0.15
        }

        # Generate cohort
        cohort = generator.generate_cohort(
            n_patients=1000,
            immune_distribution=immune_dist,
            site_distribution=site_dist,
            MIC_distribution='log_normal'
        )

        # Convert to DataFrame
        cohort_df = generator.cohort_to_dataframe(cohort)

        # Summary statistics
        print(f'\n  Cohort Summary:')
        print(f'    Total patients: {len(cohort_df)}')
        print(f'    Age: {cohort_df["age"].mean():.1f} ± {cohort_df["age"].std():.1f} years')
        print(f'    Weight: {cohort_df["weight"].mean():.1f} ± {cohort_df["weight"].std():.1f} kg')
        print(f'    CrCl: {cohort_df["crcl"].mean():.1f} ± {cohort_df["crcl"].std():.1f} mL/min')
        print(f'\n    Immune Status:')
        for status, count in cohort_df['immune_status'].value_counts().items():
            print(f'      {status}: {count} ({100*count/len(cohort_df):.1f}%)')
        print(f'\n    Infection Sites:')
        for site, count in cohort_df['infection_site'].value_counts().items():
            print(f'      {site}: {count} ({100*count/len(cohort_df):.1f}%)')
        print(f'\n    Pathogen Characteristics:')
        print(f'      Initial burden: {cohort_df["initial_burden"].median():.2e} CFU/mL (median)')
        print(f'      MIC: {cohort_df["MIC"].median():.2f} mg/L (median, range: {cohort_df["MIC"].min():.2f}-{cohort_df["MIC"].max():.2f})')
        print(f'\n    PK Parameters:')
        print(f'      CL: {cohort_df["pk_CL"].mean():.1f} ± {cohort_df["pk_CL"].std():.1f} mL/min')
        print(f'      Kp: {cohort_df["pk_Kp"].mean():.3f} ± {cohort_df["pk_Kp"].std():.3f}')

        # Save cohort
        cohort_file = f'virtual_cohort_complete_{drug}.csv'
        cohort_df.to_csv(cohort_file, index=False)
        print(f'\n  Complete cohort saved: {cohort_file}')

        # Visualize
        fig_file = f'virtual_cohort_characteristics_{drug}.png'
        visualize_cohort_characteristics(cohort_df, fig_file)
        print(f'  Visualization saved: {fig_file}')

    # Generate specialized cohorts for specific scenarios
    print('\n[Step 3.3] Generating specialized sub-cohorts...')

    # Neutropenic cohort (chemotherapy patients)
    print('\n  > Neutropenic cohort (n=200)')
    neutropenic_generator = VirtualCohortGenerator('meropenem')
    neutropenic_cohort = neutropenic_generator.generate_cohort(
        n_patients=200,
        immune_distribution={
            ImmuneStatus.NEUTROPENIC: 1.0
        },
        MIC_distribution='log_normal'
    )
    neutropenic_df = neutropenic_generator.cohort_to_dataframe(neutropenic_cohort)
    neutropenic_df.to_csv('virtual_cohort_neutropenic.csv', index=False)
    print('    Saved: virtual_cohort_neutropenic.csv')

    # Resistant pathogen cohort
    print('\n  > Resistant pathogen cohort (n=200)')
    resistant_generator = VirtualCohortGenerator('meropenem')
    resistant_cohort = resistant_generator.generate_cohort(
        n_patients=200,
        MIC_distribution='resistant_enriched'
    )
    resistant_df = resistant_generator.cohort_to_dataframe(resistant_cohort)
    resistant_df.to_csv('virtual_cohort_resistant.csv', index=False)
    print('    Saved: virtual_cohort_resistant.csv')
    print(f'    Resistant strain fraction: {100*(resistant_df["MIC"] > 4).sum()/len(resistant_df):.1f}%')

    print('\n' + '='*80)
    print('STAGE 3 COMPLETE: VIRTUAL PATIENT COHORTS')
    print('='*80)
    print('\nKey Features Implemented:')
    print('  - Multi-dimensional patient heterogeneity')
    print('  - Immune status stratification (4 phenotypes)')
    print('  - Infection site variability (5 sites)')
    print('  - Pathogen heterogeneity (MIC, initial burden)')
    print('  - Specialized sub-cohorts (neutropenic, resistant)')
    print('  - Ready for in silico trial simulations')
    print('='*80)


if __name__ == '__main__':
    run_virtual_cohort_stage()
