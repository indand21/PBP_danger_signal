"""
In silico clinical trial simulation engine
Runs large-scale QSP simulations and computes clinical endpoints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid  # np.trapz removed in NumPy 2.0
import warnings
warnings.filterwarnings('ignore')

from src.core.parameters import get_default_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation
from src.utils.virtual_cohort import VirtualPatient, ImmuneStatus


@dataclass
class TrialEndpoints:
    """Clinical trial endpoints"""
    patient_id: int
    clinical_success: bool  # Bacterial burden < threshold at 96h
    microbiologic_success: bool  # >3 log10 reduction
    resistance_emergence: bool  # SCV fraction > 10%
    inflammatory_toxicity: bool  # IL-6 > toxicity threshold
    time_to_50_percent_reduction: float  # hours
    peak_IL6: float  # pg/mL
    peak_TNF: float  # pg/mL
    final_bacterial_burden: float  # CFU/mL
    auc_bacterial_burden: float  # CFU*h/mL


class InSilicoTrial:
    """
    Execute in silico clinical trial
    """

    def __init__(self,
                 treatment_name: str,
                 drug_name: str,
                 drug_class: str,
                 regimen: DosingRegimen,
                 simulation_duration: float = 96.0):
        """
        Parameters:
            treatment_name: Label for treatment arm
            drug_name: Name of drug
            drug_class: 'static' or 'cidal'
            regimen: Dosing regimen
            simulation_duration: Hours
        """
        self.treatment_name = treatment_name
        self.drug_name = drug_name
        self.drug_class = drug_class
        self.regimen = regimen
        self.simulation_duration = simulation_duration

        # Load calibrated parameters
        self.params = get_default_parameters()
        self.pd_model = create_ode_system(self.params)

    def simulate_patient(self, patient: VirtualPatient) -> Optional[TrialEndpoints]:
        """
        Simulate a single patient and compute endpoints
        """
        try:
            # Create PK model for this patient
            pk_model = TwoCompartmentPKModel(
                CL=patient.pk_params['CL'],
                Vc=patient.pk_params['Vc'],
                Vp=patient.pk_params['Vp'],
                Q=patient.pk_params['Q'],
                Ka=patient.pk_params['Ka'],
                Kp=patient.pk_params['Kp'],
                effect_site_model=True
            )

            # Get initial conditions
            init_cond = patient.get_initial_conditions()

            # Run simulation
            result = run_simulation(
                pk_model=pk_model,
                regimen=self.regimen,
                pd_model=self.pd_model,
                initial_conditions=init_cond,
                t_span=(0, self.simulation_duration),
                drug_class=self.drug_class,
                weight_kg=patient.demographics.weight
            )

            # Extract trajectories
            t = result.t
            B_total = result.y[:, 4] + result.y[:, 5] + result.y[:, 6]  # B_rep + B_pers + B_SCV
            B_SCV = result.y[:, 6]
            IL6 = result.y[:, 9]
            TNF = result.y[:, 10]

            # Compute endpoints
            # 1. Clinical success: Bacterial burden < 1e3 CFU/mL at end
            clinical_success = B_total[-1] < 1e3

            # 2. Microbiologic success: >3 log10 reduction
            initial_burden = init_cond['B_rep']
            log_reduction = np.log10(initial_burden / max(B_total[-1], 1))
            microbiologic_success = log_reduction >= 3.0

            # 3. Resistance emergence: SCV fraction > 10% at any time
            frac_SCV = B_SCV / np.maximum(B_total, 1e-6)
            resistance_emergence = frac_SCV.max() > 0.10

            # 4. Inflammatory toxicity: IL-6 > 500 pg/mL (ARDS risk threshold)
            inflammatory_toxicity = IL6.max() > 500

            # 5. Time to 50% reduction
            idx_50 = np.where(B_total < 0.5 * initial_burden)[0]
            if len(idx_50) > 0:
                time_to_50 = t[idx_50[0]]
            else:
                time_to_50 = self.simulation_duration  # Did not achieve

            # 6. Peak cytokines
            peak_IL6 = IL6.max()
            peak_TNF = TNF.max()

            # 7. Final bacterial burden
            final_burden = B_total[-1]

            # 8. AUC of bacterial burden
            auc_burden = trapezoid(B_total, t)

            return TrialEndpoints(
                patient_id=patient.patient_id,
                clinical_success=clinical_success,
                microbiologic_success=microbiologic_success,
                resistance_emergence=resistance_emergence,
                inflammatory_toxicity=inflammatory_toxicity,
                time_to_50_percent_reduction=time_to_50,
                peak_IL6=peak_IL6,
                peak_TNF=peak_TNF,
                final_bacterial_burden=final_burden,
                auc_bacterial_burden=auc_burden
            )

        except Exception as e:
            print(f'    Warning: Simulation failed for patient {patient.patient_id}: {e}')
            return None

    def run_trial_arm(self, cohort: List[VirtualPatient],
                     max_patients: Optional[int] = None) -> pd.DataFrame:
        """
        Run trial on entire cohort
        """
        if max_patients:
            cohort = cohort[:max_patients]

        print(f'\n  Running {self.treatment_name} arm (n={len(cohort)} patients)...')

        results = []
        failed = 0

        for idx, patient in enumerate(cohort):
            if (idx + 1) % 50 == 0:
                print(f'    Progress: {idx+1}/{len(cohort)} patients simulated')

            endpoints = self.simulate_patient(patient)
            if endpoints is not None:
                # Add patient demographics to results
                result_dict = {
                    'treatment': self.treatment_name,
                    'patient_id': patient.patient_id,
                    'age': patient.demographics.age,
                    'weight': patient.demographics.weight,
                    'immune_status': patient.immune_status.value,
                    'infection_site': patient.infection_site.value,
                    'MIC': patient.pathogen_params['MIC'],
                    'initial_burden': patient.pathogen_params['initial_burden'],
                    **endpoints.__dict__
                }
                results.append(result_dict)
            else:
                failed += 1

        print(f'    Completed: {len(results)} successful, {failed} failed simulations')

        return pd.DataFrame(results)


def analyze_trial_outcomes(results_df: pd.DataFrame, stratify_by: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze trial outcomes with optional stratification
    """
    if stratify_by:
        groups = results_df.groupby(['treatment', stratify_by])
    else:
        groups = results_df.groupby('treatment')

    summary = groups.agg({
        'clinical_success': ['sum', 'mean'],
        'microbiologic_success': ['sum', 'mean'],
        'resistance_emergence': ['sum', 'mean'],
        'inflammatory_toxicity': ['sum', 'mean'],
        'time_to_50_percent_reduction': ['mean', 'median'],
        'peak_IL6': ['mean', 'median'],
        'final_bacterial_burden': ['median'],
        'patient_id': 'count'
    }).round(3)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary


def visualize_trial_results(results_df: pd.DataFrame, output_file: str = None):
    """
    Visualize trial results comparing treatments
    """
    treatments = results_df['treatment'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Clinical success rate
    ax = axes[0, 0]
    success_rates = [results_df[results_df['treatment'] == t]['clinical_success'].mean()
                    for t in treatments]
    ax.bar(range(len(treatments)), success_rates, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=0, fontsize=9)
    ax.set_ylabel('Success Rate', fontsize=10)
    ax.set_title('Clinical Success Rate', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, rate in enumerate(success_rates):
        ax.text(i, rate + 0.02, f'{100*rate:.1f}%', ha='center', fontsize=9)

    # Resistance emergence
    ax = axes[0, 1]
    resistance_rates = [results_df[results_df['treatment'] == t]['resistance_emergence'].mean()
                       for t in treatments]
    ax.bar(range(len(treatments)), resistance_rates, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=0, fontsize=9)
    ax.set_ylabel('Resistance Rate', fontsize=10)
    ax.set_title('Resistance Emergence', fontsize=11, fontweight='bold')
    ax.set_ylim([0, max(resistance_rates) * 1.5] if max(resistance_rates) > 0 else [0, 0.1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, rate in enumerate(resistance_rates):
        ax.text(i, rate + 0.005, f'{100*rate:.1f}%', ha='center', fontsize=9)

    # Inflammatory toxicity
    ax = axes[0, 2]
    toxicity_rates = [results_df[results_df['treatment'] == t]['inflammatory_toxicity'].mean()
                     for t in treatments]
    ax.bar(range(len(treatments)), toxicity_rates, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(treatments)))
    ax.set_xticklabels(treatments, rotation=0, fontsize=9)
    ax.set_ylabel('Toxicity Rate', fontsize=10)
    ax.set_title('Inflammatory Toxicity', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    for i, rate in enumerate(toxicity_rates):
        ax.text(i, rate + 0.02, f'{100*rate:.1f}%', ha='center', fontsize=9)

    # Time to 50% reduction
    ax = axes[1, 0]
    for t in treatments:
        subset = results_df[results_df['treatment'] == t]
        ax.hist(subset['time_to_50_percent_reduction'], bins=20, alpha=0.6,
               label=t, edgecolor='black')
    ax.set_xlabel('Time to 50% Reduction (hours)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Time to 50% Burden Reduction', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Peak IL-6
    ax = axes[1, 1]
    for t in treatments:
        subset = results_df[results_df['treatment'] == t]
        ax.hist(np.log10(subset['peak_IL6'] + 1), bins=20, alpha=0.6,
               label=t, edgecolor='black')
    ax.set_xlabel('Log10(Peak IL-6) [pg/mL]', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Peak Inflammatory Response', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Final bacterial burden
    ax = axes[1, 2]
    for t in treatments:
        subset = results_df[results_df['treatment'] == t]
        ax.hist(np.log10(subset['final_bacterial_burden'] + 1), bins=20, alpha=0.6,
               label=t, edgecolor='black')
    ax.set_xlabel('Log10(Final Burden) [CFU/mL]', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Final Bacterial Burden', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('In Silico Trial Results: Static vs Cidal Comparison',
                fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def run_in_silico_trials_stage():
    """
    Execute Stage 4: In silico clinical trials
    """
    print('='*80)
    print('STAGE 4: IN SILICO CLINICAL TRIALS')
    print('='*80)

    # Load virtual cohorts
    print('\n[Step 4.1] Loading virtual patient cohorts...')
    cohort_static_df = pd.read_csv('virtual_cohort_complete_doxycycline.csv')
    cohort_cidal_df = pd.read_csv('virtual_cohort_complete_meropenem.csv')
    print(f'  Doxycycline cohort: n={len(cohort_static_df)}')
    print(f'  Meropenem cohort: n={len(cohort_cidal_df)}')

    # Reconstruct virtual patient objects (simplified for trial)
    # For efficiency, we'll run on a subset
    n_trial = 200  # Run 200 patients per arm
    print(f'\n[Step 4.2] Running trial on n={n_trial} patients per arm (for demonstration)...')

    # Import necessary classes
    from src.utils.virtual_cohort import VirtualPatient, ImmuneStatus, InfectionSite
    from src.utils.population_pk import PatientCovariates
    from src.core.parameters import PKParameters

    def reconstruct_patient(row) -> VirtualPatient:
        """Reconstruct VirtualPatient from DataFrame row"""
        demographics = PatientCovariates(
            age=row['age'],
            weight=row['weight'],
            sex=row['sex'],
            creatinine=row['creatinine']
        )

        pk_params = {
            'CL': row['pk_CL'],
            'Vc': row['pk_Vc'],
            'Vp': row['pk_Vp'],
            'Q': row['pk_Q'],
            'Ka': row['pk_Ka'],
            'Kp': row['pk_Kp']
        }

        immune_status = ImmuneStatus(row['immune_status'])
        infection_site = InfectionSite(row['infection_site'])

        pathogen_params = {
            'initial_burden': row['initial_burden'],
            'MIC': row['MIC'],
            'growth_rate_factor': 1.0
        }

        return VirtualPatient(
            patient_id=row['patient_id'],
            demographics=demographics,
            pk_params=pk_params,
            immune_status=immune_status,
            infection_site=infection_site,
            pathogen_params=pathogen_params
        )

    # Reconstruct cohorts
    cohort_static = [reconstruct_patient(row) for _, row in cohort_static_df.head(n_trial).iterrows()]
    cohort_cidal = [reconstruct_patient(row) for _, row in cohort_cidal_df.head(n_trial).iterrows()]

    # Define treatment arms
    print('\n[Step 4.3] Setting up treatment arms...')

    # Doxycycline (bacteriostatic)
    regimen_static = DosingRegimen(
        dose_mg=100,
        interval_hours=12,
        start_time=0,
        n_doses=8,
        infusion_duration_min=30
    )
    trial_static = InSilicoTrial(
        treatment_name='Doxycycline (Static)',
        drug_name='doxycycline',
        drug_class='static',
        regimen=regimen_static,
        simulation_duration=96.0
    )

    # Meropenem (bactericidal)
    regimen_cidal = DosingRegimen(
        dose_mg=1000,
        interval_hours=8,
        start_time=0,
        n_doses=12,
        infusion_duration_min=30
    )
    trial_cidal = InSilicoTrial(
        treatment_name='Meropenem (Cidal)',
        drug_name='meropenem',
        drug_class='cidal',
        regimen=regimen_cidal,
        simulation_duration=96.0
    )

    # Run trials
    print('\n[Step 4.4] Executing in silico trials...')
    results_static = trial_static.run_trial_arm(cohort_static)
    results_cidal = trial_cidal.run_trial_arm(cohort_cidal)

    # Combine results
    all_results = pd.concat([results_static, results_cidal], ignore_index=True)
    all_results.to_csv('in_silico_trial_results.csv', index=False)
    print('\n  Trial results saved: in_silico_trial_results.csv')

    # Analyze outcomes
    print('\n[Step 4.5] Analyzing trial outcomes...')
    print('\n  Overall Results:')
    summary = analyze_trial_outcomes(all_results)
    print(summary)

    print('\n  Stratified by Immune Status:')
    summary_immune = analyze_trial_outcomes(all_results, stratify_by='immune_status')
    print(summary_immune)

    # Save summaries
    summary.to_csv('trial_summary_overall.csv')
    summary_immune.to_csv('trial_summary_by_immune_status.csv')

    # Visualize
    print('\n[Step 4.6] Generating visualizations...')
    visualize_trial_results(all_results, 'in_silico_trial_results.png')
    print('  Visualization saved: in_silico_trial_results.png')

    # Print key findings
    print('\n' + '='*80)
    print('STAGE 4 COMPLETE: IN SILICO TRIALS')
    print('='*80)

    static_success = results_static['clinical_success'].mean()
    cidal_success = results_cidal['clinical_success'].mean()
    static_toxicity = results_static['inflammatory_toxicity'].mean()
    cidal_toxicity = results_cidal['inflammatory_toxicity'].mean()

    print('\nKey Trial Findings:')
    print(f'  Clinical Success Rate:')
    print(f'    Doxycycline (Static): {100*static_success:.1f}%')
    print(f'    Meropenem (Cidal): {100*cidal_success:.1f}%')
    print(f'    Absolute difference: {100*abs(cidal_success - static_success):.1f}%')
    print(f'\n  Inflammatory Toxicity Rate:')
    print(f'    Doxycycline (Static): {100*static_toxicity:.1f}%')
    print(f'    Meropenem (Cidal): {100*cidal_toxicity:.1f}%')
    print(f'    Relative risk: {cidal_toxicity / max(static_toxicity, 0.01):.2f}×')
    print(f'\n  Median Time to 50% Reduction:')
    print(f'    Doxycycline: {results_static["time_to_50_percent_reduction"].median():.1f} hours')
    print(f'    Meropenem: {results_cidal["time_to_50_percent_reduction"].median():.1f} hours')

    print('\n  Context-Dependent Outcomes:')
    for status in all_results['immune_status'].unique():
        subset_static = results_static[results_static['immune_status'] == status]
        subset_cidal = results_cidal[results_cidal['immune_status'] == status]
        if len(subset_static) > 0 and len(subset_cidal) > 0:
            print(f'    {status.capitalize()}:')
            print(f'      Static success: {100*subset_static["clinical_success"].mean():.1f}%')
            print(f'      Cidal success: {100*subset_cidal["clinical_success"].mean():.1f}%')

    print('='*80)


if __name__ == '__main__':
    run_in_silico_trials_stage()
