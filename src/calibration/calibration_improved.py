"""
IMPROVED parameter calibration with better cidal mechanism
Fixes issues with damage accumulation and killing rate
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, List

from src.core.parameters import get_default_parameters, BacterialParameters
from src.calibration.calibration import SyntheticTimeKillData


class ImprovedBacterialODE:
    """
    Improved PD model with better cidal mechanism
    """

    def __init__(self, params: Dict):
        self.p_bact = params['bacteria']
        self.p_imm = params['immune']

    def rhs(self, t: float, y: np.ndarray, C_effect: float,
            k_dmg_rate: float, k_kill_cidal: float, Damage50: float,
            drug_class: str = 'cidal') -> np.ndarray:
        """
        Improved ODE with parameterized cidal mechanism

        Key improvements:
        1. k_dmg_rate is a fitted parameter (not fixed)
        2. k_kill_cidal is concentration-dependent and fitted
        3. Damage accumulation is faster and more responsive
        """
        B_rep = max(y[0], 1e-6)
        B_pers = max(y[1], 0)
        B_SCV = max(y[2], 0)
        N_eff = max(y[3], 0)
        Damage = max(y[4], 0)

        B_total = B_rep + B_pers + B_SCV

        dydt = np.zeros(5)

        # Logistic growth
        growth_term = self.p_bact.k_growth * (1.0 - B_total / self.p_bact.B_max) * B_rep

        # Immune killing (reduced for calibration simplicity)
        immune_kill = 0.01 * self.p_imm.k_kill_base * N_eff * B_rep

        # IMPROVED CIDAL MECHANISM
        if drug_class == 'cidal':
            # Damage accumulation: proportional to concentration
            # Use fitted k_dmg_rate parameter
            k_dmg = k_dmg_rate * C_effect
            k_repair = self.p_bact.k_repair
            damage_accumulation = k_dmg - k_repair * Damage

            # Damage-dependent killing with Hill function
            n_hill = 2.0
            f_damage = (Damage**n_hill) / (Damage50**n_hill + Damage**n_hill)

            # Concentration-dependent cidal kill rate (FIXED: now uses fitted parameter)
            # At high concentrations, killing should be fast
            cidal_kill = k_kill_cidal * (1 + C_effect) * f_damage * B_rep

            dydt[4] = damage_accumulation  # Damage
        else:
            cidal_kill = 0.0
            dydt[4] = 0.0

        # Replicating cells
        to_pers = self.p_bact.k_pers * B_rep
        dydt[0] = growth_term - immune_kill - cidal_kill - to_pers

        # Persisters (simplified)
        dydt[1] = to_pers - 0.01 * B_pers

        # SCVs (not relevant for calibration)
        dydt[2] = 0.0

        # Immune effectors (simplified recruitment)
        recruit = self.p_imm.k_prod * (B_total / (self.p_imm.EC50_immune + B_total))
        degrade = self.p_imm.k_deg_immune * N_eff
        dydt[3] = recruit - degrade

        return dydt


class ImprovedCalibratorCidal:
    """
    Improved calibrator for cidal drugs
    """

    def __init__(self, default_params: Dict):
        self.default_params = default_params

    def objective_function(self, param_vector: np.ndarray,
                          data: pd.DataFrame) -> float:
        """
        Objective function with improved cidal parameters

        Parameters to fit:
        - k_growth: bacterial growth rate
        - k_dmg_rate: damage accumulation rate coefficient
        - k_kill_cidal: cidal killing rate coefficient
        - Damage50: damage threshold for 50% killing
        """
        k_growth, k_dmg_rate, k_kill_cidal, Damage50 = param_vector

        params = self.default_params.copy()
        params['bacteria'].k_growth = k_growth

        model = ImprovedBacterialODE(params)

        total_error = 0.0
        n_points = 0

        for conc in data['concentration'].unique():
            subset = data[data['concentration'] == conc]

            try:
                y0 = np.array([1e6, 1e3, 0, 1e7, 0])
                t_span = (0, subset['time'].max())

                def rhs_wrapper(t, y):
                    return model.rhs(t, y, C_effect=conc,
                                   k_dmg_rate=k_dmg_rate,
                                   k_kill_cidal=k_kill_cidal,
                                   Damage50=Damage50,
                                   drug_class='cidal')

                sol = solve_ivp(rhs_wrapper, t_span, y0, method='RK45',
                               t_eval=subset['time'].values, max_step=0.1,
                               rtol=1e-6, atol=1e-8)

                if sol.success:
                    B_sim = sol.y[0] + sol.y[1]
                    B_obs = subset['CFU'].values

                    # Log-scale squared error
                    error = np.sum((np.log10(B_sim + 1) - np.log10(B_obs + 1))**2)
                    total_error += error
                    n_points += len(B_obs)
                else:
                    total_error += 1e6

            except Exception as e:
                total_error += 1e6

        if n_points > 0:
            total_error /= n_points

        return total_error

    def calibrate(self, data: pd.DataFrame) -> Dict:
        """
        Calibrate cidal parameters with improved bounds
        """
        print('\n  Calibrating IMPROVED bactericidal parameters...')

        # IMPROVED BOUNDS - allow much stronger killing
        bounds = [
            (0.3, 0.8),      # k_growth (higher range)
            (0.5, 5.0),      # k_dmg_rate (much higher - faster damage)
            (0.5, 3.0),      # k_kill_cidal (higher - stronger killing)
            (2.0, 15.0)      # Damage50 (broader range)
        ]

        print('    Parameter bounds:')
        print(f'      k_growth: {bounds[0]}')
        print(f'      k_dmg_rate: {bounds[1]} (INCREASED from 0.1-0.5)')
        print(f'      k_kill_cidal: {bounds[2]} (INCREASED from 0.5)')
        print(f'      Damage50: {bounds[3]}')

        result = differential_evolution(
            lambda p: self.objective_function(p, data),
            bounds,
            maxiter=150,     # More iterations
            popsize=20,      # Larger population
            tol=0.001,
            seed=42,
            workers=1,
            polish=True      # Final refinement
        )

        fitted = {
            'k_growth': result.x[0],
            'k_dmg_rate': result.x[1],
            'k_kill_cidal': result.x[2],
            'Damage50': result.x[3],
            'objective_value': result.fun
        }

        print(f'\n    FITTED PARAMETERS:')
        print(f'      k_growth: {fitted["k_growth"]:.4f} per hour')
        print(f'      k_dmg_rate: {fitted["k_dmg_rate"]:.4f} (damage accumulation coefficient)')
        print(f'      k_kill_cidal: {fitted["k_kill_cidal"]:.4f} (killing rate coefficient)')
        print(f'      Damage50: {fitted["Damage50"]:.2f}')
        print(f'      Final objective: {fitted["objective_value"]:.4f}')
        print(f'      Improvement: {(5.4074 - fitted["objective_value"])/5.4074 * 100:.1f}% better')

        return fitted

    def validate_fit(self, fitted_params: Dict, data: pd.DataFrame,
                    output_file: str = None):
        """
        Validate improved fit
        """
        print(f'\n  Validating improved cidal fit...')

        params = self.default_params.copy()
        params['bacteria'].k_growth = fitted_params['k_growth']

        model = ImprovedBacterialODE(params)

        concentrations = sorted(data['concentration'].unique())
        n_conc = len(concentrations)

        fig, axes = plt.subplots(2, (n_conc + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        for idx, conc in enumerate(concentrations):
            subset = data[data['concentration'] == conc]

            y0 = np.array([1e6, 1e3, 0, 1e7, 0])
            t_dense = np.linspace(0, subset['time'].max(), 200)

            def rhs_wrapper(t, y):
                return model.rhs(t, y, C_effect=conc,
                               k_dmg_rate=fitted_params['k_dmg_rate'],
                               k_kill_cidal=fitted_params['k_kill_cidal'],
                               Damage50=fitted_params['Damage50'],
                               drug_class='cidal')

            sol = solve_ivp(rhs_wrapper, (0, subset['time'].max()), y0,
                           method='RK45', t_eval=t_dense, max_step=0.1,
                           rtol=1e-6, atol=1e-8)

            if sol.success:
                B_pred = sol.y[0] + sol.y[1]

                ax = axes[idx]
                ax.semilogy(subset['time'], subset['CFU'], 'o',
                           label='Data', markersize=8, alpha=0.7, color='steelblue')
                ax.semilogy(t_dense, B_pred, '-', linewidth=2.5,
                           label='Improved Model', color='darkred')
                ax.set_xlabel('Time (hours)', fontsize=10)
                ax.set_ylabel('CFU/mL', fontsize=10)
                ax.set_title(f'{conc:.2f} mg/L ({conc/1.0:.2f}×MIC)',
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9)
                ax.set_ylim([1e2, 1e9])

                # Add annotation showing fit quality
                rmse = np.sqrt(np.mean((np.log10(B_pred[::len(B_pred)//len(subset['CFU'])][:len(subset['CFU'])]) -
                                       np.log10(subset['CFU']))**2))
                ax.text(0.05, 0.05, f'RMSE: {rmse:.2f}',
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('IMPROVED Calibration: Bactericidal Drug\n(Fixed Damage Accumulation & Killing Rate)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f'    Validation plot saved: {output_file}')

        return fig


def run_improved_calibration():
    """
    Run improved cidal calibration
    """
    print('='*80)
    print('IMPROVED CIDAL CALIBRATION')
    print('='*80)
    print('\nObjective: Fix poor cidal fit (objective 5.41 -> <1.0)')
    print('Changes:')
    print('  1. Parameterized k_dmg_rate (not hardcoded)')
    print('  2. Parameterized k_kill_cidal (concentration-dependent)')
    print('  3. Expanded parameter bounds (allow stronger killing)')
    print('  4. More optimization iterations')

    # Load existing cidal data
    print('\n[Step 1] Loading cidal time-kill data...')
    data_cidal = pd.read_csv('synthetic_data_cidal_timekill.csv')
    print(f'  Loaded {len(data_cidal)} data points')
    print(f'  Concentrations: {sorted(data_cidal["concentration"].unique())}')

    # Initialize improved calibrator
    print('\n[Step 2] Initializing improved calibrator...')
    default_params = get_default_parameters()
    calibrator = ImprovedCalibratorCidal(default_params)

    # Calibrate
    print('\n[Step 3] Running optimization...')
    fitted = calibrator.calibrate(data_cidal)

    # Validate
    print('\n[Step 4] Validating improved fit...')
    calibrator.validate_fit(fitted, data_cidal,
                           'calibration_validation_cidal_IMPROVED.png')

    # Compare with original
    print('\n[Step 5] Comparison with original calibration:')
    print('  ' + '-'*76)
    print(f'  {"Parameter":<25} {"Original":<20} {"Improved":<20} {"Change"}')
    print('  ' + '-'*76)
    print(f'  {"Objective (RMSE)":<25} {5.4074:<20.4f} {fitted["objective_value"]:<20.4f} '
          f'{(5.4074-fitted["objective_value"])/5.4074*100:>+6.1f}%')
    print(f'  {"k_growth":<25} {0.2000:<20.4f} {fitted["k_growth"]:<20.4f}')
    print(f'  {"k_dmg (coefficient)":<25} {"0.3 (fixed)":<20} {fitted["k_dmg_rate"]:<20.4f} '
          f'{fitted["k_dmg_rate"]/0.3:>6.1f}× stronger')
    print(f'  {"k_kill_cidal":<25} {"0.5 (fixed)":<20} {fitted["k_kill_cidal"]:<20.4f} '
          f'{fitted["k_kill_cidal"]/0.5:>6.1f}× stronger')
    print(f'  {"Damage50":<25} {11.67:<20.2f} {fitted["Damage50"]:<20.2f}')
    print('  ' + '-'*76)

    # Save improved parameters
    print('\n[Step 6] Saving improved parameters...')
    pd.DataFrame([fitted]).to_json('calibrated_parameters_cidal_IMPROVED.json',
                                   orient='records', indent=2)
    print('  Saved: calibrated_parameters_cidal_IMPROVED.json')

    # Recommendations
    print('\n' + '='*80)
    print('CALIBRATION IMPROVEMENT COMPLETE')
    print('='*80)
    print('\nNext Steps:')
    print('  1. Update pd_model.py to use fitted k_dmg_rate and k_kill_cidal')
    print('  2. Re-run in silico trials with improved parameters')
    print('  3. Verify grow-then-crash pattern at 1×MIC')
    print('  4. Check that supra-MIC killing is now rapid')
    print('='*80)

    return fitted


if __name__ == '__main__':
    fitted_params = run_improved_calibration()
