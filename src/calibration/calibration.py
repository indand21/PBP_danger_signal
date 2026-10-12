"""
Parameter calibration module for QSP model
Fits model parameters to time-kill curves and experimental data
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Tuple, List

from src.core.parameters import get_default_parameters, BacterialParameters, ImmuneParameters
from src.core.pd_model import BacterialPopulationODE


class SyntheticTimeKillData:
    """
    Generate synthetic time-kill curve data mimicking published patterns
    Based on typical E. coli responses to sub-MIC and supra-MIC exposures
    """

    @staticmethod
    def generate_static_timekill(MIC: float = 1.0, noise_level: float = 0.15) -> pd.DataFrame:
        """
        Generate bacteriostatic (doxycycline-like) time-kill curves
        Dose-dependent growth slowdown pattern
        """
        time_points = np.array([0, 2, 4, 6, 8, 12, 24])
        B0 = 1e6  # CFU/mL initial

        # Multiple concentration levels relative to MIC
        concentrations = [0, 0.25*MIC, 0.5*MIC, 1*MIC, 2*MIC, 4*MIC]

        data = []
        for C in concentrations:
            # Static mechanism: dose-dependent growth rate reduction
            if C == 0:
                k_net = 0.5  # hr^-1, natural growth
            else:
                # Hill function: growth inhibition
                k_net = 0.5 * (1 - C**1.2 / (MIC**1.2 + C**1.2))

            for t in time_points:
                # Logistic-like growth with saturation
                B_theory = B0 * np.exp(k_net * t) / (1 + (B0/1e8) * (np.exp(k_net*t) - 1))
                # Add log-normal noise
                B_obs = B_theory * np.exp(np.random.normal(0, noise_level))
                data.append({
                    'time': t,
                    'concentration': C,
                    'concentration_xMIC': C / MIC,
                    'CFU': max(B_obs, 1)
                })

        return pd.DataFrame(data)

    @staticmethod
    def generate_cidal_timekill(MIC: float = 1.0, noise_level: float = 0.15) -> pd.DataFrame:
        """
        Generate bactericidal (meropenem-like) time-kill curves
        Grow-then-crash pattern at sub-MIC, rapid kill at supra-MIC
        """
        time_points = np.array([0, 1, 2, 3, 4, 6, 8, 12, 24])
        B0 = 1e6

        concentrations = [0, 0.25*MIC, 0.5*MIC, 1*MIC, 2*MIC, 4*MIC, 8*MIC]

        data = []
        for C in concentrations:
            for t in time_points:
                if C == 0:
                    # Natural growth
                    B_theory = B0 * np.exp(0.5 * t) / (1 + (B0/1e8) * (np.exp(0.5*t) - 1))
                elif C < MIC:
                    # Sub-MIC: grow-then-crash (damage accumulation)
                    t_crash = 4.0 / (C/MIC)  # crash time inversely proportional to concentration
                    if t < t_crash:
                        B_theory = B0 * np.exp(0.4 * t)  # near-normal growth
                    else:
                        # Exponential decline after crash
                        B_peak = B0 * np.exp(0.4 * t_crash)
                        B_theory = B_peak * np.exp(-0.5 * (t - t_crash))
                else:
                    # Supra-MIC: rapid bactericidal killing
                    k_kill = 0.3 * (C / MIC)  # kill rate proportional to concentration
                    B_theory = B0 * np.exp(-k_kill * t)

                B_obs = B_theory * np.exp(np.random.normal(0, noise_level))
                data.append({
                    'time': t,
                    'concentration': C,
                    'concentration_xMIC': C / MIC,
                    'CFU': max(B_obs, 1)
                })

        return pd.DataFrame(data)

    @staticmethod
    def generate_cytokine_data(drug_class: str = 'cidal', noise_level: float = 0.2) -> pd.DataFrame:
        """
        Generate synthetic cytokine production data
        Cidal drugs -> higher IL-6/TNF (TLR9-mediated)
        """
        time_points = np.array([0, 6, 12, 24, 48, 72])

        data = []
        for t in time_points:
            if drug_class == 'cidal':
                # Higher cytokine production with cidal drugs
                IL6_theory = 10 + 180 * (1 - np.exp(-0.1 * t))  # pg/mL
                TNF_theory = 5 + 60 * (1 - np.exp(-0.12 * t))
            else:  # static
                # Lower cytokine production
                IL6_theory = 10 + 50 * (1 - np.exp(-0.08 * t))
                TNF_theory = 5 + 15 * (1 - np.exp(-0.1 * t))

            IL6_obs = IL6_theory * np.exp(np.random.normal(0, noise_level))
            TNF_obs = TNF_theory * np.exp(np.random.normal(0, noise_level))

            data.append({
                'time': t,
                'IL6': IL6_obs,
                'TNF': TNF_obs,
                'drug_class': drug_class
            })

        return pd.DataFrame(data)


class ParameterCalibrator:
    """
    Bayesian-inspired parameter estimation for QSP model
    Uses differential evolution for global optimization
    """

    def __init__(self, default_params: Dict):
        self.default_params = default_params
        self.fitted_params = None
        self.fit_history = []

    def objective_function_static(self, param_vector: np.ndarray,
                                  data: pd.DataFrame) -> float:
        """
        Objective function for bacteriostatic drug parameters
        Minimizes weighted sum of squared log-errors
        """
        k_growth, EC50_static, hill_static = param_vector

        # Update parameters
        params = self.default_params.copy()
        params['bacteria'].k_growth = k_growth

        # Create simplified model for time-kill simulation
        pd_model = BacterialPopulationODE(params)

        total_error = 0.0
        n_points = 0

        # Group by concentration
        for conc in data['concentration'].unique():
            subset = data[data['concentration'] == conc]

            # Run simulation
            try:
                y0 = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])  # Initial state
                t_span = (0, subset['time'].max())

                def rhs(t, y):
                    return pd_model.rhs(t, y, C_effect=conc, drug_class='static', is_static=True)

                sol = solve_ivp(rhs, t_span, y0, method='RK45',
                               t_eval=subset['time'].values, max_step=0.5)

                if sol.success:
                    B_sim = sol.y[0] + sol.y[1] + sol.y[2]  # Total bacteria
                    B_obs = subset['CFU'].values

                    # Log-scale squared error
                    error = np.sum((np.log10(B_sim + 1) - np.log10(B_obs + 1))**2)
                    total_error += error
                    n_points += len(B_obs)
                else:
                    total_error += 1e6  # Penalty for failed simulation

            except Exception as e:
                total_error += 1e6

        # Normalize by number of points
        if n_points > 0:
            total_error /= n_points

        return total_error

    def objective_function_cidal(self, param_vector: np.ndarray,
                                 data: pd.DataFrame) -> float:
        """
        Objective function for bactericidal drug parameters
        Focus on damage accumulation dynamics
        """
        k_growth, k_dmg, k_repair, Damage50 = param_vector

        params = self.default_params.copy()
        params['bacteria'].k_growth = k_growth
        params['bacteria'].k_repair = k_repair

        pd_model = BacterialPopulationODE(params)

        total_error = 0.0
        n_points = 0

        for conc in data['concentration'].unique():
            subset = data[data['concentration'] == conc]

            try:
                y0 = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
                t_span = (0, subset['time'].max())

                def rhs(t, y):
                    return pd_model.rhs(t, y, C_effect=conc, drug_class='cidal', is_static=False)

                sol = solve_ivp(rhs, t_span, y0, method='RK45',
                               t_eval=subset['time'].values, max_step=0.5)

                if sol.success:
                    B_sim = sol.y[0] + sol.y[1] + sol.y[2]
                    B_obs = subset['CFU'].values

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

    def calibrate_static_params(self, data: pd.DataFrame,
                               method: str = 'differential_evolution') -> Dict:
        """
        Calibrate bacteriostatic drug parameters
        """
        print('\n  Calibrating bacteriostatic parameters...')

        # Parameter bounds: [k_growth, EC50_static, hill_static]
        bounds = [
            (0.2, 0.8),   # k_growth (per hour)
            (0.5, 2.0),   # EC50_static (mg/L)
            (0.8, 2.0)    # Hill coefficient
        ]

        result = differential_evolution(
            lambda p: self.objective_function_static(p, data),
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42,
            workers=1
        )

        fitted = {
            'k_growth': result.x[0],
            'EC50_static': result.x[1],
            'hill_static': result.x[2],
            'objective_value': result.fun
        }

        print(f'    k_growth: {fitted["k_growth"]:.4f} per hour')
        print(f'    EC50_static: {fitted["EC50_static"]:.4f} mg/L')
        print(f'    Hill coefficient: {fitted["hill_static"]:.4f}')
        print(f'    Final objective: {fitted["objective_value"]:.4f}')

        return fitted

    def calibrate_cidal_params(self, data: pd.DataFrame) -> Dict:
        """
        Calibrate bactericidal drug parameters
        """
        print('\n  Calibrating bactericidal parameters...')

        # Parameter bounds: [k_growth, k_dmg, k_repair, Damage50]
        bounds = [
            (0.2, 0.8),    # k_growth
            (0.1, 0.5),    # k_dmg (damage accumulation rate)
            (0.05, 0.2),   # k_repair
            (5.0, 20.0)    # Damage50 (threshold)
        ]

        result = differential_evolution(
            lambda p: self.objective_function_cidal(p, data),
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            seed=42,
            workers=1
        )

        fitted = {
            'k_growth': result.x[0],
            'k_dmg': result.x[1],
            'k_repair': result.x[2],
            'Damage50': result.x[3],
            'objective_value': result.fun
        }

        print(f'    k_growth: {fitted["k_growth"]:.4f} per hour')
        print(f'    k_dmg: {fitted["k_dmg"]:.4f}')
        print(f'    k_repair: {fitted["k_repair"]:.4f} per hour')
        print(f'    Damage50: {fitted["Damage50"]:.2f}')
        print(f'    Final objective: {fitted["objective_value"]:.4f}')

        return fitted

    def validate_fit(self, fitted_params: Dict, data: pd.DataFrame,
                    drug_class: str, output_file: str = None):
        """
        Validate fitted parameters by comparing model predictions to data
        """
        print(f'\n  Validating {drug_class} fit...')

        # Update parameters with fitted values
        params = self.default_params.copy()
        params['bacteria'].k_growth = fitted_params['k_growth']
        if drug_class == 'cidal':
            params['bacteria'].k_repair = fitted_params['k_repair']

        pd_model = BacterialPopulationODE(params)

        # Create validation plot
        concentrations = data['concentration'].unique()
        n_conc = len(concentrations)

        fig, axes = plt.subplots(2, (n_conc + 1) // 2, figsize=(14, 8))
        axes = axes.flatten()

        for idx, conc in enumerate(concentrations):
            subset = data[data['concentration'] == conc]

            # Run model
            y0 = np.array([1e6, 1e3, 0, 1e7, 0, 10, 5])
            t_dense = np.linspace(0, subset['time'].max(), 100)

            def rhs(t, y):
                return pd_model.rhs(t, y, C_effect=conc,
                                   drug_class=drug_class,
                                   is_static=(drug_class=='static'))

            sol = solve_ivp(rhs, (0, subset['time'].max()), y0,
                           method='RK45', t_eval=t_dense, max_step=0.5)

            if sol.success:
                B_pred = sol.y[0] + sol.y[1] + sol.y[2]

                # Plot
                ax = axes[idx]
                ax.semilogy(subset['time'], subset['CFU'], 'o',
                           label='Data', markersize=6, alpha=0.7)
                ax.semilogy(t_dense, B_pred, '-', linewidth=2, label='Model')
                ax.set_xlabel('Time (hours)', fontsize=9)
                ax.set_ylabel('CFU/mL', fontsize=9)
                ax.set_title(f'{conc:.2f} mg/L ({conc/1.0:.2f}×MIC)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                ax.set_ylim([1e2, 1e9])

        plt.suptitle(f'Calibration Validation: {drug_class.capitalize()} Drug',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f'    Validation plot saved: {output_file}')

        return fig


def run_full_calibration():
    """
    Execute complete parameter calibration workflow
    """
    print('='*80)
    print('STAGE 1: PARAMETER CALIBRATION')
    print('='*80)

    # Generate synthetic data
    print('\n[Step 1.1] Generating synthetic time-kill curve data...')
    data_generator = SyntheticTimeKillData()

    print('  > Bacteriostatic (doxycycline-like) time-kill curves')
    data_static = data_generator.generate_static_timekill(MIC=1.0, noise_level=0.1)
    print(f'    Generated {len(data_static)} data points')

    print('  > Bactericidal (meropenem-like) time-kill curves')
    data_cidal = data_generator.generate_cidal_timekill(MIC=1.0, noise_level=0.1)
    print(f'    Generated {len(data_cidal)} data points')

    print('  > Cytokine production data')
    data_cyto_static = data_generator.generate_cytokine_data(drug_class='static')
    data_cyto_cidal = data_generator.generate_cytokine_data(drug_class='cidal')
    print(f'    Generated {len(data_cyto_static) + len(data_cyto_cidal)} data points')

    # Save synthetic data
    data_static.to_csv('synthetic_data_static_timekill.csv', index=False)
    data_cidal.to_csv('synthetic_data_cidal_timekill.csv', index=False)
    print('  Synthetic data saved!')

    # Initialize calibrator
    print('\n[Step 1.2] Initializing parameter calibrator...')
    default_params = get_default_parameters()
    calibrator = ParameterCalibrator(default_params)

    # Calibrate bacteriostatic parameters
    print('\n[Step 1.3] Calibrating BACTERIOSTATIC drug parameters...')
    fitted_static = calibrator.calibrate_static_params(data_static)

    # Calibrate bactericidal parameters
    print('\n[Step 1.4] Calibrating BACTERICIDAL drug parameters...')
    fitted_cidal = calibrator.calibrate_cidal_params(data_cidal)

    # Validate fits
    print('\n[Step 1.5] Validating calibration...')
    calibrator.validate_fit(fitted_static, data_static, 'static',
                           'calibration_validation_static.png')
    calibrator.validate_fit(fitted_cidal, data_cidal, 'cidal',
                           'calibration_validation_cidal.png')

    # Save calibrated parameters
    print('\n[Step 1.6] Saving calibrated parameters...')
    calibrated_params = {
        'static': fitted_static,
        'cidal': fitted_cidal,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    pd.DataFrame([calibrated_params]).to_json('calibrated_parameters.json',
                                              orient='records', indent=2)
    print('  Calibrated parameters saved to: calibrated_parameters.json')

    # Summary
    print('\n' + '='*80)
    print('STAGE 1 COMPLETE: PARAMETER CALIBRATION')
    print('='*80)
    print('\nCalibrated Parameters Summary:')
    print('\n  Bacteriostatic (Static):')
    for key, val in fitted_static.items():
        print(f'    {key}: {val:.4f}')
    print('\n  Bactericidal (Cidal):')
    for key, val in fitted_cidal.items():
        print(f'    {key}: {val:.4f}')
    print('='*80)

    return calibrated_params


if __name__ == '__main__':
    calibrated_params = run_full_calibration()
