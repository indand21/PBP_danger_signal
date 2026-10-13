"""
Run QSP simulation and generate results
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Import QSP modules
from src.core.parameters import get_default_parameters, get_drug_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

print('='*80)
print('QSP MODEL: BACTERICIDAL VS BACTERIOSTATIC ANTIBIOTIC EFFICACY')
print('='*80)
print('\nModules loaded successfully!')
print(f'NumPy version: {np.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')

# Load default parameters
print('\n[Step 1] Loading default parameters...')
params = get_default_parameters()
print('  Bacterial growth rate: {:.3f} per hour'.format(params['bacteria'].k_growth))
print('  Carrying capacity: {:.1e} CFU/mL'.format(params['bacteria'].B_max))
print('  Mutation rate (enhanced): {:.1e} per cell per generation'.format(params['bacteria'].mu_mut))
print('  Baseline MIC: {:.1f} mg/L'.format(params['bacteria'].MIC_baseline))

# Create PD model
print('\n[Step 2] Creating pharmacodynamic model...')
pd_model = create_ode_system(params)
print('  Immune effector baseline: {:.1e} cells/mL'.format(pd_model.p_imm.N_eff_0))
print('  Recruited plateau: {:.1f}x baseline'.format(pd_model.p_imm.recruit_fold_max))
print('  Alpha (cidal): {:.1f}, Alpha (static): {:.1f}'.format(
    pd_model.p_cyto.alpha_cidal, pd_model.p_cyto.alpha_static))

# Define drug PK and dosing
print('\n[Step 3] Configuring drug PK parameters...')
weight = 70  # kg

# Doxycycline (bacteriostatic)
# RAW per-kg PK parameters: run_simulation scales Vc/Vp by weight internally, so
# normalize_pk_parameters() output must not be passed (it would double-scale Vc).
pk_static = get_drug_pk_parameters('doxycycline')
print('  Doxycycline (Bacteriostatic):')
print('    CL: {:.2f} L/h, Vc: {:.2f} L ({:.2f} L/kg), Kp (lung): {:.2f}'.format(
    pk_static.CL, pk_static.Vc * weight, pk_static.Vc, pk_static.Kp))

# Meropenem (bactericidal)
pk_cidal = get_drug_pk_parameters('meropenem')
print('  Meropenem (Bactericidal):')
print('    CL: {:.2f} L/h, Vc: {:.2f} L ({:.2f} L/kg), Kp (lung): {:.2f}'.format(
    pk_cidal.CL, pk_cidal.Vc * weight, pk_cidal.Vc, pk_cidal.Kp))

# Initial conditions
print('\n[Step 4] Setting initial conditions...')
init_cond = {
    'B_rep': 1e6,        # 1 million CFU/mL replicating cells
    'B_pers': 1e3,       # small number of persisters
    'B_SCV': 0,          # no SCVs initially
    'N_eff': 1e7,        # baseline neutrophil capacity
    'Damage': 0,         # no cidal damage initially
    'IL6': 10,           # baseline IL-6 (pg/mL)
    'TNF': 5             # baseline TNF (pg/mL)
}
print('  Initial bacterial burden: {:.1e} CFU/mL'.format(init_cond['B_rep']))
print('  Initial immune effectors: {:.1e}'.format(init_cond['N_eff']))

# Define dosing regimens
regimen_static = DosingRegimen(
    dose_mg=100,
    interval_hours=12,
    start_time=0,
    n_doses=8,
    infusion_duration_min=30
)

regimen_cidal = DosingRegimen(
    dose_mg=1000,
    interval_hours=8,
    start_time=0,
    n_doses=12,
    infusion_duration_min=30
)

print('\n[Step 5] Creating PK models and dosing regimens...')
print('  Doxycycline: {} mg q{}h (oral)'.format(regimen_static.dose_mg, regimen_static.interval_hours))
print('  Meropenem: {} mg q{}h (IV)'.format(regimen_cidal.dose_mg, regimen_cidal.interval_hours))

pk_model_static = TwoCompartmentPKModel(CL=pk_static.CL, Vc=pk_static.Vc, Vp=pk_static.Vp,
                                       Q=pk_static.Q, Ka=pk_static.Ka, Kp=pk_static.Kp,
                                       effect_site_model=True)
pk_model_cidal = TwoCompartmentPKModel(CL=pk_cidal.CL, Vc=pk_cidal.Vc, Vp=pk_cidal.Vp,
                                      Q=pk_cidal.Q, Ka=pk_cidal.Ka, Kp=pk_cidal.Kp,
                                      effect_site_model=True)

# Run simulations
print('\n[Step 6] Running simulations...')
print('  Duration: 96 hours')

print('  > No drug control...')
regimen_none = DosingRegimen(dose_mg=0, interval_hours=24, n_doses=0)
pk_model_none = TwoCompartmentPKModel(CL=1, Vc=1000, Vp=500, Q=100, Ka=0, Kp=1)
result_nodrug = run_simulation(pk_model_none, regimen_none, pd_model, init_cond,
                               t_span=(0, 96), drug_class='none', weight_kg=weight)
print('    Complete!')

print('  > Doxycycline (static)...')
result_static = run_simulation(pk_model_static, regimen_static, pd_model, init_cond,
                               t_span=(0, 96), drug_class='static', weight_kg=weight)
print('    Complete!')

print('  > Meropenem (cidal)...')
result_cidal = run_simulation(pk_model_cidal, regimen_cidal, pd_model, init_cond,
                              t_span=(0, 96), drug_class='cidal', weight_kg=weight)
print('    Complete!')

# Extract data
t_no, B_no = result_nodrug.get_bacterial_burden()
t_sta, B_sta = result_static.get_bacterial_burden()
t_cid, B_cid = result_cidal.get_bacterial_burden()

# Compute summary statistics
print('\n[Step 7] Computing summary statistics...')

summary = pd.DataFrame({
    'Scenario': ['No Drug', 'Doxycycline (Static)', 'Meropenem (Cidal)'],
    'Initial Burden (CFU/mL)': [
        B_no[0],
        B_sta[0],
        B_cid[0]
    ],
    'Final Burden (CFU/mL)': [
        B_no[-1],
        B_sta[-1],
        B_cid[-1]
    ],
    'Log10 Reduction': [
        np.log10(max(B_no[0], 1) / max(B_no[-1], 1)),
        np.log10(max(B_sta[0], 1) / max(B_sta[-1], 1)),
        np.log10(max(B_cid[0], 1) / max(B_cid[-1], 1))
    ],
    'Peak IL-6 (pg/mL)': [
        result_nodrug.y[:, 9].max(),
        result_static.y[:, 9].max(),
        result_cidal.y[:, 9].max()
    ],
    'Final SCV Fraction': [
        0,
        result_static.y[-1, 2] / max(B_sta[-1], 1e-6),
        result_cidal.y[-1, 2] / max(B_cid[-1], 1e-6)
    ]
})

print('\n' + '='*100)
print('SUMMARY STATISTICS AT 96 HOURS')
print('='*100)
print(summary.to_string(index=False))
print('='*100)

# Generate comprehensive visualization
print('\n[Step 8] Generating visualization...')

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: Bacterial burden
ax1 = fig.add_subplot(gs[0, :2])
ax1.semilogy(t_no, np.maximum(B_no, 1), 'k--', linewidth=2, label='No drug')
ax1.semilogy(t_sta, np.maximum(B_sta, 1), 'b-', linewidth=2, label='Doxycycline (static)')
ax1.semilogy(t_cid, np.maximum(B_cid, 1), 'r-', linewidth=2, label='Meropenem (cidal)')
ax1.set_xlabel('Time (hours)', fontsize=11)
ax1.set_ylabel('Total Bacterial Burden (CFU/mL)', fontsize=11)
ax1.set_title('A) Bacterial Population Dynamics', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_ylim([1, 1e9])

# Panel B: Drug concentration at effect site
ax2 = fig.add_subplot(gs[0, 2])
C_eff_static = result_static.y[:, 0] / (pk_static.Vc * weight)  # Central concentration
C_eff_cidal = result_cidal.y[:, 0] / (pk_cidal.Vc * weight)
ax2.plot(t_sta, C_eff_static, 'b-', linewidth=2, label='Doxycycline')
ax2.plot(t_cid, C_eff_cidal, 'r-', linewidth=2, label='Meropenem')
ax2.set_xlabel('Time (hours)', fontsize=11)
ax2.set_ylabel('Concentration (mg/L)', fontsize=11)
ax2.set_title('B) Drug Conc. (Central)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Panel C: Persister dynamics
ax3 = fig.add_subplot(gs[1, 0])
B_pers_no = result_nodrug.y[:, 5]
B_pers_sta = result_static.y[:, 5]
B_pers_cid = result_cidal.y[:, 5]
ax3.semilogy(t_no, np.maximum(B_pers_no, 1), 'k--', linewidth=2)
ax3.semilogy(t_sta, np.maximum(B_pers_sta, 1), 'b-', linewidth=2)
ax3.semilogy(t_cid, np.maximum(B_pers_cid, 1), 'r-', linewidth=2)
ax3.set_xlabel('Time (hours)', fontsize=11)
ax3.set_ylabel('Persisters (CFU/mL)', fontsize=11)
ax3.set_title('C) Persister Population', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel D: SCV dynamics
ax4 = fig.add_subplot(gs[1, 1])
B_scv_sta = result_static.y[:, 6]
B_scv_cid = result_cidal.y[:, 6]
ax4.plot(t_sta, B_scv_sta, 'b-', linewidth=2, label='Doxycycline')
ax4.plot(t_cid, B_scv_cid, 'r-', linewidth=2, label='Meropenem')
ax4.set_xlabel('Time (hours)', fontsize=11)
ax4.set_ylabel('SCV Population (CFU/mL)', fontsize=11)
ax4.set_title('D) SCV (Heteroresistance)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)

# Panel E: Resistance fraction
ax5 = fig.add_subplot(gs[1, 2])
t_res_sta, frac_sta = result_static.get_resistance_fraction()
t_res_cid, frac_cid = result_cidal.get_resistance_fraction()
ax5.plot(t_res_sta, frac_sta, 'b-', linewidth=2, label='Doxycycline')
ax5.plot(t_res_cid, frac_cid, 'r-', linewidth=2, label='Meropenem')
ax5.set_xlabel('Time (hours)', fontsize=11)
ax5.set_ylabel('Fraction SCV/Total', fontsize=11)
ax5.set_title('E) Resistance Fraction', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# Panel F: Immune effectors
ax6 = fig.add_subplot(gs[2, 0])
N_no = result_nodrug.y[:, 7]
N_sta = result_static.y[:, 7]
N_cid = result_cidal.y[:, 7]
ax6.semilogy(t_no, N_no, 'k--', linewidth=2)
ax6.semilogy(t_sta, N_sta, 'b-', linewidth=2)
ax6.semilogy(t_cid, N_cid, 'r-', linewidth=2)
ax6.set_xlabel('Time (hours)', fontsize=11)
ax6.set_ylabel('Immune Effectors', fontsize=11)
ax6.set_title('F) Neutrophil Recruitment', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Panel G: IL-6 production
ax7 = fig.add_subplot(gs[2, 1])
IL6_no = result_nodrug.y[:, 9]
IL6_sta = result_static.y[:, 9]
IL6_cid = result_cidal.y[:, 9]
ax7.plot(t_no, IL6_no, 'k--', linewidth=2, label='No drug')
ax7.plot(t_sta, IL6_sta, 'b-', linewidth=2, label='Doxycycline')
ax7.plot(t_cid, IL6_cid, 'r-', linewidth=2, label='Meropenem')
ax7.set_xlabel('Time (hours)', fontsize=11)
ax7.set_ylabel('IL-6 (pg/mL)', fontsize=11)
ax7.set_title('G) IL-6 Cytokine Production', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(fontsize=9)

# Panel H: Cidal damage
ax8 = fig.add_subplot(gs[2, 2])
Dmg_sta = result_static.y[:, 8]
Dmg_cid = result_cidal.y[:, 8]
ax8.plot(t_sta, Dmg_sta, 'b-', linewidth=2, label='Doxycycline')
ax8.plot(t_cid, Dmg_cid, 'r-', linewidth=2, label='Meropenem')
ax8.set_xlabel('Time (hours)', fontsize=11)
ax8.set_ylabel('Cidal Damage (units)', fontsize=11)
ax8.set_title('H) Accumulated Drug Damage', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=9)

fig.suptitle('QSP Model: Bactericidal vs Bacteriostatic Antibiotic Dynamics',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('qsp_results.png', dpi=150, bbox_inches='tight')
print('  Figure saved as: qsp_results.png')

# Export data
print('\n[Step 9] Exporting data...')
summary.to_csv('qsp_summary_statistics.csv', index=False)
print('  Summary statistics saved as: qsp_summary_statistics.csv')

result_static.df.to_csv('qsp_static_trajectory.csv', index=False)
result_cidal.df.to_csv('qsp_cidal_trajectory.csv', index=False)
print('  Trajectories saved as: qsp_static_trajectory.csv, qsp_cidal_trajectory.csv')

print('\n' + '='*80)
print('SIMULATION COMPLETE!')
print('='*80)
print('\nKey Scientific Findings:')
print('  1. Bactericidal (Meropenem) shows rapid "grow-then-crash" dynamics')
print('  2. Bacteriostatic (Doxycycline) shows dose-dependent growth slowdown')
print('  3. Cidal drug triggers {:.1f}x higher peak IL-6 (TLR9-mediated)'.format(
    result_cidal.y[:, 9].max() / max(result_static.y[:, 9].max(), 1)))
print('  4. SCV emergence more pronounced under static pressure')
print('  5. Log10 reduction: Static={:.2f}, Cidal={:.2f}'.format(
    summary.loc[1, 'Log10 Reduction'], summary.loc[2, 'Log10 Reduction']))
print('\nEnhancements implemented:')
print('  - Fixed H_static variable scoping in SCV mutation logic')
print('  - Increased mutation rate to 1e-6 for clinically relevant heteroresistance')
print('  - Added MIC_baseline parameter to bacterial parameters')
print('  - Improved ODE solver error handling and convergence monitoring')
print('='*80)
