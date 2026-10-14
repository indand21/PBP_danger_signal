"""Joint PD/yield stress tests with a fixed exposure denominator.

No parameter is adopted and no range has a confidence level. The reference
1 mg/L is measured for the meropenem broth isolate and assumed for CAZ in this
synthetic comparison; it is not a paired susceptibility measurement. Holding
it fixed prevents an unidentified C50 from redefining the exposure index.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ds_multiarm_downstream import load_phase4, factory
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf
from src.analysis.yield_observations import observations, comparable_ratio
from src.utils.provenance import provenance


def tipping_ratio(caz_lysis, caz_filament, mero_lysis, y_caz, y_filament):
    """CAZ:MERO latent lysis-yield ratio where their DS AUCs are equal."""
    denominator = y_caz*caz_lysis + y_filament*caz_filament
    if denominator <= 0 or mero_lysis <= 0:
        raise ValueError("Positive release coefficients are required")
    return y_caz*mero_lysis/denominator


def main():
    fit_path = ROOT/'results/danger_signal/multiarm_refit.json'
    joint_path = ROOT/'results/danger_signal/joint_broth_refit.json'
    fits = json.loads(fit_path.read_text(encoding='utf-8'))
    joint = json.loads(joint_path.read_text(encoding='utf-8'))
    p4 = load_phase4()
    p4.MIC_EFFECT = 1.
    family = [x for x in calibrate_family() if x['feasible']]
    p4.P_REF = calibrate_tnf(min(family, key=lambda x:x['cost']))['params']
    nominal = p4.get_default_parameters()['bacteria']
    variants = {'adopted': {k:getattr(nominal,k) for k in
                             ['k_kill_max','kill_C50','kill_hill']}}
    variants.update({k:v['parameters'] for k,v in fits['scenarios'].items()})
    variants['joint_ABC_adequacy_failure'] = joint['best']
    obs = observations()
    direct = comparable_ratio(obs['trautmann_ceftazidime'],obs['trautmann_meropenem'])
    ratios = [.25,.5,1.,2.,3.5,direct,5.,7.,10.,40.]
    doses = {target:{a:p4.dose_for_target_ft(a,target,8.,6,mic=1.)
                     for a in ['ceftazidime','meropenem']}
             for target in [40.,60.,80.,95.]}
    rows = []
    for label, parameters in variants.items():
        params = factory(parameters)()
        for target, dose in doses.items():
            dose = dict(dose)
            sims = {a:p4.simulate(a,d,8.,6,params=params,mic=1.) for a,d in dose.items()}
            # Adaptive bacterial integration changes the time grid used for
            # crossing interpolation. Correct that numerical residual rather
            # than changing the exposure denominator with the PD parameters.
            for a,sim in sims.items():
                if abs(sim['fT']-target) >= .15:
                    d = dose[a]
                    dose[a] = float(brentq(lambda z:p4.simulate(a,z,8.,6,params=params,mic=1.)['fT']-target,
                                           .85*d,1.15*d,xtol=.1))
                    sims[a] = p4.simulate(a,dose[a],8.,6,params=params,mic=1.)
            assert all(abs(s['fT']-target)<.15 for s in sims.values())
            def coefficient(agent, y, yf):
                return p4.ds_cost(sims[agent],Y=y,Yf=yf)[1]['AUC_DS_pg_h_mL']
            ca = coefficient('ceftazidime',1.,0.)
            cf = coefficient('ceftazidime',0.,1.)
            me = coefficient('meropenem',1.,0.)
            yc, yf = p4.AGENT_YIELD['ceftazidime'], p4.Y_FILAMENT_CAZ
            stress = []
            for filament_factor in [0.,.5,1.,2.]:
                critical = tipping_ratio(ca,cf,me,yc,yf*filament_factor)
                stress.append({'filament_factor':filament_factor,
                               'critical_lysis_yield_ratio':critical,
                               'cost_ratios':[{'lysis_yield_ratio':r,
                                               'AUC_DS_ratio':r/critical} for r in ratios]})
            rows.append({'pd_scenario':label,'target_fT_pct':target,
                         'pd_parameters':{k:parameters[k] for k in
                                          ['k_kill_max','kill_C50','kill_hill']},
                         'dose_mg_q8h':dose,
                         'achieved_fT_pct':{a:s['fT'] for a,s in sims.items()},
                         'net_log10_change':{a:s['net_log10'] for a,s in sims.items()},
                         'kill_difference_log10':sims['ceftazidime']['net_log10']-sims['meropenem']['net_log10'],
                         'unit_release_AUC_coefficients':{'caz_lysis':ca,'caz_filament':cf,'mero_lysis':me},
                         'yield_stress_tests':stress})
        print(label,'complete',flush=True)
    critical = [s['critical_lysis_yield_ratio'] for r in rows for s in r['yield_stress_tests']]
    equal_null = [s['cost_ratios'][2]['AUC_DS_ratio'] for r in rows
                  for s in r['yield_stress_tests'] if s['filament_factor']==0.]
    payload = {'status':'conditional joint robustness; no defaults or Table 1 replaced',
               'mic_reference_mgL':1.,
               'mic_reference_basis':'Measured for meropenem ATCC-27853; same value assumed for CAZ. '
                                     'Fixed across all PD variants; distinct from model-predicted stasis.',
               'assumed_yield_interval':[3.5,7.],
               'direct_matched_ELISA_observation_ratio':direct,
               'direct_ratio_scope':'CAZ:MERO free LPS at 50xMIC,8h,E.coli Bort; not a latent per-CFU yield',
               'yield_stress_grid':ratios,'filament_factors':[0.,.5,1.,2.],
               'uncertainty':'Discrete structural and reading scenarios, not confidence limits. '
                             'Broth growth nuisance parameters are not transferred in vivo. '
                             'DS AUC is linear in release coefficients; cytokine AUC need not be.',
               'summary':{'n_pd_scenarios':len(variants),'n_matched_rows':len(rows),
                          'critical_lysis_yield_ratio_range':[min(critical),max(critical)],
                          'equal_yield_no_filament_cost_ratio_range':[min(equal_null),max(equal_null)],
                          'max_abs_kill_difference_log10':max(abs(r['kill_difference_log10']) for r in rows)},
               'rows':rows,
               'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [fit_path,joint_path]},
               'provenance':provenance(note='Fixed-MIC joint PD and latent-yield stress tests')}
    out = ROOT/'results/danger_signal/joint_robustness.json'
    out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(payload['summary'],indent=2),flush=True)


if __name__=='__main__':main()
