"""Figures from stored PD-context and joint-robustness results."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures/pd_context'
plt.rcParams.update({'font.size':9,'savefig.dpi':200,'savefig.bbox':'tight'})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    joint=json.loads((ROOT/'results/danger_signal/joint_broth_refit.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout='constrained')
    for panel,ax in zip('ABC',axes.ravel()[:3]):
        pts=[r for r in joint['predictions'] if r['panel']==panel]
        for c in sorted(set(r['C_mgL'] for r in pts)):
            sub=sorted([r for r in pts if r['C_mgL']==c],key=lambda r:r['time_h'])
            line=ax.plot([r['time_h'] for r in sub],[r['predicted_log10'] for r in sub],
                         '--',lw=1,label=f'{c:g} mg/L')[0]
            ax.scatter([r['time_h'] for r in sub],[r['observed_log10'] for r in sub],
                       s=22,color=line.get_color())
        ax.set(title=f'Figure 2{panel}: observed points and joint fit',
               xlabel='Time (h)',ylabel='log10 CFU/mL')
        ax.legend(fontsize=6,ncol=2,frameon=False)
    ax=axes[1,1]
    ax.semilogx([r['hill'] for r in joint['hill_profiles']],
                [r['rmse_log10'] for r in joint['hill_profiles']],'o-',color='#355d76')
    ax.axhline(.3,ls=':',color='#777777',label='Reading allowance (not SD)')
    ax.set(xlabel='Fixed bacterial Hill exponent h',ylabel='Reoptimized RMSE (log10)',
           title='Profile of fit error; no confidence interval')
    ax.legend(fontsize=7)
    fig.savefig(OUT/'joint_broth_adequacy.png');plt.close(fig)

    context=json.loads((ROOT/'results/danger_signal/pd_context.json').read_text())
    strains=[r['strain'] for r in context['growth_contexts']]
    def values(label,key):
        return [next(r[key] for r in context['scenarios'] if r['strain']==s and r['scenario']==label)
                for s in strains]
    low,mid,high=map(lambda label:np.array(values(label,'predicted_stasis_ft_pct')),
                     ['control_low','control_mid','control_high'])
    fig,ax=plt.subplots(figsize=(7.5,3.5),layout='constrained')
    x=np.arange(6)
    ax.errorbar(x-.08,mid,yerr=[mid-low,high-mid],fmt='o',capsize=4,
                color='#355d76',label='Fixed-MIC transport scenario; growth envelope')
    ax.scatter(x+.08,values('control_mid','published_stasis_ft_pct'),marker='s',
               color='#b54b42',label='Published strain thresholds')
    ax.set(xticks=x,xticklabels=strains,ylabel='Stasis threshold (% fT>MIC)',
           title='Source-timed controls; canonical PK retained, not murine replay')
    ax.legend(fontsize=8,frameon=False)
    fig.savefig(OUT/'thigh_context.png');plt.close(fig)

    robust=json.loads((ROOT/'results/danger_signal/joint_robustness.json').read_text())
    fig,ax=plt.subplots(figsize=(7.5,3.5),layout='constrained')
    for label in dict.fromkeys(r['pd_scenario'] for r in robust['rows']):
        rows=[r for r in robust['rows'] if r['pd_scenario']==label]
        crit=[next(s['critical_lysis_yield_ratio'] for s in r['yield_stress_tests']
                   if s['filament_factor']==1.) for r in rows]
        ax.plot([r['target_fT_pct'] for r in rows],crit,'o-',lw=1,ms=3,label=label)
    ax.axhline(1.,color='black',ls=':',lw=1,label='Equal lysis yields')
    ax.set(xlabel='Matched exposure target (% fT>MIC; fixed MIC 1 mg/L)',
           ylabel='Critical CAZ:MERO latent lysis-yield ratio',
           title='Above each tipping point, CAZ has greater DS AUC (nominal filament release)')
    ax.legend(fontsize=6,bbox_to_anchor=(1.01,1),loc='upper left',frameon=False)
    fig.savefig(OUT/'joint_ranking_tipping_points.png');plt.close(fig)

    # Overlay public CC BY broth figure only. Copyrighted thigh source images
    # are not copied into tracked analysis output.
    with (ROOT/'data/external_validation/meropenem_figure2bc_initial.csv').open(newline='') as f:
        readings=list(csv.DictReader(f))
    source=np.asarray(Image.open(ROOT/'data/external_validation/source_figures/figure2_embedded0.png'))
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for panel,ax,limits in zip('BC',axes,[(25,990,760,350),(25,990,1180,755)]):
        ax.imshow(source)
        rows=[r for r in readings if r['panel']==panel]
        ax.scatter([float(r['pixel_x']) for r in rows],[float(r['pixel_y']) for r in rows],
                   s=80,facecolors='none',edgecolors='#e83d34',linewidths=1)
        ax.set(xlim=limits[:2],ylim=limits[2:],title=f'Panel {panel}: recorded reading positions')
        ax.axis('off')
    fig.savefig(OUT/'figure2bc_readings.png');plt.close(fig)
    print('Wrote',OUT)


if __name__=='__main__':main()
