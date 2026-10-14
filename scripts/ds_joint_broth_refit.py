"""Joint concentration-response check using the first exposure in Fig 2A-C.

Panel-specific growth and baseline, shared kmax/C50/h. This is a model
adequacy and identification analysis, not adoption of a best optimizer point.
There is on-drug regrowth in panel B. Adequacy is assessed for the current
fixed persistence mechanism, not every possible constant-concentration model.
Missing/off-plot markers are not invented as assay-censored data.
"""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import ds_multiarm_refit as single
from src.utils.provenance import provenance


def points():
    _,c,t,y,labels=single.load_points()
    result={"A":(c,t,y,labels)}
    with (ROOT/"data/external_validation/meropenem_figure2bc_initial.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    for panel in ["B","C"]:
        use=[r for r in rows if r["panel"]==panel]
        # Coincident control markers are not two independently digitised data.
        dedup=[];seen=set()
        for r in use:
            key=(r["concentration_mgL"],r["time_h"],r["pixel_x"],r["pixel_y"])
            if key not in seen: dedup.append(r);seen.add(key)
        result[panel]=tuple(np.array([float(r[k]) for r in dedup])
                            for k in ["concentration_mgL","time_h","value"])+([r["arm"] for r in dedup],)
    return result


def main():
    data=points()
    low=np.r_[single.LOW[:3],np.tile([np.log(.1),3.0],3)]
    high=np.r_[single.HIGH[:3],np.tile([np.log(2.),5.8],3)]
    def residual(x):
        return np.concatenate([single.predict(np.r_[x[:3],x[3+2*j:5+2*j]],c,t)-y
                               for j,(c,t,y,_) in enumerate(data.values())])
    starts=[np.r_[np.log([k,c,h]),np.log(.75),4.8,np.log(1.),4.1,np.log(1.2),4.1]
            for k,c,h in [(2.,.7,1.),(3.,.9,4.),(4.,.4,8.),(6.,1.,20.)]]
    fits=[]
    for start in starts:
        fit=least_squares(residual,start,bounds=(low,high),diff_step=1e-4,max_nfev=220,
                          ftol=1e-8,xtol=1e-8,gtol=1e-7)
        fits.append(fit)
    best=min(fits,key=lambda f:np.sum(f.fun**2))
    profiles=[]
    for h in [.5,.9,1.,2.,4.,8.,16.,29.49,64.]:
        free=[i for i in range(9) if i!=2]
        candidates=[]
        for init in [best.x,starts[1],starts[2]]:
            def expand(z):
                x=init.copy();x[2]=np.log(h);x[free]=z;return x
            f=least_squares(lambda z:residual(expand(z)),init[free],bounds=(low[free],high[free]),
                            diff_step=1e-4,max_nfev=180)
            candidates.append((float(np.sqrt(np.mean(f.fun**2))),expand(f.x),bool(f.success)))
        rmse,x,success=min(candidates,key=lambda z:z[0])
        profiles.append({"hill":h,"rmse_log10":rmse,"parameters_transformed":x.tolist(),"success":success})
    predictions=[]
    for j,(panel,(c,t,y,labels)) in enumerate(data.items()):
        pred=single.predict(np.r_[best.x[:3],best.x[3+2*j:5+2*j]],c,t)
        predictions.extend({"panel":panel,"arm":lab,"C_mgL":float(ci),"time_h":float(ti),
                            "observed_log10":float(yi),"predicted_log10":float(pi)}
                           for ci,ti,yi,pi,lab in zip(c,t,y,pred,labels))
    pars=dict(zip(["k_kill_max","kill_C50","kill_hill"],np.exp(best.x[:3]).tolist()))
    pars["panels"]={p:{"growth_h_inv":float(np.exp(best.x[3+2*j])),
                        "initial_total_log10":float(best.x[4+2*j])} for j,p in enumerate(data)}
    sensitivities = {}
    for label in ['exclude_overlaid_C_high_doses','omit_B_2x_adequacy_diagnostic']:
        def subset_residual(x):
            parts=[]
            for j,(panel,(c,t,y,labels)) in enumerate(data.items()):
                keep=np.ones(len(c),dtype=bool)
                if panel=='C' and label=='exclude_overlaid_C_high_doses':
                    keep=(c<1.)|((c==1.)&(t<3.2))
                if panel=='B' and label=='omit_B_2x_adequacy_diagnostic':keep=c!=2.
                pred=single.predict(np.r_[x[:3],x[3+2*j:5+2*j]],c[keep],t[keep])
                parts.append(pred-y[keep])
            return np.concatenate(parts)
        candidates=[least_squares(subset_residual,start,bounds=(low,high),
                    diff_step=1e-4,max_nfev=220) for start in [best.x,starts[1],starts[2]]]
        fit=min(candidates,key=lambda f:np.sum(f.fun**2))
        sensitivities[label]={
            'parameters':dict(zip(['k_kill_max','kill_C50','kill_hill'],np.exp(fit.x[:3]).tolist())),
            'rmse_log10':float(np.sqrt(np.mean(fit.fun**2))),
            'max_abs_residual_log10':float(np.max(np.abs(fit.fun))),
            'n_points':len(fit.fun),'success':bool(fit.success),
            'scope':'Reading/model-adequacy sensitivity; excluded observations are not invalidated'}
    payload={"status":"exploratory model-adequacy check; not adopted",
             "best":{**pars,"rmse_log10":float(np.sqrt(np.mean(best.fun**2))),
                      "max_abs_residual_log10":float(np.max(np.abs(best.fun))),"success":bool(best.success)},
             "hill_profiles":profiles,"predictions":predictions,
             "scenario_sensitivity":sensitivities,
             "structural_check":{"panel":"B","arm":"2xMIC","observed_growth_2_to_4_8h_log10":2.3,
                                 "reading_allowance_per_point_log10":.3,
                                 "interpretation":"The joint best fit fails the stated reading allowance "
                                                  "under the current fixed persistence mechanism. "
                                                  "This is model inadequacy, not proof of resistance "
                                                  "or a general impossibility for all kill models."},
             "uncertainty":"Digitisation and overlap sensitivity; no sampling-error model or confidence interval",
             "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                 ["data/external_validation/meropenem_figure2a_multiarm.csv",
                  "data/external_validation/meropenem_figure2bc_initial.csv"]},
             "analysis_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "provenance":provenance(note="Joint first-exposure broth windows, per-panel baseline and growth")}
    (ROOT/"results/danger_signal/joint_broth_refit.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload["best"],indent=2),flush=True)


if __name__=="__main__":main()
