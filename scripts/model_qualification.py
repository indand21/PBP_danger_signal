"""Model qualification: Tier 3 (PK metrics vs published ranges) and
Tier 5 (PK/PD index: dose-fractionation time-dependence + stasis %fT>MIC).

Parameters are used as-is (no tuning). All comparisons are against pre-specified
published ranges/targets. Results are reported honestly, whichever way they fall.
"""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.parameters import get_default_parameters, get_drug_pk_parameters, normalize_pk_parameters
from src.core.pd_model import create_ode_system
from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
from src.core.simulation import run_simulation

WEIGHT = 70.0


# --------------------------------------------------------------------------
# Shared: analytical PLASMA concentration profile (matches the model's PK)
# --------------------------------------------------------------------------
def plasma_profile(drug, dose_mg, interval_h, n_doses, inf_min, t_end, npts=6000):
    pk = normalize_pk_parameters(get_drug_pk_parameters(drug), WEIGHT)
    CL, Vc, Ka = pk["CL"], pk["Vc"], pk["Ka"]
    k = CL / Vc
    is_oral = Ka > 0
    inf_h = inf_min / 60.0
    dose_times = [i * interval_h for i in range(n_doses)]
    t = np.linspace(0, t_end, npts)
    C = np.zeros_like(t)
    for i, ti in enumerate(t):
        c = 0.0
        for dt in dose_times:
            if ti <= dt:
                continue
            tau = ti - dt
            if is_oral:
                if abs(Ka - k) > 1e-9:
                    c += dose_mg / Vc * Ka / (Ka - k) * (np.exp(-k * tau) - np.exp(-Ka * tau))
                else:
                    c += dose_mg / Vc * Ka * tau * np.exp(-Ka * tau)
            else:
                if tau <= inf_h and inf_h > 0:
                    R0 = dose_mg / inf_h
                    c += R0 / (k * Vc) * (1 - np.exp(-k * tau))
                else:
                    R0 = dose_mg / inf_h if inf_h > 0 else 0.0
                    Cend = (R0 / (k * Vc) * (1 - np.exp(-k * inf_h))) if inf_h > 0 else dose_mg / Vc
                    c += Cend * np.exp(-k * (tau - inf_h))
        C[i] = c
    return t, C, k


# --------------------------------------------------------------------------
# Tier 3: single-dose PK metrics vs published ranges
# --------------------------------------------------------------------------
PUBLISHED = {
    "meropenem":   dict(dose=1000, inf=30, ref="Drusano 1995 / Nicolau",
                        Cmax=(49, 62), thalf=(0.8, 1.2), AUC=(70, 90)),
    "doxycycline": dict(dose=200, inf=0, ref="Agwuh & MacGowan 2006",
                        Cmax=(3, 5), thalf=(16, 18), AUC=(90, 113)),
}


def tier3():
    print("=" * 72)
    print("TIER 3 - PK metrics vs published ranges (single dose, no tuning)")
    print("=" * 72)
    for drug, ref in PUBLISHED.items():
        pk = normalize_pk_parameters(get_drug_pk_parameters(drug), WEIGHT)
        t, C, k = plasma_profile(drug, ref["dose"], 24, 1, ref["inf"], t_end=96)
        cmax = C.max()
        thalf = np.log(2) / k
        auc = ref["dose"] / pk["CL"]          # single-dose AUC0-inf (1-compartment)
        tmax = t[C.argmax()]

        def mark(v, lo, hi):
            return "in range" if lo <= v <= hi else ("LOW" if v < lo else "HIGH")

        print(f"\n{drug}  ({ref['ref']}):")
        print(f"  Cmax  = {cmax:6.2f} mg/L   published {ref['Cmax']}   -> {mark(cmax, *ref['Cmax'])}")
        print(f"  t1/2  = {thalf:6.2f} h      published {ref['thalf']}   -> {mark(thalf, *ref['thalf'])}")
        print(f"  AUC   = {auc:6.1f} mg.h/L published {ref['AUC']}   -> {mark(auc, *ref['AUC'])}")
        print(f"  tmax  = {tmax:6.2f} h  (info)")


# --------------------------------------------------------------------------
# Tier 5: PK/PD index (meropenem, cidal, immune OFF = neutropenic emulation)
# --------------------------------------------------------------------------
IMMUNE_OFF_IC = dict(B_rep=1e6, B_pers=1e3, B_SCV=0, N_eff=1e2,
                     Damage=0, IL6=10, TNF=5, PAMP=0, D_host=0)


def _net_logchange_constant(C_eff):
    """24 h log10 change in total burden at CONSTANT effect-site conc, immune off."""
    pd = create_ode_system(get_default_parameters())
    y0 = np.array([1e6, 1e3, 0, 1e2, 0, 10, 5, 0, 0], dtype=float)
    sol = solve_ivp(lambda t, y: pd.rhs(t, y, C_effect=C_eff, drug_class="cidal"),
                    (0, 24), y0, method="LSODA", max_step=0.1, rtol=1e-6, atol=1e-8)
    B0 = y0[:3].sum()
    Bf = max(sol.y[:3, -1].sum(), 1e-6)
    return np.log10(Bf / B0)


def effective_mic():
    """Effect-site and plasma effective MIC (constant conc giving 24 h stasis)."""
    mic_es = brentq(_net_logchange_constant, 1e-4, 10.0, xtol=1e-4)
    Kp = get_drug_pk_parameters("meropenem").Kp
    return mic_es, mic_es / Kp


def _logkill_regimen(dose_mg, interval, n_doses, inf_min=30):
    # RAW per-kg params; run_simulation scales Vc/Vp by weight (passing normalized
    # output here double-scales Vc -> t1/2 ~57 h, see run_simulation CONVENTION).
    p = get_drug_pk_parameters("meropenem")
    pkm = TwoCompartmentPKModel(CL=p.CL, Vc=p.Vc, Vp=p.Vp, Q=p.Q, Ka=p.Ka, Kp=p.Kp,
                                effect_site_model=True)
    reg = DosingRegimen(dose_mg=dose_mg, interval_hours=interval, start_time=0,
                        n_doses=n_doses, infusion_duration_min=inf_min)
    pd = create_ode_system(get_default_parameters())
    r = run_simulation(pkm, reg, pd, dict(IMMUNE_OFF_IC), t_span=(0, 24),
                       drug_class="cidal", weight_kg=WEIGHT)
    _, B = r.get_bacterial_burden()
    return np.log10(max(B[-1], 1e-6) / B[0])


def _fT_over_mic(dose_mg, interval, n_doses, mic_plasma, inf_min=30):
    t, C, _ = plasma_profile("meropenem", dose_mg, interval, n_doses, inf_min, t_end=24, npts=4000)
    return float(np.mean(C > mic_plasma) * 100.0)


def tier5():
    print("\n" + "=" * 72)
    print("TIER 5 - PK/PD index (meropenem, immune OFF)")
    print("=" * 72)
    mic_es, mic_pl = effective_mic()
    print(f"\nModel effective MIC: {mic_es:.3f} mg/L (effect-site) = {mic_pl:.3f} mg/L (plasma-equivalent)")

    # 5a. Dose-fractionation: fixed total daily dose 3 g, vary interval.
    print("\n[5a] Dose-fractionation at fixed 3 g/day (time-dependence signature):")
    print("     regimen           %fT>MIC   24h log10 change")
    fracs = [("3 g q24 (x1)", 3000, 24, 1), ("1.5 g q12 (x2)", 1500, 12, 2),
             ("1 g q8 (x3)", 1000, 8, 3), ("0.75 g q6 (x4)", 750, 6, 4)]
    for label, d, iv, n in fracs:
        ft = _fT_over_mic(d, iv, n, mic_pl)
        lk = _logkill_regimen(d, iv, n)
        print(f"     {label:18s} {ft:6.1f}    {lk:+.2f}")

    # 5b. Stasis target: sweep total dose at q8, find %fT>MIC where log change = 0.
    print("\n[5b] Stasis %fT>MIC (q8 dosing, sweep dose):")
    pts = []
    for d in [30, 60, 100, 150, 200, 300, 450, 700, 1000]:
        ft = _fT_over_mic(d, 8, 3, mic_pl)
        lk = _logkill_regimen(d, 8, 3)
        pts.append((ft, lk))
        print(f"     {d:5d} mg q8   %fT>MIC={ft:5.1f}   log10 change={lk:+.2f}")
    pts = sorted(pts)
    fts = np.array([p[0] for p in pts]); lks = np.array([p[1] for p in pts])
    stasis = np.nan
    for i in range(len(lks) - 1):
        if lks[i] >= 0 >= lks[i + 1] or lks[i] <= 0 <= lks[i + 1]:
            # linear interp of %fT>MIC at log change = 0
            if lks[i] != lks[i + 1]:
                stasis = fts[i] + (0 - lks[i]) * (fts[i + 1] - fts[i]) / (lks[i + 1] - lks[i])
            break
    print(f"\n  Model stasis target ~ {stasis:.0f}% fT>MIC")
    print("  Published (Craig/Andes) carbapenem: stasis ~20% fT>MIC, 1-log kill ~40%")


if __name__ == "__main__":
    tier3()
    tier5()
