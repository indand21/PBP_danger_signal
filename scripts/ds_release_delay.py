"""Does a delay between killing and endotoxin release change the results?

The model releases endotoxin into plasma at the moment of killing. The one
data-fitted release model for a beta-lactam, Thorsted et al. 2020 (cefuroxime
against E. coli, 52 time-kill experiments; full text PMC7179275), found
instead that release after antibiotic-induced killing is delayed, through ONE
transit compartment with a transit rate of 0.131 per hour (RSE 27%), a mean
transit time of 7.63 h, while release during growth is immediate. This script
asks which of the paper's statements survive that structure.

The delay sits between lysis-coupled release and the plasma endotoxin
compartment:

    A_1' = r_lysis(t) - (n/MTT) A_1
    A_i' = (n/MTT) (A_{i-1} - A_i)                     i = 2..n
    D'   = (n/MTT) A_n + r_filament(t) - k D

n = 1 is the cefuroxime structure. MTT is varied from 0.5 to 16 h, which
spans the 0.5-6 h delays that study cites from earlier work and the
approximate 95% interval of its own estimate (5.0-16.2 h). n = 3 at the same
means is a check on the shape: a lag with less tail. The sub-MIC term here is
release from surviving, filamenting cells, not release on killing, so it stays
immediate; one variant delays it as well.

Nothing is re-solved. Release is a one-way output of the bacterial trajectory
and the endotoxin compartment is linear, so the delayed endpoint of each
stored regimen is computed exactly from its release input: the chain is
integrated in closed form (one matrix exponential per step) over the same
piecewise-linear input the ODE solver interpolates. With MTT = 0 this is
ds_endpoint_windows.exact_window_aucs, which is checked on every arm. Each
regimen must first reproduce its stored 48-h AUC.

Three regimen sets are re-read, each the one a claim in the paper rests on:
  analysis 3 (Table 1)   the agent ratio at matched exposure;
  analysis 4 (Table 2)   index correlations over the 48-regimen library, and
                         the agent offset at matched %fT>MIC;
  analysis 5, MIC x1     the within-agent exposure direction on the 48-h and
    (Figure 4)           first-hour windows.

What a delay can change follows from mass balance. Over [0, T]
    released = k AUC_D(T) + D(T) + in_transit(T),
so a delay lowers the 48-h AUC by the endotoxin still in transit at 48 h
(divided by k) and leaves exposure to infinity, released / k, unchanged. On
the 48-h endpoint it is a finite-horizon effect, largest for regimens that
release late; on short windows it is a change of timing.

Writes results/danger_signal/release_delay.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_endpoint_windows import (  # noqa: E402
    REPRODUCTION_RTOL, exact_window_aucs, load_phase4_module)
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"
OUT = ROOT / "results/danger_signal/release_delay.json"
AGENTS = ("ceftazidime", "meropenem")
WINDOWS_H = (1.0, 2.0, 4.0, 8.0, 24.0, 48.0)
REFERENCE_TARGET = 40.0
LIBRARY_DOSES = {6.0: 8, 8.0: 6, 12.0: 4, 24.0: 2}   # interval -> n doses, as analysis 4

# Thorsted et al. 2020, Table 1: transit rate for release after antibiotic-
# induced killing of susceptible and filamentous cells, one compartment.
THORSTED_KTR_PER_H = 0.131
THORSTED_KTR_RSE = 0.27
THORSTED_MTT_H = 7.63                     # as reported; 1 / 0.131 = 7.634
MTT_GRID_H = (0.5, 1.0, 2.0, 4.0, THORSTED_MTT_H, 12.0, 16.0)
N_GRID = (1, 3)
PRIMARY = "n1_mtt7.63"

ENDPOINTS = ("0-1h", "0-24h", "0-48h", "peak", "max_arrival")


# --------------------------------------------------------------------------
# exact integration of the delayed compartment
# --------------------------------------------------------------------------
def chain_generator(k, mtt, n, delay_filament=False):
    """Generator of the augmented linear system integrated over one step.

    State [A_1..A_n, D, I, a_l, b_l, a_f, b_f]: n transit amounts, plasma
    endotoxin above baseline, its running integral, and each input's value
    and slope on the current step (a' = b, b' = 0). A piecewise-linear input
    is then integrated exactly by one matrix exponential per step.
    Returns (H, n), with n = 0 when MTT = 0 (no delay).
    """
    n = 0 if mtt == 0 else int(n)
    m = n + 2
    iD, iI, il, iF = n, n + 1, m, m + 2
    H = np.zeros((m + 4, m + 4))
    if n:
        ktr = n / mtt
        for i in range(n):
            H[i, i] = -ktr
            if i:
                H[i, i - 1] = ktr
        H[iD, n - 1] = ktr
        H[0, il] = 1.0
        H[0 if delay_filament else iD, iF] = 1.0
    else:
        H[iD, il] = 1.0
        H[iD, iF] = 1.0
    H[iD, iD] = -k
    H[iI, iD] = 1.0
    H[il, il + 1] = 1.0
    H[iF, iF + 1] = 1.0
    return H, n


def delayed_endotoxin(t, r_lysis, r_filament, k, mtt=0.0, n=1,
                      delay_filament=False, windows=WINDOWS_H):
    """Plasma endotoxin above baseline when lysis release passes a transit chain.

    ``r_lysis`` and ``r_filament`` are appearance rates (pg/mL/h of plasma) on
    ``t``, taken as piecewise linear, exactly as the ODE solver interpolates
    them. Returns the trajectory on the grid (with window ends inserted), the
    exact AUC at each window end, and the mass-balance terms.
    """
    t = np.asarray(t, float)
    extra = [w for w in windows
             if t[0] < w <= t[-1] and np.min(np.abs(t - w)) > 1e-9]
    grid = np.unique(np.concatenate([t, extra]))
    ul = np.interp(grid, t, np.asarray(r_lysis, float))
    uf = np.interp(grid, t, np.asarray(r_filament, float))
    H, n = chain_generator(k, mtt, n, delay_filament)
    m = n + 2
    h = np.diff(grid)
    # One exponential per distinct step length: the solver grid repeats them.
    _, first, inv = np.unique(np.round(h, 12), return_index=True,
                              return_inverse=True)
    E = expm(h[first][:, None, None] * H[None])[:, :m, :]
    sl, sf = np.diff(ul) / h, np.diff(uf) / h
    x = np.zeros((len(grid), m))
    z = np.zeros(m + 4)
    for i in range(len(h)):
        z[:m] = x[i]
        z[m], z[m + 1], z[m + 2], z[m + 3] = ul[i], sl[i], uf[i], sf[i]
        x[i + 1] = E[inv[i]] @ z
    D, area, transit = x[:, n], x[:, n + 1], x[:, :n].sum(axis=1)
    arrival = ((n / mtt) * x[:, n - 1] + (0.0 if delay_filament else uf)
               if n else ul + uf)
    u = ul + uf
    released = float(np.sum(h * (u[1:] + u[:-1]) / 2.0))   # exact: u is linear
    idx = {w: int(np.argmin(np.abs(grid - w))) for w in windows if w <= grid[-1] + 1e-9}
    residual = released - (k * area[-1] + D[-1] + transit[-1])
    return {
        "grid": grid, "D": D, "arrival": arrival,
        "auc": {w: float(area[i]) for w, i in idx.items()},
        "released": released, "in_transit_end": float(transit[-1]),
        "D_end": float(D[-1]),
        "mass_balance_rel_residual": float(abs(residual) / max(released, 1e-300)),
    }


def summarize_run(run):
    """The endpoints compared across delays, keyed as ENDPOINTS."""
    kp, ka = int(np.argmax(run["D"])), int(np.argmax(run["arrival"]))
    return {
        "window_AUC": {f"0-{w:g}h": v for w, v in run["auc"].items()},
        "peak": float(run["D"][kp]),
        "t_peak_h": float(run["grid"][kp]),
        "max_arrival": float(run["arrival"][ka]),
        "t_max_arrival_h": float(run["grid"][ka]),
        "in_transit_fraction_at_end": run["in_transit_end"] / run["released"],
        "first_hour_share": run["auc"][1.0] / run["auc"][48.0],
        "mass_balance_rel_residual": run["mass_balance_rel_residual"],
    }


def value(s, key):
    return s["window_AUC"][key] if key in s["window_AUC"] else s[key]


# --------------------------------------------------------------------------
# the stored regimens
# --------------------------------------------------------------------------
def arm_inputs(p4, agent, dose, interval, n_doses, mic, stored_auc, where):
    """Re-simulate one stored arm and split its release into per-yield inputs."""
    sim = p4.simulate(agent, dose, interval, n_doses, mic=mic)
    out, summ = p4.ds_cost(sim, agent=agent, cytokines=False)
    rel = abs(summ["AUC_DS_pg_h_mL"] - stored_auc) / stored_auc
    if rel > REPRODUCTION_RTOL:
        raise RuntimeError(f"{where}: reproduces the stored 48-h AUC only to "
                           f"{rel:.2e}; not the same simulation")
    q = p4.P_REF
    if q.DS_baseline != 0.0:
        raise RuntimeError("the endpoint is AUC of DS, which equals the excess "
                           "this script integrates only at zero baseline")
    to_conc = 1e-6 / q.V_ds_L * q.translocation_fraction
    B = np.asarray(sim["B_rep"], float)
    lys_unit = np.asarray(sim["kill"], float) * B * p4.V_FOCUS * to_conc
    fil_unit = np.asarray(out["phi"], float) * B * p4.V_FOCUS * to_conc
    Y, Yf = p4.AGENT_YIELD[agent], p4.AGENT_FILAMENT[agent]
    src = (np.asarray(out["release_lysis"]) + np.asarray(out["release_filament"])) * 1e6 * to_conc
    if not np.allclose(Y * lys_unit + Yf * fil_unit, src, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"{where}: per-yield split does not rebuild the release input")
    # The independent exact integrator of ds_endpoint_windows, for the
    # zero-delay identity check below.
    ref48 = exact_window_aucs(sim["t"], src, q.k_ds_clear, (48.0,))[48.0]
    return {"where": where, "agent": agent, "dose_mg": dose, "interval_h": interval,
            "t": np.asarray(sim["t"], float), "lys_unit": lys_unit, "fil_unit": fil_unit,
            "Y": Y, "Yf": Yf, "fT": sim["fT"], "net_log10": sim["net_log10"],
            "controlled": bool(sim["net_log10"] < 0.0),
            "stored_auc48": stored_auc, "reproduced_rel_diff": rel,
            "exact_auc48_independent": ref48}


def run_arm(arm, cfg, k, Y=None, Yf=None):
    Y = arm["Y"] if Y is None else Y
    Yf = arm["Yf"] if Yf is None else Yf
    return summarize_run(delayed_endotoxin(
        arm["t"], Y * arm["lys_unit"], Yf * arm["fil_unit"], k,
        mtt=cfg["mtt_h"], n=cfg["n"], delay_filament=cfg["delay_filament"]))


def configs():
    yield {"label": "immediate", "mtt_h": 0.0, "n": 0, "delay_filament": False}
    for n in N_GRID:
        for mtt in MTT_GRID_H:
            yield {"label": f"n{n}_mtt{mtt:g}", "mtt_h": mtt, "n": n,
                   "delay_filament": False}
    yield {"label": f"n1_mtt{THORSTED_MTT_H:g}_filament_delayed",
           "mtt_h": THORSTED_MTT_H, "n": 1, "delay_filament": True}


# --------------------------------------------------------------------------
# the three claims, re-read under one delay
# --------------------------------------------------------------------------
def table1(a3_arms, cfg, k, y_caz, ratios):
    rows = []
    for target, (caz, mero) in sorted(a3_arms.items()):
        c = run_arm(caz, cfg, k)["window_AUC"]["0-48h"]
        m_unit = run_arm(mero, cfg, k, Y=1.0, Yf=0.0)["window_AUC"]["0-48h"]
        rows.append({"target_fT_pct": target,
                     "ratio_AUC48": {f"{r:g}": c / (y_caz / r * m_unit) for r in ratios}})
    flat = [v for r in rows for v in r["ratio_AUC48"].values()]
    return {"rows": rows, "range": [min(flat), max(flat)]}


def offset_at_matched_index(lib, cost):
    """Analysis 4's offset: CAZ/MERO cost at matched %fT>MIC, log-interpolated."""
    per = {}
    for agent in AGENTS:
        sub = sorted((i for i, r in enumerate(lib) if r["agent"] == agent),
                     key=lambda i: lib[i]["fT_pct"])
        per[agent] = (np.array([lib[i]["fT_pct"] for i in sub]),
                      np.array([cost[i] for i in sub]))
    lo = max(v[0].min() for v in per.values())
    hi = min(v[0].max() for v in per.values())
    ratios = []
    for g in np.linspace(lo, hi, 7):
        val = {a: float(np.exp(np.interp(g, x, np.log(c)))) for a, (x, c) in per.items()}
        ratios.append(val["ceftazidime"] / val["meropenem"])
    return [min(ratios), max(ratios)]


CANDIDATES = ("fT_pct", "fAUC_MIC", "fCmax_MIC", "net_log10_change",
              "cum_killed_cfu_per_mL", "yield_weighted_kill")


def table2(lib, lib_arms, cfg, k):
    runs = [run_arm(a, cfg, k) for a in lib_arms]
    cost = np.array([r["window_AUC"]["0-48h"] for r in runs])
    pooled = {c: float(spearmanr([r[c] for r in lib], cost)[0]) for c in CANDIDATES}
    within = {}
    for agent in AGENTS:
        ix = [i for i, r in enumerate(lib) if r["agent"] == agent]
        within[agent] = {c: float(spearmanr([lib[i][c] for i in ix], cost[ix])[0])
                         for c in CANDIDATES}
    transit = [r["in_transit_fraction_at_end"] for r in runs]
    return {"spearman_pooled": pooled, "spearman_within": within,
            "offset_at_matched_fT": offset_at_matched_index(lib, cost),
            "max_in_transit_fraction_at_48h": max(transit),
            "cost_AUC48": cost.tolist()}


def figure4(a5_arms, cfg, k, targets):
    s = {key: run_arm(arm, cfg, k) for key, arm in a5_arms.items()}
    divergent = {e: 0 for e in ENDPOINTS}
    steps = []
    for agent in AGENTS:
        for lo, hi in zip(targets, targets[1:]):
            a, b = s[(agent, lo)], s[(agent, hi)]
            better = a5_arms[(agent, hi)]["net_log10"] < a5_arms[(agent, lo)]["net_log10"]
            fold = {e: value(b, e) / value(a, e) for e in ENDPOINTS}
            for e in ENDPOINTS:
                divergent[e] += int(better and fold[e] > 1.0)
            steps.append({"agent": agent, "from_fT": lo, "to_fT": hi,
                          "kill_improved": bool(better), "fold_change": fold})
    top, lowest = max(targets), min(targets)
    per_agent, ratio_vals, share = {}, [], []
    for agent in AGENTS:
        ctl = [t for t in targets if a5_arms[(agent, t)]["controlled"]]
        fail = [t for t in targets if not a5_arms[(agent, t)]["controlled"]]
        a48 = {t: s[(agent, t)]["window_AUC"]["0-48h"] for t in targets}
        med = float(np.median([a48[t] for t in ctl]))
        ref = s[(agent, REFERENCE_TARGET)]
        per_agent[agent] = {
            "failing_over_controlled_fold": (float(np.median([a48[t] for t in fail])) / med
                                             if fail else None),
            "change_pct_reference_to_top": {
                e: 100.0 * (value(s[(agent, top)], e) / value(ref, e) - 1.0)
                for e in ENDPOINTS},
            "first_hour_change_pct_failing_to_reference":
                100.0 * (value(ref, "0-1h") / value(s[(agent, lowest)], "0-1h") - 1.0),
            "t_peak_h_reference": ref["t_peak_h"],
            "in_transit_fraction_at_48h": {f"{t:g}": s[(agent, t)]["in_transit_fraction_at_end"]
                                           for t in targets},
        }
        share += [s[(agent, t)]["first_hour_share"] for t in ctl]
    controlled = [t for t in targets if all(a5_arms[(a, t)]["controlled"] for a in AGENTS)]
    agent_ratio = {f"{t:g}": {e: value(s[("ceftazidime", t)], e) / value(s[("meropenem", t)], e)
                              for e in ENDPOINTS} for t in targets}
    ratio_vals = [v for t in controlled for v in agent_ratio[f"{t:g}"].values()]
    return {"divergent_steps_by_endpoint": divergent, "n_steps": len(steps),
            "steps": steps, "per_agent": per_agent, "agent_ratio": agent_ratio,
            "agent_ratio_range_controlled": [min(ratio_vals), max(ratio_vals)],
            "first_hour_share_controlled_range": [min(share), max(share)],
            "arms": {f"{a} {t:g}": s[(a, t)] for (a, t) in s}}


# --------------------------------------------------------------------------
def main():
    p4 = load_phase4_module()
    family = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(family, key=lambda e: e["cost"]))["params"]
    k = p4.P_REF.k_ds_clear
    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    mic = phase4["meta"]["mic_effect_site_mgL"]
    p4.MIC_EFFECT = mic
    y_caz = p4.AGENT_YIELD["ceftazidime"]
    ratios = tuple(phase4["meta"]["yield_ratio_range"])

    # analysis 3: matched-exposure pairs (Table 1)
    a3_arms, a3_stored = {}, {}
    for row in phase4["analysis3_matched_exposure"]["rows"]:
        if not row.get("evaluated"):
            continue
        t = row["target_fT_pct"]
        a3_arms[t] = tuple(
            arm_inputs(p4, a, row["arms"][a]["dose_mg"], 8.0, 6, mic,
                       row["arms"][a]["AUC_DS_pg_h_mL"], f"analysis3 {a} {t:g}%")
            for a in AGENTS)
        a3_stored[t] = {f"{r['yield_ratio']:g}": r["ratio_AUC_DS"]
                        for r in row["cost_ratio_range"]}
    print(f"  analysis 3: {2 * len(a3_arms)} arms", flush=True)

    # analysis 4: the 48-regimen library (Table 2)
    a4 = phase4["analysis4_cost_index"]
    lib = a4["library"]
    lib_arms = [arm_inputs(p4, r["agent"], r["dose_mg"], r["interval_h"],
                           LIBRARY_DOSES[r["interval_h"]], mic, r["AUC_DS_pg_h_mL"],
                           f"analysis4 {r['agent']} {r['dose_mg']:g} mg q{r['interval_h']:g}h")
                for r in lib]
    print(f"  analysis 4: {len(lib_arms)} arms", flush=True)

    # analysis 5, MIC x1 stratum (Figure 4)
    base = sorted((c for c in phase4["analysis5_divergence_map"]["cells"]
                   if c["mic_factor"] == 1.0), key=lambda c: c["target_fT_pct"])
    targets = [c["target_fT_pct"] for c in base]
    a5_arms = {(a, c["target_fT_pct"]): arm_inputs(
        p4, a, c[a]["dose_mg"], 8.0, 6, c["mic_mgL"], c[a]["AUC_DS_pg_h_mL"],
        f"analysis5 {a} {c['target_fT_pct']:g}%") for a in AGENTS for c in base}
    print(f"  analysis 5: {len(a5_arms)} arms", flush=True)

    all_arms = [x for pair in a3_arms.values() for x in pair] + lib_arms + list(a5_arms.values())

    results, worst_mb = {}, 0.0
    for cfg in configs():
        t1 = table1(a3_arms, cfg, k, y_caz, ratios)
        t2 = table2(lib, lib_arms, cfg, k)
        f4 = figure4(a5_arms, cfg, k, targets)
        worst_mb = max([worst_mb] + [a["mass_balance_rel_residual"] for a in f4["arms"].values()])
        results[cfg["label"]] = {"config": cfg, "table1": t1, "table2": t2, "figure4": f4}
        print(f"  {cfg['label']:32s} T1 {t1['range'][0]:.2f}-{t1['range'][1]:.2f}  "
              f"div48 {f4['divergent_steps_by_endpoint']['0-48h']}  "
              f"div1h {f4['divergent_steps_by_endpoint']['0-1h']}  "
              f"rho_fT {t2['spearman_pooled']['fT_pct']:+.2f}  "
              f"rho_yw {t2['spearman_pooled']['yield_weighted_kill']:+.3f}  "
              f"offset {t2['offset_at_matched_fT'][0]:.2f}-{t2['offset_at_matched_fT'][1]:.2f}",
              flush=True)

    # ---- numerical checks -------------------------------------------------
    imm = results["immediate"]
    # zero delay against the independent integrator, on every arm
    zero_gap = max(abs(run_arm(a, imm["config"], k)["window_AUC"]["0-48h"]
                       - a["exact_auc48_independent"]) / a["exact_auc48_independent"]
                   for a in all_arms)
    stored_gap = max(abs(run_arm(a, imm["config"], k)["window_AUC"]["0-48h"]
                         - a["stored_auc48"]) / a["stored_auc48"] for a in all_arms)
    t1_gap = max(abs(row["ratio_AUC48"][r] / a3_stored[row["target_fT_pct"]][r] - 1.0)
                 for row in imm["table1"]["rows"] for r in row["ratio_AUC48"])
    stored_rho = {c: a4["candidates"][c]["spearman_rho"] for c in CANDIDATES}
    rho_gap = max(abs(imm["table2"]["spearman_pooled"][c] - stored_rho[c]) for c in CANDIDATES)
    off_gap = max(abs(x / y - 1.0) for x, y in zip(
        imm["table2"]["offset_at_matched_fT"],
        a4["cost_offset_at_matched_index"]["ratio_at_matched_index"]))
    for cfg_label, r in results.items():
        for arm in r["figure4"]["arms"].values():
            worst_mb = max(worst_mb, arm["mass_balance_rel_residual"])

    # ---- headline: the cefuroxime structure, and the range over the grid ----
    def span(fn, labels):
        vals = [fn(results[l]) for l in labels]
        return [min(vals), max(vals)]

    n1 = [f"n1_mtt{m:g}" for m in MTT_GRID_H]
    every = [l for l in results if l != "immediate"]
    P = results[PRIMARY]
    pa = P["figure4"]["per_agent"]

    # How far any delay moves a Table 1 ratio, against the same exact
    # quadrature at zero delay (the stored ratios differ from that by the
    # trapezoid error, up to ~0.1%, which is not an effect of the delay).
    def t1_change(label):
        return max(abs(row["ratio_AUC48"][r] / row0["ratio_AUC48"][r] - 1.0)
                   for row, row0 in zip(results[label]["table1"]["rows"],
                                        imm["table1"]["rows"])
                   for r in row["ratio_AUC48"])

    def ratio_range(label, keys):
        f4 = results[label]["figure4"]
        ctl = [t for t in targets if all(a5_arms[(a, t)]["controlled"] for a in AGENTS)]
        v = [f4["agent_ratio"][f"{t:g}"][e] for t in ctl for e in keys]
        return [min(v), max(v)]

    def steps_raising(label, agent, endpoint):
        return sum(1 for s in results[label]["figure4"]["steps"]
                   if s["agent"] == agent and s["kill_improved"]
                   and s["fold_change"][endpoint] > 1.0)

    ctl_targets = [f"{t:g}" for t in targets
                   if all(a5_arms[(a, t)]["controlled"] for a in AGENTS)]
    fail_targets = [f"{t:g}" for t in targets
                    if not any(a5_arms[(a, t)]["controlled"] for a in AGENTS)]
    headline = {
        "primary_config": PRIMARY,
        "table1_ratio_range": P["table1"]["range"],
        "table1_ratio_range_immediate": imm["table1"]["range"],
        "divergent_48h_steps": P["figure4"]["divergent_steps_by_endpoint"]["0-48h"],
        "divergent_48h_steps_max_any_delay": max(
            results[l]["figure4"]["divergent_steps_by_endpoint"]["0-48h"] for l in every),
        "divergent_first_hour_steps": P["figure4"]["divergent_steps_by_endpoint"]["0-1h"],
        "divergent_peak_steps": P["figure4"]["divergent_steps_by_endpoint"]["peak"],
        "divergent_max_arrival_steps": P["figure4"]["divergent_steps_by_endpoint"]["max_arrival"],
        "n_steps": P["figure4"]["n_steps"],
        "failing_over_controlled_fold": {a: pa[a]["failing_over_controlled_fold"] for a in AGENTS},
        "failing_over_controlled_fold_immediate": {
            a: imm["figure4"]["per_agent"][a]["failing_over_controlled_fold"] for a in AGENTS},
        "auc48_change_pct_reference_to_top": {
            a: pa[a]["change_pct_reference_to_top"]["0-48h"] for a in AGENTS},
        "first_hour_change_pct_reference_to_top": {
            a: pa[a]["change_pct_reference_to_top"]["0-1h"] for a in AGENTS},
        "peak_change_pct_reference_to_top": {
            a: pa[a]["change_pct_reference_to_top"]["peak"] for a in AGENTS},
        "max_arrival_change_pct_reference_to_top": {
            a: pa[a]["change_pct_reference_to_top"]["max_arrival"] for a in AGENTS},
        "first_hour_change_pct_failing_to_reference": {
            a: pa[a]["first_hour_change_pct_failing_to_reference"] for a in AGENTS},
        "first_hour_share_pct_controlled_range": [
            100 * v for v in P["figure4"]["first_hour_share_controlled_range"]],
        "t_peak_h_reference": {a: pa[a]["t_peak_h_reference"] for a in AGENTS},
        "agent_ratio_range_controlled": P["figure4"]["agent_ratio_range_controlled"],
        "agent_ratio_range_controlled_24_48h": ratio_range(PRIMARY, ("0-24h", "0-48h")),
        "agent_ratio_range_controlled_short": ratio_range(PRIMARY, ("0-1h", "peak", "max_arrival")),
        "agent_ratio_range_controlled_immediate": ratio_range(
            "immediate", ("0-1h", "0-24h", "0-48h", "peak", "max_arrival")),
        "table1_max_abs_rel_change_pct_any_delay": 100 * max(t1_change(l) for l in every),
        "table1_max_abs_rel_change_pct_primary": 100 * t1_change(PRIMARY),
        "in_transit_pct_at_48h_controlled_max": 100 * max(
            pa[a]["in_transit_fraction_at_48h"][t] for a in AGENTS for t in ctl_targets),
        "in_transit_pct_at_48h_failing_range": [
            100 * min(pa[a]["in_transit_fraction_at_48h"][t] for a in AGENTS for t in fail_targets),
            100 * max(pa[a]["in_transit_fraction_at_48h"][t] for a in AGENTS for t in fail_targets)],
        "immediate_steps_raising": {
            e: imm["figure4"]["divergent_steps_by_endpoint"][e] for e in ENDPOINTS},
        "first_hour_steps_raising_by_agent": {
            label: {a: steps_raising(label, a, "0-1h") for a in AGENTS}
            for label in results},
        "spearman_pooled_fT": P["table2"]["spearman_pooled"]["fT_pct"],
        "spearman_pooled_yield_weighted_kill": P["table2"]["spearman_pooled"]["yield_weighted_kill"],
        "spearman_within_fT": {a: P["table2"]["spearman_within"][a]["fT_pct"] for a in AGENTS},
        "offset_at_matched_fT": P["table2"]["offset_at_matched_fT"],
        "offset_at_matched_fT_immediate": imm["table2"]["offset_at_matched_fT"],
        "max_in_transit_pct_at_48h_library": 100 * P["table2"]["max_in_transit_fraction_at_48h"],
        "in_transit_pct_at_48h_figure4": {
            a: {t: 100 * v for t, v in pa[a]["in_transit_fraction_at_48h"].items()}
            for a in AGENTS},
        "over_mtt_grid_n1": {
            "mtt_h": list(MTT_GRID_H),
            "table1_ratio_range": [span(lambda r: r["table1"]["range"][0], n1)[0],
                                   span(lambda r: r["table1"]["range"][1], n1)[1]],
            "spearman_pooled_yield_weighted_kill_min": span(
                lambda r: r["table2"]["spearman_pooled"]["yield_weighted_kill"], n1)[0],
            "spearman_pooled_fT_range": span(
                lambda r: r["table2"]["spearman_pooled"]["fT_pct"], n1),
            "failing_over_controlled_fold": {
                a: span(lambda r, a=a: r["figure4"]["per_agent"][a]["failing_over_controlled_fold"], n1)
                for a in AGENTS},
            "divergent_first_hour_steps": span(
                lambda r: r["figure4"]["divergent_steps_by_endpoint"]["0-1h"], n1),
            "offset_range": [span(lambda r: r["table2"]["offset_at_matched_fT"][0], n1)[0],
                             span(lambda r: r["table2"]["offset_at_matched_fT"][1], n1)[1]],
        },
        "over_every_delay": {
            "n_configs": len(every),
            "divergent_48h_steps_max": max(
                results[l]["figure4"]["divergent_steps_by_endpoint"]["0-48h"] for l in every),
            "divergent_peak_steps_max": max(
                results[l]["figure4"]["divergent_steps_by_endpoint"]["peak"] for l in every),
            "divergent_max_arrival_steps_max": max(
                results[l]["figure4"]["divergent_steps_by_endpoint"]["max_arrival"] for l in every),
            "spearman_pooled_yield_weighted_kill_min": span(
                lambda r: r["table2"]["spearman_pooled"]["yield_weighted_kill"], every)[0],
            "failing_over_controlled_fold": {
                a: span(lambda r, a=a: r["figure4"]["per_agent"][a]["failing_over_controlled_fold"], every)
                for a in AGENTS},
            "meropenem_first_hour_steps_raising_min": min(
                steps_raising(l, "meropenem", "0-1h") for l in every),
            "table1_ratio_range": [span(lambda r: r["table1"]["range"][0], every)[0],
                                   span(lambda r: r["table1"]["range"][1], every)[1]],
            "offset_range": [span(lambda r: r["table2"]["offset_at_matched_fT"][0], every)[0],
                             span(lambda r: r["table2"]["offset_at_matched_fT"][1], every)[1]],
            "agent_ratio_range_controlled": [
                span(lambda r: r["figure4"]["agent_ratio_range_controlled"][0], every)[0],
                span(lambda r: r["figure4"]["agent_ratio_range_controlled"][1], every)[1]],
        },
    }

    # One compact row per delay, for the supplement's table.
    by_delay = []
    for label, r in results.items():
        f4, cfg = r["figure4"], r["config"]
        by_delay.append({
            "label": label, "mtt_h": cfg["mtt_h"], "n": cfg["n"],
            "filament_delayed": cfg["delay_filament"],
            "table1_ratio_range": r["table1"]["range"],
            "offset_at_matched_fT": r["table2"]["offset_at_matched_fT"],
            "spearman_pooled_fT": r["table2"]["spearman_pooled"]["fT_pct"],
            "spearman_pooled_yield_weighted_kill": r["table2"]["spearman_pooled"]["yield_weighted_kill"],
            "divergent_steps_by_endpoint": f4["divergent_steps_by_endpoint"],
            "failing_over_controlled_fold": {a: f4["per_agent"][a]["failing_over_controlled_fold"]
                                             for a in AGENTS},
            "auc48_change_pct_reference_to_top": {
                a: f4["per_agent"][a]["change_pct_reference_to_top"]["0-48h"] for a in AGENTS},
            "first_hour_share_controlled_range": f4["first_hour_share_controlled_range"],
            "first_hour_steps_raising": {a: steps_raising(label, a, "0-1h") for a in AGENTS},
            "first_hour_change_pct_reference_to_top": {
                a: f4["per_agent"][a]["change_pct_reference_to_top"]["0-1h"] for a in AGENTS},
            "table1_max_abs_rel_change_pct": 100 * t1_change(label),
            "agent_ratio_range_controlled": f4["agent_ratio_range_controlled"],
            "max_in_transit_fraction_at_48h_library": r["table2"]["max_in_transit_fraction_at_48h"],
        })

    ktr_lo = THORSTED_KTR_PER_H * (1 - 1.96 * THORSTED_KTR_RSE)
    ktr_hi = THORSTED_KTR_PER_H * (1 + 1.96 * THORSTED_KTR_RSE)
    payload = {
        "status": "re-reading of phase 4's stored regimens under a delay between "
                  "killing and release; no dose re-solved, no parameter of the "
                  "bacterial or release layers changed",
        "endpoint": "plasma endotoxin above the pre-treatment level, uncalibrated "
                    "units; windows are AUCs from the first dose",
        "delay_structure": {
            "equations": "A_1' = r_lysis - (n/MTT) A_1; A_i' = (n/MTT)(A_{i-1} - A_i); "
                         "D' = (n/MTT) A_n + r_filament - k D",
            "source": "Thorsted A et al. Antimicrob Agents Chemother 2020;64:e02070-19, "
                      "PMC7179275, Results and Table 1: one transit compartment for "
                      "release after antibiotic-induced killing of susceptible and "
                      "filamentous cells; growth-associated release immediate",
            "thorsted_transit_rate_per_h": THORSTED_KTR_PER_H,
            "thorsted_transit_rate_rse": THORSTED_KTR_RSE,
            "thorsted_mtt_h": THORSTED_MTT_H,
            "thorsted_mtt_approx_95pct_h": [1 / ktr_hi, 1 / ktr_lo],
            "approx_interval_method": "Wald interval on the transit rate, "
                                      "rate x (1 +/- 1.96 RSE), inverted",
            "mtt_grid_h": list(MTT_GRID_H),
            "n_grid": list(N_GRID),
            "filament_term": "kept immediate: in this model it is release from "
                             "surviving filamenting cells, not release on killing; "
                             "one variant delays it as well",
            "k_ds_clear_per_h": k,
        },
        "headline": headline,
        "by_delay": by_delay,
        "results": {l: {"config": r["config"], "table1": r["table1"],
                        "table2": {kk: v for kk, v in r["table2"].items() if kk != "cost_AUC48"},
                        "figure4": {kk: v for kk, v in r["figure4"].items()
                                    if kk != "arms" or l in ("immediate", PRIMARY)}}
                    for l, r in results.items()},
        "numerical_check": {
            "max_rel_diff_reproduced_vs_stored_48h_AUC":
                max(a["reproduced_rel_diff"] for a in all_arms),
            "reproduction_tolerance": REPRODUCTION_RTOL,
            "max_rel_diff_zero_delay_vs_independent_exact": zero_gap,
            "max_rel_diff_zero_delay_exact_vs_stored_trapezoid": stored_gap,
            "max_rel_diff_table1_ratio_zero_delay_vs_stored": t1_gap,
            "max_abs_diff_pooled_spearman_zero_delay_vs_stored": rho_gap,
            "max_rel_diff_offset_zero_delay_vs_stored": off_gap,
            "max_mass_balance_rel_residual": worst_mb,
            "n_arms": len(all_arms),
            "method": "closed-form integration (matrix exponential per step) of the "
                      "transit chain and the endotoxin compartment over the same "
                      "piecewise-linear release input the ODE solver interpolates",
        },
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {PHASE4.name: hashlib.sha256(PHASE4.read_bytes()).hexdigest()},
        "provenance": provenance(note="Delay between killing and endotoxin release, "
                                      "re-reading phase 4 regimens"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(headline, indent=1), flush=True)
    print(json.dumps(payload["numerical_check"], indent=1), flush=True)


if __name__ == "__main__":
    main()
