"""Joint uncertainty: which assumptions decide the ranking, and which the outcome?

The robustness evidence so far is a collection of one-at-a-time sweeps, some
of them replicates of each other (the susceptibility sweep reproduces one
simulation per exposure target; cytokine parameters cannot reach the
endotoxin endpoint, which is upstream of them). This script replaces that
picture with one joint analysis over the assumptions that can matter, varied
together:

  starting burden            10^5-10^8 CFU/mL
  effector pool              0-1 x default (0 = neutropenic)
  immune recruitment plateau 1-5 x the pool (the admissible band)
  dosing interval            4, 6, 8, 12 h over 48 h
  infusion duration          0.5-3.5 h
  clearance, both agents     0.5-2 x default (profile shape at matched fT)
  meropenem kill rate        0.5-2 x ceftazidime's
  meropenem C50              0.5-2 x ceftazidime's
  release delay              one transit compartment, mean 0-15 h
  meropenem sub-MIC term     0-1 x ceftazidime's coefficient

At each point the two agents are matched on %fT>MIC (40, 60 and 80%)
against ONE fixed MIC reference, the model's effective MIC stored with
Table 1, never re-derived per scenario, which would cancel the potency
difference being varied (see scripts/ds_agent_pd_map.py). Because the
reference is fixed, a matched dose depends on pharmacokinetics alone and is
solved once per agent, target, interval, infusion and clearance; a dose
needing more than 60 g is recorded as unattainable, not forced.

Screening is by Morris elementary effects (4 levels, 30 trajectories, fixed
seed) on three outcomes per target: log10 rho_critical, the
ceftazidime:meropenem lysis-yield ratio at which 48-h release is equal (the
ranking holds under the assumed 3.5-7 range below 3.5 and reverses above 7),
and each agent's 48-h net change (it controls the infection below zero).
mu* ranks influence, sigma flags interaction or curvature. The two most
influential factors for the ranking, and the two for control, are then
mapped jointly at 9 levels with everything else at its default. Every
screening point is also read directly: where the ranking holds, depends,
reverses, and where an agent fails.

Release is linear in the yields and a one-way output of the bacterial
trajectory, so the delay and the meropenem sub-MIC coefficient are applied to
each simulated trajectory exactly, without re-simulating. Simulations are
independent and run in a process pool (--workers, default all cores but
one); every result is keyed by its inputs, so the output does not depend on
the number of workers or the order they finish in.

Writes results/danger_signal/joint_uncertainty.json.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_endpoint_windows import load_phase4_module  # noqa: E402
from scripts.ds_release_delay import delayed_endotoxin  # noqa: E402
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.analysis.danger_signal import filamentation_activity  # noqa: E402
from src.core.parameters import get_default_parameters  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"
KILL_MATCHED = ROOT / "results/danger_signal/kill_matched.json"
OUT = ROOT / "results/danger_signal/joint_uncertainty.json"

TARGETS = (40.0, 60.0, 80.0)
ASSUMED = (3.5, 7.0)
FT_TOL = 0.15                    # %fT>MIC, as ds_agent_pd_map.py
DOSE_BRACKET_MG = (1.0, 60000.0)  # as phase 4's dose_for_target_ft
T_END = 48.0
LEVELS = 4
DELTA = LEVELS / (2.0 * (LEVELS - 1))   # 2/3, Morris's recommended step
N_TRAJECTORIES = 30
SEED = 20261117
MAP_POINTS = 9
#: log10 rho_critical where the ranking reverses at every yield ratio (the
#: meropenem sub-MIC term alone outweighs ceftazidime's release); capped so
#: that elementary effects stay finite. The cap is recorded with each use.
LOG_RSTAR_CAP = 3.0

FACTORS = (
    {"name": "log10_burden", "label": "starting burden, log10 CFU/mL",
     "lo": 5.0, "hi": 8.0, "scale": "linear", "baseline": 7.0},
    {"name": "immune_pool_fraction", "label": "effector pool, x default",
     "lo": 0.0, "hi": 1.0, "scale": "linear", "baseline": 1.0},
    {"name": "immune_recruit_fold", "label": "immune recruitment plateau, x pool",
     "lo": 1.0, "hi": 5.0, "scale": "linear", "baseline": 3.0},
    {"name": "interval_h", "label": "dosing interval, h",
     "levels": (4.0, 6.0, 8.0, 12.0), "baseline": 8.0},
    {"name": "infusion_h", "label": "infusion duration, h",
     "lo": 0.5, "hi": 3.5, "scale": "linear", "baseline": 0.5},
    {"name": "clearance_factor", "label": "clearance, x default",
     "lo": 0.5, "hi": 2.0, "scale": "log", "baseline": 1.0},
    {"name": "mero_kill_factor", "label": "meropenem kill rate, x ceftazidime",
     "lo": 0.5, "hi": 2.0, "scale": "log", "baseline": 1.0},
    {"name": "mero_c50_factor", "label": "meropenem C50, x ceftazidime",
     "lo": 0.5, "hi": 2.0, "scale": "log", "baseline": 1.0},
    {"name": "release_mtt_h", "label": "release delay, mean transit h",
     "lo": 0.0, "hi": 15.0, "scale": "linear", "baseline": 0.0},
    {"name": "mero_filament_fraction", "label": "meropenem sub-MIC term, x ceftazidime",
     "lo": 0.0, "hi": 1.0, "scale": "linear", "baseline": 0.0},
)
NAMES = tuple(f["name"] for f in FACTORS)
#: Applied to the simulated trajectory afterwards; they need no new simulation.
POST = ("release_mtt_h", "mero_filament_fraction")
SIM = tuple(n for n in NAMES if n not in POST)
PK_KEYS = ("interval_h", "infusion_h", "clearance_factor")
BASELINE = {f["name"]: f["baseline"] for f in FACTORS}


def value(factor, u):
    """Factor value at the normalized level u in [0, 1]."""
    if "levels" in factor:
        return float(factor["levels"][int(round(u * (len(factor["levels"]) - 1)))])
    lo, hi = factor["lo"], factor["hi"]
    if factor["scale"] == "log":
        return float(lo * (hi / lo) ** u)
    return float(lo + (hi - lo) * u)


def n_doses(interval_h):
    return int(round(T_END / interval_h))


def verdict(r_star):
    if r_star is None or r_star > ASSUMED[1]:
        return "reverses"
    if r_star < ASSUMED[0]:
        return "holds"
    return "depends_on_yield"


def log_r_star(r_star):
    return LOG_RSTAR_CAP if r_star is None else min(math.log10(r_star), LOG_RSTAR_CAP)


# --------------------------------------------------------------------------
# worker state: one phase 4 module per process, configured per task
# --------------------------------------------------------------------------
_W = {}


def init_worker(p_ref, mic):
    p4 = load_phase4_module()
    p4.MIC_EFFECT = mic
    p4.P_REF = p_ref
    _W.update(p4=p4, mic=mic, pk=p4.get_drug_pk_parameters, ic=dict(p4.IC))


def configure(sc):
    """Point the phase 4 machinery at one scenario's PK and initial state."""
    p4, orig, ic0 = _W["p4"], _W["pk"], _W["ic"]
    f = float(sc["clearance_factor"])
    p4.get_drug_pk_parameters = lambda name, f=f: replace(orig(name), CL=orig(name).CL * f)
    b = 10.0 ** sc["log10_burden"]
    ic = dict(ic0)
    ic.update(B_rep=b, B_pers=b * ic0["B_pers"] / ic0["B_rep"],
              N_eff=ic0["N_eff"] * sc["immune_pool_fraction"])
    p4.IC = ic


def params_for(agent, sc):
    p = get_default_parameters()
    p["immune"].N_eff_0 *= sc["immune_pool_fraction"]
    p["immune"].recruit_fold_max = float(sc["immune_recruit_fold"])
    if agent == "meropenem":
        p["bacteria"].k_kill_max *= sc["mero_kill_factor"]
        p["bacteria"].kill_C50 *= sc["mero_c50_factor"]
    return p


def run(agent, dose, sc):
    return _W["p4"].simulate(agent, float(dose), sc["interval_h"], n_doses(sc["interval_h"]),
                             params=params_for(agent, sc), mic=_W["mic"],
                             inf_min=60.0 * sc["infusion_h"], t_end=T_END)


def dose_task(key):
    """Dose matching one target, for one agent and one PK setting (PK-only)."""
    agent, target, interval, infusion, clearance = key
    sc = dict(BASELINE, interval_h=interval, infusion_h=infusion, clearance_factor=clearance)
    configure(sc)

    def gap(z):
        return run(agent, math.exp(z), sc)["fT"] - target

    lo, hi = (math.log(d) for d in DOSE_BRACKET_MG)
    if gap(hi) < 0:
        return key, None
    return key, float(math.exp(brentq(gap, lo, hi, xtol=1e-7)))


def unit_inputs(sim):
    """Release input per unit lysis yield and per unit filament yield."""
    p4, q = _W["p4"], _W["p4"].P_REF
    to_conc = 1e-6 / q.V_ds_L * q.translocation_fraction * p4.V_FOCUS
    b = np.asarray(sim["B_rep"], float)
    lys = np.asarray(sim["kill"], float) * b * to_conc
    fil = np.asarray(filamentation_activity(sim["C_eff"], _W["mic"], q.filament_window), float) * b * to_conc
    return np.asarray(sim["t"], float), lys, fil


def unit_aucs(t, lys, fil, mtt):
    """Exact 48-h AUC per unit lysis and filament yield; lysis release delayed."""
    k = _W["p4"].P_REF.k_ds_clear
    zero = np.zeros_like(lys)
    a = delayed_endotoxin(t, lys, zero, k, mtt=mtt, n=1, windows=(T_END,))
    b = delayed_endotoxin(t, zero, fil, k, mtt=mtt, n=1, windows=(T_END,))
    return a["auc"][T_END], b["auc"][T_END], max(a["mass_balance_rel_residual"],
                                                 b["mass_balance_rel_residual"])


def matched(agent, dose, target, sc):
    """Simulate at the shared PK-only dose; re-solve only the grid residual."""
    sim = run(agent, dose, sc)
    redosed = False
    if abs(sim["fT"] - target) >= FT_TOL:
        dose = float(brentq(lambda z: run(agent, z, sc)["fT"] - target,
                            0.85 * dose, 1.15 * dose, xtol=0.01))
        sim, redosed = run(agent, dose, sc), True
    if abs(sim["fT"] - target) >= FT_TOL:
        raise RuntimeError(f"{agent} at {target}% missed its exposure target: {sim['fT']:.3f}")
    return sim, dose, redosed


def scenario_task(args):
    """All targets for one simulated setting, read under each requested
    (delay, meropenem sub-MIC fraction) pair."""
    sc, doses, post_pairs = args
    configure(sc)
    p4 = _W["p4"]
    yc, yf = p4.AGENT_YIELD["ceftazidime"], p4.Y_FILAMENT_CAZ
    b0 = 10.0 ** sc["log10_burden"] * (1.0 + _W["ic"]["B_pers"] / _W["ic"]["B_rep"])
    out = {}
    for target in TARGETS:
        d = doses[f"{target:g}"]
        if d["ceftazidime"] is None or d["meropenem"] is None:
            out[f"{target:g}"] = {"attainable": False}
            continue
        arms, inputs, n_redosed = {}, {}, 0
        for agent in ("ceftazidime", "meropenem"):
            sim, dose, redosed = matched(agent, d[agent], target, sc)
            n_redosed += int(redosed)
            net = float(sim["net_log10"])
            arms[agent] = {"dose_mg": dose, "fT_pct": float(sim["fT"]),
                           "fCmax_MIC": float(sim["fCmax_MIC"]), "net_log10_change": net,
                           "controlled": bool(net < 0.0),
                           "killed_over_initial_burden": float(sim["cum_killed_cfu_per_mL"] / b0)}
            inputs[agent] = unit_inputs(sim)
        reads, worst_mb = {}, 0.0
        for mtt, frac in post_pairs:
            alc, afc, mb1 = unit_aucs(*inputs["ceftazidime"], mtt)
            alm, afm, mb2 = unit_aucs(*inputs["meropenem"], mtt)
            worst_mb = max(worst_mb, mb1, mb2)
            den = yc * alc + yf * afc - frac * yf * afm
            r_star = yc * alm / den if den > 0 else None
            reads[post_label((mtt, frac))] = {"r_star": r_star, "verdict": verdict(r_star),
                                          "log10_r_star_capped": log_r_star(r_star)}
        out[f"{target:g}"] = {"attainable": True, "ceftazidime": arms["ceftazidime"],
                              "meropenem": arms["meropenem"], "reads": reads,
                              "n_redosed": n_redosed, "mass_balance_rel_residual": worst_mb}
    return out


# --------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------
def morris_design(rng):
    """Morris trajectories on a LEVELS-level grid: k+1 points, one factor per step."""
    k = len(FACTORS)
    grid = np.arange(LEVELS) / (LEVELS - 1)
    starts = grid[grid + DELTA <= 1.0 + 1e-12]
    trajectories = []
    for _ in range(N_TRAJECTORIES):
        base = rng.choice(starts, size=k)
        sign = rng.choice([-1.0, 1.0], size=k)
        u = np.where(sign > 0, base, base + DELTA)
        order = rng.permutation(k)
        points, steps = [u.copy()], []
        for i in order:
            u = u.copy()
            u[i] += sign[i] * DELTA
            points.append(u)
            steps.append({"factor": NAMES[i], "delta_u": float(sign[i] * DELTA)})
        trajectories.append({"u": [p.round(12).tolist() for p in points], "steps": steps})
    return trajectories


def scenario_from_u(u):
    return {f["name"]: value(f, ui) for f, ui in zip(FACTORS, u)}


def sim_key(sc):
    return tuple(round(float(sc[n]), 12) for n in SIM)


def pk_key(sc):
    return tuple(round(float(sc[n]), 12) for n in PK_KEYS)


def evaluate(scenarios, pool, mapper, log):
    """Simulate every distinct setting once and read each scenario from it."""
    pks = sorted({pk_key(sc) for sc in scenarios})
    dose_keys = [(a, t) + pk for pk in pks for a in ("ceftazidime", "meropenem") for t in TARGETS]
    missing = [k for k in dose_keys if k not in pool["doses"]]
    t0 = time.time()
    for key, dose in mapper(dose_task, missing):
        pool["doses"][key] = dose
    log(f"  {len(missing)} dose solves in {time.time() - t0:.0f} s")

    groups = {}
    for sc in scenarios:
        groups.setdefault(sim_key(sc), (sc, set()))[1].add(post_pair(sc))
    tasks = []
    for key, (sc, pairs) in groups.items():
        need = sorted(pairs - pool["read_pairs"].get(key, set()))
        if not need:
            continue
        doses = {f"{t:g}": {a: pool["doses"][(a, t) + pk_key(sc)] for a in ("ceftazidime", "meropenem")}
                 for t in TARGETS}
        tasks.append((key, (sc, doses, need)))
    t0 = time.time()
    results = list(mapper(scenario_task, [t[1] for t in tasks]))
    for (key, (_, _, need)), res in zip(tasks, results):
        merged = pool["sims"].setdefault(key, {})
        for t, v in res.items():
            if t in merged and merged[t]["attainable"]:
                merged[t]["reads"].update(v["reads"])
            else:
                merged[t] = v
        pool["read_pairs"].setdefault(key, set()).update(need)
    log(f"  {len(tasks)} settings simulated in {time.time() - t0:.0f} s")
    return [read(sc, pool) for sc in scenarios]


def post_pair(sc):
    return (round(float(sc["release_mtt_h"]), 12), round(float(sc["mero_filament_fraction"]), 12))


def post_label(pair):
    return f"{pair[0]!r}|{pair[1]!r}"


def read(sc, pool):
    sims = pool["sims"][sim_key(sc)]
    post = post_label(post_pair(sc))
    out = {"factors": sc, "targets": {}}
    for t in TARGETS:
        s = sims[f"{t:g}"]
        if not s["attainable"]:
            out["targets"][f"{t:g}"] = {"attainable": False}
            continue
        r = s["reads"][post]
        out["targets"][f"{t:g}"] = {
            "attainable": True, "r_star": r["r_star"], "verdict": r["verdict"],
            "log10_r_star_capped": r["log10_r_star_capped"],
            "ceftazidime": s["ceftazidime"], "meropenem": s["meropenem"],
            "net_difference_mero_minus_caz": s["meropenem"]["net_log10_change"]
            - s["ceftazidime"]["net_log10_change"],
            "mass_balance_rel_residual": s["mass_balance_rel_residual"],
            "n_redosed": s["n_redosed"],
        }
    return out


OUTCOMES = {
    "log10_r_star": lambda r: r["log10_r_star_capped"],
    "net_ceftazidime": lambda r: r["ceftazidime"]["net_log10_change"],
    "net_meropenem": lambda r: r["meropenem"]["net_log10_change"],
}


def elementary_effects(trajectories, evaluated):
    """mu*, mu and sigma per factor, outcome and target, over usable steps."""
    ee = {o: {t: {n: [] for n in NAMES} for t in map("{:g}".format, TARGETS)} for o in OUTCOMES}
    skipped = 0
    for traj, pts in zip(trajectories, evaluated):
        for step, a, b in zip(traj["steps"], pts[:-1], pts[1:]):
            for t in map("{:g}".format, TARGETS):
                ra, rb = a["targets"][t], b["targets"][t]
                if not (ra["attainable"] and rb["attainable"]):
                    skipped += 1
                    continue
                for o, fn in OUTCOMES.items():
                    ee[o][t][step["factor"]].append((fn(rb) - fn(ra)) / step["delta_u"])
    stats = {o: {t: {n: ({"mu_star": float(np.mean(np.abs(v))), "mu": float(np.mean(v)),
                          "sigma": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
                         if v else None)
                     for n, v in per.items()}
                 for t, per in by_t.items()}
             for o, by_t in ee.items()}
    return stats, skipped


def top_two(stats, outcome, target):
    s = stats[outcome][target]
    ranked = sorted((n for n in NAMES if s[n]), key=lambda n: -s[n]["mu_star"])
    return ranked[:2]


def map_scenarios(pair):
    axes = []
    for name in pair:
        f = FACTORS[NAMES.index(name)]
        us = (np.linspace(0, 1, len(f["levels"])) if "levels" in f
              else np.linspace(0, 1, MAP_POINTS))
        axes.append([value(f, u) for u in us])
    return axes, [dict(BASELINE, **{pair[0]: x, pair[1]: y}) for y in axes[1] for x in axes[0]]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731

    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    mic = phase4["meta"]["mic_effect_site_mgL"]
    family = [e for e in calibrate_family() if e["feasible"]]
    p_ref = calibrate_tnf(min(family, key=lambda e: e["cost"]))["params"]
    init_worker(p_ref, mic)          # the main process reads results with it too

    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                       initargs=(p_ref, mic))

        def mapper(fn, items):
            return executor.map(fn, items, chunksize=1)
    else:
        def mapper(fn, items):
            return map(fn, items)

    pool = {"doses": {}, "sims": {}, "read_pairs": {}}
    t_start = time.time()
    try:
        # ---- the baseline, against Table 1 and kill_matched.json ---------
        log("baseline")
        base = [dict(BASELINE)]
        base_eval = evaluate(base, pool, mapper, log)[0]
        km = json.loads(KILL_MATCHED.read_text(encoding="utf-8"))
        stored = {r["target_fT_pct"]: r for r in phase4["analysis3_matched_exposure"]["rows"]}
        configure(BASELINE)
        table1_check = {}
        for t in TARGETS:
            row = stored[t]
            sims = {a: run(a, row["arms"][a]["dose_mg"], BASELINE) for a in ("ceftazidime", "meropenem")}
            alc, afc, _ = unit_aucs(*unit_inputs(sims["ceftazidime"]), 0.0)
            alm, _, _ = unit_aucs(*unit_inputs(sims["meropenem"]), 0.0)
            yc, yf = _W["p4"].AGENT_YIELD["ceftazidime"], _W["p4"].Y_FILAMENT_CAZ
            r_star = yc * alm / (yc * alc + yf * afc)
            want = 3.5 / km["table1"][f"{t:g}"]["decomposition"]["3.5"]["ratio"]
            table1_check[f"{t:g}"] = {
                "r_star_at_stored_doses": r_star,
                "rel_diff_vs_kill_matched": abs(r_star / want - 1.0),
                "solved_over_stored_dose": {
                    a: pool["doses"][(a, t) + pk_key(BASELINE)] / row["arms"][a]["dose_mg"]
                    for a in ("ceftazidime", "meropenem")},
                "r_star_at_solved_doses": base_eval["targets"][f"{t:g}"]["r_star"],
            }

        # ---- screening -----------------------------------------------------
        rng = np.random.default_rng(SEED)
        trajectories = morris_design(rng)
        flat = [scenario_from_u(u) for tr in trajectories for u in tr["u"]]
        log(f"screening: {len(flat)} points")
        evaluated_flat = evaluate(flat, pool, mapper, log)
        k1 = len(FACTORS) + 1
        evaluated = [evaluated_flat[i * k1:(i + 1) * k1] for i in range(N_TRAJECTORIES)]
        stats, skipped = elementary_effects(trajectories, evaluated)

        # ---- the two interaction maps ----------------------------------------
        # The ranking map takes the two factors that move rho_critical most.
        # The control map takes the two that move CEFTAZIDIME's outcome most:
        # its PD is the adopted one, so they are the host and regimen factors,
        # whereas meropenem's outcome is dominated by its own PD, which the
        # ranking map already covers.
        pair_rank = top_two(stats, "log10_r_star", "40")
        pair_ctl = top_two(stats, "net_ceftazidime", "40")
        maps = {}
        for label, pair in (("ranking", pair_rank), ("control", pair_ctl)):
            axes, scs = map_scenarios(pair)
            log(f"map {label}: {pair[0]} x {pair[1]} ({len(scs)} points)")
            maps[label] = {"factors": pair, "axes": axes, "points": evaluate(scs, pool, mapper, log)}
    finally:
        if executor is not None:
            executor.shutdown()
    elapsed = time.time() - t_start

    # ---- reading every screening point directly ---------------------------
    def cells(points):
        return [(t, p["factors"], p["targets"][t]) for p in points for t in p["targets"]]

    every = cells(evaluated_flat)
    att = [c for c in every if c[2]["attainable"]]
    verdicts = {v: sum(1 for c in att if c[2]["verdict"] == v)
                for v in ("holds", "depends_on_yield", "reverses")}
    not_holding = [c for c in att if c[2]["verdict"] != "holds"]
    fails = {a: [c for c in att if not c[2][a]["controlled"]] for a in ("ceftazidime", "meropenem")}

    def factor_ranges(group):
        return {n: [min(c[1][n] for c in group), max(c[1][n] for c in group)] for n in NAMES} if group else None

    def share_by_level(group, base_group, name):
        f = FACTORS[NAMES.index(name)]
        vals = sorted({round(c[1][name], 9) for c in base_group})
        return {f"{v:g}": (sum(1 for c in group if round(c[1][name], 9) == v)
                           / max(1, sum(1 for c in base_group if round(c[1][name], 9) == v)))
                for v in vals} if f else None

    def disadvantage(c):
        return c[2]["net_difference_mero_minus_caz"]

    def killed_ratio(c):
        return (c[2]["meropenem"]["killed_over_initial_burden"]
                / c[2]["ceftazidime"]["killed_over_initial_burden"])

    def outcome_class(c):
        return (("ceftazidime_controls" if c[2]["ceftazidime"]["controlled"] else "ceftazidime_fails")
                + "|" + ("meropenem_controls" if c[2]["meropenem"]["controlled"] else "meropenem_fails"))

    def counts(group, key):
        out = {}
        for c in group:
            out[key(c)] = out.get(key(c), 0) + 1
        return dict(sorted(out.items()))

    threshold = min(disadvantage(c) for c in not_holding) if not_holding else None
    below = [c for c in att if disadvantage(c) < threshold] if not_holding else att
    no_sub = [c for c in not_holding if c[1]["mero_filament_fraction"] == 0.0]
    both_ok = [c for c in not_holding if c[2]["ceftazidime"]["controlled"]
               and c[2]["meropenem"]["controlled"]]
    level = lambda name: (lambda c: f"{round(c[1][name], 6):g}")  # noqa: E731
    ctl_map = maps["control"]
    peaks = {f"{p['factors'][ctl_map['factors'][1]]:g}": {
        "dose_mg": p["targets"]["40"]["ceftazidime"]["dose_mg"],
        "fCmax_MIC": p["targets"]["40"]["ceftazidime"]["fCmax_MIC"],
        "net_log10_change": p["targets"]["40"]["ceftazidime"]["net_log10_change"]}
        for p in ctl_map["points"]
        if p["factors"][ctl_map["factors"][0]] == BASELINE[ctl_map["factors"][0]]
        and p["targets"]["40"]["attainable"]} if ctl_map["factors"][1] == "interval_h" else None

    headline = {
        "n_points": len(flat), "n_cell_targets": len(every), "n_attainable": len(att),
        "n_unattainable": len(every) - len(att),
        "verdict_counts": verdicts,
        "n_not_holding": len(not_holding),
        "not_holding_min_mero_net_disadvantage": (min(c[2]["net_difference_mero_minus_caz"]
                                                      for c in not_holding) if not_holding else None),
        "not_holding_with_both_controlled": sum(1 for c in not_holding
                                                if c[2]["ceftazidime"]["controlled"]
                                                and c[2]["meropenem"]["controlled"]),
        "not_holding_factor_ranges": factor_ranges(not_holding),
        "not_holding_share_by_level": {n: share_by_level(not_holding, att, n) for n in NAMES},
        "n_failing": {a: len(v) for a, v in fails.items()},
        "failing_share_by_level": {a: {n: share_by_level(v, att, n) for n in NAMES}
                                   for a, v in fails.items()},
        "holding_r_star_max": max((c[2]["r_star"] for c in att if c[2]["verdict"] == "holds"),
                                  default=None),
        "unattainable_by_target": counts([c for c in every if not c[2]["attainable"]], lambda c: c[0]),
        "outcome_class_counts": counts(att, outcome_class),
        "not_holding_by_outcome_class": counts(not_holding, outcome_class),
        "not_holding_min_killed_ratio_mero_over_caz": (min(killed_ratio(c) for c in not_holding)
                                                       if not_holding else None),
        "holding_max_mero_net_disadvantage": max(disadvantage(c) for c in att if c[2]["verdict"] == "holds"),
        # Below the smallest meropenem disadvantage at which the ranking was
        # lost anywhere, how many comparisons there are and how close they come.
        "comparable_outcomes": {"threshold_log10": threshold, "n": len(below),
                                "all_hold": all(c[2]["verdict"] == "holds" for c in below),
                                "max_r_star": max(c[2]["r_star"] for c in below)},
        "not_holding_without_mero_subMIC": {
            "n": len(no_sub),
            "min_mero_net_disadvantage": min(disadvantage(c) for c in no_sub) if no_sub else None,
            "min_killed_ratio_mero_over_caz": min(killed_ratio(c) for c in no_sub) if no_sub else None},
        "not_holding_both_controlled": {
            "n": len(both_ok),
            "min_mero_net_disadvantage": min(disadvantage(c) for c in both_ok) if both_ok else None,
            "min_killed_ratio_mero_over_caz": min(killed_ratio(c) for c in both_ok) if both_ok else None,
            "n_with_mero_subMIC": sum(1 for c in both_ok if c[1]["mero_filament_fraction"] > 0)},
        "failing_by_target": {a: counts(v, lambda c: c[0]) for a, v in fails.items()},
        "ceftazidime_failing_by_level": {
            n: {"failing": counts(fails["ceftazidime"], level(n)), "all": counts(att, level(n))}
            for n in ("immune_pool_fraction", "interval_h", "log10_burden")},
        "meropenem_failing_by_level": {
            n: {"failing": counts(fails["meropenem"], level(n)), "all": counts(att, level(n))}
            for n in ("mero_c50_factor", "mero_kill_factor")},
        "ceftazidime_at_40_by_interval_default_host": peaks,
        "maps_summary": {lab: {f"{t:g}": {"verdicts": counts([p["targets"][f"{t:g}"] for p in m["points"]
                                                               if p["targets"][f"{t:g}"]["attainable"]],
                                                              lambda r: r["verdict"]),
                                         "ceftazidime_fails": sum(1 for p in m["points"]
                                                                  if p["targets"][f"{t:g}"]["attainable"]
                                                                  and not p["targets"][f"{t:g}"]["ceftazidime"]["controlled"]),
                                         "meropenem_fails": sum(1 for p in m["points"]
                                                                if p["targets"][f"{t:g}"]["attainable"]
                                                                and not p["targets"][f"{t:g}"]["meropenem"]["controlled"]),
                                         "n": len(m["points"])}
                               for t in TARGETS}
                         for lab, m in maps.items()},
        "ranking_top_two_at_40": pair_rank,
        "control_top_two_at_40": pair_ctl,
        "mu_star_rank_log10_r_star_at_40": sorted(
            ((n, stats["log10_r_star"]["40"][n]["mu_star"]) for n in NAMES if stats["log10_r_star"]["40"][n]),
            key=lambda x: -x[1]),
    }
    numerical_check = {
        "table1_r_star_max_rel_diff_vs_kill_matched": max(v["rel_diff_vs_kill_matched"]
                                                          for v in table1_check.values()),
        "baseline_solved_over_stored_dose_range": [
            min(x for v in table1_check.values() for x in v["solved_over_stored_dose"].values()),
            max(x for v in table1_check.values() for x in v["solved_over_stored_dose"].values())],
        "max_mass_balance_rel_residual": max(c[2]["mass_balance_rel_residual"] for c in att),
        "n_redosed": sum(c[2]["n_redosed"] for c in att),
        "morris_steps_skipped_unattainable": skipped,
    }
    payload = {
        "design": {"factors": [{k: v for k, v in f.items()} for f in FACTORS],
                   "targets_fT_pct": list(TARGETS), "mic_reference_mgL": mic,
                   "levels": LEVELS, "delta": DELTA, "n_trajectories": N_TRAJECTORIES,
                   "seed": SEED, "map_points": MAP_POINTS, "assumed_yield_range": list(ASSUMED),
                   "log10_r_star_cap": LOG_RSTAR_CAP, "dose_bracket_mg": list(DOSE_BRACKET_MG),
                   "post_processed_factors": list(POST)},
        "baseline": base_eval, "table1_check": table1_check,
        "morris": {"trajectories": trajectories, "points": evaluated, "effects": stats},
        "maps": maps,
        "headline": headline,
        "numerical_check": numerical_check,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (PHASE4, KILL_MATCHED)},
        "provenance": provenance(note="Joint uncertainty: Morris screening and interaction maps"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    log(json.dumps({"headline": headline, "numerical_check": numerical_check}, indent=1))
    # Not written to the file: neither affects a result, and the file should
    # reproduce byte for byte whatever the machine or pool size.
    log(f"elapsed {elapsed:.0f} s on {args.workers} worker(s)")


if __name__ == "__main__":
    main()
