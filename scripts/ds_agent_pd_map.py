"""Where does the endotoxin ranking reverse when the two agents' PD differ?

Every comparison in the paper gives ceftazidime and meropenem one kill
function. That isolates the release yield, but it also assumes the agents
kill alike at equal %fT>MIC, which is the assumption most favourable to
reading any release difference as a yield difference. This script lets the
agents differ in maximum kill rate, C50, Hill exponent and persister
switching, and maps where the ranking survives.

Fixed MIC reference. Exposure is %fT>MIC against 1 mg/L for both agents:
measured for meropenem against the calibration isolate (P. aeruginosa
ATCC-27853), assumed for ceftazidime. The same value is the denominator of the
sub-MIC window. It is NOT re-derived from each scenario's own stasis
concentration, as phase 4 does: that would make C50/MIC identical by
construction and cancel exactly the potency difference being tested (the
invariance scripts/ds_endpoint_windows.py documents). With the reference
fixed, the matched doses depend on pharmacokinetics alone, so they are solved
once per target and shared by every scenario.

What varies. Meropenem keeps a PD parameter set; ceftazidime's is the same set
with k_kill_max and kill_C50 multiplied by factors from 1/4 to 4 (a 9 x 9
map), and, in separate sweeps at equal kill rate and C50, its Hill exponent
(1-8) and persister switching rate (0-16 x). A mirror map holds ceftazidime at
the adopted set and scales meropenem's kill rate and C50 over the same range,
because the two are not symmetric: what decides release is whether each agent
controls the infection or fails, and that depends on absolute potency, not
only on the ratio. The map is repeated coarsely
(5 x 5) on each of the eight meropenem PD sets of the multi-arm broth refit
(results/danger_signal/multiarm_refit.json), so meropenem's own uncertainty
enters the envelope; one of the eight carries the adopted kill parameters and
must reproduce the matching cells of the main map.

What is reported. Release is linear in the yields, so each cell is solved
once at unit yields and the ranking follows in closed form: r*, the
ceftazidime:meropenem lysis-yield ratio at which 48-h release is equal
(tipping_ratio, as in scripts/ds_joint_robustness.py). Under the assumed
3.5-7 range the ranking holds if r* < 3.5, reverses if r* > 7, and depends on
where in the range the true ratio lies in between. Both agents' 48-h net
change and cumulative killing are reported with it, because a PD difference
changes the bacterial outcome as well as the release. Endotoxin AUCs are
integrated exactly (ds_endpoint_windows.exact_window_aucs); the shared-PD
cell is checked against the trapezoid values of ds_joint_robustness.py.

Data check. Tsuji 2003 is the only source with paired kill data for these two
agents: P. aeruginosa SR24, viable counts at 6 h at 0.1, 1 and 10 x each
agent's measured MIC, with an untreated control. Three single-time points per
agent do not identify a kill rate, C50 and Hill exponent (the per-agent fits
run to their bounds, recorded below), so the data are used as a check, not a
fit: each cell's between-agent 6-h differences are predicted in a broth
model, with growth fixed by Tsuji's control, and compared with the observed
differences. A cell is "closer to Tsuji" when that misfit is smaller than the
shared kill function's. Viable counts cannot tell killing from filamentation,
which suppresses colony counts without killing, so ceftazidime's apparent
sub-MIC activity in those data may not be killing at all.

Writes results/danger_signal/agent_pd_map.json.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_endpoint_windows import exact_window_aucs  # noqa: E402
from scripts.ds_joint_robustness import tipping_ratio  # noqa: E402
from scripts.ds_multiarm_downstream import load_phase4  # noqa: E402
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.analysis.danger_signal import filamentation_activity  # noqa: E402
from src.core.parameters import get_default_parameters  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

REFIT = ROOT / "results/danger_signal/multiarm_refit.json"
JOINT = ROOT / "results/danger_signal/joint_robustness.json"
REGISTRY = ROOT / "data/danger_signal/registry.json"
OUT = ROOT / "results/danger_signal/agent_pd_map.json"

MIC_REF = 1.0
TARGETS = (40.0, 60.0, 80.0, 95.0)
FACTORS = tuple(float(2.0 ** e) for e in np.linspace(-2.0, 2.0, 9))
COARSE = tuple(float(2.0 ** e) for e in np.linspace(-2.0, 2.0, 5))
HILL_SWEEP = (1.0, 2.0, 4.0, 8.0)
KPERS_FACTORS = (0.0, 0.25, 1.0, 4.0, 16.0)
ASSUMED = (3.5, 7.0)
FT_TOL = 0.15                    # %fT>MIC, as ds_joint_robustness.py
PD_KEYS = ("k_kill_max", "kill_C50", "kill_hill", "k_pers")
REACTIVATION = 0.05              # per hour, fixed in pd_model.py
TSUJI_LEVELS = ("0.1x", "1x", "10x")
TSUJI_XMIC = (0.1, 1.0, 10.0)
TSUJI_LOG10_INOCULUM = 5.0       # "1 x 10^5 cfu/mL" (Tsuji 2003, Discussion)


# --------------------------------------------------------------------------
# broth model for the Tsuji check
# --------------------------------------------------------------------------
def broth_net_change(pd, conc_xmic, growth, t_end=6.0,
                     log10_inoculum=TSUJI_LOG10_INOCULUM, persisters=100.0):
    """Net log10 change after t_end h at a constant concentration (x MIC).

    The immune-free broth structure of ds_multiarm_refit.predict: replicating
    and persister pools, logistic growth to the model's B_max, Hill killing of
    the replicating pool only, switching at k_pers and reactivation at 0.05/h.
    With MIC_REF = 1 mg/L, kill_C50 in mg/L is also C50 in multiples of MIC.
    """
    b_max = get_default_parameters()["bacteria"].B_max
    c = float(conc_xmic) * MIC_REF
    k, c50, h, kp = (float(pd[key]) for key in PD_KEYS)
    kill = k * c ** h / (c ** h + c50 ** h) if c > 0 else 0.0
    total0 = 10.0 ** log10_inoculum

    def rhs(_, y):
        rep, pers = y
        transfer = kp * rep - REACTIVATION * pers
        return [growth * (1.0 - (rep + pers) / b_max) * rep - kill * rep - transfer,
                transfer]

    sol = solve_ivp(rhs, (0.0, t_end), [total0 - persisters, persisters],
                    method="LSODA", rtol=1e-9, atol=1e-6, max_step=0.25)
    if not sol.success or sol.y[:, -1].min() < -1e-6:
        raise RuntimeError("broth integration failed")
    return float(np.log10(max(sol.y[:, -1].sum(), 1e-10)) - log10_inoculum)


def tsuji_data():
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = reg["datasets"] if isinstance(reg, dict) and "datasets" in reg else \
        next(v for v in reg.values() if isinstance(v, list))
    e = next(x for x in entries if x.get("id") == "tsuji2003_sr24_release")
    cc = e["constant_concentration_6h"]
    obs = {a: [cc[a][lvl][0] for lvl in TSUJI_LEVELS] for a in ("ceftazidime", "meropenem")}
    return {"control_6h": cc["control"][0], "mic_mg_L": e["mic_mg_L"], "net_change_6h": obs,
            "difference_caz_minus_mero": [c - m for c, m in zip(obs["ceftazidime"], obs["meropenem"])]}


def tsuji_digest(data):
    """Hash of exactly the Tsuji values the analysis reads."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def tsuji_growth(control_6h):
    nodrug = {"k_kill_max": 1.0, "kill_C50": 1.0, "kill_hill": 4.0, "k_pers": 0.01}
    return float(brentq(lambda g: broth_net_change(nodrug, 0.0, g) - control_6h, 0.05, 5.0,
                        xtol=1e-10))


def tsuji_identifiability(data, growth, k_pers):
    """Try to fit each agent's PD to its three 6-h points; report what binds."""
    low, high = np.log([0.05, 0.005, 0.3]), np.log([20.0, 50.0, 30.0])
    names = ("k_kill_max", "kill_C50_xMIC", "kill_hill")
    out = {}
    for agent, y in data["net_change_6h"].items():
        best = None
        for start in [(k, c, h) for k in (1.0, 4.0) for c in (0.3, 1.0, 3.0) for h in (1.0, 4.0)]:
            def resid(z):
                pd = dict(zip(PD_KEYS, [*np.exp(z), k_pers]))
                return [broth_net_change(pd, x, growth) - o for x, o in zip(TSUJI_XMIC, y)]
            s = least_squares(resid, np.log(start), bounds=(low, high), diff_step=1e-4)
            if best is None or s.cost < best.cost:
                best = s
        out[agent] = {
            "fitted": dict(zip(names, map(float, np.exp(best.x)))),
            "residuals_log10": [float(r) for r in best.fun],
            "bounds_reached": [names[i] for i, z in enumerate(best.x)
                               if min(z - low[i], high[i] - z) < 0.01],
        }
    return out


def tsuji_misfit(caz_pd, mero_pd, data, growth):
    """RMS misfit of predicted CAZ-minus-MERO 6-h differences against Tsuji's."""
    pred = [broth_net_change(caz_pd, x, growth) - broth_net_change(mero_pd, x, growth)
            for x in TSUJI_XMIC]
    err = np.array(pred) - np.array(data["difference_caz_minus_mero"])
    return float(np.sqrt(np.mean(err ** 2))), [float(p) for p in pred]


# --------------------------------------------------------------------------
# in-host simulation at matched exposure
# --------------------------------------------------------------------------
def params_for(pd):
    p = get_default_parameters()
    for key in PD_KEYS:
        setattr(p["bacteria"], key, float(pd[key]))
    return p


def simulate_matched(p4, agent, dose, target, pd):
    """Simulate at the shared matched dose; re-solve only the grid residual.

    The adaptive bacterial grid moves with the PD parameters and with it the
    crossing interpolation of %fT>MIC, so a dose matched once can miss its
    target by a few hundredths. That residual is corrected by re-solving the
    dose for this simulation, never by moving the MIC reference.
    """
    params = params_for(pd)

    def run(d):
        return p4.simulate(agent, d, 8.0, 6, params=params, mic=MIC_REF)

    sim = run(dose)
    if abs(sim["fT"] - target) >= FT_TOL:
        dose = float(brentq(lambda z: run(z)["fT"] - target, 0.85 * dose, 1.15 * dose, xtol=0.1))
        sim = run(dose)
    if abs(sim["fT"] - target) >= FT_TOL:
        raise RuntimeError(f"{agent} at {target}% missed its exposure target: {sim['fT']:.3f}")
    return sim, dose


def unit_release(p4, sim):
    """Exact 48-h endotoxin AUC per unit lysis yield and per unit filament yield."""
    q = p4.P_REF
    to_conc = 1e-6 / q.V_ds_L * q.translocation_fraction
    b = np.asarray(sim["B_rep"], float)
    lysis = np.asarray(sim["kill"], float) * b * p4.V_FOCUS * to_conc
    phi = filamentation_activity(sim["C_eff"], sim["mic"], q.filament_window)
    filament = np.asarray(phi, float) * b * p4.V_FOCUS * to_conc
    k = q.k_ds_clear
    return (exact_window_aucs(sim["t"], lysis, k, (48.0,))[48.0],
            exact_window_aucs(sim["t"], filament, k, (48.0,))[48.0])


def outcome(sim, dose, target):
    return {"dose_mg_q8h": dose, "achieved_fT_pct": float(sim["fT"]),
            "net_log10_change": float(sim["net_log10"]),
            "controlled": bool(sim["net_log10"] < 0.0),
            "cum_killed_cfu_per_mL": float(sim["cum_killed_cfu_per_mL"])}


def verdict(r_star):
    if r_star < ASSUMED[0]:
        return "holds"
    if r_star > ASSUMED[1]:
        return "reverses"
    return "depends_on_yield"


class Runner:
    def __init__(self, p4, doses, tsuji, growth):
        self.p4, self.doses, self.tsuji, self.growth = p4, doses, tsuji, growth
        self.mero_cache = {}
        self.n_redosed = 0

    def mero(self, mero_pd):
        key = tuple(round(float(mero_pd[k]), 12) for k in PD_KEYS)
        if key not in self.mero_cache:
            runs = {}
            for t in TARGETS:
                sim, dose = simulate_matched(self.p4, "meropenem", self.doses[t]["meropenem"], t, mero_pd)
                self.n_redosed += int(dose != self.doses[t]["meropenem"])
                lys, _ = unit_release(self.p4, sim)
                runs[t] = {"lysis_unit_AUC": lys, **outcome(sim, dose, t)}
            self.mero_cache[key] = runs
        return self.mero_cache[key]

    def cell(self, caz_pd, mero_pd, label):
        mero = self.mero(mero_pd)
        y_caz, y_fil = self.p4.AGENT_YIELD["ceftazidime"], self.p4.Y_FILAMENT_CAZ
        misfit, pred = tsuji_misfit(caz_pd, mero_pd, self.tsuji, self.growth)
        per_target = {}
        for t in TARGETS:
            sim, dose = simulate_matched(self.p4, "ceftazidime", self.doses[t]["ceftazidime"], t, caz_pd)
            self.n_redosed += int(dose != self.doses[t]["ceftazidime"])
            lys, fil = unit_release(self.p4, sim)
            m = mero[t]
            r_star = tipping_ratio(lys, fil, m["lysis_unit_AUC"], y_caz, y_fil)
            caz = outcome(sim, dose, t)
            per_target[f"{t:g}"] = {
                "r_star": r_star,
                "r_star_no_subMIC_term": tipping_ratio(lys, fil, m["lysis_unit_AUC"], y_caz, 0.0),
                "verdict": verdict(r_star),
                "release_ratio_at_assumed_yields": [ASSUMED[0] / r_star, ASSUMED[1] / r_star],
                "ceftazidime": caz,
                "meropenem": {k: v for k, v in m.items() if k != "lysis_unit_AUC"},
                "killed_ratio_caz_over_mero": caz["cum_killed_cfu_per_mL"] / m["cum_killed_cfu_per_mL"],
                "unit_release_AUC": {"caz_lysis": lys, "caz_filament": fil,
                                     "mero_lysis": m["lysis_unit_AUC"]},
            }
        return {"label": label, "ceftazidime_pd": {k: float(caz_pd[k]) for k in PD_KEYS},
                "meropenem_pd": {k: float(mero_pd[k]) for k in PD_KEYS},
                "tsuji_difference_misfit_log10": misfit,
                "tsuji_predicted_difference_caz_minus_mero": pred,
                "targets": per_target}


#: Per-process state for the cell pool: each worker builds its own Runner.
_W = {}


def init_worker(p_ref, doses, tsuji, growth):
    p4 = load_phase4()
    p4.MIC_EFFECT = MIC_REF
    p4.P_REF = p_ref
    _W["run"] = Runner(p4, doses, tsuji, growth)


def cell_task(args):
    """One map cell. Depends only on its arguments (the meropenem cache only
    avoids repeat work), so the pool reproduces the serial output exactly."""
    return _W["run"].cell(*args)


def n_redosed(cells, doses):
    """Simulations whose dose had to be re-solved off the shared matched dose
    (meropenem counted once per distinct PD, as it is simulated once)."""
    caz = sum(1 for c in cells for t in TARGETS
              if c["targets"][f"{t:g}"]["ceftazidime"]["dose_mg_q8h"] != doses[t]["ceftazidime"])
    mero = {(tuple(sorted(c["meropenem_pd"].items())), t) for c in cells for t in TARGETS
            if c["targets"][f"{t:g}"]["meropenem"]["dose_mg_q8h"] != doses[t]["meropenem"]}
    return caz + len(mero)


def scaled(base, kill=1.0, c50=1.0, hill=None, kpers=1.0):
    return {"k_kill_max": base["k_kill_max"] * kill, "kill_C50": base["kill_C50"] * c50,
            "kill_hill": base["kill_hill"] if hill is None else hill,
            "k_pers": base["k_pers"] * kpers}


def summarise_map(cells, shared_misfit):
    by_target = {}
    for t in TARGETS:
        key = f"{t:g}"
        rs = [c["targets"][key]["r_star"] for c in cells]
        counts = {v: sum(1 for c in cells if c["targets"][key]["verdict"] == v)
                  for v in ("holds", "depends_on_yield", "reverses")}
        rev = [c for c in cells if c["targets"][key]["verdict"] != "holds"]
        by_target[key] = {
            "r_star_range": [min(rs), max(rs)], "verdict_counts": counts,
            "not_holding_with_meropenem_failing": sum(
                1 for c in rev if not c["targets"][key]["meropenem"]["controlled"]),
            "not_holding_with_ceftazidime_failing": sum(
                1 for c in rev if not c["targets"][key]["ceftazidime"]["controlled"]),
            "not_holding_with_both_controlled": sum(
                1 for c in rev if c["targets"][key]["meropenem"]["controlled"]
                and c["targets"][key]["ceftazidime"]["controlled"]),
        }
    closer = [c for c in cells if c["tsuji_difference_misfit_log10"] < shared_misfit]
    closer_r = [c["targets"][f"{t:g}"]["r_star"] for c in closer for t in TARGETS]
    return {"n_cells": len(cells), "by_target": by_target,
            "n_holding_at_every_target": sum(
                1 for c in cells if all(c["targets"][f"{t:g}"]["verdict"] == "holds" for t in TARGETS)),
            "tsuji_closer_than_shared": {
                "n_cells": len(closer),
                "r_star_range": [min(closer_r), max(closer_r)] if closer else None,
                "n_not_holding_any_target": sum(
                    1 for c in closer if any(c["targets"][f"{t:g}"]["verdict"] != "holds" for t in TARGETS)),
                "best": min(cells, key=lambda c: c["tsuji_difference_misfit_log10"])["label"],
            }}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    p4 = load_phase4()
    p4.MIC_EFFECT = MIC_REF
    family = [x for x in calibrate_family() if x["feasible"]]
    p4.P_REF = calibrate_tnf(min(family, key=lambda x: x["cost"]))["params"]
    default = get_default_parameters()["bacteria"]
    adopted = {k: float(getattr(default, k)) for k in PD_KEYS}

    tsuji = tsuji_data()
    growth = tsuji_growth(tsuji["control_6h"])
    ident = tsuji_identifiability(tsuji, growth, adopted["k_pers"])
    shared_misfit = float(np.sqrt(np.mean(np.square(tsuji["difference_caz_minus_mero"]))))
    print(f"  Tsuji broth growth {growth:.3f}/h; shared-PD misfit {shared_misfit:.3f}", flush=True)

    doses = {t: {a: p4.dose_for_target_ft(a, t, 8.0, 6, mic=MIC_REF)
                 for a in ("ceftazidime", "meropenem")} for t in TARGETS}

    # Every cell of every map is independent: build the whole task list, run
    # it in a process pool (--workers; 1 runs serially), and split it back.
    refit = json.loads(REFIT.read_text(encoding="utf-8"))
    bases = {name: {"k_kill_max": sc["parameters"]["k_kill_max"],
                    "kill_C50": sc["parameters"]["kill_C50"],
                    "kill_hill": sc["parameters"]["kill_hill"], "k_pers": adopted["k_pers"]}
             for name, sc in refit["scenarios"].items()}
    groups = {
        "main": [(scaled(adopted, fk, fc), adopted, f"kill x{fk:.3g}, C50 x{fc:.3g}")
                 for fk in FACTORS for fc in FACTORS],
        "mirror": [(adopted, scaled(adopted, fk, fc), f"MERO kill x{fk:.3g}, C50 x{fc:.3g}")
                   for fk in FACTORS for fc in FACTORS],
        "hill": [(scaled(adopted, hill=h), adopted, f"CAZ hill {h:g}") for h in HILL_SWEEP],
        "kpers": [(scaled(adopted, kpers=f), adopted, f"CAZ k_pers x{f:g}") for f in KPERS_FACTORS],
        **{f"env:{name}": [(scaled(base, fk, fc), base, f"{name}: kill x{fk:.3g}, C50 x{fc:.3g}")
                           for fk in COARSE for fc in COARSE] for name, base in bases.items()},
    }
    tasks = [(g, a) for g, items in groups.items() for a in items]
    print(f"  {len(tasks)} cells on {args.workers} worker(s)", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(p4.P_REF, doses, tsuji, growth)) as ex:
            results = list(ex.map(cell_task, [a for _, a in tasks], chunksize=1))
    else:
        init_worker(p4.P_REF, doses, tsuji, growth)
        results = [cell_task(a) for _, a in tasks]
    done = {g: [] for g in groups}
    for (g, _), c in zip(tasks, results):
        done[g].append(c)
    main_map, mirror_map = done["main"], done["mirror"]
    hill_sweep, kpers_sweep = done["hill"], done["kpers"]
    envelope = {}
    for name, base in bases.items():
        cells = done[f"env:{name}"]
        # With equal PD the predicted difference is zero, so the shared-PD
        # misfit is the same for every meropenem base.
        envelope[name] = {"meropenem_pd": base, "refit_rmse": refit["scenarios"][name].get("rmse"),
                          "summary": summarise_map(cells, shared_misfit), "cells": cells}
    print("  all maps done", flush=True)

    # ---- checks ---------------------------------------------------------
    shared = next(c for c in main_map if c["label"] == "kill x1, C50 x1")
    joint = json.loads(JOINT.read_text(encoding="utf-8"))
    joint_adopted = {r["target_fT_pct"]: next(s["critical_lysis_yield_ratio"]
                                              for s in r["yield_stress_tests"] if s["filament_factor"] == 1.0)
                     for r in joint["rows"] if r["pd_scenario"] == "adopted"}
    vs_joint = max(abs(shared["targets"][f"{t:g}"]["r_star"] / joint_adopted[t] - 1.0) for t in TARGETS)
    twin = next(n for n, e in envelope.items()
                if all(abs(e["meropenem_pd"][k] / adopted[k] - 1.0) < 1e-9 for k in PD_KEYS))
    twin_gap = 0.0
    for c in envelope[twin]["cells"]:
        m = next(x for x in main_map
                 if all(abs(x["ceftazidime_pd"][k] / c["ceftazidime_pd"][k] - 1.0) < 1e-12
                        for k in ("k_kill_max", "kill_C50", "kill_hill")))
        for t in TARGETS:
            a, b = c["targets"][f"{t:g}"]["r_star"], m["targets"][f"{t:g}"]["r_star"]
            twin_gap = max(twin_gap, abs(a / b - 1.0))

    main_summary = summarise_map(main_map, shared_misfit)
    mirror_summary = summarise_map(mirror_map, shared_misfit)
    mirror_shared = next(c for c in mirror_map if c["label"] == "MERO kill x1, C50 x1")
    mirror_gap = max(abs(mirror_shared["targets"][f"{t:g}"]["r_star"]
                         / shared["targets"][f"{t:g}"]["r_star"] - 1.0) for t in TARGETS)
    env_r = [c["targets"][f"{t:g}"]["r_star"] for e in envelope.values() for c in e["cells"] for t in TARGETS]

    # What every reversal has in common. Release follows cumulative killed
    # biomass, so the ranking can only turn where meropenem kills far more
    # biomass than ceftazidime, which here means where it controls the
    # infection far less well. Measured over every cell of every map.
    every = (main_map + mirror_map + hill_sweep + kpers_sweep
             + [c for e in envelope.values() for c in e["cells"]])
    rows = [v for c in every for v in c["targets"].values()]
    turned = [v for v in rows if v["verdict"] != "holds"]
    held = [v for v in rows if v["verdict"] == "holds"]

    def gap(v):  # meropenem's 48-h net change minus ceftazidime's; >0 favours CAZ
        return v["meropenem"]["net_log10_change"] - v["ceftazidime"]["net_log10_change"]

    sweeps_r = [v["r_star"] for c in hill_sweep + kpers_sweep for v in c["targets"].values()]
    closer = [c for c in main_map + mirror_map if c["tsuji_difference_misfit_log10"] < shared_misfit]
    headline = {
        "n_cell_targets": len(rows),
        "main_map_all_hold": main_summary["n_holding_at_every_target"] == main_summary["n_cells"],
        "main_map_r_star_max": max(main_summary["by_target"][k]["r_star_range"][1]
                                   for k in main_summary["by_target"]),
        "mirror_not_holding_by_target": {k: v["verdict_counts"]["reverses"] + v["verdict_counts"]["depends_on_yield"]
                                         for k, v in mirror_summary["by_target"].items()},
        "mirror_reverses_by_target": {k: v["verdict_counts"]["reverses"]
                                      for k, v in mirror_summary["by_target"].items()},
        "mirror_not_holding_both_controlled_by_target": {
            k: v["not_holding_with_both_controlled"] for k, v in mirror_summary["by_target"].items()},
        "envelope_all_hold": all(e["summary"]["n_holding_at_every_target"] == e["summary"]["n_cells"]
                                 for e in envelope.values()),
        "envelope_r_star_max": max(env_r),
        "sweeps_r_star_range": [min(sweeps_r), max(sweeps_r)],
        "shared_r_star_range": [min(shared["targets"][f"{t:g}"]["r_star"] for t in TARGETS),
                                max(shared["targets"][f"{t:g}"]["r_star"] for t in TARGETS)],
        "n_not_holding": len(turned),
        "not_holding_ceftazidime_failing": sum(1 for v in turned if not v["ceftazidime"]["controlled"]),
        "not_holding_min_meropenem_disadvantage_log10": min(gap(v) for v in turned) if turned else None,
        "not_holding_max_meropenem_net_log10": max(v["meropenem"]["net_log10_change"] for v in turned) if turned else None,
        "not_holding_min_killed_ratio_mero_over_caz": min(1.0 / v["killed_ratio_caz_over_mero"] for v in turned) if turned else None,
        "holding_max_meropenem_disadvantage_log10": max(gap(v) for v in held),
        "tsuji_closer_cells": len(closer),
        "tsuji_closer_r_star_range": [min(v["r_star"] for c in closer for v in c["targets"].values()),
                                      max(v["r_star"] for c in closer for v in c["targets"].values())] if closer else None,
        "tsuji_closer_all_hold": all(v["verdict"] == "holds" for c in closer for v in c["targets"].values()),
    }
    env_shared = [c["targets"][f"{t:g}"]["r_star"] for e in envelope.values() for c in e["cells"]
                  if c["label"].endswith("kill x1, C50 x1") for t in TARGETS]
    payload = {
        "status": "conditional PD-difference map; no default, dose or Table 1 value replaced",
        "question": "At matched %fT>MIC against a fixed MIC reference, how different must the two "
                    "agents' bacterial PD be before the endotoxin ranking reverses?",
        "design": {
            "mic_reference_mgL": MIC_REF,
            "mic_reference_basis": "Measured for meropenem against P. aeruginosa ATCC-27853 (the "
                                   "calibration isolate); the same value assumed for ceftazidime. "
                                   "Fixed in every scenario, never re-derived from a scenario's stasis.",
            "targets_fT_pct": list(TARGETS),
            "matched_doses_mg_q8h": {f"{t:g}": d for t, d in doses.items()},
            "ceftazidime_factor_grid": list(FACTORS), "coarse_factor_grid": list(COARSE),
            "hill_sweep": list(HILL_SWEEP), "k_pers_factors": list(KPERS_FACTORS),
            "meropenem_reference_pd": adopted,
            "assumed_yield_ratio_range": list(ASSUMED),
            "r_star": "CAZ:MERO lysis-yield ratio at which 48-h endotoxin AUC is equal, with the "
                      "sub-MIC term at its model value; ranking holds if r* < 3.5, reverses if r* > 7",
            "endotoxin_integration": "exact (ds_endpoint_windows.exact_window_aucs), 48 h",
        },
        "tsuji_check": {
            "source": "Tsuji M et al. J Antimicrob Chemother 2003;51:353-9, P. aeruginosa SR24, "
                      "constant concentrations, 6 h (data/danger_signal/registry.json)",
            "observed": tsuji, "broth_growth_per_h": growth,
            "log10_inoculum": TSUJI_LOG10_INOCULUM,
            "per_agent_fit_attempt": ident,
            "identifiable": not any(v["bounds_reached"] for v in ident.values()),
            "shared_pd_misfit_log10": shared_misfit,
            "caveat": "Viable counts do not separate killing from filamentation, which suppresses "
                      "colony counts without killing; ceftazidime's activity at 0.1 x MIC may be "
                      "filament formation rather than kill.",
        },
        "headline": headline,
        "main_map": {"summary": main_summary, "cells": main_map},
        "mirror_map": {"summary": mirror_summary, "cells": mirror_map,
                       "note": "ceftazidime at the adopted PD; meropenem's kill rate and C50 "
                               "scaled by the same factors"},
        "hill_sweep": hill_sweep,
        "k_pers_sweep": kpers_sweep,
        "meropenem_envelope": {
            "scenarios": {n: {k: v for k, v in e.items() if k != "cells"} for n, e in envelope.items()},
            "cells": {n: e["cells"] for n, e in envelope.items()},
            "r_star_range_all": [min(env_r), max(env_r)],
            "r_star_range_shared_pd": [min(env_shared), max(env_shared)],
        },
        "numerical_check": {
            "max_rel_diff_shared_cell_r_star_vs_joint_robustness_trapezoid": vs_joint,
            "envelope_twin_of_adopted_pd": twin,
            "max_rel_diff_envelope_twin_vs_main_map": twin_gap,
            "max_rel_diff_mirror_shared_cell_vs_main_shared_cell": mirror_gap,
            "n_doses_resolved_for_grid_residual": n_redosed(every, doses),
            "ft_tolerance_pct": FT_TOL,
        },
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (REFIT, JOINT)},
        # The registry is a shared ledger edited for other reasons, so only the
        # Tsuji values this analysis reads are hashed, not the whole file.
        "tsuji_data_sha256": tsuji_digest(tsuji),
        "provenance": provenance(note="Agent-specific PD map at a fixed MIC reference"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"headline": headline,
                      "numerical_check": payload["numerical_check"],
                      "tsuji": {"identifiable": payload["tsuji_check"]["identifiable"],
                                "shared_misfit": shared_misfit},
                      "envelope_r_star": payload["meropenem_envelope"]["r_star_range_all"],
                      "envelope_shared": payload["meropenem_envelope"]["r_star_range_shared_pd"]},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
