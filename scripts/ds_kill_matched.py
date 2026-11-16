"""Matched on cumulative killing, and what the endotoxin ratio is made of.

Table 1 matches the two agents on %fT>MIC. That leaves cumulative
drug-induced killing, the quantity lysis-coupled release is proportional to,
unequal by up to 11%, and it leaves the sub-MIC term with ceftazidime alone,
so the Table 1 ratio mixes the assumed yield difference with other
differences. This script separates them in the Table 1 setting (the model's
effective MIC, q8h for 48 h, 30-minute infusions, the adopted shared kill
function, immediate release), in two steps.

1. Matched on killing. For each Table 1 target, one agent is held at its
   matched dose and the other's dose is solved so that the two do the same
   cumulative drug-induced killing over 48 h; both directions are solved.
   Killing is not monotonic in dose. It is near zero at trace doses, peaks
   where therapy fails and a regrowing burden is killed partially and
   continuously, and falls to a plateau at high doses, where the burden
   present plus what grows before it is cleared is killed once. Any killing
   level above the plateau is therefore reached at two doses, one of them a
   failing regimen. Every solution is found from a log-spaced dose scan; each
   local extremum of the scan is refined, so that a pair of solutions lying
   between two scan points would be found as well. Each solution is reported
   with its bacterial outcome.

2. Decomposed. Release is linear in the yields. Over [0, T] the lysis part
   obeys released = c K, with K the cumulative killing and c a unit
   conversion, and the 48-h AUC is (released - D(T) - in transit(T)) / k.
   The ceftazidime:meropenem ratio of 48-h AUC therefore factors exactly:

     R = (Y_caz/Y_mero) (K_caz/K_mero) (tau_caz/tau_mero) (1+S_caz)/(1+S_mero)
           yield          killing         timing            sub-MIC

   tau = k AUC_lysis / released_lysis is the share of the released endotoxin
   that the 48-h AUC captures (below one when release comes late, or waits
   in transit under a delay), and S is an agent's sub-MIC AUC over its own
   lysis AUC. The factors multiply, so ln R is a sum and each term's share of
   it needs no ordering convention. The product is checked against the
   directly computed ratio, which tests released = c K on every arm.

The timing factor is re-read under every release-delay structure of
scripts/ds_release_delay.py, and the sub-MIC factor under every meropenem
filament fraction of scripts/ds_filament_structure.py. The killing factor
depends on neither: release is a one-way output of the bacterial trajectory.

Each simulation takes about 0.15 s and the analysis needs about 840, so it
runs serially in about three minutes. A process pool would save about a
minute of that, and a GPU nothing: the model's right-hand side is scalar
Python, solved one trajectory at a time.

Writes results/danger_signal/kill_matched.json.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_endpoint_windows import (  # noqa: E402
    REPRODUCTION_RTOL, exact_window_aucs, load_phase4_module)
from scripts.ds_filament_structure import FRACTIONS as MERO_FILAMENT_FRACTIONS  # noqa: E402
from scripts.ds_release_delay import PRIMARY as DELAY_PRIMARY  # noqa: E402
from scripts.ds_release_delay import configs as delay_configs  # noqa: E402
from scripts.ds_release_delay import delayed_endotoxin  # noqa: E402
from src.analysis.cytokine_calibration import calibrate_family, calibrate_tnf  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

PHASE4 = ROOT / "results/danger_signal/phase4_analyses.json"
RELEASE_DELAY = ROOT / "results/danger_signal/release_delay.json"
OUT =ROOT / "results/danger_signal/kill_matched.json"
AGENTS = ("ceftazidime", "meropenem")
INTERVAL_H, N_DOSES = 8.0, 6
SCAN_MG = tuple(float(d) for d in np.geomspace(1.0, 60000.0, 161))
DOSE_RTOL = 1e-10
#: Cumulative killing is the trapezoid of kill rate x burden on the solver's
#: adaptive output grid (max_step 0.1 h, rtol 1e-6), and that grid moves with
#: the dose. On the high-dose plateau, where killing changes by a few parts in
#: 10^5 per percent of dose, the value jitters by up to 2e-4 between
#: neighbouring doses (measured). Equal killing is therefore resolved to 1e-3,
#: and scan extrema less prominent than that are solver jitter, not shape.
MATCH_RTOL = 1e-3
EXTREMUM_MIN_PROMINENCE = 1e-3
#: The stored Table 1 ratios come from trapezoid quadrature; the exact
#: integration here differs from them by that quadrature error only.
STORED_RATIO_RTOL = 5e-3
FACTORS = ("yield", "killing", "timing", "sub_mic")
IMMEDIATE = {"label": "immediate", "mtt_h": 0.0, "n": 0, "delay_filament": False}


# --------------------------------------------------------------------------
# simulations, cached by (agent, dose)
# --------------------------------------------------------------------------
class Arms:
    """One simulation per agent and dose, split into per-unit-yield inputs."""

    def __init__(self, p4):
        self.p4 = p4
        q = p4.P_REF
        if q.DS_baseline != 0.0:
            raise RuntimeError("the endpoint is AUC of DS, which equals the excess "
                               "integrated here only at zero baseline")
        self.k = q.k_ds_clear
        self.to_conc = 1e-6 / q.V_ds_L * q.translocation_fraction
        self.sims, self.arms = {}, {}

    def sim(self, agent, dose):
        key = (agent, float(dose))
        if key not in self.sims:
            self.sims[key] = self.p4.simulate(agent, float(dose), INTERVAL_H, N_DOSES)
        return self.sims[key]

    def killing(self, agent, dose):
        return float(self.sim(agent, dose)["cum_killed_cfu_per_mL"])

    def arm(self, agent, dose):
        key = (agent, float(dose))
        if key in self.arms:
            return self.arms[key]
        p4, sim = self.p4, self.sim(agent, dose)
        out, summ = p4.ds_cost(sim, agent=agent)
        b = np.asarray(sim["B_rep"], float)
        scale = p4.V_FOCUS * self.to_conc
        lys = np.asarray(sim["kill"], float) * b * scale
        fil = np.asarray(out["phi"], float) * b * scale
        y, yf = p4.AGENT_YIELD[agent], p4.AGENT_FILAMENT[agent]
        src = (np.asarray(out["release_lysis"])
               + np.asarray(out["release_filament"])) * 1e6 * self.to_conc
        if not np.allclose(y * lys + yf * fil, src, rtol=1e-12, atol=0.0):
            raise RuntimeError(f"{agent} {dose:g} mg: per-yield split does not "
                               "rebuild the release input")
        net = float(sim["net_log10"])
        arm = {
            "agent": agent, "dose_mg_q8h": float(dose), "fT_pct": float(sim["fT"]),
            "net_log10_change": net,
            "log10_final_burden_cfu_per_mL": float(np.log10(p4.B_TOTAL_0) + net),
            "controlled": bool(net < 0.0),
            "cum_killed_cfu_per_mL": float(sim["cum_killed_cfu_per_mL"]),
            "killed_over_initial_burden": float(sim["cum_killed_cfu_per_mL"] / p4.B_TOTAL_0),
            "AUC_DS_model_pg_h_mL": float(summ["AUC_DS_pg_h_mL"]),
            # The independent integrator, at zero delay, for the identity check.
            "exact_auc48_independent": float(exact_window_aucs(
                sim["t"], src, self.k, (48.0,))[48.0]),
            "_t": np.asarray(sim["t"], float), "_lys": lys, "_fil": fil, "_auc": {},
        }
        self.arms[key] = arm
        return arm

    def unit_aucs(self, arm, cfg):
        """48-h AUC per unit lysis yield and per unit filament yield, under ``cfg``."""
        label = cfg["label"]
        if label not in arm["_auc"]:
            zero = np.zeros_like(arm["_lys"])
            kw = dict(mtt=cfg["mtt_h"], n=cfg["n"], delay_filament=cfg["delay_filament"],
                      windows=(48.0,))
            lys = delayed_endotoxin(arm["_t"], arm["_lys"], zero, self.k, **kw)
            fil = delayed_endotoxin(arm["_t"], zero, arm["_fil"], self.k, **kw)
            arm["_auc"][label] = {
                "lys": lys["auc"][48.0], "fil": fil["auc"][48.0],
                "released_lys": lys["released"],
                "mass_balance": max(lys["mass_balance_rel_residual"],
                                    fil["mass_balance_rel_residual"]),
            }
        return arm["_auc"][label]


def public(arm):
    return {k: v for k, v in arm.items() if not k.startswith("_")}


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------
def decompose(arms, caz, mero, cfg, yield_ratio, mero_filament_fraction=0.0):
    """CAZ:MERO 48-h AUC ratio and its four exact factors."""
    p4, k = arms.p4, arms.k
    yc = p4.AGENT_YIELD["ceftazidime"]
    ym = yc / yield_ratio
    yfc = p4.Y_FILAMENT_CAZ
    yfm = mero_filament_fraction * p4.Y_FILAMENT_CAZ
    a, b = arms.unit_aucs(caz, cfg), arms.unit_aucs(mero, cfg)
    ratio = (yc * a["lys"] + yfc * a["fil"]) / (ym * b["lys"] + yfm * b["fil"])
    tau = {"ceftazidime": k * a["lys"] / a["released_lys"],
           "meropenem": k * b["lys"] / b["released_lys"]}
    sub = {"ceftazidime": yfc * a["fil"] / (yc * a["lys"]),
           "meropenem": yfm * b["fil"] / (ym * b["lys"])}
    factors = {
        "yield": float(yield_ratio),
        "killing": caz["cum_killed_cfu_per_mL"] / mero["cum_killed_cfu_per_mL"],
        "timing": tau["ceftazidime"] / tau["meropenem"],
        "sub_mic": (1.0 + sub["ceftazidime"]) / (1.0 + sub["meropenem"]),
    }
    product = float(np.prod([factors[f] for f in FACTORS]))
    log_r = float(np.log(ratio))
    return {
        "ratio": float(ratio),
        "factors": factors,
        "share_of_log_ratio": {f: float(np.log(factors[f]) / log_r) for f in FACTORS},
        "tau": tau,
        "sub_mic_over_lysis_AUC": sub,
        "identity_rel_residual": abs(product / ratio - 1.0),
        "mass_balance_rel_residual": max(a["mass_balance"], b["mass_balance"]),
    }


def at_yield_range(arms, caz, mero, cfg, ratios, fraction=0.0):
    return {f"{r:g}": decompose(arms, caz, mero, cfg, r, fraction) for r in ratios}


# --------------------------------------------------------------------------
# solving for equal killing
# --------------------------------------------------------------------------
def scan(arms, agent):
    """Killing over the dose scan, and each local extremum refined."""
    d = np.array(SCAN_MG)
    kill = np.array([arms.killing(agent, x) for x in d])
    extrema, jitter = [], []
    for i in range(1, len(d) - 1):
        if (kill[i] - kill[i - 1]) * (kill[i + 1] - kill[i]) < 0:
            prominence = min(abs(kill[i] - kill[i - 1]), abs(kill[i + 1] - kill[i])) / kill[i]
            if prominence < EXTREMUM_MIN_PROMINENCE:
                jitter.append({"dose_mg_q8h": float(d[i]), "prominence_rel": float(prominence)})
                continue
            is_max = kill[i] > kill[i - 1]
            sign = -1.0 if is_max else 1.0
            res = minimize_scalar(lambda z: sign * arms.killing(agent, float(np.exp(z))),
                                  bounds=(np.log(d[i - 1]), np.log(d[i + 1])),
                                  method="bounded", options={"xatol": 1e-7})
            x = float(np.exp(res.x))
            extrema.append({"kind": "max" if is_max else "min", "scan_index": i,
                            "dose_mg_q8h": x, "cum_killed_cfu_per_mL": arms.killing(agent, x)})
    # Past the peak the curve should only fall; any rise is jitter, and its
    # size is what EXTREMUM_MIN_PROMINENCE has to exceed.
    top = int(np.argmax(kill))
    rises = np.diff(kill[top:]) / kill[top + 1:]
    return {"dose_mg_q8h": d.tolist(), "cum_killed_cfu_per_mL": kill.tolist(),
            "fT_pct": [float(arms.sim(agent, x)["fT"]) for x in d],
            "net_log10_change": [float(arms.sim(agent, x)["net_log10"]) for x in d],
            "extrema": extrema,
            "jitter_extrema": {"n": len(jitter),
                               "max_prominence_rel": max((j["prominence_rel"] for j in jitter), default=0.0),
                               "dose_mg_q8h_range": ([min(j["dose_mg_q8h"] for j in jitter),
                                                      max(j["dose_mg_q8h"] for j in jitter)]
                                                     if jitter else None)},
            "max_rel_rise_after_peak": float(max(rises.max(), 0.0)) if len(rises) else 0.0}


def solve_equal_killing(arms, agent, sc, target):
    """Every dose on the scan range at which ``agent`` kills ``target`` CFU/mL."""
    d = np.array(sc["dose_mg_q8h"])
    f = np.array(sc["cum_killed_cfu_per_mL"]) / target - 1.0
    brackets = {(d[i], d[i + 1]) for i in range(len(d) - 1) if f[i] * f[i + 1] < 0}
    exact = {float(d[i]) for i in range(len(d)) if f[i] == 0.0}
    added_from_extrema = 0
    for e in sc["extrema"]:
        i = e["scan_index"]
        fx = e["cum_killed_cfu_per_mL"] / target - 1.0
        # A pair of solutions between scan points: the neighbours and the
        # scan point agree in sign, the refined extremum does not.
        if f[i - 1] * f[i] > 0 and f[i] * f[i + 1] > 0 and fx * f[i] < 0:
            brackets |= {(d[i - 1], e["dose_mg_q8h"]), (e["dose_mg_q8h"], d[i + 1])}
            added_from_extrema += 2
    roots = sorted(exact | {
        float(brentq(lambda x: arms.killing(agent, x) / target - 1.0, lo, hi,
                     xtol=1e-12, rtol=DOSE_RTOL)) for lo, hi in brackets})
    return roots, added_from_extrema


def conditioning(arms, agent, dose, half_width=0.05, n=11):
    """How sharply equal killing fixes the dose.

    A quadratic in log dose is fitted to ln K over +/-5% of the solution. Its
    slope is the elasticity d ln K / d ln dose, its RMS residual the solver
    jitter in K there, and their quotient the relative dose resolution. A
    two-point difference would not do: on the plateau the jitter is as large
    as the change it would be measuring.
    """
    x = np.linspace(-half_width, half_width, n)
    lk = np.log([arms.killing(agent, dose * float(np.exp(v))) for v in x])
    coef = np.polyfit(x, lk, 2)
    slope = float(coef[1])
    noise = float(np.sqrt(np.mean((lk - np.polyval(coef, x)) ** 2)))
    return {"elasticity_dlnK_dlnDose": slope, "killing_jitter_rel_rms": noise,
            "dose_resolution_rel": noise / abs(slope) if slope != 0.0 else None}


# --------------------------------------------------------------------------
def main():
    p4 = load_phase4_module()
    family = [e for e in calibrate_family() if e["feasible"]]
    p4.P_REF = calibrate_tnf(min(family, key=lambda e: e["cost"]))["params"]
    phase4 = json.loads(PHASE4.read_text(encoding="utf-8"))
    p4.MIC_EFFECT = phase4["meta"]["mic_effect_site_mgL"]
    ratios = tuple(phase4["meta"]["yield_ratio_range"])
    arms = Arms(p4)
    delays = list(delay_configs())
    if delays[0]["label"] != "immediate":
        raise RuntimeError("the delay grid must start at immediate release")

    # ---- Table 1 pairs ---------------------------------------------------
    table1, stored = {}, {}
    for row in phase4["analysis3_matched_exposure"]["rows"]:
        if not row.get("evaluated"):
            continue
        t = f"{row['target_fT_pct']:g}"
        pair = {}
        for agent in AGENTS:
            arm = arms.arm(agent, row["arms"][agent]["dose_mg"])
            gap = abs(arm["AUC_DS_model_pg_h_mL"] / row["arms"][agent]["AUC_DS_pg_h_mL"] - 1.0)
            if gap > REPRODUCTION_RTOL:
                raise RuntimeError(f"Table 1 {agent} {t}%: stored AUC reproduced only to {gap:.2e}")
            pair[agent] = arm
        table1[t] = pair
        stored[t] = {f"{r['yield_ratio']:g}": r["ratio_AUC_DS"] for r in row["cost_ratio_range"]}
    print(f"Table 1: {len(table1)} pairs reproduced", flush=True)

    # ---- the killing curve and every equal-killing dose ------------------
    scans = {a: scan(arms, a) for a in AGENTS}
    for a, sc in scans.items():
        print(f"  {a}: {len(sc['extrema'])} extremum/extrema "
              + ", ".join(f"{e['kind']} {e['cum_killed_cfu_per_mL'] / p4.B_TOTAL_0:.2f}x B0 "
                          f"at {e['dose_mg_q8h']:.1f} mg" for e in sc["extrema"]), flush=True)
    peak = {a: max(sc["extrema"], key=lambda e: e["cum_killed_cfu_per_mL"])["dose_mg_q8h"]
            for a, sc in scans.items()}

    matched, n_from_extrema, worst_match = {}, 0, 0.0
    for t, pair in table1.items():
        matched[t] = {}
        for solved, held in (("meropenem", "ceftazidime"), ("ceftazidime", "meropenem")):
            target = pair[held]["cum_killed_cfu_per_mL"]
            roots, added = solve_equal_killing(arms, solved, scans[solved], target)
            n_from_extrema += added
            sols = []
            for dose in roots:
                arm = arms.arm(solved, dose)
                worst_match = max(worst_match, abs(arm["cum_killed_cfu_per_mL"] / target - 1.0))
                caz, mero = (arm, pair[held]) if solved == "ceftazidime" else (pair[held], arm)
                sols.append({
                    "branch": "below_peak" if dose < peak[solved] else "above_peak",
                    "arm": public(arm),
                    "killing_mismatch_rel": arm["cum_killed_cfu_per_mL"] / target - 1.0,
                    "conditioning": conditioning(arms, solved, dose),
                    "net_log10_difference_caz_minus_mero":
                        caz["net_log10_change"] - mero["net_log10_change"],
                    "decomposition": at_yield_range(arms, caz, mero, IMMEDIATE, ratios),
                    "_pair": (caz, mero),
                })
            matched[t][f"{solved}_solved"] = {"held": public(pair[held]),
                                              "target_cum_killed_cfu_per_mL": target,
                                              "solutions": sols}
            print(f"  {t}%: {solved} matched to {held}'s killing: "
                  + "; ".join(f"{s['branch']} {s['arm']['dose_mg_q8h']:.1f} mg, "
                              f"fT {s['arm']['fT_pct']:.1f}%, net {s['arm']['net_log10_change']:+.2f}, "
                              f"R {s['decomposition']['3.5']['ratio']:.2f}-{s['decomposition']['7']['ratio']:.2f}"
                              for s in sols), flush=True)

    # ---- every comparison, under every delay and filament structure ------
    comparisons = {f"table1 {t}%": (pair["ceftazidime"], pair["meropenem"])
                   for t, pair in table1.items()}
    for t, block in matched.items():
        for direction, m in block.items():
            for s in m["solutions"]:
                name = f"kill-matched {t}% {direction} {s['branch']}"
                if name in comparisons:
                    raise RuntimeError(f"two solutions on one branch: {name}")
                comparisons[name] = s["_pair"]
    by_delay = {}
    for cfg in delays:
        by_delay[cfg["label"]] = {
            name: {r: {k: v for k, v in d.items() if k in ("ratio", "factors", "tau")}
                   for r, d in at_yield_range(arms, caz, mero, cfg, ratios).items()}
            for name, (caz, mero) in comparisons.items()}
    by_filament = {}
    for frac in MERO_FILAMENT_FRACTIONS:
        by_filament[f"{frac:g}"] = {
            name: {r: {k: v for k, v in d.items() if k in ("ratio", "factors")}
                   for r, d in at_yield_range(arms, caz, mero, IMMEDIATE, ratios, frac).items()}
            for name, (caz, mero) in comparisons.items()}

    table1_out = {}
    for t, pair in table1.items():
        dec = at_yield_range(arms, pair["ceftazidime"], pair["meropenem"], IMMEDIATE, ratios)
        table1_out[t] = {"ceftazidime": public(pair["ceftazidime"]),
                         "meropenem": public(pair["meropenem"]),
                         "decomposition": dec,
                         "stored_ratio": stored[t]}
    for block in matched.values():
        for m in block.values():
            for s in m["solutions"]:
                s.pop("_pair")

    # ---- numerical checks -------------------------------------------------
    all_arms = list(arms.arms.values())
    every_dec = ([d for v in table1_out.values() for d in v["decomposition"].values()]
                 + [d for b in matched.values() for m in b.values()
                    for s in m["solutions"] for d in s["decomposition"].values()])
    for cfg in delays:
        for name, (caz, mero) in comparisons.items():
            every_dec += list(at_yield_range(arms, caz, mero, cfg, ratios).values())
    for frac in MERO_FILAMENT_FRACTIONS:
        for name, (caz, mero) in comparisons.items():
            every_dec += list(at_yield_range(arms, caz, mero, IMMEDIATE, ratios, frac).values())
    zero_delay_gap = max(
        abs((arm["_auc"]["immediate"]["lys"] * p4.AGENT_YIELD[arm["agent"]]
             + arm["_auc"]["immediate"]["fil"] * p4.AGENT_FILAMENT[arm["agent"]])
            / arm["exact_auc48_independent"] - 1.0) for arm in all_arms)
    stored_gap = max(abs(v["decomposition"][r]["ratio"] / v["stored_ratio"][r] - 1.0)
                     for v in table1_out.values() for r in v["stored_ratio"])
    if stored_gap > STORED_RATIO_RTOL:
        raise RuntimeError(f"Table 1 ratios reproduced only to {stored_gap:.2e}")
    if worst_match > MATCH_RTOL:
        raise RuntimeError(f"killing matched only to {worst_match:.2e}")
    numerical_check = {
        "max_identity_rel_residual": max(d["identity_rel_residual"] for d in every_dec),
        "max_mass_balance_rel_residual": max(d["mass_balance_rel_residual"] for d in every_dec
                                             if "mass_balance_rel_residual" in d),
        "max_rel_diff_zero_delay_vs_exact_window_aucs": zero_delay_gap,
        "max_rel_diff_table1_ratio_vs_stored_trapezoid": stored_gap,
        "max_rel_killing_mismatch_at_solutions": worst_match,
        "n_scan_extrema": {a: len(sc["extrema"]) for a, sc in scans.items()},
        "scan_jitter_extrema": {a: sc["jitter_extrema"] for a, sc in scans.items()},
        "max_rel_rise_after_peak": {a: sc["max_rel_rise_after_peak"] for a, sc in scans.items()},
        "n_solutions_found_only_by_refining_extrema": n_from_extrema,
        "n_simulations": len(arms.sims),
    }

    # ---- headline --------------------------------------------------------
    sols = [(t, direction, s) for t, b in matched.items() for direction, m in b.items()
            for s in m["solutions"]]
    above = [x for x in sols if x[2]["branch"] == "above_peak"]
    below = [x for x in sols if x[2]["branch"] == "below_peak"]

    def rng(vals):
        vals = list(vals)
        return [min(vals), max(vals)] if vals else None

    def ratio_span(group):
        return rng(s["decomposition"][r]["ratio"] for _, _, s in group for r in s["decomposition"])

    t1 = table1_out
    table1_names = [f"table1 {t}%" for t in t1]
    controlled_pairs = [n for n, (c, m) in comparisons.items() if c["controlled"] and m["controlled"]]
    failing_pairs = [n for n in comparisons if n not in controlled_pairs]

    def timing_span(names, labels):
        return rng(by_delay[l][n][r]["factors"]["timing"] for l in labels for n in names
                   for r in by_delay[l][n])

    def delay_ratio_change(names):
        return max(abs(by_delay[l][n][r]["ratio"] / by_delay["immediate"][n][r]["ratio"] - 1.0)
                   for l in by_delay for n in names for r in by_delay[l][n])

    headline = {
        "table1_factors": {t: {f: v["decomposition"]["3.5"]["factors"][f] for f in FACTORS[1:]}
                           for t, v in t1.items()},
        "table1_killing_factor_range": rng(v["decomposition"]["3.5"]["factors"]["killing"] for v in t1.values()),
        "table1_sub_mic_factor_range": rng(v["decomposition"]["3.5"]["factors"]["sub_mic"] for v in t1.values()),
        "table1_timing_factor_range": rng(v["decomposition"]["3.5"]["factors"]["timing"] for v in t1.values()),
        "table1_ratio_range": rng(d["ratio"] for v in t1.values() for d in v["decomposition"].values()),
        "table1_share_pct": {
            f: [100 * min(d["share_of_log_ratio"][f] for v in t1.values() for d in v["decomposition"].values()),
                100 * max(d["share_of_log_ratio"][f] for v in t1.values() for d in v["decomposition"].values())]
            for f in FACTORS},
        "table1_non_yield_factor_range": rng(
            v["decomposition"]["3.5"]["ratio"] / 3.5 for v in t1.values()),
        "n_solutions": len(sols),
        "n_solutions_by_target_and_direction": {
            t: {direction: len(m["solutions"]) for direction, m in b.items()} for t, b in matched.items()},
        "killing_peak_over_initial_burden": {
            a: max(e["cum_killed_cfu_per_mL"] for e in sc["extrema"]) / p4.B_TOTAL_0 for a, sc in scans.items()},
        "killing_peak_dose_mg_q8h": peak,
        "killing_plateau_over_initial_burden_at_max_dose": {
            a: sc["cum_killed_cfu_per_mL"][-1] / p4.B_TOTAL_0 for a, sc in scans.items()},
        "above_peak": {
            "n": len(above),
            "all_controlled": all(s["arm"]["controlled"] for _, _, s in above),
            "ratio_range": ratio_span(above),
            "timing_factor_range": rng(s["decomposition"]["3.5"]["factors"]["timing"] for _, _, s in above),
            "sub_mic_factor_range": rng(s["decomposition"]["3.5"]["factors"]["sub_mic"] for _, _, s in above),
            "fT_pct_range": rng(s["arm"]["fT_pct"] for _, _, s in above),
            "abs_net_log10_difference_max": max(abs(s["net_log10_difference_caz_minus_mero"]) for _, _, s in above),
            "elasticity_range": rng(s["conditioning"]["elasticity_dlnK_dlnDose"] for _, _, s in above),
            "dose_resolution_rel_max": max(s["conditioning"]["dose_resolution_rel"] for _, _, s in above),
            "dose_over_table1_dose": {
                f"{t} {direction}": s["arm"]["dose_mg_q8h"]
                / t1[t][direction.replace("_solved", "")]["dose_mg_q8h"]
                for t, direction, s in above},
        },
        "below_peak": {
            "n": len(below),
            "all_failing": all(not s["arm"]["controlled"] for _, _, s in below),
            "ratio_range": ratio_span(below),
            "net_log10_change_range": rng(s["arm"]["net_log10_change"] for _, _, s in below),
            "net_log10_difference_range": rng(s["net_log10_difference_caz_minus_mero"] for _, _, s in below),
            "timing_factor_range": rng(s["decomposition"]["3.5"]["factors"]["timing"] for _, _, s in below),
            "sub_mic_factor_range": rng(s["decomposition"]["3.5"]["factors"]["sub_mic"] for _, _, s in below),
            "dose_mg_q8h_range": rng(s["arm"]["dose_mg_q8h"] for _, _, s in below),
        },
        "by_direction": {d: {b: group_summary([x for x in sols if x[1] == d and x[2]["branch"] == b],
                                              d, t1, rng, ratio_span)
                             for b in ("above_peak", "below_peak")}
                         for d in ("meropenem_solved", "ceftazidime_solved")},
        # Meropenem has no sub-MIC term and timing is near one, so at equal
        # killing its failing and controlled regimens give the same ratio.
        "meropenem_failing_vs_controlled_ratio_max_rel_diff": max(
            abs(b["decomposition"][r]["ratio"] / a["decomposition"][r]["ratio"] - 1.0)
            for blk in matched.values()
            for a, b in [sorted(blk["meropenem_solved"]["solutions"],
                                key=lambda s: s["branch"])]
            for r in a["decomposition"]),
        "delay": {
            "primary": DELAY_PRIMARY,
            "max_ratio_change_pct_table1_any_delay": 100 * delay_ratio_change(table1_names),
            "timing_factor_range_controlled_pairs_primary": timing_span(controlled_pairs, [DELAY_PRIMARY]),
            "timing_factor_range_controlled_pairs_any_delay": timing_span(controlled_pairs, list(by_delay)),
            "timing_factor_range_failing_pairs_any_delay": timing_span(failing_pairs, list(by_delay)),
            "max_ratio_change_pct_controlled_pairs_any_delay": 100 * delay_ratio_change(controlled_pairs),
            "max_ratio_change_pct_failing_pairs_any_delay": 100 * delay_ratio_change(failing_pairs),
        },
        "meropenem_filament": {
            f: {"table1_ratio_range": rng(by_filament[f][f"table1 {t}%"][r]["ratio"]
                                          for t in t1 for r in ratios_keys(ratios)),
                "above_peak_ratio_range": rng(
                    by_filament[f][n][r]["ratio"] for n in comparisons
                    if n.endswith("above_peak") for r in ratios_keys(ratios)),
                "sub_mic_factor_range_table1": rng(
                    by_filament[f][f"table1 {t}%"][r]["factors"]["sub_mic"]
                    for t in t1 for r in ratios_keys(ratios))}
            for f in by_filament},
        "dose_resolution_rel_max_all": max(s["conditioning"]["dose_resolution_rel"] for _, _, s in sols),
        "meropenem_filament_controlled_ratio_min": min(
            by_filament[f][n][r]["ratio"] for f in by_filament for n in controlled_pairs
            for r in ratios_keys(ratios)),
        "controlled_comparisons": controlled_pairs,
        "comparisons_with_a_failing_arm": failing_pairs,
    }

    # The Table 1 pairs under every delay, against ds_release_delay.py's own
    # calculation of the same quantity by a different route (full release
    # through the chain, not per-yield parts).
    stored_delay = json.loads(RELEASE_DELAY.read_text(encoding="utf-8"))["headline"]
    numerical_check["abs_diff_pct_table1_delay_change_vs_release_delay"] = abs(
        headline["delay"]["max_ratio_change_pct_table1_any_delay"]
        - stored_delay["table1_max_abs_rel_change_pct_any_delay"])

    payload = {
        "design": {
            "setting": "Table 1: effective MIC from phase4_analyses.json, q8h x 6 doses, "
                       "30-min infusion, adopted shared kill function, immediate release",
            "mic_effect_site_mgL": p4.MIC_EFFECT,
            "yield_ratio_range": list(ratios),
            "dose_scan_mg_q8h": [SCAN_MG[0], SCAN_MG[-1], len(SCAN_MG)],
            "dose_rtol": DOSE_RTOL,
            "match_rtol": MATCH_RTOL,
            "initial_burden_cfu_per_mL": p4.B_TOTAL_0,
            "factorization": "R = (Y_caz/Y_mero)(K_caz/K_mero)(tau_caz/tau_mero)"
                             "(1+S_caz)/(1+S_mero); tau = k AUC_lysis/released_lysis, "
                             "S = Y_filament AUC_filament / (Y_lysis AUC_lysis)",
            "delay_configs": [c["label"] for c in delays],
            "meropenem_filament_fractions": list(MERO_FILAMENT_FRACTIONS),
        },
        "killing_curve": scans,
        "table1": table1_out,
        "kill_matched": matched,
        "by_delay": by_delay,
        "by_meropenem_filament_fraction": by_filament,
        "headline": headline,
        "numerical_check": numerical_check,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (PHASE4, RELEASE_DELAY)},
        "provenance": provenance(note="Comparison matched on cumulative killing, and an "
                                      "exact decomposition of the agent ratio"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(headline, indent=1), flush=True)
    print(json.dumps(numerical_check, indent=1), flush=True)


def ratios_keys(ratios):
    return [f"{r:g}" for r in ratios]


def group_summary(group, direction, t1, rng, ratio_span):
    """One direction and branch of the equal-killing solutions, all targets."""
    solved = direction.replace("_solved", "")
    return {
        "n": len(group),
        "all_controlled": all(s["arm"]["controlled"] for _, _, s in group),
        "ratio_range": ratio_span(group),
        "killing_factor_range": rng(s["decomposition"]["3.5"]["factors"]["killing"] for _, _, s in group),
        "sub_mic_factor_range": rng(s["decomposition"]["3.5"]["factors"]["sub_mic"] for _, _, s in group),
        "dose_over_table1_dose_range": rng(s["arm"]["dose_mg_q8h"] / t1[t][solved]["dose_mg_q8h"]
                                           for t, _, s in group),
        "fT_pct_range": rng(s["arm"]["fT_pct"] for _, _, s in group),
        "net_log10_difference_caz_minus_mero_range": rng(
            s["net_log10_difference_caz_minus_mero"] for _, _, s in group),
    }


if __name__ == "__main__":
    main()
