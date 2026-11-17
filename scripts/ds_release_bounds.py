"""What the published paired release experiments bound, and what they cannot.

Plan item 5. Two studies report free LPS together with viable counts in the
same cultures:

  Yamaguchi et al. 1999 (calibration source): P. aeruginosa PAO1, inoculum
    1e6 CFU/mL, isepamicin, gentamicin, piperacillin, ceftazidime and
    imipenem at 1/4, 1 and 4 x MIC, sampled at 2, 4 and 6 h (digitized from
    Figures 2 and 3; data/danger_signal/yamaguchi1999_pao1_release.csv).
  Jackson and Kropp 1992 (held out): three isolates, imipenem against
    ceftazidime at 2 x MIC (and 0.5 x for one), sampled to 24 h
    (transcribed from Tables 1, 2, 4 and 5; jackson1992_pa_release.csv).

Observation operator. Free LPS in a culture filtrate is cumulative: nothing
clears it in vitro. It is the sum of what the inoculum carried, what live
cells shed while growing (the untreated controls show this is substantial),
and what drug action liberates. Only the total and the viable count are
measured. The drug-attributable release per killed cell is therefore not
observed but bounded, by two facts that need no kinetic model:

  killed biomass   K >= N0 - N(t)            (cells now missing were killed)
                   K <= N_ctl-fold x N0 - N(t) (survivors cannot outgrow the
                                               untreated control)
  background       B <= the smallest free LPS any arm of the same experiment
                        reached at that time, and B >= the arm's own 0-h value
                        (or 0 where none was measured).

The per-killed release y = (L - B)/K then lies in [(L - B_max)/K_max,
(L - B_min)/K_min], and the ceftazidime:imipenem ratio has the guaranteed
lower bound y_CAZ,min / y_IPM,max: every unfavorable choice is made against
ceftazidime. Bounds are only informative where the control barely grew
(Yamaguchi to 4 h); where it grew 100-fold (Jackson) the upper bound on K is
loose, so a central reading is reported alongside, which takes K at its lower
bound and B at the arm's own starting value. Readings carry no replicate
error, and the reported SEs are kept but not propagated.

Whole-study withholding. Jackson enters nothing above; its central ratios
are then compared with Yamaguchi's bounds as a retrospective check.

What stays unidentified is listed in the output: the absolute yield per
killed cell (assay scale and the upper bound on K), the split between
lysis-coupled and kill-independent release, background shedding as a
function of growth, and any transfer from broth free LPS to in vivo exposure.

Writes results/danger_signal/release_bounds.json.
"""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.held_out_validation import AGENT_EFFECT_ESTIMATES  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

YAM = ROOT / "data/danger_signal/yamaguchi1999_pao1_release.csv"
JACK = ROOT / "data/danger_signal/jackson1992_pa_release.csv"
OUT = ROOT / "results/danger_signal/release_bounds.json"
YAM_INOCULUM = 1e6
#: A count plotted on the axis floor is at or below detection: read as 1 CFU/mL,
#: which can only make the inferred killing larger (conservative for the
#: denominator agent, irrelevant for ceftazidime, which never reaches it).
FLOOR_CFU = 1.0
ASSUMED = tuple(AGENT_EFFECT_ESTIMATES["recommended_interval"])


def rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


def bounds(L, n0, n, ctl_fold, b_min, b_max):
    """Per-killed release bounds for one arm and time."""
    k_lo = max(0.0, n0 - n)
    k_hi = max(k_lo, n0 * ctl_fold - n)
    r_lo, r_hi = max(0.0, L - b_max), max(0.0, L - b_min)
    return {"free_lps_ng_mL": L, "n0": n0, "n": n, "killed_lo": k_lo, "killed_hi": k_hi,
            "attributable_lo": r_lo, "attributable_hi": r_hi,
            "per_killed_lo": r_lo / k_hi if k_hi > 0 else None,
            "per_killed_hi": r_hi / k_lo if k_lo > 0 else None,
            "per_killed_central": r_hi / k_lo if k_lo > 0 else None}


def ratio(a, b, which):
    """Ratio of per-killed release, a over b: 'lower' (guaranteed) or 'central'."""
    if which == "lower":
        num, den = a["per_killed_lo"], b["per_killed_hi"]
    else:
        num, den = a["per_killed_central"], b["per_killed_central"]
    if num is None or den is None or den == 0:
        return None
    return num / den


# --------------------------------------------------------------------------
def yamaguchi():
    data = rows(YAM)
    get = {}
    for r in data:
        get[(r["agent"], float(r["concentration_xMIC"]), float(r["time_h"]), r["series"])] = (
            float(r["value"]), r["quality"])
    out = {}
    for t in (2.0, 4.0, 6.0):
        lps_t = [v for (a, x, tt, s), (v, _) in get.items() if tt == t and s == "free_lps_ng_mL"]
        b_max = min(lps_t)
        ctl_fold = 10.0 ** get[("control", 0.0, t, "log10_cfu_mL")][0] / YAM_INOCULUM
        for (a, x, tt, s), (v, q) in get.items():
            if tt != t or s != "free_lps_ng_mL" or a == "control":
                continue
            cfu, cq = get[(a, x, t, "log10_cfu_mL")]
            n = FLOOR_CFU if cq == "at_axis_floor" else 10.0 ** cfu
            out[f"{a}|{x:g}|{t:g}"] = {**bounds(v, YAM_INOCULUM, n, ctl_fold, 0.0, b_max),
                                      "background_max": b_max, "control_fold": ctl_fold,
                                      "count_quality": cq}
    return out


def jackson():
    data = rows(JACK)
    tables = {}
    for r in data:
        tables.setdefault(r["table"], []).append(r)
    out = {}
    for tab, rs in tables.items():
        val = {}
        for r in rs:
            if r["value"] == "":
                continue
            val[(r["agent"], float(r["time_h"]), r["series"])] = (float(r["value"]), r["quality"])
        times = sorted({float(r["time_h"]) for r in rs if float(r["time_h"]) > 0})
        for t in times:
            if ("control", t, "log10_cfu_mL") not in val:
                continue                              # ND counts: nothing to bound with
            ctl_fold = 10.0 ** (val[("control", t, "log10_cfu_mL")][0]
                                - val[("control", 0.0, "log10_cfu_mL")][0])
            b_max = min(val[(a, t, "free_lps_ng_mL")][0] for a in ("control", "imipenem", "ceftazidime"))
            for a in ("imipenem", "ceftazidime"):
                if (a, t, "log10_cfu_mL") not in val:
                    continue
                n0 = 10.0 ** val[(a, 0.0, "log10_cfu_mL")][0]
                n = 10.0 ** val[(a, t, "log10_cfu_mL")][0]
                b0 = val[(a, 0.0, "free_lps_ng_mL")][0]
                out[f"T{tab}|{a}|{t:g}"] = {**bounds(val[(a, t, "free_lps_ng_mL")][0], n0, n, ctl_fold,
                                                     b0, max(b0, b_max)),
                                            "background_min": b0, "background_max": max(b0, b_max),
                                            "control_fold": ctl_fold,
                                            "count_quality": val[(a, t, "log10_cfu_mL")][1],
                                            "isolate": rs[0]["isolate"],
                                            "concentration_xMIC": float(next(r["concentration_xMIC"] for r in rs
                                                                             if r["agent"] == a))}
    return out


def main():
    yam = yamaguchi()
    jack = jackson()

    # ceftazidime (and piperacillin) against imipenem, Yamaguchi, same concentration and time
    yam_ratios = {}
    for num in ("ceftazidime", "piperacillin"):
        for x in (0.25, 1.0, 4.0):
            for t in (2.0, 4.0, 6.0):
                a, b = yam[f"{num}|{x:g}|{t:g}"], yam[f"imipenem|{x:g}|{t:g}"]
                yam_ratios[f"{num}|{x:g}|{t:g}"] = {
                    "lower_bound": ratio(a, b, "lower"), "central": ratio(a, b, "central"),
                    "raw_free_lps_ratio": a["free_lps_ng_mL"] / b["free_lps_ng_mL"],
                    "log10_count_difference": math.log10(a["n"]) - math.log10(b["n"])}
    jack_ratios = {}
    for key, a in jack.items():
        tab, agent, t = key.split("|")
        if agent != "ceftazidime":
            continue
        b = jack.get(f"{tab}|imipenem|{t}")
        if b is None:
            continue
        jack_ratios[f"{tab}|{t}"] = {"lower_bound": ratio(a, b, "lower"), "central": ratio(a, b, "central"),
                                     "raw_free_lps_ratio": a["free_lps_ng_mL"] / b["free_lps_ng_mL"],
                                     "isolate": a["isolate"], "concentration_xMIC": a["concentration_xMIC"],
                                     "control_fold": a["control_fold"]}

    def pick(d, keys, field):
        vals = [d[k][field] for k in keys if d[k][field] is not None]
        return [min(vals), max(vals)] if vals else None

    killing_keys = [f"ceftazidime|{x:g}|4" for x in (1.0, 4.0)]
    two_x = [k for k, v in jack_ratios.items() if v["concentration_xMIC"] == 2.0]
    headline = {
        "assumed_yield_ratio_range": list(ASSUMED),
        "yamaguchi_caz_ipm_lower_bound_4h": {f"{x:g}x": yam_ratios[f"ceftazidime|{x:g}|4"]["lower_bound"]
                                             for x in (1.0, 4.0)},
        "yamaguchi_pipc_ipm_lower_bound_4h": {f"{x:g}x": yam_ratios[f"piperacillin|{x:g}|4"]["lower_bound"]
                                              for x in (1.0, 4.0)},
        "yamaguchi_caz_ipm_central_4h": {f"{x:g}x": yam_ratios[f"ceftazidime|{x:g}|4"]["central"]
                                         for x in (1.0, 4.0)},
        "yamaguchi_caz_ipm_lower_bound_6h": {f"{x:g}x": yam_ratios[f"ceftazidime|{x:g}|6"]["lower_bound"]
                                             for x in (1.0, 4.0)},
        "yamaguchi_control_fold": {f"{t:g}h": yam[f"imipenem|1|{t:g}"]["control_fold"] for t in (2.0, 4.0, 6.0)},
        "yamaguchi_count_difference_caz_minus_ipm_4h": {
            f"{x:g}x": yam_ratios[f"ceftazidime|{x:g}|4"]["log10_count_difference"] for x in (1.0, 4.0)},
        "yamaguchi_raw_ratio_caz_ipm_1_4x_4_6h": pick(yam_ratios, [f"ceftazidime|{x:g}|{t:g}"
                                                                 for x in (1.0, 4.0) for t in (4.0, 6.0)],
                                                      "raw_free_lps_ratio"),
        "yamaguchi_sub_mic": {
            a: {"log10_cfu_6h": math.log10(yam[f"{a}|0.25|6"]["n"]),
                "free_lps_6h": yam[f"{a}|0.25|6"]["free_lps_ng_mL"],
                "killed_lower_bound_6h": yam[f"{a}|0.25|6"]["killed_lo"]}
            for a in ("ceftazidime", "piperacillin", "imipenem")},
        "yamaguchi_aminoglycoside_free_lps_4h": pick(yam, [f"{a}|{x:g}|4" for a in ("isepamicin", "gentamicin")
                                                          for x in (0.25, 1.0, 4.0)], "free_lps_ng_mL"),
        "yamaguchi_aminoglycoside_log10_cfu_4h": [min(math.log10(yam[f"{a}|{x:g}|4"]["n"])
                                                      for a in ("isepamicin", "gentamicin") for x in (1.0, 4.0)),
                                                  max(math.log10(yam[f"{a}|{x:g}|4"]["n"])
                                                      for a in ("isepamicin", "gentamicin") for x in (1.0, 4.0))],
        "jackson_central_2x": {k: jack_ratios[k]["central"] for k in two_x},
        "jackson_central_2x_range": pick(jack_ratios, two_x, "central"),
        "jackson_raw_ratio_2x_range": pick(jack_ratios, two_x, "raw_free_lps_ratio"),
        "jackson_control_fold_2x_range": pick(jack_ratios, two_x, "control_fold"),
        "jackson_lower_bound_2x_max": max(v["lower_bound"] for k, v in jack_ratios.items()
                                          if k in two_x and v["lower_bound"] is not None),
    }
    # The ratio is not one number: per isolate, the latest time with a defined
    # central ratio, and every time before 4 h.
    latest = {}
    for k, v in jack_ratios.items():
        tab, t = k.split("|")
        if v["concentration_xMIC"] == 2.0 and v["central"] is not None:
            if tab not in latest or float(t) > latest[tab]["time_h"]:
                latest[tab] = {"time_h": float(t), "central": v["central"], "isolate": v["isolate"]}
    headline["jackson_latest_central_2x"] = latest
    headline["jackson_central_before_4h"] = {k: v["central"] for k, v in jack_ratios.items()
                                             if float(k.split("|")[1]) < 4.0 and v["central"] is not None}
    headline["jackson_central_4h_onward_range"] = pick(
        jack_ratios, [k for k, v in jack_ratios.items() if float(k.split("|")[1]) >= 4.0
                      and v["concentration_xMIC"] == 2.0], "central")
    # Both sources together, at 1 x MIC or above: before 4 h and from 4 h on.
    early, late = {}, {}
    for x in (1.0, 4.0):
        for t in (2.0, 4.0, 6.0):
            v = yam_ratios[f"ceftazidime|{x:g}|{t:g}"]["central"]
            if v is not None:
                (early if t < 4.0 else late)[f"yamaguchi|{x:g}x|{t:g}h"] = v
    for k, v in jack_ratios.items():
        tab, t = k.split("|")
        if v["concentration_xMIC"] == 2.0 and v["central"] is not None:
            (early if float(t) < 4.0 else late)[f"jackson|{tab}|{t}h"] = v["central"]
    headline["central_before_4h_both"] = early
    headline["central_before_4h_both_range"] = [min(early.values()), max(early.values())]
    headline["central_4h_onward_both_range"] = [min(late.values()), max(late.values())]
    # Where a central reading is undefined because ceftazidime had not yet
    # lowered the count, the guaranteed lower bound can still be informative.
    informative = {k: v for k, v in jack_ratios.items()
                   if v["concentration_xMIC"] == 2.0 and v["central"] is None
                   and v["lower_bound"] is not None and v["lower_bound"] > 1.0}
    headline["jackson_early_lower_bound_without_net_kill"] = {
        k: {"lower_bound": v["lower_bound"], "control_fold": v["control_fold"], "isolate": v["isolate"],
            "ceftazidime_free_lps": jack[f"{k.split('|')[0]}|ceftazidime|{k.split('|')[1]}"]["free_lps_ng_mL"]}
        for k, v in informative.items()}
    yl = min(headline["yamaguchi_caz_ipm_lower_bound_4h"].values())
    jc = headline["jackson_central_2x_range"]
    headline["yamaguchi_lower_bound_4h_min"] = yl
    headline["lower_bound_exceeds_assumed_range"] = bool(yl > ASSUMED[1])
    headline["jackson_central_all_above_yamaguchi_bound"] = bool(jc and jc[0] >= yl)
    headline["n_jackson_central_2x"] = len(headline["jackson_central_2x"])

    payload = {
        "design": {"yamaguchi_inoculum_cfu_per_mL": YAM_INOCULUM, "floor_cfu_per_mL": FLOOR_CFU,
                   "killed_bounds": "N0 - N(t) <= K <= N0 x control fold - N(t)",
                   "background_bounds": "arm's own 0-h value (0 where unmeasured) <= B <= least free LPS "
                                        "of any arm of the same experiment at that time",
                   "lower_bound": "per-killed release of ceftazidime at its smallest over the bounds, "
                                  "of imipenem at its largest",
                   "central": "K at its lower bound and B at its minimum, for both agents"},
        "yamaguchi": yam, "jackson": jack,
        "ratios": {"yamaguchi": yam_ratios, "jackson": jack_ratios},
        "headline": headline,
        "unidentified": [
            "absolute release per killed cell: the assay's ng scale and the upper bound on killed biomass",
            "the split between lysis-coupled and kill-independent (filament) release",
            "background shedding as a function of growth (controls show it is large and phase-dependent)",
            "transfer from broth free LPS to in vivo plasma exposure, which is where the assumed range comes from"],
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (YAM, JACK)},
        "provenance": provenance(note="Bounds on per-killed-cell free LPS release from paired published data"),
    }
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(headline, indent=1))


if __name__ == "__main__":
    main()
