"""Experimental context for PD calibration, separate from study defaults.

Sabet 2020 reports control growth over 12 h postinfection, with treatment
starting at 2 h: the primary control interval is 10 h. The paper gives only
an unpaired 1.16-2.08 log10 range across strains, so this constrains growth;
it does not identify six exact growth rates. Measured MIC is never recomputed
from a fitted in-vivo growth rate.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from src.core.parameters import get_default_parameters, immune_free_parameters
from src.core.pd_model import BacterialPopulationODE

ROOT = Path(__file__).resolve().parents[2]
CONTROL_ELAPSED_H = 10.0
CONTROL_DELTA_RANGE = (1.16, 2.08)
THIGH_CAPACITY_CFU = 1e10  # context assumption, not the broth CFU/mL capacity


def load_thigh_strains():
    with (ROOT / "data/external_validation/sabet2020_strains.csv").open(newline="") as f:
        return [{k:(v if k=="strain" else float(v) if v else None)
                 for k,v in r.items()} for r in csv.DictReader(f)]


def untreated_delta(growth, initial_total, elapsed_h, carrying_capacity=THIGH_CAPACITY_CFU):
    """Predict control growth in count units per thigh, immune-free.

    Cytokine states are irrelevant here and are not interpreted. The bacterial
    mass-balance equations are unchanged by expressing all counts per thigh.
    """
    if elapsed_h <= 0 or initial_total <= 100 or carrying_capacity <= initial_total:
        raise ValueError("Invalid control interval, inoculum or carrying capacity")
    p = immune_free_parameters(get_default_parameters())
    p["bacteria"].k_growth = float(growth)
    p["bacteria"].B_max = float(carrying_capacity)
    model = BacterialPopulationODE(p)
    initial = np.array([initial_total-100,100,0,0,0,0,0,0,0.])
    sol = solve_ivp(lambda t,y:model.rhs(t,y,0.,"cidal"),(0,elapsed_h),initial,
                    method="LSODA",rtol=1e-8,atol=1e-8,max_step=.1)
    if not sol.success: raise RuntimeError(sol.message)
    return float(np.log10(sol.y[:3,-1].sum()/initial_total))


def growth_from_control(delta, initial_total, elapsed_h=CONTROL_ELAPSED_H,
                        carrying_capacity=THIGH_CAPACITY_CFU):
    """Invert a control endpoint, without using a treated CFU or stasis target."""
    if delta < 0: raise ValueError("Control growth must be nonnegative")
    if initial_total*10**delta >= carrying_capacity:
        raise ValueError("Observed control burden exceeds the context carrying capacity")
    return float(brentq(lambda g:untreated_delta(g,initial_total,elapsed_h,carrying_capacity)-delta,
                        0.,5.,xtol=1e-9))


def strain_scenario(strain, growth, *, interval_h=6., carrying_capacity=THIGH_CAPACITY_CFU,
                    hill=None, n_grid=18):
    """A fixed-MIC, count-per-thigh context under canonical human PK.

    Dose/interval identity for each plotted point is unavailable. This is
    consequently a transport sensitivity, not a replay of the murine study.
    C50/MIC is transferred from the calibration isolate as an explicit
    assumption; an isolate's measured MIC is an input, not a fitted endpoint.
    """
    base = get_default_parameters()["bacteria"]
    overrides = {"k_growth":float(growth), "B_max":float(carrying_capacity),
                 "kill_C50":base.kill_C50 * strain["mic_mgL"] / 1.0}
    if hill is not None: overrides["kill_hill"] = float(hill)
    return {"index":"ft_mic", "mic_reference_mgL":strain["mic_mgL"],
            "mic_reference_basis":"measured broth MIC, Sabet Table 1",
            "initial_burden":10**strain["baseline_log10_cfu_thigh"]-100,
            "initial_persister":100., "interval_hours":float(interval_h),
            "t_end":24., "infusion_min":30., "weight_kg":70., "n_grid":n_grid,
            "bacterial_parameters":overrides,
            "transport_assumptions":"canonical human effect-site PK, not murine IP PK; "
                                    "C50/MIC transferred from P. aeruginosa"}
