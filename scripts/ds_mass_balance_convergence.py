"""Grid-refinement check on the danger-signal compartment's mass balance.

The compartment satisfies, exactly,

    k_ds * integral(D dt) = integral(src dt) - [D(T) - D(0)]

so the discrepancy between the two sides measures integration error alone.
Since plan item 8 the AUC is the exact integral for the piecewise-linear
release, and the residual is rounding on every grid. Until then it was a
trapezoid over the sampled signal, which left a residual of up to 4e-3
across the regimen library; this script keeps both, so the attribution of
that old residual to quadrature (second-order convergence) stays on record
beside the exact result that replaced it.

A deliberately transient-rich driver is used (killing switched on and off on
an eight-hourly cycle, as dosing does), because a smooth driver would
understate the error.

Writes results/danger_signal/mass_balance_convergence.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.danger_signal import (  # noqa: E402
    DangerSignalParameters, check_release_mass_balance, simulate_danger_signal,
)

OUT = os.path.join("results", "danger_signal", "mass_balance_convergence.json")
GRIDS = (300, 600, 1200, 2400, 4800, 9600)
T_END = 24.0
INTERVAL_H = 8.0


def run():
    p = DangerSignalParameters(Y_lysis=4.93, Y_filament=1.5, V_focus_mL=10.0)
    cells = []
    for n in GRIDS:
        t = np.linspace(0.0, T_END, n)
        on = (np.sin(2 * np.pi * t / INTERVAL_H) > 0).astype(float)
        out = simulate_danger_signal(t, np.full(n, 1e7), 0.8 * on,
                                     4.0 * on + 0.1, 1.0, p)
        trap = check_release_mass_balance(out, p, use_exact=False)
        exact = check_release_mass_balance(out, p)
        cells.append({"n_points": n, "dt_h": float(T_END / (n - 1)),
                      "abs_relative_residual":
                          abs(float(trap["relative_residual"])),
                      "abs_relative_residual_exact":
                          abs(float(exact["relative_residual"]))})
    for prev, cur in zip(cells, cells[1:]):
        cur["reduction_vs_previous"] = (prev["abs_relative_residual"]
                                        / cur["abs_relative_residual"])
    return {
        "driver": "constant burden 1e7 CFU/mL, kill switched on/off on an "
                  f"{INTERVAL_H:.0f} h cycle, {T_END:.0f} h horizon",
        "identity": "k_ds * AUC_D = integral(src dt) - [D(T) - D(0)]",
        "grids": cells,
        "coarsest_residual": cells[0]["abs_relative_residual"],
        "finest_residual": cells[-1]["abs_relative_residual"],
        "max_residual_exact_auc": max(c["abs_relative_residual_exact"] for c in cells),
        "refinement_factor": GRIDS[-1] / GRIDS[0],
        "interpretation":
            "With the AUC taken by a trapezoid over the sampled signal (the "
            "method before plan item 8) the residual falls by about four per "
            "doubling, i.e. at second order: it was discretization error, "
            "not a defect in the identity. With the exact AUC now used it is "
            "rounding on every grid.",
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = run()
    print(f"{'n':>7s} {'dt (h)':>9s} {'|rel residual|':>16s} {'ratio':>8s}")
    for c in payload["grids"]:
        r = c.get("reduction_vs_previous")
        print(f"{c['n_points']:7d} {c['dt_h']:9.4f} "
              f"{c['abs_relative_residual']:16.3e} "
              f"{('-' if r is None else f'{r:.2f}'):>8s}")
    print(f"\n{payload['interpretation']}")
    from src.utils.provenance import provenance
    from src.analysis.exact_release import NUMERICS
    payload["provenance"] = provenance(solver=NUMERICS)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
