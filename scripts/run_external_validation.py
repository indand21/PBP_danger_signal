"""
Run curve-level external validation of the QSP antibiotic model against
digitized published datasets.

Usage:
    python scripts/run_external_validation.py                # validation set only
    python scripts/run_external_validation.py --all-roles    # + calibration set
    python scripts/run_external_validation.py --no-plots

Outputs a JSON report and per-dataset overlay figures. See
data/external_validation/README.md for how to add datasets.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.external_validation import run_external_validation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-roles", action="store_true",
                    help="Include calibration-role datasets in the report (context only).")
    ap.add_argument("--no-plots", action="store_true", help="Skip overlay figures.")
    args = ap.parse_args()

    roles = ("calibration", "validation") if args.all_roles else ("validation",)
    run_external_validation(roles=roles, make_plots=not args.no_plots, verbose=True)


if __name__ == "__main__":
    main()
