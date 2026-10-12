"""
Command-line interface for the QSP Antibiotic Model.

Usage:
    python cli.py simulate --drug meropenem --dose 1000 --interval 8
    python cli.py cohort --n-patients 50 --drug meropenem
    python cli.py sensitivity --params k_growth,k_pers --samples 128
    python cli.py validate
    python cli.py checkpoints list

Subcommands:
    simulate    - Run a single simulation
    cohort      - Run a virtual patient cohort
    sensitivity - Run global sensitivity analysis
    validate    - Run literature validation pipeline
    checkpoints - Manage checkpoints (list, load, delete)
"""

import argparse
import sys
import json
import os
from typing import Optional


def cmd_simulate(args):
    """Run a single simulation."""
    from src.core.parameters import get_default_parameters, get_drug_pk_parameters
    from src.core.pd_model import BacterialPopulationODE
    from src.core.pk_models import TwoCompartmentPKModel, DosingRegimen
    from src.core.simulation import run_simulation

    print(f"Running simulation: {args.drug} ({args.drug_class})")
    print(f"  Dose: {args.dose}mg q{args.interval}h x {args.n_doses}")
    print(f"  Weight: {args.weight}kg, Immune: {args.immune}")

    params = get_default_parameters()
    pd_model = BacterialPopulationODE(params)

    pk_params = get_drug_pk_parameters(args.drug)
    pk_model = TwoCompartmentPKModel(
        CL=pk_params.CL, Vc=pk_params.Vc, Vp=pk_params.Vp,
        Q=pk_params.Q, Ka=pk_params.Ka, Kp=pk_params.Kp,
        effect_site_model=True,
    )

    regimen = DosingRegimen(
        dose_mg=args.dose,
        interval_hours=args.interval,
        start_time=0.0,
        n_doses=args.n_doses,
        infusion_duration_min=args.infusion,
    )

    ic = {
        "B_rep": args.burden,
        "B_pers": args.burden * 0.001,
        "B_SCV": 0,
        "N_eff": args.immune,
        "Damage": 0,
        "IL6": 10,
        "TNF": 5,
        "PAMP": 0,
    }

    t_span = (0, args.n_doses * args.interval + 24)

    result = run_simulation(
        pk_model=pk_model,
        regimen=regimen,
        pd_model=pd_model,
        initial_conditions=ic,
        t_span=t_span,
        drug_class=args.drug_class,
        weight_kg=args.weight,
    )

    # Extract key metrics
    import numpy as np
    t, B = result.get_bacterial_burden()
    _, _, il6 = result.get_cytokines()
    _, frac_scv = result.get_resistance_fraction()

    log_B = np.log10(np.maximum(B, 1e-10))

    print(f"\nResults:")
    print(f"  Final burden: {log_B[-1]:.2f} log10 CFU/mL")
    print(f"  Peak burden:  {log_B.max():.2f} log10 CFU/mL")
    print(f"  Min burden:   {log_B.min():.2f} log10 CFU/mL")
    print(f"  Peak IL-6:    {il6.max():.0f} pg/mL")
    print(f"  Final IL-6:   {il6[-1]:.0f} pg/mL")
    print(f"  Peak SCV:     {frac_scv.max():.4f}")

    # Save if requested
    if args.output:
        import pandas as pd
        df = result.df
        df.to_csv(args.output, index=False)
        print(f"\nSaved trajectory to: {args.output}")

    # Plot if requested
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(t, log_B, "b-", linewidth=2)
        axes[0, 0].set_xlabel("Time (hours)")
        axes[0, 0].set_ylabel("Bacterial burden (log10 CFU/mL)")
        axes[0, 0].set_title("Bacterial Burden")
        axes[0, 0].axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="VAP threshold")
        axes[0, 0].legend()

        axes[0, 1].plot(t, il6, "r-", linewidth=2)
        axes[0, 1].set_xlabel("Time (hours)")
        axes[0, 1].set_ylabel("IL-6 (pg/mL)")
        axes[0, 1].set_title("IL-6")

        axes[1, 0].plot(t, frac_scv, "g-", linewidth=2)
        axes[1, 0].set_xlabel("Time (hours)")
        axes[1, 0].set_ylabel("SCV fraction")
        axes[1, 0].set_title("Resistance (SCV)")

        # PK profile
        axes[1, 1].plot(t, result.y[:, 0] / 1000, "m-", linewidth=2, label="Central")
        axes[1, 1].set_xlabel("Time (hours)")
        axes[1, 1].set_ylabel("Amount (mg)")
        axes[1, 1].set_title("PK Profile")
        axes[1, 1].legend()

        fig.suptitle(f"{args.drug.title()} {args.drug_class} — {args.dose}mg q{args.interval}h", fontsize=14)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to: {args.plot}")


def cmd_cohort(args):
    """Run a virtual patient cohort."""
    from src.utils.parallel_sim import generate_cohort, run_cohort, aggregate_results

    print(f"Generating cohort of {args.n_patients} virtual patients...")
    patients = generate_cohort(
        n_patients=args.n_patients,
        drug_name=args.drug,
        drug_class=args.drug_class,
        seed=args.seed,
    )

    def progress(done, total):
        if done % max(1, total // 5) == 0 or done == total:
            print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    print("Running simulations...")
    results = run_cohort(
        patients,
        parallel=not args.sequential,
        max_workers=args.workers,
        progress_callback=progress,
    )

    agg = aggregate_results(results)

    print(f"\nCohort Summary:")
    print(f"  Success rate: {agg['success_rate']:.1%} ({agg['n_success']}/{agg['n_total']})")
    if agg["metrics_mean"]:
        print(f"  Mean final burden: {agg['metrics_mean'].get('final_burden_log10', 'N/A'):.2f} log10 CFU/mL")
        print(f"  Std final burden:  {agg['metrics_std'].get('final_burden_log10', 'N/A'):.2f}")
        print(f"  Mean peak IL-6:    {agg['metrics_mean'].get('peak_il6', 'N/A'):.0f} pg/mL")

    # Save results
    if args.output:
        from src.utils.parallel_sim import results_to_dataframe
        df = results_to_dataframe(results)
        df.to_csv(args.output, index=False)
        print(f"\nSaved results to: {args.output}")

    # Save aggregation
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(agg, f, indent=2, default=str)
        print(f"Saved summary to: {args.summary}")

    # Plot
    if args.plot:
        from src.utils.parallel_sim import plot_cohort_summary, plot_kinetics_overlay
        plot_cohort_summary(results, save_path=args.plot)
        print(f"Saved plot to: {args.plot}")
        if args.kinetics_plot:
            plot_kinetics_overlay(results, save_path=args.kinetics_plot)
            print(f"Saved kinetics plot to: {args.kinetics_plot}")


def cmd_sensitivity(args):
    """Run global sensitivity analysis."""
    from src.analysis.sensitivity_analysis import run_sensitivity_analysis, plot_sobol_indices

    param_names = args.params.split(",") if args.params else None

    print(f"Running Sobol sensitivity analysis...")
    print(f"  Parameters: {param_names or 'all defaults'}")
    print(f"  Metric: {args.metric}")
    print(f"  Samples: {args.samples}")

    result = run_sensitivity_analysis(
        param_names=param_names,
        drug_name=args.drug,
        drug_class=args.drug_class,
        metric=args.metric,
        n_samples=args.samples,
        calc_second_order=not args.first_order_only,
        seed=args.seed,
        print_progress=True,
    )

    # Save results
    if args.output:
        import numpy as np
        Si = result["Si"]
        output_data = {
            "names": result["problem"]["names"],
            "S1": Si["S1"].tolist(),
            "S1_conf": Si["S1_conf"].tolist(),
            "ST": Si["ST"].tolist(),
            "ST_conf": Si["ST_conf"].tolist(),
        }
        if "S2" in Si:
            output_data["S2"] = Si["S2"].tolist()
            output_data["S2_conf"] = Si["S2_conf"].tolist()

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to: {args.output}")

    # Plot
    if args.plot:
        plot_sobol_indices(result, save_path=args.plot, show=False)
        print(f"Saved plot to: {args.plot}")


def cmd_validate(args):
    """Run literature validation pipeline."""
    from src.analysis.literature_validation import run_full_validation, save_validation_report, plot_validation_summary

    print("Running literature validation pipeline...")
    results = run_full_validation(
        drug_cidal=args.drug_cidal,
        drug_static=args.drug_static,
        verbose=True,
    )

    # Save results
    if args.output:
        save_validation_report(results, args.output)
        print(f"\nSaved report to: {args.output}")

    # Plot
    if args.plot:
        plot_validation_summary(results, save_path=args.plot, show=False)
        print(f"Saved plot to: {args.plot}")


def cmd_checkpoints(args):
    """Manage checkpoints."""
    from src.utils.checkpoint import CheckpointManager

    mgr = CheckpointManager(args.checkpoint_dir)

    if args.ckpt_action == "list":
        ckpts = mgr.list_checkpoints(checkpoint_type=args.type)
        if not ckpts:
            print("No checkpoints found.")
            return

        print(f"Checkpoints ({len(ckpts)}):")
        print("-" * 80)
        for c in ckpts:
            size = mgr.get_size(c.checkpoint_id)
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024*1024):.1f} MB"
            print(f"  {c.checkpoint_id}")
            print(f"    Type: {c.checkpoint_type} | Items: {c.n_items} | Size: {size_str}")
            print(f"    Created: {c.created_at}")
            print(f"    Description: {c.description}")
            if c.tags:
                print(f"    Tags: {c.tags}")
            print()

    elif args.ckpt_action == "delete":
        try:
            meta = mgr.get_metadata(args.id)
            mgr.delete_checkpoint(args.id)
            print(f"Deleted checkpoint: {args.id}")
        except FileNotFoundError:
            print(f"Checkpoint not found: {args.id}")

    elif args.ckpt_action == "info":
        try:
            meta = mgr.get_metadata(args.id)
            size = mgr.get_size(args.id)
            print(f"Checkpoint: {meta.checkpoint_id}")
            print(f"  Type: {meta.checkpoint_type}")
            print(f"  Created: {meta.created_at}")
            print(f"  Description: {meta.description}")
            print(f"  Items: {meta.n_items}")
            print(f"  Status: {meta.status}")
            print(f"  Size: {size / 1024:.1f} KB")
            if meta.tags:
                print(f"  Tags: {meta.tags}")
        except FileNotFoundError:
            print(f"Checkpoint not found: {args.id}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="qsp-antibiotics",
        description="QSP Antibiotic Model — Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py simulate --drug meropenem --dose 1000 --interval 8 --plot sim.png
  python cli.py cohort --n-patients 50 --drug meropenem --plot cohort.png
  python cli.py sensitivity --params k_growth,k_pers --samples 128
  python cli.py validate --output validation.json
  python cli.py checkpoints list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- simulate ---
    p_sim = subparsers.add_parser("simulate", help="Run a single simulation")
    p_sim.add_argument("--drug", default="meropenem", help="Drug name (default: meropenem)")
    p_sim.add_argument("--drug-class", default="cidal", choices=["cidal", "static"],
                       help="Drug class (default: cidal)")
    p_sim.add_argument("--dose", type=float, default=1000, help="Dose in mg (default: 1000)")
    p_sim.add_argument("--interval", type=float, default=8, help="Dosing interval in hours (default: 8)")
    p_sim.add_argument("--n-doses", type=int, default=12, help="Number of doses (default: 12)")
    p_sim.add_argument("--infusion", type=float, default=60, help="Infusion duration in minutes (default: 60)")
    p_sim.add_argument("--weight", type=float, default=70, help="Patient weight in kg (default: 70)")
    p_sim.add_argument("--immune", type=float, default=1e7, help="Immune effector count (default: 1e7)")
    p_sim.add_argument("--burden", type=float, default=1e5, help="Initial bacterial burden (default: 1e5)")
    p_sim.add_argument("--output", "-o", help="Save trajectory to CSV file")
    p_sim.add_argument("--plot", "-p", help="Save plot to file")
    p_sim.set_defaults(func=cmd_simulate)

    # --- cohort ---
    p_cohort = subparsers.add_parser("cohort", help="Run a virtual patient cohort")
    p_cohort.add_argument("--n-patients", type=int, default=50, help="Number of patients (default: 50)")
    p_cohort.add_argument("--drug", default="meropenem", help="Drug name (default: meropenem)")
    p_cohort.add_argument("--drug-class", default="cidal", choices=["cidal", "static"])
    p_cohort.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_cohort.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    p_cohort.add_argument("--sequential", action="store_true", help="Run sequentially")
    p_cohort.add_argument("--output", "-o", help="Save results to CSV")
    p_cohort.add_argument("--summary", "-s", help="Save aggregation summary to JSON")
    p_cohort.add_argument("--plot", "-p", help="Save summary plot")
    p_cohort.add_argument("--kinetics-plot", help="Save kinetics overlay plot")
    p_cohort.set_defaults(func=cmd_cohort)

    # --- sensitivity ---
    p_sa = subparsers.add_parser("sensitivity", help="Run sensitivity analysis")
    p_sa.add_argument("--params", help="Comma-separated parameter names (default: all)")
    p_sa.add_argument("--drug", default="meropenem", help="Drug name (default: meropenem)")
    p_sa.add_argument("--drug-class", default="cidal", choices=["cidal", "static"])
    p_sa.add_argument("--metric", default="auc_burden",
                      choices=["auc_burden", "peak_burden", "final_burden", "auc_il6", "peak_il6", "peak_resistance"],
                      help="Output metric (default: auc_burden)")
    p_sa.add_argument("--samples", type=int, default=128, help="Base sample count (default: 128)")
    p_sa.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_sa.add_argument("--first-order-only", action="store_true", help="Skip second-order indices")
    p_sa.add_argument("--output", "-o", help="Save results to JSON")
    p_sa.add_argument("--plot", "-p", help="Save plot")
    p_sa.set_defaults(func=cmd_sensitivity)

    # --- validate ---
    p_val = subparsers.add_parser("validate", help="Run literature validation")
    p_val.add_argument("--drug-cidal", default="meropenem", help="Cidal drug (default: meropenem)")
    p_val.add_argument("--drug-static", default="doxycycline", help="Static drug (default: doxycycline)")
    p_val.add_argument("--output", "-o", help="Save report to JSON")
    p_val.add_argument("--plot", "-p", help="Save plot")
    p_val.set_defaults(func=cmd_validate)

    # --- checkpoints ---
    p_ckpt = subparsers.add_parser("checkpoints", help="Manage checkpoints")
    p_ckpt.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    ckpt_sub = p_ckpt.add_subparsers(dest="ckpt_action", help="Checkpoint actions")

    p_list = ckpt_sub.add_parser("list", help="List checkpoints")
    p_list.add_argument("--type", choices=["simulation", "cohort", "custom"], help="Filter by type")

    p_delete = ckpt_sub.add_parser("delete", help="Delete a checkpoint")
    p_delete.add_argument("id", help="Checkpoint ID to delete")

    p_info = ckpt_sub.add_parser("info", help="Show checkpoint details")
    p_info.add_argument("id", help="Checkpoint ID")

    p_ckpt.set_defaults(func=cmd_checkpoints)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
