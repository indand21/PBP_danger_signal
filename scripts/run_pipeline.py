"""Regenerate every danger-signal results file, in dependency order (plan item 8).

Each script declares the scripts whose results it reads, and starts as soon
as those have finished, so independent work overlaps (``--jobs`` scripts at a
time; the three heaviest also use their own process pools). Every script's
output goes to a log file, and the run stops at the first failure.

Afterwards it checks what the results must satisfy to be committed: every
results file parses as JSON and is written with LF line endings (the
fingerprints recorded in them are of LF bytes; see tests/test_line_endings.py).

GPU acceleration does not apply: the work is many small, stiff, serial ODE
solves with scalar Python right-hand sides, not array arithmetic, so the
speed comes from solving less (PK-only dose search, exact endotoxin update,
no cytokine solve where only endotoxin is reported) and from running
independent work side by side.

usage: python scripts/run_pipeline.py [--jobs N] [--only NAME ...] [--log-dir DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "danger_signal"

#: script -> (results it writes, scripts whose results it reads)
PIPELINE = {
    "ds_pilot_acceptance": (["pilot_acceptance.json"], []),
    "ds_phase2_build": (["phase2_build.json"], []),
    "ds_phase3_validate": (["phase3_validation.json"], []),
    "ds_phase4_analyses": (["phase4_analyses.json"], []),
    "ds_phase5_tolerance": (["phase5_tolerance.json"], []),
    "ds_mass_balance_convergence": (["mass_balance_convergence.json"], []),
    "ds_bacterial_refit": (["bacterial_refit.json"], []),
    "ds_pk_refit": (["pk_refit.json"], []),
    "ds_validation_targets": (["validation_targets.json"], ["ds_pd_context"]),
    "ds_multiarm_refit": (["multiarm_refit.json"], []),
    "ds_joint_broth_refit": (["joint_broth_refit.json"], []),
    "ds_broth_replay": (["broth_replay.json"], []),
    "ds_release_bounds": (["release_bounds.json"], []),
    "ds_filament_structure": (["filament_structure.json"], []),
    "ds_structural_sensitivity": (["structural_sensitivity.json"], []),
    "ds_pd_context": (["pd_context.json"], []),
    "ds_endpoint_windows": (["endpoint_windows.json"], ["ds_phase4_analyses"]),
    "ds_release_delay": (["release_delay.json"], ["ds_phase4_analyses"]),
    "ds_cytokine_ridge": (["cytokine_ridge.json"], ["ds_phase4_analyses"]),
    "ds_numerical_assurance": (["numerical_assurance.json"], ["ds_phase4_analyses"]),
    "ds_multiarm_downstream": (["multiarm_downstream.json"],
                               ["ds_phase4_analyses", "ds_multiarm_refit"]),
    "ds_joint_robustness": (["joint_robustness.json"],
                            ["ds_joint_broth_refit", "ds_multiarm_refit"]),
    "ds_kill_matched": (["kill_matched.json"], ["ds_phase4_analyses", "ds_release_delay"]),
    "ds_agent_pd_map": (["agent_pd_map.json"], ["ds_joint_robustness", "ds_multiarm_refit"]),
    "ds_joint_uncertainty": (["joint_uncertainty.json"], ["ds_phase4_analyses", "ds_kill_matched"]),
    # figures, from the stored results
    "ds_phase4_figures": ([], ["ds_phase4_analyses", "ds_endpoint_windows"]),
    "ds_agent_pd_figure": ([], ["ds_agent_pd_map"]),
    "ds_kill_matched_figure": ([], ["ds_kill_matched"]),
    "ds_joint_uncertainty_figure": ([], ["ds_joint_uncertainty"]),
    "ds_broth_replay_figure": ([], ["ds_broth_replay"]),
    "ds_pd_review_figures": ([], ["ds_joint_broth_refit", "ds_joint_robustness", "ds_pd_context"]),
}


def ready_scripts(todo, done, active):
    """Scripts in ``todo`` that may start now.

    A dependency is met when it has finished in this run, or when it is not
    part of this run at all (``--only``: its results must already be
    current). A dependency still waiting or still running is not met.
    ``todo`` maps script -> dependencies; ``active`` is the set of script
    names running now.
    """
    return [s for s, deps in todo.items()
            if all(d in done or (d not in todo and d not in active) for d in deps)]


def run_one(script, log_dir):
    log = Path(log_dir) / f"{script}.log"
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py")], cwd=ROOT,
                            stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    return script, rc, time.perf_counter() - t0, log


def check_outputs(names):
    """Every listed results file parses and has no CRLF."""
    bad = []
    for n in names:
        b = (RESULTS / n).read_bytes()
        if b"\r\n" in b:
            bad.append(f"{n}: CRLF line endings")
        try:
            json.loads(b.decode("utf-8"))
        except ValueError as e:
            bad.append(f"{n}: not valid JSON ({e})")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--only", nargs="*", help="run only these scripts (names without .py); "
                                              "their inputs must already be current")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()
    todo = {s: deps for s, (_, deps) in PIPELINE.items() if not args.only or s in args.only}
    for s, deps in todo.items():
        for d in deps:
            if d not in PIPELINE:
                raise SystemExit(f"{s} depends on unknown {d}")
    log_dir = args.log_dir or tempfile.mkdtemp(prefix="ds_pipeline_")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"logs in {log_dir}; {len(todo)} scripts, up to {args.jobs} at a time", flush=True)
    done, running, timings, t_start = set(), {}, [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        while todo or running:
            ready = ready_scripts(todo, done, set(running.values()))
            for s in ready[:max(0, args.jobs - len(running))]:
                running[ex.submit(run_one, s, log_dir)] = s
                del todo[s]
                print(f"  start  {s}", flush=True)
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in finished:
                script, rc, dt, log = f.result()
                del running[f]
                timings.append((script, dt))
                print(f"  {'done' if rc == 0 else 'FAILED'}   {script:32s} {dt:7.1f} s  "
                      f"(+{(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)
                if rc != 0:
                    print(f"\n{script} failed; last lines of {log}:", flush=True)
                    print("".join(log.read_text(encoding="utf-8").splitlines(True)[-25:]))
                    os._exit(1)
                bad = check_outputs(PIPELINE[script][0])
                if bad:
                    print("\noutput check failed:\n  " + "\n  ".join(bad))
                    os._exit(1)
                done.add(script)
    total = time.perf_counter() - t_start
    print(f"\nall {len(timings)} scripts done in {total / 60:.1f} min wall clock "
          f"({sum(d for _, d in timings) / 60:.1f} min of script time)")


if __name__ == "__main__":
    main()
