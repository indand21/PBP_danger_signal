"""The pipeline runner's dependency declarations match what the scripts read."""
import re
from pathlib import Path

from scripts.run_pipeline import PIPELINE

ROOT = Path(__file__).resolve().parents[1]
#: A results file named as a path or as a whole quoted string (a mention
#: inside prose does not count as a read).
RESULT_NAME = re.compile(r"""(?:danger_signal/|["'])([a-z0-9_]+\.json)["']""")
#: Results files no pipeline script produces, and why.
NOT_REGENERATED = {
    "pd_reassessment_checks.json": "historical QA snapshot from a one-off, untracked script "
                                   "(tmp/final_pd_checks.py); its fingerprints describe the "
                                   "files as they were then",
}


def test_every_results_file_has_exactly_one_producer():
    produced = [n for outs, _ in PIPELINE.values() for n in outs]
    assert len(produced) == len(set(produced))
    on_disk = {p.name for p in (ROOT / "results/danger_signal").glob("*.json")}
    assert on_disk - set(NOT_REGENERATED) <= set(produced), on_disk - set(produced)


def test_every_results_file_a_script_reads_comes_from_a_declared_dependency():
    producer = {n: s for s, (outs, _) in PIPELINE.items() for n in outs}
    for script, (outs, deps) in PIPELINE.items():
        src = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
        for name in set(RESULT_NAME.findall(src)):
            if name in producer and name not in outs:
                assert producer[name] in deps, f"{script} reads {name} without depending on {producer[name]}"


def test_a_script_waits_until_its_dependencies_have_finished():
    """A dependency still queued or still running is not met; a finished one
    is, and so is one excluded from the run (--only)."""
    from scripts.run_pipeline import ready_scripts
    todo = {"b": ["a"], "c": ["b"], "d": []}
    assert ready_scripts(todo, done=set(), active={"a"}) == ["d"]          # a running
    assert ready_scripts({"a": [], **todo}, done=set(), active=set()) == ["a", "d"]  # a queued
    assert ready_scripts(todo, done={"a"}, active=set()) == ["b", "d"]     # a finished
    assert ready_scripts({"c": ["b"]}, done=set(), active=set()) == ["c"]  # b not in this run


def test_the_dependency_graph_has_no_cycle():
    seen, stack = set(), set()

    def visit(s):
        assert s not in stack, f"cycle through {s}"
        if s in seen:
            return
        stack.add(s)
        for d in PIPELINE[s][1]:
            visit(d)
        stack.discard(s)
        seen.add(s)

    for s in PIPELINE:
        visit(s)
