"""Every tracked text file is on disk with LF line endings.

Results record SHA-256 fingerprints of the bytes of their input files, and
the tests check them. Git stores text with LF. A file with CRLF on disk, which
is what a script run on Windows writes by default, has a fingerprint no fresh
clone can reproduce, yet every local check passes. That is how 17 recorded
fingerprints came to fail in fresh clones, on Windows or on Linux depending on
the file. .gitattributes makes every checkout LF; this test catches a file
that a script has since rewritten with CRLF, before its fingerprint is
recorded or committed.

To fix a reported file, convert it to LF; if a results file recorded its
fingerprint in the meantime, regenerate that results file too.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_text_files_have_lf_line_endings():
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "--eol", "-z"], cwd=ROOT,
                         capture_output=True, check=True).stdout
    crlf = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        info, path = rec.split(b"\t", 1)
        index_eol, worktree_eol = info.split()[:2]
        if index_eol == b"i/lf" and worktree_eol in (b"w/crlf", b"w/mixed"):
            crlf.append(path.decode("utf-8"))
    assert not crlf, f"{len(crlf)} tracked text file(s) have CRLF on disk: {crlf[:10]}"


def test_gitattributes_checks_text_out_with_lf():
    rules = (ROOT / ".gitattributes").read_text(encoding="utf-8").split("\n")
    assert "* text=auto eol=lf" in rules
