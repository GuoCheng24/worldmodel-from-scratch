"""Tests for the thing that checks everything else.

check_claims.py enforces every number in the README and had no tests of its
own. That is how it came to promise, in the README, that "an empty selection
exits non-zero - a check that matched nothing is not a check that passed",
while `--lessons=9` printed "12/12 backed" and exited 0.
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "check_claims.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=str(ROOT))


def test_a_lesson_that_does_not_exist_is_an_error():
    """The guarantee the README makes. It did not hold until it was tested."""
    r = run("--lessons=9", "/dev/null")
    assert r.returncode != 0, r.stdout
    assert "no claims for lesson 9" in r.stdout, r.stdout


def test_a_typo_for_two_lessons_is_an_error():
    """`--lessons=12` means lesson twelve, not lessons one and two, and there
    is no lesson twelve. Silently checking nothing and passing is the failure
    this catches."""
    r = run("--lessons=12", "/dev/null")
    assert r.returncode != 0, r.stdout


def test_one_bad_lesson_among_good_ones_is_still_an_error():
    """Asserting only on the exit code was not enough here: with an empty
    input file lesson 1's own claims all fail, so this returned non-zero for
    the wrong reason and passed with the guard removed. It has to see the
    message."""
    r = run("--lessons=1,7", "/dev/null")
    assert r.returncode != 0, r.stdout
    assert "no claims for lesson 7" in r.stdout, r.stdout


def test_no_files_is_an_error_not_a_pass():
    r = run("--lessons=1")
    assert r.returncode != 0


def test_an_unreadable_file_is_an_error():
    r = run(str(ROOT / "no-such-file.txt"))
    assert r.returncode != 0
    assert "cannot read" in r.stdout


def test_every_claim_pattern_compiles():
    """A pattern that does not compile would take the whole check down; one
    that compiles but cannot match is the subtler failure below."""
    sys.path.insert(0, str(ROOT))
    import check_claims
    for lesson, stable, name, pattern in check_claims.CLAIMS:
        re.compile(pattern)


def test_no_pattern_contains_a_unicode_escape_written_in_a_raw_string():
    r"""A raw string does not process \uXXXX.

    Two claim patterns were once written as r"...加..." and matched the
    literal characters backslash-u-5-2-a-0, which appear in no README. They
    could never have failed, and a check that cannot fail is worse than no
    check. Non-ASCII patterns are written as real characters instead.
    """
    bad = []
    for f in [SCRIPT] + sorted((ROOT / "tests").glob("*.py")):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r"""\br(['"])(?:(?!\1).)*\1""", line):
                if re.search(r"\\u[0-9a-fA-F]{4}", m.group(0)):
                    bad.append("%s:%d  %s" % (f.relative_to(ROOT), n, m.group(0)[:60]))
    assert not bad, "raw strings with \\uXXXX in them:\n  " + "\n  ".join(bad)
