"""Markdown that renders as something other than what it says.

A pipe inside a table cell ends the cell, even inside backticks. `E[post | pre
intensity]` in a three-column table rendered as four columns with the formula
cut in half, and nothing but looking at the rendered page would have shown it.

Write it as `\\|`. The obvious alternative, `&#124;`, fixes the table and then
renders the entity literally inside the code span, which is worse than the bug.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)


def test_there_are_documents_to_check():
    assert len(DOCS) >= 3


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_table_cells_do_not_contain_bare_pipes(doc):
    bad = []
    for n, line in enumerate(doc.read_text().splitlines(), 1):
        if not line.lstrip().startswith("|"):
            continue
        for m in re.finditer(r"`([^`]*)`", line):
            # GFM accepts a backslash-escaped pipe, including inside a code
            # span. An HTML entity does not work there - it renders literally.
            if re.search(r"(?<!\\)\|", m.group(1)):
                bad.append("line %d: %s" % (n, m.group(0)))
    assert not bad, (
        "%s has a pipe inside a table cell, which ends the cell wherever it "
        "appears - write it as &#124;:\n  %s"
        % (doc.relative_to(ROOT), "\n  ".join(bad)))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_table_rows_have_a_consistent_column_count(doc):
    """A row with the wrong number of columns silently loses or shifts a cell."""
    rows, bad = [], []
    for n, line in enumerate(doc.read_text().splitlines() + [""], 1):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            # An escaped pipe is content, not a separator, so it must not be
            # counted as one - which is the mistake this check made first.
            rows.append((n, len(re.findall(r"(?<!\\)\|", s))))
            continue
        if len(rows) >= 2:
            counts = {c for _, c in rows}
            if len(counts) > 1:
                bad.append("lines %d-%d: column counts %s"
                           % (rows[0][0], rows[-1][0], sorted(counts)))
        rows = []
    assert not bad, "%s:\n  %s" % (doc.relative_to(ROOT), "\n  ".join(bad))
