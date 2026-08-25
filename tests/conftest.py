"""Put the repository root on sys.path so `import wm` works however pytest is run.

`python -m pytest` inserts the working directory into sys.path; a bare `pytest`
does not, and neither does running the suite by absolute path from somewhere
else. Without this the suite passes one way and fails to collect the other two,
which is exactly what happened here: it was verified locally with `python -m
pytest` and CI ran a bare `pytest`, so a green local run sat next to a red
first build.

This file lives in tests/ rather than the repository root because pytest
collects conftest.py from the rootdir down to the test files, and the rootdir
depends on how it was invoked - a root-level conftest is skipped when the suite
is run by absolute path from another directory. This one is always collected.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
