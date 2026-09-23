"""The tutorial notebooks in `examples/` must keep running.

The notebooks are the only end-to-end walkthrough of the library, and they are
committed with their outputs, so nothing notices when an API change breaks them
unless the code is actually run. This runs their code cells in order, **up to
the first cell that would call the LLM**: a cell tagged `calls-llm`, or one that calls
`.predict(`. Everything before it is free and offline, which is also what the notebooks
promise the reader.

Only the standard library is needed to read a notebook, so this works without
Jupyter installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))
LLM_TAG = "calls-llm"          # cell tag for LLM calls that a text match cannot tell apart

pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="examples/ is not shipped with this install")


def _free_code_cells(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    out = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if ".predict(" in src or LLM_TAG in c.get("metadata", {}).get("tags", []):
            break
        out.append(src)
    return out


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_without_calling_the_LLM(path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    cells = _free_code_cells(path)
    assert len(cells) >= 3, "the notebook lost its code cells"
    ns: dict = {"__name__": "__notebook__"}
    for k, src in enumerate(cells):
        try:
            exec(compile(src, f"{path.name}[code cell {k}]", "exec"), ns)  # noqa: S102
        except Exception as e:                       # say which cell, not just which line
            raise AssertionError(f"{path.name}: code cell {k} failed: {e!r}\n\n{src}") from e


def test_notebooks_are_committed_with_outputs_and_without_local_paths():
    for path in NOTEBOOKS:
        text = path.read_text(encoding="utf-8")
        cells = json.loads(text)["cells"]
        ran = [c for c in cells if c["cell_type"] == "code" and c.get("outputs")]
        assert ran, f"{path.name} has no outputs; commit it after running every cell"
        for marker in ("/home/", "/Users/", "/work/", "/tmp/"):
            assert marker not in text, f"{path.name} leaks a local path ({marker})"
