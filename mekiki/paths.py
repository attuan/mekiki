"""Resolve file locations in one place (works both installed and inside the repository).

After `pip install`, the parent of the parent of `__file__` is **site-packages, not the
repository**. Paths such as `ROOT / "sampledata" / ...` then either do not exist or are
not writable. This module absorbs that difference.

Every function searches in the same order:

1. **An explicit location** (environment variable). The escape hatch for moving things
   on CI or a shared machine.
2. **If running inside a clone of the repository, a place inside it** (`sampledata/`).
3. Otherwise, **the per-user standard location** (the equivalent of `~/.cache/mekiki`).

Step 2 exists **for money**: the LLM cache is accumulated with real charges, and a
location that moved between runs would re-bill every row (see the top of `mekiki/llm.py`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Markers that identify the repository root. Both must exist to count as a source checkout.
_ROOT_MARKERS = ("pyproject.toml", "mekiki")


def repo_root() -> Path | None:
    """The root of the source repository if running inside it, otherwise None.

    Returns None when installed into site-packages. Callers use that to give up
    on "repository-relative" locations and switch to the per-user ones.
    """
    root = Path(__file__).resolve().parent.parent
    if all((root / m).exists() for m in _ROOT_MARKERS):
        return root
    return None


def user_cache_root() -> Path:
    """The per-OS user cache location (down to the `mekiki` subdirectory).

    platformdirs would do this, but it is not worth one more dependency for so
    little. Only three cases are handled: XDG, macOS and Windows.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base or Path.home() / "AppData" / "Local") / "mekiki" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mekiki"
    base = os.environ.get("XDG_CACHE_HOME")
    return Path(base or Path.home() / ".cache") / "mekiki"


def cache_dir(name: str) -> Path:
    """Return the cache location for the purpose `name` (without creating it).

    Under `MEKIKI_CACHE_DIR` if set; under `sampledata/processed/` inside the
    repository; otherwise the per-user location.
    """
    override = os.environ.get("MEKIKI_CACHE_DIR")
    if override:
        return Path(override).expanduser() / name
    root = repo_root()
    if root is not None:
        return root / "sampledata" / "processed" / name
    return user_cache_root() / name


def find_dotenv(filename: str = ".env") -> Path | None:
    """Locate the settings file (dotenv format by default). **Does not read it.**

    Walks upward from the current directory, because after installation the file
    lives "inside the user's project", not "next to the library". When running
    inside the repository, the repository root is also checked.
    """
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        candidate = d / filename
        if candidate.is_file():
            return candidate
    root = repo_root()
    if root is not None and (root / filename).is_file():
        return root / filename
    return None


def sample_data(name: str) -> Path:
    """Return the location of a bundled sample file (one file in `sampledata/sample/`).

    **Does not raise when the file is missing.** It returns the most likely
    candidate so the caller can check `exists()` and print guidance that says
    where it looked.

    Inside the repository this points to `sampledata/sample/`. In the
    distribution it points to `mekiki/data/` bundled in the package (pyproject's
    force-include copies the same repository files there).
    """
    root = repo_root()
    if root is not None:
        in_repo = root / "sampledata" / "sample" / name
        if in_repo.is_file():
            return in_repo
    packaged = Path(__file__).resolve().parent / "data" / name
    return packaged


def display_path(path: Path) -> str:
    """Short form for display: relative inside the repository, absolute outside it."""
    root = repo_root()
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)
