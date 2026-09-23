"""Tests for file lookup (mekiki/paths.py).

Three properties to protect.

1. **Inside the repository, locations are fixed.**
   The LLM cache is accumulated with real charges; if its location moved,
   every row would be re-billed on each re-run
2. **Nothing breaks outside the repository (pip-installed state).**
   There is no sampledata/ next to site-packages
3. **Locations can be moved with an environment variable.** The escape hatch
   for shared machines and CI

Run: .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

from pathlib import Path

import mekiki
from mekiki import paths


def test_cache_inside_repository_points_to_the_usual_place(monkeypatch):
    monkeypatch.delenv("MEKIKI_CACHE_DIR", raising=False)
    root = paths.repo_root()
    assert root is not None, "tests are expected to run from inside the repository"
    assert paths.cache_dir("llm_cache") == root / "sampledata" / "processed" / "llm_cache"


def test_environment_variable_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MEKIKI_CACHE_DIR", str(tmp_path))
    assert paths.cache_dir("llm_cache") == tmp_path / "llm_cache"


def test_cache_outside_repository_is_per_user(monkeypatch, tmp_path):
    """Reproduce the site-packages state by faking the package location."""
    monkeypatch.delenv("MEKIKI_CACHE_DIR", raising=False)
    fake = tmp_path / "site-packages" / "mekiki" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.touch()
    monkeypatch.setattr(paths, "__file__", str(fake))
    assert paths.repo_root() is None
    got = paths.cache_dir("llm_cache")
    assert got == paths.user_cache_root() / "llm_cache"
    # Must not point to an unwritable or nonexistent place
    assert got.is_absolute()


def test_bundled_data_is_found_inside_and_outside_repository(monkeypatch, tmp_path):
    name = "vehicles_sample500.csv"
    in_repo = paths.sample_data(name)
    assert in_repo.is_file(), "inside the repository it points to sampledata/sample/"

    # The distribution copies it to mekiki/data/ (pyproject's force-include)
    fake = tmp_path / "site-packages" / "mekiki" / "paths.py"
    fake.parent.mkdir(parents=True)
    fake.touch()
    monkeypatch.setattr(paths, "__file__", str(fake))
    assert paths.sample_data(name) == fake.parent / "data" / name


def test_settings_file_is_searched_upward_from_cwd(monkeypatch, tmp_path):
    """When used installed, the settings live in the user's project."""
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    (tmp_path / "config.local").write_text("X=1\n", encoding="utf-8")
    monkeypatch.chdir(deep)
    assert paths.find_dotenv("config.local") == tmp_path / "config.local"
    assert paths.find_dotenv("name_that_should_not_exist.local") is None


def test_display_path_is_relative_inside_repository():
    root = paths.repo_root()
    assert paths.display_path(root / "mekiki" / "paths.py") == "mekiki/paths.py"
    assert paths.display_path(Path("/tmp/x.csv")) == "/tmp/x.csv"


def test_type_hints_reach_the_user():
    """Without py.typed, mypy / pyright ignore the type hints even if written."""
    assert (Path(mekiki.__file__).parent / "py.typed").is_file()
