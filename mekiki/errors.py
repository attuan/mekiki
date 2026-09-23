"""Exceptions raised by mekiki."""


class MekikiError(Exception):
    """A problem the user can fix, such as a configuration mistake.

    The message must say **what to do next**.
    In particular, a few hundred labelled rows are often not available in real
    datasets, so do not guess silently around teacher labels.
    """


class MekikiWarning(UserWarning):
    """Not serious enough to stop, but proceeding silently would mislead the reader.

    Duplicate-record detection (`mekiki.leakage`) uses this. **It is not an
    exception because duplicates are sometimes intentional, and stopping would
    make "just run it for now" impossible.** The point is to be noticed; it can
    be switched off with `check_leakage=False`.
    """


def missing_extra(module: str, extra: str) -> MekikiError:
    """Build the exception for a missing optional dependency.

    The only required dependencies are pandas / numpy / scikit-learn;
    LightGBM, XGBoost, anthropic and so on are added with `pip install mekiki[...]`
    (`pyproject.toml`). **Raising a bare ImportError leaves the user without
    any idea of what to install.** Every lazy import goes through here so the
    message includes the fix.
    """
    return MekikiError(
        f"{module} is not installed. Install it with `pip install \"mekiki[{extra}]\"` "
        f"(the only required dependencies are pandas / numpy / scikit-learn; "
        f"{module} is optional).")
