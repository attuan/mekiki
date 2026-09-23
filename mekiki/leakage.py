"""Duplicate-record detection (leak prevention).

**When the same item is in both train and test, it is not prediction but reading off the
answer.** Duplicates are common in real data: in wine reviews the same tasting note is
attached to several titles (7.7% of rows), in pet-adoption profiles one shelter reuses the
same description, and in the Craigslist used-car listings the same car is posted in
several regions (one VIN appeared up to 261 times, price included). In that last case,
after filtering, 145,997 of 346,371 rows (42%) were duplicates, and random-split
cross-validation with them left in makes **MAE look 4.2% better and R² 0.03 better**
(R² 0.880 -> 0.914).

Moreover, **the more unstructured text is used as a feature, the larger the inflation.**
mekiki is precisely a tool for turning text into features, so instead of relying on the
user's discipline the **library detects it**.
By the same reasoning, amounts in free text are already masked by default (`mask_amounts`).
Masking duplicates would break training itself, so here we **stop at a warning.**

    from mekiki import check_duplicates, check_overlap

    print(check_duplicates(df))              # duplicates within one table
    print(check_overlap(train, test))        # duplicates straddling train and test

`SemanticEncoder` and `EvidencePredictor` call this automatically on fit / predict and raise
a `MekikiWarning` when something is found (`check_leakage=False` disables it).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from mekiki.errors import MekikiWarning

#: Duplicate rate above which a warning is raised. 0 warns on even a single duplicate
DEFAULT_WARN_RATE = 0.0


def _keys(df: pd.DataFrame, keys: Sequence[str] | None,
          ignore: Sequence[str] | None) -> list[str]:
    """Decide which columns are compared. The default is "all columns must match".

    Columns whose value **differs even for the same item**, such as id, region or posting
    date, are removed with `ignore`. For the Craigslist used-car listings,
    `ignore=["id", "region", "state"]` fits reality.
    """
    cols = list(keys) if keys else list(df.columns)
    if ignore:
        cols = [c for c in cols if c not in set(ignore)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns to compare are missing from the data: {missing}")
    return cols


def _fingerprint(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """A fingerprint built from row contents. Instant even for tens of thousands of rows."""
    return pd.util.hash_pandas_object(df[list(cols)].astype(str), index=False)


@dataclass
class DuplicateReport:
    """Duplicates within one table. Printing it gives a readable summary."""

    n_rows: int
    n_duplicate_rows: int             # rows beyond the first of each group (= droppable rows)
    n_groups: int                     # number of groups with 2 or more rows
    largest_group: int
    columns: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_duplicate_rows / self.n_rows if self.n_rows else 0.0

    @property
    def ok(self) -> bool:
        return self.n_duplicate_rows == 0

    def __str__(self) -> str:
        if self.ok:
            return (f"No duplicates ({self.n_rows:,} rows / "
                    f"compared on {len(self.columns)} columns)")
        return "\n".join([
            f"**{self.n_duplicate_rows:,} duplicate rows found**"
            f"({self.rate:.1%} of {self.n_rows:,} rows / "
            f"{self.n_groups:,} groups / largest group {self.largest_group:,} rows)",
            "  When several rows have identical contents, random-split cross-validation"
            " puts the same record in both train and test.",
            "  That turns prediction into reading off the answer, and the score"
            " comes out better than the real ability (measured: R² 0.880 -> 0.914).",
            "  -> Collapse to one row per record before evaluating.",
        ])


@dataclass
class OverlapReport:
    """Duplicates straddling train and test. **This is the actual harm.**"""

    n_train: int
    n_test: int
    n_leaked_rows: int                # test rows whose contents also appear in train
    columns: list[str] = field(default_factory=list)
    examples: list[int] = field(default_factory=list)   # row numbers on the test side

    @property
    def rate(self) -> float:
        return self.n_leaked_rows / self.n_test if self.n_test else 0.0

    @property
    def ok(self) -> bool:
        return self.n_leaked_rows == 0

    def __str__(self) -> str:
        if self.ok:
            return (f"No overlap between train and test"
                    f" (train {self.n_train:,} / test {self.n_test:,} rows)")
        head = ", ".join(str(i) for i in self.examples[:5])
        return "\n".join([
            f"**{self.n_leaked_rows:,} test rows ({self.rate:.1%}) have the same "
            f"contents as train** (compared on {len(self.columns)} columns / "
            f"e.g. rows {head})",
            "  Those rows are being solved with the answer already known,"
            " so the evaluation comes out better than the real ability.",
            "  -> Collapse to one row per record before splitting, or keep each"
            " group on one side of the split (GroupKFold etc.).",
        ])


def check_duplicates(df: pd.DataFrame, keys: Sequence[str] | None = None,
                     ignore: Sequence[str] | None = None) -> DuplicateReport:
    """Count how many rows within one table have identical contents.

    Parameters
    ----------
    keys:
        Columns to compare. All columns if omitted. If there is a single identifier
        column such as a serial number or a VIN, `keys=["VIN"]` is the most accurate
        (in the used-car data one VIN was duplicated up to 261 times).
    ignore:
        Columns excluded from the comparison. Put in columns whose value **differs
        even for the same item**, such as id or region.
    """
    cols = _keys(df, keys, ignore)
    fp = _fingerprint(df, cols)
    counts = fp.value_counts()
    dup_groups = counts[counts > 1]
    return DuplicateReport(
        n_rows=len(df),
        n_duplicate_rows=int(dup_groups.sum() - len(dup_groups)),
        n_groups=int(len(dup_groups)),
        largest_group=int(dup_groups.max()) if len(dup_groups) else 1,
        columns=cols)


def check_overlap(train: pd.DataFrame, test: pd.DataFrame,
                  keys: Sequence[str] | None = None,
                  ignore: Sequence[str] | None = None) -> OverlapReport:
    """Count the test rows whose contents also appear in train.

    Only columns present in **both** train and test are compared
    (test may lack the target column).
    """
    cols = [c for c in _keys(train, keys, ignore) if c in test.columns]
    if not cols:
        raise KeyError("train and test have no columns in common.")
    known = set(_fingerprint(train, cols).tolist())
    fp = _fingerprint(test, cols)
    hit = [i for i, v in enumerate(fp.tolist()) if v in known]
    return OverlapReport(n_train=len(train), n_test=len(test),
                         n_leaked_rows=len(hit), columns=cols, examples=hit)


def warn_if_leaky(report: DuplicateReport | OverlapReport,
                  where: str = "", rate: float = DEFAULT_WARN_RATE) -> bool:
    """Warn if the report shows a problem. Returns True when a warning was raised.

    **Not an exception.** Duplicates are sometimes intentional, and stopping would
    make "just run it for now" impossible. The point is to be noticed.
    """
    if report.ok or report.rate <= rate:
        return False
    prefix = f"[{where}] " if where else ""
    warnings.warn(prefix + str(report), MekikiWarning, stacklevel=3)
    return True
