"""Text clean-up before embedding.

**Constant tokens are dropped by default.**
In our measurements, dropping tokens whose corpus frequency is at or above a threshold

  - lowered the MAE of 5-nearest-neighbour price prediction from 31.45 to 28.92 (10k JPY)
  - lowered the MAE of supervised learning (embeddings fed to LightGBM) from 13.68 to 13.29
    (10k JPY)
  - also shortened embedding time

so none of the three measurements got worse.
Instead of a human writing a stop-word list it is **decided from the data**,
which fits `SemanticEncoder`'s "no hand-written rules" intent.

On the other hand, the "cut at the first delimiter" rule hurt downstream accuracy in the same
measurement, so it is not a default. Removing more is not always better.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# Characters separating a run of tokens: full-width space, slash, middle dot etc.
# (independent of language or domain)
SEP = r"[\s/・,、|｜]+"
IDEO_SPACE = "　"


def tokenize(text: str) -> list[str]:
    """Split on delimiters. No morphological analysis (to stay language-independent)."""
    return [t for t in re.split(SEP, text.replace(IDEO_SPACE, " ")) if t]


def constant_tokens(texts: pd.Series, threshold: float = 0.9) -> list[str]:
    """Return tokens whose frequency is at or above threshold (tokens that do not tell rows apart).

    A token appearing in every row, such as a product name repeated in every title, says nothing
    about which row resembles which, yet it aligns the vector directions.
    """
    n = max(len(texts), 1)
    df = Counter()
    for s in texts.fillna(""):
        df.update(set(tokenize(s)))
    return sorted(w for w, k in df.items() if k / n >= threshold)


def drop_constant_tokens(texts: pd.Series, stop: list[str] | None = None,
                         threshold: float = 0.9) -> tuple[pd.Series, list[str]]:
    """Return the texts with constant tokens dropped, plus the list of dropped tokens.

    Passing stop uses that list (**so the tokens decided at fit time are reused at inference**.
    Recounting on inference data with few rows drops different tokens and shifts the
    representation).
    """
    if stop is None:
        stop = constant_tokens(texts, threshold)
    stopset = set(stop)
    out = texts.fillna("").map(
        lambda s: " ".join(t for t in tokenize(s) if t not in stopset))
    return out, list(stop)
