"""Components that turn text into vectors. Every vectorizer is swappable.

Encoders are swappable: anything satisfying the `Vectorizer` protocol will do.
Three are bundled.

  CharTfidfVectorizer          ... the default, runs with no extra dependencies. Compresses the
                              TF-IDF of character n-grams with SVD. Empirically robust to
                              unseen values
  SentenceTransformerVectorizer... multilingual-e5-small etc. Needs torch, so it is an optional
                              extra (`pip install "mekiki[embed]"`)
  PrecomputedVectorizer        ... only reads an embedding parquet that is already computed

**Why the default is not an embedding model.** Measured on values absent from the training
data (Craigslist used cars), character TF-IDF (3,715) beat the embedding (3,873), and
overall the two were equal to marginally different. Making torch mandatory would stop anyone
from trying it, so the default is the dependency-free one and embeddings come in as a swap.

Every vectorizer returns **vectors normalised to length 1**.
That way the dot product is directly the cosine similarity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from mekiki.errors import MekikiError


@runtime_checkable
class Vectorizer(Protocol):
    """Text series -> vectors. Same fit / transform shape as scikit-learn."""

    name: str

    def fit(self, texts: pd.Series) -> Vectorizer: ...

    def transform(self, texts: pd.Series) -> np.ndarray: ...


def _l2_normalize(V: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (V / norm).astype("float32")


class CharTfidfVectorizer:
    """Default vectorizer: TF-IDF of character n-grams compressed with SVD.

    It does not split into words, so Japanese and English are handled alike,
    and `f-250 lariat` decomposes into fragments `f-2` `250` `lar` even when unseen.
    Its robustness to unseen words comes from this property
    (measured: values absent from training data 3,715 vs embedding 3,873).
    """

    def __init__(self, n_components: int = 256, ngram_range: tuple[int, int] = (2, 4),
                 min_df: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_state = random_state
        self.name = f"char_tfidf_svd{n_components}"
        self._vec = None
        self._svd = None

    def fit(self, texts: pd.Series) -> CharTfidfVectorizer:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                                    min_df=self.min_df)
        X = self._vec.fit_transform(texts.fillna(""))
        # The SVD output dimension can exceed neither the vocabulary size nor the row count
        dim = int(min(self.n_components, X.shape[1] - 1, max(X.shape[0] - 1, 1)))
        self._svd = TruncatedSVD(n_components=max(dim, 1), random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        if self._vec is None or self._svd is None:
            raise MekikiError("CharTfidfVectorizer has not been fitted yet.")
        return _l2_normalize(self._svd.transform(self._vec.transform(texts.fillna(""))))


class SentenceTransformerVectorizer:
    """Uses a sentence-transformers model (requires torch).

    torch is not a required dependency; install it with `pip install "mekiki[embed]"`.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small",
                 prefix: str = "query: ", max_seq_length: int = 128,
                 batch_size: int = 64):
        self.model_name = model_name
        self.prefix = prefix
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.name = model_name
        self._model = None

    def fit(self, texts: pd.Series) -> SentenceTransformerVectorizer:
        # A pretrained model: learns nothing from the corpus (only loads)
        self._load()
        return self

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - environment dependent
                raise MekikiError(
                    "sentence-transformers is not installed. "
                    'Install it with `pip install "mekiki[embed]"`, '
                    "or use CharTfidfVectorizer / PrecomputedVectorizer.") from e
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def transform(self, texts: pd.Series) -> np.ndarray:
        model = self._load()
        vecs = model.encode((self.prefix + texts.fillna("")).tolist(),
                            batch_size=self.batch_size, normalize_embeddings=True,
                            show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype="float32"))


class PrecomputedVectorizer:
    """Vectorizer that only reads precomputed embeddings.

    For when a text-to-vector table already exists, such as
    `sampledata/processed/*_emb_*.parquet`. **It cannot handle unseen text**,
    so `missing="error"` makes that case noticeable.
    """

    def __init__(self, table: pd.DataFrame, text_col: str, name: str = "precomputed",
                 missing: str = "error"):
        self.text_col = text_col
        self.name = name
        self.missing = missing
        cols = [c for c in table.columns if c != text_col]
        self._lookup = {t: i for i, t in enumerate(table[text_col].astype(str))}
        self._V = _l2_normalize(table[cols].to_numpy(dtype="float32"))

    def fit(self, texts: pd.Series) -> PrecomputedVectorizer:
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        out = np.zeros((len(texts), self._V.shape[1]), dtype="float32")
        missing = []
        for r, t in enumerate(texts.fillna("").astype(str)):
            i = self._lookup.get(t)
            if i is None:
                missing.append(t)
                continue
            out[r] = self._V[i]
        if missing and self.missing == "error":
            raise MekikiError(
                f"{len(missing)} texts are absent from the precomputed embeddings"
                f" (e.g. {missing[0][:40]!r}). "
                "Rebuild the embeddings, or use missing='zero' to treat them as zero vectors.")
        return out


class CachedVectorizer:
    """Wraps any vectorizer with a per-text cache.

    You only pay for a text once: the same string is never computed twice.
    Passing `cache_dir` persists it as parquet.
    """

    def __init__(self, vectorizer: Vectorizer, cache_dir: str | Path | None = None):
        self.vectorizer = vectorizer
        self.name = getattr(vectorizer, "name", type(vectorizer).__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, np.ndarray] = {}
        self._loaded = False

    def _path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(self.name.encode()).hexdigest()[:12]
        return self.cache_dir / f"mekiki_cache_{key}.parquet"

    def _load(self) -> None:
        p = self._path()
        if self._loaded or p is None or not p.exists():
            self._loaded = True
            return
        tb = pd.read_parquet(p)
        V = tb.drop(columns=["text"]).to_numpy(dtype="float32")
        for t, v in zip(tb["text"].astype(str), V, strict=True):
            self._cache[t] = v
        self._loaded = True

    def save(self) -> None:
        p = self._path()
        if p is None or not self._cache:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        texts = list(self._cache)
        V = np.vstack([self._cache[t] for t in texts])
        tb = pd.DataFrame(V, columns=[f"v{i}" for i in range(V.shape[1])])
        tb.insert(0, "text", texts)
        tb.to_parquet(p, index=False)

    def fit(self, texts: pd.Series) -> CachedVectorizer:
        self.vectorizer.fit(texts)
        # Refitting the vectorizer invalidates the cache contents
        self._cache.clear()
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        self._load()
        s = texts.fillna("").astype(str)
        todo = sorted({t for t in s if t not in self._cache})
        if todo:
            V = self.vectorizer.transform(pd.Series(todo))
            for t, v in zip(todo, V, strict=True):
                self._cache[t] = v
        return np.vstack([self._cache[t] for t in s]).astype("float32")

    @property
    def n_cached(self) -> int:
        return len(self._cache)
