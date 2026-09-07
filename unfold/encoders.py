"""テキストをベクトルにする部品（設計書の「Everything is swappable」に相当）。

エンコーダは差し替え可能で、`Encoder` プロトコルを満たせば何でもよい。
同梱するのは3つ。

  CharTfidfEncoder          … 追加依存なしで動く既定。文字 n-gram の TF-IDF を
                              特異値分解（SVD）で圧縮する。実測で未知の値に強い
  SentenceTransformerEncoder… multilingual-e5-small など。torch が要るので
                              主環境では入らない（このプロジェクトでは .venv-embed 側）
  PrecomputedEncoder        … すでに計算済みの埋め込み parquet を読むだけ

**なぜ既定が埋め込みモデルではないのか。** 2026-08-29 の実測で、Craigslist の
訓練データに無い値では文字 TF-IDF（3,715）が埋め込み（3,873）に勝っており、全体でも
両者は同等〜わずかな差だった。torch を必須にすると誰も試せなくなるので、
既定は依存なしで動く方にして、埋め込みは差し替えで使う。

すべてのエンコーダは**長さ1に正規化したベクトル**を返す。
こうしておくと内積がそのままコサイン類似度になる。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from unfold.errors import UnfoldError


@runtime_checkable
class Encoder(Protocol):
    """テキスト列 → ベクトル。scikit-learn と同じ fit / transform の形。"""

    name: str

    def fit(self, texts: pd.Series) -> "Encoder": ...

    def transform(self, texts: pd.Series) -> np.ndarray: ...


def _l2_normalize(V: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (V / norm).astype("float32")


class CharTfidfEncoder:
    """文字 n-gram の TF-IDF を SVD で圧縮する既定エンコーダ。

    単語に分割しないので日本語でも英語でも同じ扱いができ、
    `f-250 lariat` を知らなくても `f-2` `250` `lar` という断片に分解できる。
    未知語に強いのはこの性質による（実測: 訓練データに無い値 3,715 対 埋め込み 3,873）。
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

    def fit(self, texts: pd.Series) -> "CharTfidfEncoder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                                    min_df=self.min_df)
        X = self._vec.fit_transform(texts.fillna(""))
        # SVD の出力次元は、語彙数と行数のどちらも超えられない
        dim = int(min(self.n_components, X.shape[1] - 1, max(X.shape[0] - 1, 1)))
        self._svd = TruncatedSVD(n_components=max(dim, 1), random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        if self._vec is None or self._svd is None:
            raise UnfoldError("CharTfidfEncoder はまだ fit されていません。")
        return _l2_normalize(self._svd.transform(self._vec.transform(texts.fillna("")))) 


class SentenceTransformerEncoder:
    """sentence-transformers のモデルを使う（torch が必要）。

    このプロジェクトでは主環境に torch を入れていないので、
    使うときは `.venv-embed` 側から呼ぶこと。
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

    def fit(self, texts: pd.Series) -> "SentenceTransformerEncoder":
        # 学習済みモデルなので、コーパスからは何も学ばない（読み込むだけ）
        self._load()
        return self

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - 環境依存
                raise UnfoldError(
                    "sentence-transformers が入っていません。"
                    "主環境ではなく .venv-embed から実行するか、"
                    "CharTfidfEncoder / PrecomputedEncoder を使ってください。") from e
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def transform(self, texts: pd.Series) -> np.ndarray:
        model = self._load()
        vecs = model.encode((self.prefix + texts.fillna("")).tolist(),
                            batch_size=self.batch_size, normalize_embeddings=True,
                            show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype="float32"))


class PrecomputedEncoder:
    """計算済みの埋め込みを読むだけのエンコーダ。

    `sampledata/processed/*_emb_*.parquet` のように、テキストとベクトルの
    対応表がすでにある場合に使う。**未知のテキストは扱えない**ので、
    そのときは `missing="error"` で気づけるようにしてある。
    """

    def __init__(self, table: pd.DataFrame, text_col: str, name: str = "precomputed",
                 missing: str = "error"):
        self.text_col = text_col
        self.name = name
        self.missing = missing
        cols = [c for c in table.columns if c != text_col]
        self._lookup = {t: i for i, t in enumerate(table[text_col].astype(str))}
        self._V = _l2_normalize(table[cols].to_numpy(dtype="float32"))

    def fit(self, texts: pd.Series) -> "PrecomputedEncoder":
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
            raise UnfoldError(
                f"計算済み埋め込みに無いテキストが {len(missing)} 件あります"
                f"（例: {missing[0][:40]!r}）。"
                "埋め込みを作り直すか、missing='zero' で0ベクトルとして扱ってください。")
        return out


class CachedEncoder:
    """任意のエンコーダに、テキスト単位のキャッシュをかぶせる。

    設計書の「caches everything so you only pay for a row once」に相当する。
    同じ文字列は二度計算しない。`cache_dir` を渡すと parquet で永続化する。
    """

    def __init__(self, encoder: Encoder, cache_dir: str | Path | None = None):
        self.encoder = encoder
        self.name = getattr(encoder, "name", type(encoder).__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, np.ndarray] = {}
        self._loaded = False

    def _path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1(self.name.encode()).hexdigest()[:12]
        return self.cache_dir / f"unfold_cache_{key}.parquet"

    def _load(self) -> None:
        p = self._path()
        if self._loaded or p is None or not p.exists():
            self._loaded = True
            return
        tb = pd.read_parquet(p)
        V = tb.drop(columns=["テキスト"]).to_numpy(dtype="float32")
        for t, v in zip(tb["テキスト"].astype(str), V):
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
        tb.insert(0, "テキスト", texts)
        tb.to_parquet(p, index=False)

    def fit(self, texts: pd.Series) -> "CachedEncoder":
        self.encoder.fit(texts)
        # エンコーダを学習し直したらキャッシュの中身は無効になる
        self._cache.clear()
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        self._load()
        s = texts.fillna("").astype(str)
        todo = sorted({t for t in s if t not in self._cache})
        if todo:
            V = self.encoder.transform(pd.Series(todo))
            for t, v in zip(todo, V):
                self._cache[t] = v
        return np.vstack([self._cache[t] for t in s]).astype("float32")

    @property
    def n_cached(self) -> int:
        return len(self._cache)
