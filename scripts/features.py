"""ベースライン共通の特徴量づくり。

シエンタ（単一車種・日本語）と Craigslist（複数車種・英語）の両方で
同じ手続きを使うために、run_baselines.py から切り出した。
**ここを変えると両データセットの過去の数字が引き直しになる。**

原則は1つだけ: 前処理は必ず fold 内の train だけで fit する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

SEED = 42

LGBM_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=SEED,
    verbose=-1,
)

# TF-IDF の設定。単語版と文字版で「表記ゆれをどう吸収するか」が違う。
TFIDF_KW = {
    "word": dict(analyzer="word", token_pattern=r"\S+", min_df=5, max_features=300),
    "char": dict(analyzer="char_wb", ngram_range=(2, 4), min_df=10, max_features=1000),
}


def numeric_frame(df: pd.DataFrame, num: list[str], boo: list[str]) -> pd.DataFrame:
    """数値列と bool 列を取り出す。bool は 0/1 にするだけ。"""
    out = df[num].astype("float64").copy()
    for c in boo:
        out[c] = df[c].astype("float64")
    return out


def as_category(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    """train に出た水準だけをカテゴリとして固定する。

    test にしか無い水準（train に無かった値）は NaN になり、
    LightGBM は欠損として扱う。train を見て決めるのが原則。
    """
    tr, te = train[cols].copy(), test[cols].copy()
    for c in cols:
        cats = pd.Index(tr[c].dropna().unique())
        tr[c] = pd.Categorical(tr[c], categories=cats)
        te[c] = pd.Categorical(te[c].where(te[c].isin(cats)), categories=cats)
    return tr, te


def tfidf_frames(train: pd.DataFrame, test: pd.DataFrame, col: str, kind: str):
    """テキスト列を TF-IDF に変換する。fit は train のみ。

    欠損は空文字にする。記載が無いこと自体が情報なので行は落とさない。
    """
    vec = TfidfVectorizer(**TFIDF_KW[kind])
    A = vec.fit_transform(train[col].fillna("").to_numpy()).toarray()
    B = vec.transform(test[col].fillna("").to_numpy()).toarray()
    names = [f"tfidf_{i}" for i in range(A.shape[1])]
    return (pd.DataFrame(A, columns=names, index=train.index),
            pd.DataFrame(B, columns=names, index=test.index))


def predict_median(target: str):
    """何も見ずに train の中央値を返す。これを下回るモデルは無意味。"""
    def fit_predict(train, test):
        return np.full(len(test), train[target].median())
    return fit_predict


def make_linear(target: str, num: list[str], boo: list[str], cat: list[str]):
    """線形回帰 + one-hot ダミー = 従来手法の再現。

    Ridge にしているのは、水準数の多い列を one-hot したときに
    最小二乗が不安定になるのを防ぐため。正則化以外は素の線形回帰と同じ。
    """
    def fit_predict(train, test):
        Xtr_n = numeric_frame(train, num, boo)
        Xte_n = numeric_frame(test, num, boo)
        med = Xtr_n.median()
        Xtr_n, Xte_n = Xtr_n.fillna(med), Xte_n.fillna(med)

        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xtr_c = enc.fit_transform(train[cat].fillna("欠損").astype(str))
        Xte_c = enc.transform(test[cat].fillna("欠損").astype(str))

        model = Ridge(alpha=1.0)
        model.fit(np.hstack([Xtr_n.to_numpy(), Xtr_c]), train[target])
        return model.predict(np.hstack([Xte_n.to_numpy(), Xte_c]))
    return fit_predict


def make_lgbm(target: str, num: list[str], boo: list[str], cat: list[str],
              text_col: str | None = None, text_kind: str | None = None,
              log_target: bool = False, extra=None):
    """LightGBM。text_col を渡すと TF-IDF 列を、extra を渡すと任意の行列を足す。

    extra は (train_df, test_df) -> (DataFrame, DataFrame) の関数で、
    埋め込みなど「事前に計算しておいた密行列」を差し込むのに使う。
    """
    def fit_predict(train, test):
        Xtr = numeric_frame(train, num, boo)
        Xte = numeric_frame(test, num, boo)
        if cat:
            Ctr, Cte = as_category(train, test, cat)
            Xtr = pd.concat([Xtr, Ctr], axis=1)
            Xte = pd.concat([Xte, Cte], axis=1)
        if text_col:
            Ttr, Tte = tfidf_frames(train, test, text_col, text_kind)
            Xtr = pd.concat([Xtr, Ttr], axis=1)
            Xte = pd.concat([Xte, Tte], axis=1)
        if extra is not None:
            Etr, Ete = extra(train, test)
            Xtr = pd.concat([Xtr, Etr], axis=1)
            Xte = pd.concat([Xte, Ete], axis=1)

        y = train[target].to_numpy(dtype=float)
        if log_target:
            y = np.log(y)

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, y, categorical_feature=cat or "auto")
        pred = model.predict(Xte)
        return np.exp(pred) if log_target else pred
    return fit_predict
