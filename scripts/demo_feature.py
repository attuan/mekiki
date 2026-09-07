"""機能A（Feature）を実データで測る — 教師ラベルの起点を3通り比べる。

    .venv/bin/python scripts/demo_feature.py

**何を確かめたいか。** 仕様書は「すでに 200 件の正解ラベルがある」前提で書かれているが、
中古車データにその正解は存在しない（PRD 機能A の 01「教師ラベル」）。
そこで起点の作り方を変えて、同じ 5-fold・同じ下流モデルで比べる。

  (b) ラベル0件 … `values=[...]` に値の名前だけ宣言し、その名前を参照点にする
  (a) ラベル200件 … 仕様書の前提どおりの件数だけ人手ラベルを与える
  (a') ラベル全件 … 上限の参考（訓練データのラベルを全部使う）
  (c) 埋め込み列  … ラベルという概念を使わず、そのまま特徴量にする

比較の基準は既存のベースライン。

  A2  従来列のみ（テキストを使わない）               MAE 17.51 万円
  Ab1 ＋正規表現で抜いたグレード名（人手のルール）    MAE 13.60 万円

**ここで作る列は、Ab1 のグレード名列を正規表現なしで作り直したものにあたる。**
2つの指標を出す。

  中間ラベルの accuracy … 作った列が、正規表現版のグレード名とどれだけ一致するか
  下流の MAE            … その列を足したときに価格予測がどれだけ良くなるか

「機能A の評価を中間ラベルで見るか下流で見るか」（`docs/2026-08-29-denoise.md`）の材料にもなる。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_protocol import (  # noqa: E402
    LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, N_SPLITS, SEED, TARGET,
    cross_validate, load_dataset,
)
from features import LGBM_PARAMS, as_category as _as_category, numeric_frame as _numeric_frame  # noqa: E402

from unfold import Feature  # noqa: E402

COL = "タイトル"
GENERATED = "グレード_生成"

# (b) の入口で利用者が宣言する値。カタログを見れば書ける範囲の粒度にする。
# 正規表現版（116水準）より粗いことに注意。
DECLARED_VALUES = [
    "G", "Z", "X", "G クエロ", "ファンベース G", "ファンベース G クエロ",
    "G セーフティ エディション", "X ウェルキャブ", "Z E-Four 4WD", "G 4WD",
]
N_SEED_LABELS = 200   # 仕様書が前提にしている件数


def make_feature(mode: str, n_labels: int | None = None) -> Feature:
    """起点の作り方だけが違う Feature を返す。"""
    if mode in ("labelname", "human"):
        return Feature(source=COL, type="category", values=DECLARED_VALUES,
                       k="auto", threshold=0.9, name=GENERATED)
    if mode == "embedding":
        return Feature(source=COL, type="embedding", name="タイトル埋め込み")
    raise ValueError(mode)


def _labels_for(train: pd.DataFrame, n_labels: int | None) -> pd.Series:
    """訓練データのラベルを n_labels 件だけ残し、残りを欠損にする。

    「200件だけ人手でラベルを付けた」状況を再現する。どの200件を選ぶかで
    結果が変わるので seed を固定する。
    """
    y = train["グレード名"].astype("object").reset_index(drop=True)
    if n_labels is None or n_labels >= len(y):
        return y
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(y), size=n_labels, replace=False)
    masked = pd.Series([np.nan] * len(y), dtype="object")
    masked.iloc[keep] = y.iloc[keep].to_numpy()
    return masked


def make_lgbm_with_feature(mode: str, n_labels: int | None = None,
                           extra_cat: str | None = GENERATED):
    """fold の中で Feature を fit し、生成列を足して LightGBM を学習する。"""
    def fit_predict(train, test):
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        num, boo, cat = list(LEGACY_NUM), list(LEGACY_BOOL), list(LEGACY_CAT)
        Xtr = _numeric_frame(train, num, boo)
        Xte = _numeric_frame(test, num, boo)

        if mode == "regex":                       # 人手のルール（比較の基準）
            tr, te = train.assign(), test.assign()
            cat = cat + ["グレード名"]
        elif mode == "none":
            tr, te = train, test
        elif mode == "embedding":
            f = Feature(source=COL, type="embedding", name="emb").fit(train)
            Etr, Ete = f.transform(train), f.transform(test)
            Xtr = pd.concat([Xtr, Etr.set_index(Xtr.index)], axis=1)
            Xte = pd.concat([Xte, Ete.set_index(Xte.index)], axis=1)
            tr, te = train, test
        else:                                      # 機能A で列を作る
            f = make_feature(mode)
            y = _labels_for(train, n_labels) if mode == "human" else None
            f.fit(train, y)
            tr = train.assign(**{extra_cat: f.transform(train).astype(str)})
            te = test.assign(**{extra_cat: f.transform(test).astype(str)})
            cat = cat + [extra_cat]

        if cat:
            Ctr, Cte = _as_category(tr, te, cat)
            Xtr = pd.concat([Xtr, Ctr], axis=1)
            Xte = pd.concat([Xte, Cte], axis=1)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, train[TARGET], categorical_feature=cat or "auto")
        return model.predict(Xte)
    return fit_predict


def intermediate_accuracy(df: pd.DataFrame, mode: str,
                          n_labels: int | None = None) -> dict:
    """生成した列が、正規表現版のグレード名とどれだけ一致するかを測る。

    宣言した10値に無いグレード（116水準のうち残り）は正解になりようがないので、
    **宣言値に含まれる行だけに絞った accuracy** も併せて出す。
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    acc, acc_in, esc = [], [], []
    for tr_idx, te_idx in kf.split(df):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        f = make_feature(mode)
        y = _labels_for(train, n_labels) if mode == "human" else None
        f.fit(train, y)
        pred = f.transform(test).astype(str).to_numpy()
        truth = test["グレード名"].astype(str).to_numpy()
        acc.append(float((pred == truth).mean()))
        m = np.isin(truth, DECLARED_VALUES)
        acc_in.append(float((pred[m] == truth[m]).mean()) if m.any() else np.nan)
        esc.append(f.status()["レビュー待ち"] / len(test))
    return {"accuracy": float(np.mean(acc)),
            "宣言値に限った accuracy": float(np.nanmean(acc_in)),
            "エスカレーション率": float(np.mean(esc))}


def confidence_is_useful(df: pd.DataFrame, mode: str,
                         n_labels: int | None = None) -> pd.DataFrame:
    """confidence が「当たりやすい行」を見分けられているかを確かめる。

    信頼度ルーティング（設計書の AdaptivePredictor）は、confidence が低い行だけを
    LLM に回すことで費用を抑える設計になっている。**それが成り立つのは、
    confidence が実際の正しさと相関している場合だけ**なので、先に確かめる。

    やり方: confidence の低い順に r% を切り捨てて、残った行の accuracy を見る。
    r を上げるほど accuracy が上がるなら、confidence は routing の信号として使える。
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    conf_all, ok_all = [], []
    for tr_idx, te_idx in kf.split(df):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        f = make_feature(mode)
        f.fit(train, _labels_for(train, n_labels) if mode == "human" else None)
        pred = f.transform(test).astype(str).to_numpy()
        conf_all.append(f.confidence().to_numpy())
        ok_all.append(pred == test["グレード名"].astype(str).to_numpy())
    conf = np.concatenate(conf_all)
    ok = np.concatenate(ok_all)
    rows = []
    for r in (0.0, 0.1, 0.2, 0.3, 0.5):
        cut = np.quantile(conf, r)
        keep = conf > cut if r > 0 else np.ones(len(conf), bool)
        rows.append({"LLM に回す割合": f"{r:.0%}",
                     "自力で答える行": int(keep.sum()),
                     "その行の accuracy": round(float(ok[keep].mean()), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset()
    print(f"\n宣言した値（利用者が書く想定）: {DECLARED_VALUES}")
    covered = df["グレード名"].isin(DECLARED_VALUES).mean()
    print(f"この10値で実データの {covered:.1%} をカバーする"
          f"（残りは 116 水準の裾）\n")

    print("=" * 78)
    print("1. 中間ラベルの精度 — 正規表現版のグレード名とどれだけ一致するか")
    print("=" * 78)
    modes = [("(b) ラベル0件・値の名前だけ", "labelname", None),
             (f"(a) 人手ラベル{N_SEED_LABELS}件", "human", N_SEED_LABELS),
             ("(a') 人手ラベル全件", "human", None)]
    rows = []
    for label, mode, n in modes:
        m = intermediate_accuracy(df, mode, n)
        rows.append({"起点": label, **m})
        print(f"  {label:<22} accuracy {m['accuracy']:.3f}"
              f"  宣言値に限れば {m['宣言値に限った accuracy']:.3f}"
              f"  エスカレーション {m['エスカレーション率']:.1%}")
    pd.DataFrame(rows).to_csv(Path("results") / "feature_intermediate.csv",
                              index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("1b. confidence は「当たりやすい行」を見分けられているか（routing の前提）")
    print("=" * 78)
    for label, mode, n in modes[:2]:
        print(f"\n  {label}")
        print(confidence_is_useful(df, mode, n).to_string(index=False))

    print("\n" + "=" * 78)
    print("2. 下流の MAE — その列を足すと価格予測がどれだけ良くなるか")
    print("=" * 78)
    runs = [
        ("A2' 従来列のみ（テキストを使わない）", make_lgbm_with_feature("none"),
         "比較の下限。既測 17.51"),
        ("Ab1' ＋正規表現のグレード名（人手のルール）",
         make_lgbm_with_feature("regex"), "比較の上限。既測 13.60"),
        ("F(b) ＋機能A・ラベル0件（値の名前だけ）",
         make_lgbm_with_feature("labelname"), "unfold Feature。教師ラベルを一切使わない"),
        (f"F(a) ＋機能A・人手ラベル{N_SEED_LABELS}件",
         make_lgbm_with_feature("human", N_SEED_LABELS), "仕様書が前提にしている件数"),
        ("F(a') ＋機能A・人手ラベル全件",
         make_lgbm_with_feature("human", None), "上限の参考"),
        ("F(c) ＋機能A・埋め込み列（ラベルなし）",
         make_lgbm_with_feature("embedding"), "type='embedding'。分類を経由しない"),
    ]
    for name, fn, note in runs:
        print(f"[{name}]")
        cross_validate(name, fn, df, note=note)
        print()

    print("=" * 78)
    print("3. 来歴の確認（設計書の explain / status）")
    print("=" * 78)
    f = make_feature("labelname")
    f.fit(df)
    out = f.transform(df)
    print(f"\n生成した列の分布:\n{out.value_counts().head(6).to_string()}\n")
    print(f.explain(0))
    print(f"\nstatus: {f.status()}")
    print(f"cost:   {f.cost()}")
    q = f.review_queue()
    print(f"レビュー待ち {len(q):,} 件（先頭3件）:")
    if len(q):
        print(q.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
