"""機能A（Feature）を複数車種データで測る — P1 の残り。

    .venv/bin/python scripts/demo_feature_vehicles.py

シエンタ（単一車種）での測定は `docs/2026-08-29-feature-skeleton.md` に済んでいる。
そこで出た結論のうち、**「値の名前だけを起点にすれば人手ラベルより強い」は
シエンタ固有の可能性がある**と書いた。グレード名がタイトルに文字どおり
書かれているためで、Craigslist の `model`（19,739種類の自由記述）では
同じ条件が成り立たない。ここではそれを実際に確かめる。

比較の基準（すべて同じ 60,000 行・同じ 5-fold・既測）:

    A2 LightGBM・構造化列のみ            3,234 USD
    B2 + model を手書きルールで正規化      2,783 USD  ← 機能A が置き換えたい相手
    C2 + model の文字TF-IDF               2,640 USD
    D1 + model の埋め込み（e5-small）      2,620 USD
    D4 + 埋め込みと文字TF-IDF の併用        2,596 USD  ← 現在の最良

問いは3つ。

  1. 機能A の API を通しても、素で書いた特徴量と同じ精度が出るか（実装の健全性）
  2. 値の名前だけの起点は、自由記述データでも人手ラベルに勝つか（教師ラベルの起点の一般性）
  3. confidence によるルーティングは複数車種でも機能するか（信頼度ルーティングの前提）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from eval_protocol import (  # noqa: E402
    N_SPLITS, SEED, VEHICLES, cross_validate, load_dataset,
)
from features import LGBM_PARAMS, as_category as _as_category, numeric_frame as _numeric_frame  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    BOOL, CAT, N_SAMPLE, NUM, RULE_COL, TEXT, normalize_model,
)

from unfold import CharTfidfEncoder, Feature, PrecomputedEncoder  # noqa: E402

TARGET = VEHICLES.target
EMB_PATH = ROOT / "sampledata" / "processed" / "vehicles_emb_model_e5small.parquet"
GENERATED = "model_generated"
N_DECLARED = 60       # 利用者が宣言する値の数
N_SEED_LABELS = 200   # 仕様書が前提にしている人手ラベルの件数


def declared_values(df: pd.DataFrame) -> list[str]:
    """利用者が `values=[...]` に書く想定の値を作る。

    「よく出る model を60個書き出す」という、カタログを見れば書ける作業を模す。
    **価格は一切見ていない**ので目的変数のリークにはあたらない。
    """
    return (df[RULE_COL].value_counts().head(N_DECLARED).index.tolist())


def _labels_for(train: pd.DataFrame, n_labels: int | None) -> pd.Series:
    """訓練データのうち n_labels 件だけラベルを残す（人手ラベルの再現）。"""
    y = train[RULE_COL].astype("object").reset_index(drop=True)
    if n_labels is None or n_labels >= len(y):
        return y
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(y), size=n_labels, replace=False)
    masked = pd.Series([np.nan] * len(y), dtype="object")
    masked.iloc[keep] = y.iloc[keep].to_numpy()
    return masked


def make_run(mode: str, values: list[str], n_labels: int | None = None,
             emb_table: pd.DataFrame | None = None):
    """fold の中で Feature を fit し、生成列を足して LightGBM を学習する。"""
    def fit_predict(train, test):
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        num, boo, cat = list(NUM), list(BOOL), list(CAT)
        Xtr = _numeric_frame(train, num, boo)
        Xte = _numeric_frame(test, num, boo)
        tr, te = train, test

        if mode == "none":
            pass
        elif mode == "rule":
            cat = cat + [RULE_COL]
        elif mode in ("embedding", "embedding_e5"):
            if mode == "embedding":
                f = Feature(source=TEXT, type="embedding", name="memb")
            else:
                # 既存の e5 埋め込みを API 経由で使う。前処理は切る
                # （前処理をかけると文字列が変わり、対応表を引けなくなる）
                f = Feature(source=TEXT, type="embedding", name="memb",
                            preprocess=False,
                            encoder=PrecomputedEncoder(emb_table, TEXT,
                                                       name="e5small"))
            f.fit(train)
            Etr, Ete = f.transform(train), f.transform(test)
            Xtr = pd.concat([Xtr, Etr.set_index(Xtr.index)], axis=1)
            Xte = pd.concat([Xte, Ete.set_index(Xte.index)], axis=1)
        else:                                   # labelname / human
            f = Feature(source=TEXT, type="category", values=values,
                        k="auto", name=GENERATED)
            f.fit(train, _labels_for(train, n_labels) if mode == "human" else None)
            tr = train.assign(**{GENERATED: f.transform(train).astype(str)})
            te = test.assign(**{GENERATED: f.transform(test).astype(str)})
            cat = cat + [GENERATED]

        Ctr, Cte = _as_category(tr, te, cat)
        Xtr = pd.concat([Xtr, Ctr], axis=1)
        Xte = pd.concat([Xte, Cte], axis=1)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, train[TARGET], categorical_feature=cat)
        return model.predict(Xte)
    return fit_predict


def intermediate(df: pd.DataFrame, values: list[str], mode: str,
                 n_labels: int | None = None) -> dict:
    """中間ラベルの精度と、confidence によるルーティングの効きを測る。

    正解は手書きルールの正規化結果（シエンタでの「正規表現で抜いたグレード名」に相当）。
    宣言した60値に無い車種は当てようがないので、宣言値に絞った accuracy も出す。
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    conf_all, ok_all, in_all = [], [], []
    for tr_idx, te_idx in kf.split(df):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        f = Feature(source=TEXT, type="category", values=values, k="auto",
                    name=GENERATED)
        f.fit(train, _labels_for(train, n_labels) if mode == "human" else None)
        pred = f.transform(test).astype(str).to_numpy()
        truth = test[RULE_COL].astype(str).to_numpy()
        ok_all.append(pred == truth)
        conf_all.append(f.confidence().to_numpy())
        in_all.append(np.isin(truth, values))
    ok = np.concatenate(ok_all)
    conf = np.concatenate(conf_all)
    inside = np.concatenate(in_all)
    curve = []
    for r in (0.0, 0.1, 0.3, 0.5):
        cut = np.quantile(conf, r)
        keep = conf > cut if r > 0 else np.ones(len(conf), bool)
        curve.append((f"{r:.0%}", int(keep.sum()), round(float(ok[keep].mean()), 3)))
    return {"accuracy": float(ok.mean()),
            "宣言値に限った accuracy": float(ok[inside].mean()),
            "ルーティング曲線": curve}


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)
    values = declared_values(df)
    covered = df[RULE_COL].isin(values).mean()
    print(f"\nmodel {df[TEXT].nunique():,} 種類 → 手書きルール正規化後 "
          f"{df[RULE_COL].nunique():,} 種類")
    print(f"宣言した値 {len(values)} 個で実データの {covered:.1%} をカバー")
    print(f"  例: {values[:12]}")

    emb_table = pd.read_parquet(EMB_PATH) if EMB_PATH.exists() else None
    if emb_table is None:
        print(f"\n注意: {EMB_PATH.name} が無いので e5 版はとばします")

    print("\n" + "=" * 78)
    print("1. 中間ラベルの精度とルーティング（正解＝手書きルールの正規化結果）")
    print("=" * 78)
    for label, mode, n in [("(b) ラベル0件・値の名前だけ", "labelname", None),
                           (f"(a) 人手ラベル{N_SEED_LABELS}件", "human", N_SEED_LABELS)]:
        m = intermediate(df, values, mode, n)
        print(f"\n  {label}: accuracy {m['accuracy']:.3f}"
              f"（宣言値に限れば {m['宣言値に限った accuracy']:.3f}）")
        print("    LLM に回す割合 / 自力で答える行 / accuracy")
        for r, n_keep, acc in m["ルーティング曲線"]:
            print(f"      {r:>4}  {n_keep:>7,}  {acc:.3f}")

    print("\n" + "=" * 78)
    print("2. 下流の MAE（USD）")
    print("=" * 78)
    runs = [
        ("A2' 構造化列のみ", make_run("none", values), "比較の下限。既測 3,234"),
        ("B2' + 手書きルール正規化", make_run("rule", values),
         "機能A が置き換えたい相手。既測 2,783"),
        ("F(b)' + 機能A・ラベル0件（値の名前だけ）",
         make_run("labelname", values), "unfold Feature。教師ラベルを使わない"),
        (f"F(a)' + 機能A・人手ラベル{N_SEED_LABELS}件",
         make_run("human", values, N_SEED_LABELS), "仕様書が前提にしている件数"),
        ("F(c)' + 機能A・埋め込み列（既定エンコーダ）",
         make_run("embedding", values), "文字TF-IDF+SVD256。C2(2,640) と比較"),
    ]
    if emb_table is not None:
        runs.append(("F(c-e5)' + 機能A・埋め込み列（e5-small）",
                     make_run("embedding_e5", values, emb_table=emb_table),
                     "既存の埋め込みを API 経由で使う。D1(2,620) と比較"))

    for name, fn, note in runs:
        print(f"[{name}]")
        t0 = time.time()
        cross_validate(name, fn, df, note=note, dataset=VEHICLES)
        print(f"  ({time.time() - t0:.0f} 秒)\n")


if __name__ == "__main__":
    main()
