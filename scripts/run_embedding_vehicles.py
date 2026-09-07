"""埋め込みを特徴量にして精度を測る（複数車種・主環境 .venv で実行）。

前提:
    .venv/bin/python scripts/clean_vehicles.py
    .venv-embed/bin/python scripts/embed_vehicles.py
    .venv/bin/python scripts/run_baselines_vehicles.py
    .venv/bin/python scripts/run_embedding_vehicles.py   # ← これ

シエンタ（単一車種）で出た結論はこうだった。

  ・タイトルの埋め込みだけで、正規表現で抜いたグレード名と同等の精度が出た
  ・しかし文字TF-IDF が強く、埋め込みを足しても最良値は更新できなかった
  ・理由は「単一車種では正規表現が完璧に効いてしまう」から

複数車種ではその前提が崩れる。model は 19,739 種類あり、6割が1回しか出てこない。
テストに出てくる model の約1割は訓練データに存在しない（未知語）。
そこで問いは2つ。

  D1「ルールを書かずに model を扱えるか」★本命
      手書きルールで正規化した B2、文字TF-IDF の C2 と、同じ土俵で比べる。

  未知の model の行だけを取り出して MAE を比べる ★複数車種で初めて測れる問い
      カテゴリ変数は未知語を欠損としてしか扱えない。TF-IDF は訓練語彙に無い
      部分文字列を落とす。埋め込みは学習済みモデルなので未知の文字列にも
      ベクトルを出せる。この差が出るなら、それが複数車種に広げた意味になる。
      得意分野が分かれるなら併用（D4）で両取りできるはず、というのが次の一手。

埋め込みは学習済みモデルの出力で価格を一切見ていないため、全データ分を先に
計算しておいても交差検証のリークにはならない。PCA は fold の中で fit する。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    ROOT, VEHICLES, cross_validate, load_dataset, show_leaderboard,
)
from features import make_lgbm  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    BOOL, CAT, N_SAMPLE, NUM, RULE_COL, TEXT, normalize_model,
)

TARGET = VEHICLES.target
EMB_PATH = ROOT / "sampledata" / "processed" / "vehicles_emb_model_e5small.parquet"
EMB_PREFIX = "memb_"


def attach_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """model の埋め込みを横付けする。

    埋め込みは「model の種類」ごとに計算してあるので、突き合わせは
    行番号ではなく文字列そのものをキーにする（間引いた行でもずれない）。
    """
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH.name} がありません。先に埋め込みを計算してください:\n"
            f"  .venv-embed/bin/python scripts/embed_vehicles.py"
        )
    emb = pd.read_parquet(EMB_PATH)
    emb.columns = [TEXT] + [f"{EMB_PREFIX}{i}" for i in range(emb.shape[1] - 1)]
    out = df.merge(emb, on=TEXT, how="left")
    if len(out) != len(df):
        raise ValueError(f"結合で行数が変わりました {len(df)} → {len(out)}")
    miss = out[f"{EMB_PREFIX}0"].isna().sum()
    if miss:
        raise ValueError(f"埋め込みが見つからない model が {miss} 件あります")
    return out


def emb_extra(pca_dim: int | None = None):
    """make_lgbm の extra フック用。埋め込み列を渡す関数を作る。"""
    def extra(train: pd.DataFrame, test: pd.DataFrame):
        cols = [c for c in train.columns if c.startswith(EMB_PREFIX)]
        Etr = train[cols].to_numpy(dtype="float32")
        Ete = test[cols].to_numpy(dtype="float32")
        names = cols
        if pca_dim:
            # PCA は訓練データの分布を使うので必ず fold の中で fit する
            pca = PCA(n_components=pca_dim, random_state=42)
            Etr, Ete = pca.fit_transform(Etr), pca.transform(Ete)
            names = [f"{EMB_PREFIX}pc{i}" for i in range(Etr.shape[1])]
        return (pd.DataFrame(Etr, columns=names, index=train.index),
                pd.DataFrame(Ete, columns=names, index=test.index))
    return extra


# --- 未知の model だけを取り出した比較 ---------------------------------

def unseen_mask(df: pd.DataFrame, fold: np.ndarray, col: str) -> np.ndarray:
    """各行について「自分が test だった fold の train に、その値が無かったか」。

    cross_validate と同じ KFold を使っているので、fold 番号さえあれば
    訓練側の集合は復元できる。
    """
    mask = np.zeros(len(df), dtype=bool)
    vals = df[col].to_numpy()
    for f in np.unique(fold):
        te = fold == f
        seen = set(vals[~te])
        mask[te] = [v not in seen for v in vals[te]]
    return mask


def breakdown(df: pd.DataFrame, results: dict[str, dict]) -> pd.DataFrame:
    """既知／未知の model に分けて MAE を出す。"""
    y = df[TARGET].to_numpy(dtype=float)
    fold = next(iter(results.values()))["fold"]
    unseen = unseen_mask(df, fold, TEXT)
    print(f"\ntest 行のうち model が train に無かった行: "
          f"{unseen.sum():,} / {len(df):,}（{unseen.mean()*100:.1f}%）")

    rows = []
    for name, r in results.items():
        err = np.abs(r["pred"] - y)
        rows.append({
            "手法": name,
            "全体MAE": err.mean(),
            "既知MAE": err[~unseen].mean(),
            "未知MAE": err[unseen].mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)
    df = attach_embeddings(df)
    print(f"  埋め込み: {EMB_PATH.name} を model で結合（384次元）")

    runs = [
        ("D1 構造化列+model の埋め込み",
         make_lgbm(TARGET, NUM, BOOL, CAT, extra=emb_extra()),
         "★本命。B2(手書きルール)・C2(文字TF-IDF)と同じ土俵での比較"),
        ("D2 D1のPCA64次元版",
         make_lgbm(TARGET, NUM, BOOL, CAT, extra=emb_extra(pca_dim=64)),
         "384次元を64次元に圧縮。次元数が効いているかの確認"),
        ("D3 全部乗せ(model カテゴリ+埋め込み)",
         make_lgbm(TARGET, NUM, BOOL, CAT + [TEXT], extra=emb_extra()),
         "model をカテゴリとしても埋め込みとしても入れた場合"),
        ("D4 埋め込み+文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char", extra=emb_extra()),
         "未知の model に強い文字TF-IDF と、既知に強い埋め込みの併用"),
    ]

    oof: dict[str, dict] = {}
    print()
    for name, fn, note in runs:
        print(f"[{name}]")
        t0 = time.time()
        out: dict = {}
        cross_validate(name, fn, df, note=note, dataset=VEHICLES, oof_out=out)
        oof[name] = out
        print(f"  ({time.time() - t0:.0f} 秒)\n")

    # 既知／未知の内訳を出すために、比較対象も同じ df で回し直す（記録はしない）
    compare = [
        ("B2 手書きルール", make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL])),
        ("C2 文字TF-IDF", make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char")),
    ]
    for name, fn in compare:
        out = {}
        cross_validate(name, fn, df, record=False, verbose=False,
                       dataset=VEHICLES, oof_out=out)
        oof[name] = out

    order = ["B2 手書きルール", "C2 文字TF-IDF", "D1 構造化列+model の埋め込み",
             "D4 埋め込み+文字TF-IDF"]
    tbl = breakdown(df, {k: oof[k] for k in order})
    print()
    print("=" * 78)
    print("既知／未知の model で分けた MAE（USD）")
    print("=" * 78)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    tbl.to_csv(ROOT / "results" / "vehicles_unseen_breakdown.csv",
               index=False, encoding="utf-8-sig")

    print()
    lb = show_leaderboard(VEHICLES)
    print(lb[["手法", "MAE", "MAE_std", "RMSE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
