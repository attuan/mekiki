"""TabPFN に埋め込みを足す — PRD が「別の問い」として残していたもの。

    TABPFN_DISABLE_TELEMETRY=1 .venv-embed/bin/python scripts/run_tabpfn_emb.py --probe
    TABPFN_DISABLE_TELEMETRY=1 .venv-embed/bin/python scripts/run_tabpfn_emb.py --dims 64

**このスクリプトは .venv-embed で動かす**（torch が要る。run_tabpfn.py と同じ）。

すでに分かっていること（`docs/2026-08-29-tabpfn.md`）:

  ・構造化列だけなら TabPFN 12.95 と LightGBM 12.88 は実質同着
  ・テキストを足した LightGBM（12.21）には TabPFN が 0.74 届かない

つまり「表の列だけを与える限り、器を替えても頭打ちは動かない」。ではその頭打ちは、
**非構造テキストを足したときにも同じか**。LightGBM は埋め込み384次元を渡されると
価格に効く次元を選んで使えた（13.68 → 12.38）。TabPFN は事前学習済みの推論器で、
その場で特徴選択の学習をしない。**この差が効くなら、非構造データを扱う土俵では
LightGBM の方が有利**ということになり、機能A の下流に何を置くかの判断材料になる。

埋め込みは定数語を除いた版（usedsienta_emb_titlev2）を使う。
2026-08-29 の測定で下流精度が最も良かった版である。
PCA は必ず fold の中で fit する（訓練データの分布を使うため）。

TabPFN は特徴量が増えると重くなるので、次元数を段階的に上げて測る。
夜間に流すことを想定し、1つ終わるごとに leaderboard に記録する
（途中で打ち切られても、そこまでの結果は残る）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_protocol import (  # noqa: E402
    DATA, SEED, TARGET, cross_validate, load_dataset,
)
from run_tabpfn import build_xy  # noqa: E402

EMB_PATH = ROOT / "sampledata" / "processed" / "usedsienta_emb_titlev2_e5small.parquet"


def load_embeddings(n_rows: int) -> np.ndarray:
    """埋め込みを load_dataset() と同じ行に揃える（価格欠損6行を落とす）。"""
    keep = pd.read_parquet(DATA, columns=[TARGET])[TARGET].notna().to_numpy()
    V = pd.read_parquet(EMB_PATH).drop(columns=["行番号"]).to_numpy(dtype="float32")
    V = V[keep]
    if len(V) != n_rows:
        raise ValueError(f"行数が合いません {len(V)} != {n_rows}")
    return V


def make_tabpfn_emb(V: np.ndarray, dims: int | None, n_estimators: int = 1):
    """構造化列フル + 埋め込み（PCA で dims 次元に圧縮）を TabPFN に渡す。"""
    from sklearn.decomposition import PCA
    from tabpfn import TabPFNRegressor

    def fit_predict(train, test):
        A, B, y, cat_idx = build_xy(train, test, TARGET)
        Etr, Ete = V[train["_行番号"].to_numpy()], V[test["_行番号"].to_numpy()]
        if dims is not None and dims < Etr.shape[1]:
            pca = PCA(n_components=dims, random_state=SEED)
            Etr, Ete = pca.fit_transform(Etr), pca.transform(Ete)
        A = np.column_stack([A, Etr]).astype("float64")
        B = np.column_stack([B, Ete]).astype("float64")
        model = TabPFNRegressor(device="cpu", n_estimators=n_estimators,
                                categorical_features_indices=cat_idx,
                                random_state=SEED)
        model.fit(A, y)
        return model.predict(B)
    return fit_predict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=64,
                    help="埋め込みを何次元に圧縮するか（384 で圧縮なし）")
    ap.add_argument("--n-estimators", type=int, default=1)
    ap.add_argument("--probe", action="store_true",
                    help="1 fold だけ小さく回して所要時間を見る")
    args = ap.parse_args()

    df = load_dataset()
    df["_行番号"] = np.arange(len(df))
    V = load_embeddings(len(df))
    print(f"埋め込み: {EMB_PATH.name} {V.shape}")

    fn = make_tabpfn_emb(V, args.dims, args.n_estimators)

    if args.probe:
        # 訓練500行・予測100行で1回だけ回し、本番の所要時間を外挿する
        small_tr = df.iloc[:500].reset_index(drop=True)
        small_te = df.iloc[500:600].reset_index(drop=True)
        t0 = time.time()
        fn(small_tr, small_te)
        dt = time.time() - t0
        # TabPFN の推論は訓練行数にほぼ比例する。本番は訓練4,405行 × 5 fold
        print(f"probe: 訓練500行で {dt:.1f} 秒 "
              f"→ 5-fold の見積もり {dt * (4405 / 500) * 5 / 60:.0f} 分")
        return 0

    import tabpfn  # 版は記録に残す。2.2.1 と 8.5.0 では数字が違う
    name = f"G  TabPFN・構造化列フル+タイトル埋め込み{args.dims}次元"
    cross_validate(name, fn, df,
                   note=f"tabpfn {tabpfn.__version__} / CPU / "
                        f"n_estimators={args.n_estimators} / "
                        f"埋め込みは定数語除去版を PCA{args.dims}。"
                        f"LightGBM の D3(12.38) と同じ問いを TabPFN で")
    return 0


if __name__ == "__main__":
    sys.exit(main())
