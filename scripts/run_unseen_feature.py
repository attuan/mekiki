"""P2 — 未知の model だけで機能A を測り直す。

    .venv/bin/python scripts/run_unseen_feature.py

**なぜ未知語を切り出すのか。** テスト行の約1割は、訓練データに1度も出てこない
model を持つ。これは実運用そのもの（新型車、書き方の揺れ、珍しいグレード）で、
かつ手法ごとに得意不得意がはっきり分かれる場所である。既測（`docs/2026-08-29-vehicles-multi.md`）:

    手法              既知の model   未知の model
    手書きルール          2,623        4,118
    文字TF-IDF           2,511      **3,715**   ← 未知に強い
    e5 埋め込み        **2,470**       3,873    ← 既知に強い
    併用                 2,455        3,768

未知語こそ埋め込みの独壇場だと想定していたが、実際は逆だった。文字TF-IDF は
`f-250 lariat` を知らなくても `f-2` `250` `lar` という既知の断片に分解できるのに対し、
e5 は文全体を1つのベクトルにするため固有名詞そのものを知らないと分解しきれない。

**P2 の問い**: 機能A の既定エンコーダ（文字 n-gram TF-IDF を SVD で圧縮）は、
この分岐のどちら側に着地するか。文字 n-gram を使うので未知に強いはずだが、
SVD で 256 次元に潰す過程で断片の情報が失われている可能性がある。
潰れているなら、機能A は既定のままでは未知語に弱いことになる。

条件は P1（`scripts/demo_feature_vehicles.py`）と完全に同じ 60,000 行・5-fold。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from eval_protocol import VEHICLES, cross_validate, load_dataset  # noqa: E402
from features import make_lgbm  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    BOOL, CAT, N_SAMPLE, NUM, RULE_COL, TEXT, normalize_model,
)
from run_embedding_vehicles import (  # noqa: E402
    EMB_PATH, attach_embeddings, breakdown, emb_extra,
)

from unfold import Feature, PrecomputedEncoder  # noqa: E402

TARGET = VEHICLES.target


def feature_extra(emb_table: pd.DataFrame | None = None):
    """機能A の埋め込み列を make_lgbm の extra フックとして渡す。

    Feature は fold の train だけで fit する（エンコーダの語彙も train 由来）。
    emb_table を渡すと既存の e5 埋め込みを引くだけの構成になる。
    """
    def extra(train: pd.DataFrame, test: pd.DataFrame):
        if emb_table is None:
            f = Feature(source=TEXT, type="embedding", name="uf")
        else:
            f = Feature(source=TEXT, type="embedding", name="uf",
                        preprocess=False,
                        encoder=PrecomputedEncoder(emb_table, TEXT, name="e5small"))
        f.fit(train)
        return f.transform(train), f.transform(test)
    return extra


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)
    emb_table = pd.read_parquet(EMB_PATH)
    df = attach_embeddings(df)

    # 記録する新しい手法（機能A 側）
    new_runs = [
        ("F(c)' 機能A・埋め込み列（既定エンコーダ）",
         make_lgbm(TARGET, NUM, BOOL, CAT, extra=feature_extra()),
         "P2: 未知語にどちら側で着地するか"),
        ("F(c-e5)' 機能A・埋め込み列（e5-small）",
         make_lgbm(TARGET, NUM, BOOL, CAT, extra=feature_extra(emb_table)),
         "P2: API 経由の e5。D1 と同じ挙動になるはず"),
        ("F(d)' 機能A・埋め込み列＋文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char", extra=feature_extra()),
         "P2: 併用（D4 に対応）。機能A の列に素の文字TF-IDF を足す"),
    ]
    # 内訳を出すためだけに回し直す比較対象（すでに記録済みなので record=False）
    compare = [
        ("B2 手書きルール", make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL])),
        ("C2 文字TF-IDF", make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char")),
        ("D1 e5 埋め込み", make_lgbm(TARGET, NUM, BOOL, CAT, extra=emb_extra())),
        ("D4 e5 埋め込み+文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char", extra=emb_extra())),
    ]

    oof: dict[str, dict] = {}
    for name, fn, note in new_runs:
        print(f"[{name}]")
        t0 = time.time()
        out: dict = {}
        cross_validate(name, fn, df, note=note, dataset=VEHICLES, oof_out=out)
        oof[name] = out
        print(f"  ({time.time() - t0:.0f} 秒)\n")

    for name, fn in compare:
        print(f"[{name}]（内訳のための再実行・記録しない）")
        t0 = time.time()
        out = {}
        cross_validate(name, fn, df, record=False, verbose=False,
                       dataset=VEHICLES, oof_out=out)
        oof[name] = out
        print(f"  MAE {np.abs(out['pred'] - df[TARGET]).mean():,.0f}"
              f"  ({time.time() - t0:.0f} 秒)\n")

    order = ["B2 手書きルール", "C2 文字TF-IDF", "D1 e5 埋め込み",
             "D4 e5 埋め込み+文字TF-IDF",
             "F(c)' 機能A・埋め込み列（既定エンコーダ）",
             "F(c-e5)' 機能A・埋め込み列（e5-small）",
             "F(d)' 機能A・埋め込み列＋文字TF-IDF"]
    tbl = breakdown(df, {k: oof[k] for k in order})
    print("=" * 78)
    print("既知／未知の model で分けた MAE（USD）")
    print("=" * 78)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    tbl.to_csv(ROOT / "results" / "feature_unseen_breakdown.csv",
               index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
