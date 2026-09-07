"""Craigslist 全 200,374 行で線を引き直す（間引きなし）。

    .venv/bin/python scripts/run_fullscale.py            # 本番（全行）
    .venv/bin/python scripts/run_fullscale.py --sample 5000   # 動作確認用

これまでの複数車種の測定はすべて 60,000 行に間引いたものだった。
PRD には「S1・S2 の線は行数とセットで固定する」と書いてある。行数を変えると
**未知語率が 10.7% から 6.4% に下がり、P2 の指標の意味が変わる**ためで、
最終レポートの数字を全行で出すならいつかは引き直す必要がある。

測るのは、間引き版で意味があった構成だけに絞る。

    A2 構造化列のみ / B2 手書きルール / C2 文字TF-IDF
    F(c) 機能A の埋め込み列 / F(d) 機能A + 文字TF-IDF

**e5 埋め込みの構成（D1・D4）は入れない。** 20万行 × 384次元を pandas に載せると
メモリが厳しく、夜間に落ちると全部が無駄になる。機能A の既定エンコーダは
P2 で e5 より未知語に強いと分かっているので、比較の主役はこちらでよい。

未知／既知の内訳も同時に出す（P2 と同じ手続き。全行では未知語率が下がるので、
「行数を増やすと未知語問題がどれだけ軽くなるか」も分かる）。
"""

from __future__ import annotations

import argparse
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
    BOOL, CAT, NUM, RULE_COL, TEXT, normalize_model,
)
from run_embedding_vehicles import breakdown  # noqa: E402
from run_unseen_feature import feature_extra  # noqa: E402

TARGET = VEHICLES.target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="動作確認用に行数を絞る（既定は全行）")
    args = ap.parse_args()

    df = load_dataset(dataset=VEHICLES, sample=args.sample)
    df[RULE_COL] = df[TEXT].map(normalize_model)
    tag = f"{len(df):,}行"
    print(f"\nmodel {df[TEXT].nunique():,} 種類 → 手書きルール正規化後 "
          f"{df[RULE_COL].nunique():,} 種類\n")

    runs = [
        (f"[全行] A2 構造化列のみ",
         make_lgbm(TARGET, NUM, BOOL, CAT), "全行での下限"),
        (f"[全行] B2 + 手書きルール正規化",
         make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL]), "機能A が置き換えたい相手"),
        (f"[全行] C2 + model の文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char"), "ルールを書かない従来手法"),
        (f"[全行] F(c) + 機能A・埋め込み列",
         make_lgbm(TARGET, NUM, BOOL, CAT, extra=feature_extra()), "unfold Feature"),
        (f"[全行] F(d) + 機能A・埋め込み列＋文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char", extra=feature_extra()),
         "併用。60,000行では 2,588 で最良だった"),
    ]

    oof: dict[str, dict] = {}
    for name, fn, note in runs:
        print(f"[{name}]", flush=True)
        t0 = time.time()
        out: dict = {}
        cross_validate(name, fn, df, note=f"{note}（{tag}）", dataset=VEHICLES,
                       oof_out=out)
        oof[name] = out
        print(f"  ({time.time() - t0:.0f} 秒)\n", flush=True)

    tbl = breakdown(df, oof)
    print("=" * 78)
    print(f"既知／未知の model で分けた MAE（USD・{tag}）")
    print("=" * 78)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    dst = ROOT / "results" / "fullscale_unseen_breakdown.csv"
    tbl.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n保存: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
