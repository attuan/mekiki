"""ベースライン（Craigslist 複数車種）— `model` 列の扱い方を6通り比べる。

**問い**: 出品者が自由に書いた車種の記述（`model`, 19,739種類）を、
どう表現すれば価格予測に効くか。

このデータの `model` は、選択肢から選ばせた列ではなく1行の自由入力である。

    'f-150 raptor arizona raptor*rust free*icon level kit*tech pkg*...'
    '2500 slt / quad cab / 4x4 / leather / 5.9 l high output / ...'
    'x5 3.0i awd 126k miles 3.0l v6 pano roof heated leather'

車種の芯・グレード・装備・宣伝文が区切りも語順も揃わないまま混ざっていて、
6割は1回しか出てこない。カテゴリとして持てば未知の値だらけになり、
人手のルールで正規化すれば書き切れない。ここが unfold 機能A の本番になる。

はしご:

    0  中央値                    予測しない場合の下限
    A1 線形回帰・構造化列        従来手法の再現。model を使わない
    A2 LightGBM・構造化列        モデルを木に替えただけの効果
    B1 + model そのまま          19,739水準をそのままカテゴリに突っ込む
    B2 + model を手書きルールで正規化   区切り記号以降を捨てて先頭2語
    C1 + model の単語TF-IDF      ルールを書かずに語で持つ
    C2 + model の文字TF-IDF      同上・部分文字列で表記ゆれを吸収
    E  B2 + description の単語TF-IDF   自由記述本体も足した上限

使う列と、その列を採る理由は scripts/clean_vehicles.py と
eval_protocol.py の VEHICLES_* にまとめてある。

実行:
    .venv/bin/python scripts/run_baselines_vehicles.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    VEHICLES, VEHICLES_BOOL, VEHICLES_CAT, VEHICLES_LONG_TEXT, VEHICLES_NUM,
    VEHICLES_TEXT, cross_validate, load_dataset, show_leaderboard,
)
from features import make_lgbm, make_linear, predict_median  # noqa: E402

TARGET = VEHICLES.target

# 実験に使う行数。20万行を全部使うと TF-IDF・埋め込みまで含めた比較が
# 現実的な時間に収まらないので、seed 固定で間引く。
N_SAMPLE = 60_000

# 使う列は eval_protocol.VEHICLES_* が正本（なぜその列かはそちらに書いてある）。
# ここでは短い別名を置くだけで、他のスクリプトもこの名前で読み込んでいる。
NUM = VEHICLES_NUM
BOOL = VEHICLES_BOOL
CAT = VEHICLES_CAT
TEXT = VEHICLES_TEXT              # model（本実験の主役）
DESC = VEHICLES_LONG_TEXT         # description（自由記述本体）

RULE_COL = "model_rule_normalized"


# --- 手書きルールによる model の正規化 --------------------------------

# 「正規表現で車種の芯だけ抜く」という、人手で書ける範囲のルールを素直に書く。
# これが人間側の線で、機能A はこれを上回れるかで評価する。
_SEP = re.compile(r"[*/|!,()\[\]]")           # ここから先は宣伝文とみなす
_NOISE = re.compile(r"[^a-z0-9\- ]+")


def normalize_model(s: str) -> str:
    """区切り記号以降を捨て、先頭2語だけを車種の芯とみなす。"""
    if not isinstance(s, str):
        return ""
    head = _SEP.split(s)[0]
    head = _NOISE.sub(" ", head.lower())
    return " ".join(head.split()[:2])


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)

    print(f"  model: {df[TEXT].nunique():,} 種類 "
          f"→ 手書きルール正規化後 {df[RULE_COL].nunique():,} 種類")
    print(f"  manufacturer {df['manufacturer'].nunique()} 社")

    runs = [
        ("0  中央値", predict_median(TARGET), "予測しない場合の下限"),
        ("A1 線形回帰・構造化列",
         make_linear(TARGET, NUM, BOOL, CAT),
         "Ridge + one-hot。model を使わない従来手法の再現"),
        ("A2 LightGBM・構造化列",
         make_lgbm(TARGET, NUM, BOOL, CAT),
         "A1と同じ列。モデルを木に替えた効果"),
        ("B1 + model そのまま",
         make_lgbm(TARGET, NUM, BOOL, CAT + [TEXT]),
         f"自由記述{df[TEXT].nunique():,}水準をそのままカテゴリ扱い"),
        ("B2 + model を手書きルールで正規化",
         make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL]),
         f"区切り記号以降を捨て先頭2語。{df[RULE_COL].nunique():,}水準"),
        ("C1 + model の単語TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "word"),
         "min_df=5, max_features=300。ルールを書かずに語で持つ"),
        ("C2 + model の文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char"),
         "char_wb 2-4gram, max_features=1000。表記ゆれを部分文字列で吸収"),
        ("E  B2 + description の単語TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL], DESC, "word"),
         "自由記述本体（平均2,972文字）を足した場合の上積み"),
    ]

    print()
    for name, fn, note in runs:
        print(f"[{name}]")
        t0 = time.time()
        cross_validate(name, fn, df, note=note, dataset=VEHICLES)
        print(f"  ({time.time() - t0:.0f} 秒)\n")

    print("=" * 78)
    print(f"リーダーボード（{VEHICLES.leaderboard.name}）")
    print("=" * 78)
    lb = show_leaderboard(VEHICLES)
    print(lb[["手法", "MAE", "MAE_std", "RMSE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
