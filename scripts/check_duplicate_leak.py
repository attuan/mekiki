"""重複排除をサボるとスコアがどれだけ水増しされるかを測る。

Craigslist のデータは、同じ車が複数の region に出稿されている。
同一 VIN が最大 261 件あり、価格まで同一だった。フィルタ後 346,371 行のうち
145,997 行（42%）が重複で、これを残したままランダム分割の交差検証をすると
**まったく同じ車が train と test の両方に入る**（＝答えを見て答える）。

同じ手法・同じ行数で、重複ありとなしを並べる。差がそのままリークの実害。

    .venv/bin/python scripts/clean_vehicles.py --no-dedup   # 先に重複あり版を作る
    .venv/bin/python scripts/check_duplicate_leak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import ROOT, SEED, VEHICLES, cross_validate  # noqa: E402
from features import make_lgbm  # noqa: E402
from run_baselines_vehicles import BOOL, CAT, N_SAMPLE, NUM, TEXT  # noqa: E402

WITHDUP = ROOT / "sampledata" / "processed" / "vehicles_multi_withdup.parquet"
TARGET = VEHICLES.target


def main() -> None:
    if not WITHDUP.exists():
        raise SystemExit(
            "重複あり版がありません。先に:\n"
            "  .venv/bin/python scripts/clean_vehicles.py --no-dedup"
        )
    dup = pd.read_parquet(WITHDUP).sample(n=N_SAMPLE, random_state=SEED)
    dup = dup.reset_index(drop=True)
    print(f"重複あり {N_SAMPLE:,} 行を抽出（VIN の重複 "
          f"{dup['VIN'].duplicated().sum():,} 件）")

    runs = [
        ("A2 LightGBM・構造化列", make_lgbm(TARGET, NUM, BOOL, CAT)),
        ("C2 + model の文字TF-IDF", make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char")),
    ]
    for name, fn in runs:
        print(f"\n[※重複あり {name}]")
        cross_validate(f"※重複あり {name}", fn, dup, dataset=VEHICLES,
                       note="重複排除をしない場合。正しい値は同名の行を参照")


if __name__ == "__main__":
    main()
