"""アブレーション — ベースライン B の伸びが「どの列」から来たのかを分解する。

run_baselines.py で A2（従来列）→ B（構造化列フル）に MAE が大きく縮んだが、
足した列は4つ（グレード名・色_基本・車検残月数・装備数）あり、
まとめて足しただけではどれが効いたのか分からない。

ここでは A2 の列に1列ずつ足して測り、寄与を分ける。
**グレード名と装備数はどちらもタイトル文字列から正規表現で取り出した列**なので、
その寄与はそのまま「非構造テキストを特徴量にすることの価値」＝
unfold 機能A が自動化しようとしている作業の価値にあたる。

実行:
    .venv/bin/python scripts/run_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_protocol import (  # noqa: E402
    LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, TARGET, cross_validate, load_dataset,
)
from features import make_lgbm  # noqa: E402


def main() -> None:
    df = load_dataset()

    # A2 と同じ条件から出発して、1列ずつ足す
    runs = [
        ("Ab0 A2再掲（従来列のみ）", LEGACY_NUM, LEGACY_CAT,
         "比較の起点"),
        ("Ab1 +グレード名", LEGACY_NUM, LEGACY_CAT + ["グレード名"],
         "タイトルから正規表現で抽出した列。非構造テキスト由来"),
        ("Ab2 +装備数", LEGACY_NUM + ["装備数"], LEGACY_CAT,
         "タイトルの装備羅列の語数。非構造テキスト由来"),
        ("Ab3 +色_基本", LEGACY_NUM, LEGACY_CAT + ["色_基本"],
         "構造化列。101水準あり正規化が甘い"),
        ("Ab4 +車検残月数", LEGACY_NUM + ["車検残月数"], LEGACY_CAT,
         "構造化列。欠損2,926件（車検整備付の車）"),
    ]

    print()
    for name, num, cat, note in runs:
        print(f"[{name}]")
        cross_validate(name, make_lgbm(TARGET, num, LEGACY_BOOL, cat), df, note=note)
        print()


if __name__ == "__main__":
    main()
