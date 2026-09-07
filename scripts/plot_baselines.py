"""ベースライン結果の作図。results/*.png を吐く。

results/leaderboard.csv を読むだけなので、モデルの再学習はしない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import LEADERBOARD, ROOT  # noqa: E402
from plot_style import use_japanese_font  # noqa: E402

use_japanese_font()

OUT = ROOT / "results"


def _latest(lb: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """同じ手法名が複数回あれば最後の実行を採る。"""
    last = lb.drop_duplicates("手法", keep="last").set_index("手法")
    return last.loc[names]


def ladder(lb: pd.DataFrame) -> None:
    names = ["A1 線形回帰・従来列", "A2 LightGBM・従来列", "B  LightGBM・構造化列フル",
             "C1 B+装備テキスト(単語TF-IDF)", "C2 B+装備テキスト(文字TF-IDF)"]
    labels = ["A1 線形回帰\n従来列", "A2 LightGBM\n従来列", "B +構造化列\n(グレード等)",
              "C1 +装備テキスト\n単語TF-IDF", "C2 +装備テキスト\n文字TF-IDF"]
    d = _latest(lb, names)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, d["MAE"], yerr=d["MAE_std"], capsize=4,
                  color=["#c0c0c0", "#8fb8d8", "#4a7fb0", "#2e6b4f", "#2e6b4f"])
    for b, v in zip(bars, d["MAE"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")

    # 隣との差を矢印で書き込む
    vals = d["MAE"].to_numpy()
    for i in range(1, len(vals)):
        ax.annotate(f"−{vals[i-1]-vals[i]:.2f}",
                    xy=(i - 0.5, (vals[i - 1] + vals[i]) / 2 - 3.5),
                    ha="center", fontsize=9, color="#b03030")

    ax.set_ylabel("MAE（万円・5-fold CV）")
    ax.set_title("ベースラインのはしご — シエンタ 5,507件 / 価格中央値 185.7万円",
                 fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_ladder.png", dpi=150)
    print("保存:", OUT / "baseline_ladder.png")


def ablation(lb: pd.DataFrame) -> None:
    names = ["Ab1 +グレード名", "Ab3 +色_基本", "Ab4 +車検残月数", "Ab2 +装備数"]
    labels = ["グレード名\n(テキスト由来)", "色_基本", "車検残月数", "装備数\n(テキスト由来)"]
    base = _latest(lb, ["Ab0 A2再掲（従来列のみ）"])["MAE"].iloc[0]
    d = _latest(lb, names)
    gain = base - d["MAE"]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2e6b4f", "#9aa0a6", "#9aa0a6", "#2e6b4f"]
    bars = ax.barh(labels[::-1], gain.to_numpy()[::-1], color=colors[::-1])
    for b, v in zip(bars, gain.to_numpy()[::-1]):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"−{v:.2f} 万円",
                va="center", fontsize=10)
    ax.set_xlabel(f"MAE の改善量（万円）— 起点 A2 = {base:.2f} 万円")
    ax.set_title("1列ずつ足したときの効き方（緑＝非構造テキストから作った列）",
                 fontsize=12)
    ax.set_xlim(0, gain.max() * 1.3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ablation.png", dpi=150)
    print("保存:", OUT / "ablation.png")


def main() -> None:
    lb = pd.read_csv(LEADERBOARD, encoding="utf-8-sig")
    ladder(lb)
    ablation(lb)


if __name__ == "__main__":
    main()


def embedding_compare(lb: pd.DataFrame) -> None:
    """埋め込み vs 正規表現 / TF-IDF の対比図。run_embedding.py の後に呼ぶ。"""
    pairs = [
        ("グレード情報の取り出し方\n（従来列 + α）",
         [("正規表現で\nグレード名を抽出", "Ab1 +グレード名", "#4a7fb0"),
          ("タイトルの埋め込み\n（ルールなし）", "D2 従来列+タイトル埋め込み(グレード名なし)", "#2e6b4f")]),
        ("装備テキストの入れ方\n（構造化列フル + α）",
         [("文字TF-IDF", "C2 B+装備テキスト(文字TF-IDF)", "#4a7fb0"),
          ("埋め込み", "D1 B+装備テキスト埋め込み", "#2e6b4f")]),
    ]
    last = lb.drop_duplicates("手法", keep="last").set_index("手法")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, items) in zip(axes, pairs):
        labels = [i[0] for i in items]
        vals = [last.loc[i[1], "MAE"] for i in items]
        errs = [last.loc[i[1], "MAE_std"] for i in items]
        colors = [i[2] for i in items]
        bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}",
                    ha="center", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("MAE（万円）")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(f"差 {abs(vals[0]-vals[1]):.2f}", xy=(0.5, max(vals) * 1.12),
                    ha="center", fontsize=10, color="#b03030")
    fig.suptitle("埋め込みは人手のルールに並ぶか（緑＝埋め込み）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "embedding_compare.png", dpi=150)
    print("保存:", OUT / "embedding_compare.png")
