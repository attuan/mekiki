"""ノイズ除去と近傍検索の結果を作図する（results/denoise_*.png）。

数値は測定済みのものをそのまま置く（再学習はしない）。出典は
docs/2026-08-29-denoise.md と results/leaderboard.csv。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import ROOT  # noqa: E402
from plot_style import use_japanese_font  # noqa: E402

use_japanese_font()
OUT = ROOT / "results"

# 版 → (グレード分離AUC, 近傍5件の価格MAE, 教師ありD2'のMAE)
VARIANTS = {
    "V0 生": (0.637, 31.45, 13.68),
    "V1 既出語除去": (0.659, 30.44, 13.68),
    "V2 定数語除去": (0.707, 28.92, 13.29),
    "V3 先頭区切り": (0.711, 32.05, 13.79),
    "V23 V3+V2": (0.781, 31.86, 13.58),
}

# 近傍の選び方 → MAE
NEIGHBOUR = [
    ("意味的近傍のみ\n（生）", 31.45, "tab:red"),
    ("意味的近傍のみ\n（定数語除去）", 28.92, "tab:red"),
    ("数値距離のみ\n（埋め込みなし）", 24.45, "tab:gray"),
    ("数値距離を混ぜる\n（案a）", 18.19, "tab:blue"),
    ("価格帯で絞る\n（案b）", 18.32, "tab:blue"),
    ("参考: LightGBM\n構造化列フル", 12.88, "tab:green"),
]


def fig_variants() -> None:
    """分離の良さと価格の当たりやすさが一致しないことを1枚で見せる。"""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, (auc, knn, _) in VARIANTS.items():
        ax.scatter(auc, knn, s=90, zorder=3)
        ax.annotate(name, (auc, knn), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.set_xlabel("グレードの分離 AUC（右ほど「似ている」がグレードを言い当てる）")
    ax.set_ylabel("近傍5件の価格中央値の MAE（下ほど価格が当たる・万円）")
    ax.set_title("削るほどグレードは分離できるが、価格が当たるとは限らない",
                 fontsize=12)
    ax.grid(alpha=0.3)
    ax.annotate("分離は最良なのに\n価格では負ける", (0.781, 31.86),
                textcoords="offset points", xytext=(-110, -28), fontsize=9,
                color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    fig.tight_layout()
    fig.savefig(OUT / "denoise_auc_vs_mae.png", dpi=150)
    print("保存: results/denoise_auc_vs_mae.png")


def fig_neighbour() -> None:
    """近傍の選び方を変えたときの MAE。機能B に渡す証拠の質。"""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = [n for n, _, _ in NEIGHBOUR]
    vals = [v for _, v, _ in NEIGHBOUR]
    colors = [c for _, _, c in NEIGHBOUR]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylabel("MAE（万円・低いほど良い）")
    ax.set_title("機能B に渡す「似た事例」の質 — 距離の作り方で MAE が 31.45 → 18.19",
                 fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 36)
    fig.tight_layout()
    fig.savefig(OUT / "denoise_neighbour.png", dpi=150)
    print("保存: results/denoise_neighbour.png")


if __name__ == "__main__":
    fig_variants()
    fig_neighbour()
