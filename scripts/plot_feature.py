"""機能A の測定結果を作図する（results/feature_*.png）。

数値は scripts/demo_feature.py の実測をそのまま置く（再学習はしない）。
出典は docs/2026-08-29-feature-skeleton.md。
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

LADDER = [
    ("従来列のみ\n（テキスト不使用）", 17.51, "tab:gray"),
    ("人手ラベル200件\n（仕様書の前提）", 16.72, "tab:red"),
    ("人手ラベル全件\n4,405件", 16.16, "tab:red"),
    ("値の名前だけ\nラベル0件", 14.41, "tab:blue"),
    ("埋め込み列\n分類を経由しない", 13.97, "tab:blue"),
    ("正規表現\n（人手のルール）", 13.60, "tab:green"),
]

ROUTING = [(0, 0.785), (10, 0.819), (20, 0.848), (30, 0.869), (50, 0.904)]


def fig_ladder() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels = [n for n, _, _ in LADDER]
    vals = [v for _, v, _ in LADDER]
    bars = ax.bar(labels, vals, color=[c for _, _, c in LADDER])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                ha="center", fontsize=10)
    ax.set_ylabel("MAE（万円・低いほど良い）")
    ax.set_ylim(12, 18.5)
    ax.set_title("教師ラベルの起点で下流の精度がどう変わるか（シエンタ・5-fold）",
                 fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_label_source.png", dpi=150)
    print("保存: results/feature_label_source.png")


def fig_routing() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = [r for r, _ in ROUTING]
    y = [a for _, a in ROUTING]
    ax.plot(x, y, marker="o", color="tab:blue")
    for xi, yi in ROUTING:
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("confidence が低い順に LLM へ回す割合（%）")
    ax.set_ylabel("自力で答えた行の accuracy")
    ax.set_title("confidence は「当たりやすい行」を見分けられている\n"
                 "（信頼度ルーティングの前提・LLM 呼び出しは0回）", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_routing.png", dpi=150)
    print("保存: results/feature_routing.png")


VEHICLES_LADDER = [
    ("構造化列のみ", 3233.5, "tab:gray"),
    ("手書きルール\n正規化", 2783.0, "tab:green"),
    ("機能A\n人手ラベル200件", 2781.4, "tab:red"),
    ("機能A\n値の名前だけ", 2789.4, "tab:red"),
    ("機能A\n埋め込み列", 2601.1, "tab:blue"),
    ("従来の最良\n（併用・手組み）", 2596.0, "tab:orange"),
]


def fig_vehicles() -> None:
    """複数車種（Craigslist 60,000行）での機能A。"""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    labels = [n for n, _, _ in VEHICLES_LADDER]
    vals = [v for _, v, _ in VEHICLES_LADDER]
    bars = ax.bar(labels, vals, color=[c for _, _, c in VEHICLES_LADDER])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v:,.0f}",
                ha="center", fontsize=10)
    ax.set_ylabel("MAE（USD・低いほど良い）")
    ax.set_ylim(2400, 3400)
    ax.set_title("複数車種での機能A — 埋め込み列だけが手書きルールを超える\n"
                 "（Craigslist 60,000行・5-fold）", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "feature_vehicles.png", dpi=150)
    print("保存: results/feature_vehicles.png")


UNSEEN = [
    ("手書きルール", 2623, 4118),
    ("文字TF-IDF", 2511, 3715),
    ("e5 埋め込み", 2470, 3873),
    ("従来の最良\n（併用・手組み）", 2455, 3768),
    ("機能A\n既定エンコーダ", 2464, 3730),
]


def fig_unseen() -> None:
    """既知／未知の model で分けた MAE（P2）。"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(UNSEEN))
    w = 0.38
    known = [k for _, k, _ in UNSEEN]
    unseen = [u for _, _, u in UNSEEN]
    b1 = ax.bar(x - w / 2, known, w, label="既知の model", color="tab:blue")
    b2 = ax.bar(x + w / 2, unseen, w, label="未知の model", color="tab:red")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                    f"{b.get_height():,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in UNSEEN], fontsize=9)
    ax.set_ylabel("MAE（USD・低いほど良い）")
    ax.set_ylim(2000, 4400)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("機能A の既定エンコーダは未知語に強い側に着地した\n"
                 "（未知の model は test 行の 10.7%）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "feature_unseen.png", dpi=150)
    print("保存: results/feature_unseen.png")


if __name__ == "__main__":
    fig_ladder()
    fig_routing()
    fig_vehicles()
    fig_unseen()
