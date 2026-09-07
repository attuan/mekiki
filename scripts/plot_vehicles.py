"""複数車種（Craigslist）の結果の作図。results/vehicles_*.png を吐く。

再学習はしない。CSV を読んで描くだけ。

    .venv/bin/python scripts/plot_vehicles.py

## 出典が2つあることに注意

Craigslist の測定値には**混ぜてはいけない2系統**がある。

| 系統 | 出典 | 何者か |
|---|---|---|
| 旧分割 | `leaderboard_vehicles.csv` の 8/29〜8/30 の行 | mac 測定・32行 |
| 現分割 | `leaderboard_vehicles.csv` の 9/1 の行 | node 測定。ベースライン（0〜C2） |
| 現分割 | `results/s1_recheck.csv` | mac 測定。埋め込み以降（D1〜F+） |

**同じ手法名が旧分割と現分割の両方にある**（0・A1・A2・B1・B2・C1・C2 の7つ）。
`drop_duplicates(keep="last")` で引くと黙って新しい方を拾い、旧分割の図に
現分割の値が1本だけ混ざる。だから**日付で系統を選んでから**引くこと。

`clean_vehicles.py` は 9/1 まで書き出しの並びを固定しておらず、DuckDB が並列に読む
たびに parquet の行順が変わっていた。そのため `load_dataset(sample=60_000)` が
実行ごとに別の6万行を返しており、**8/31 以前の Craigslist の値は現在のコードでは
再現できない**（CLAUDE.md「中間データの並び順を変えない」/ `run_s1_recheck.py`）。

そこで作図も2系統に分けてある。

- `vehicles_ladder / vehicles_unseen / vehicles_vs_sienta` — **旧分割**。
  `docs/2026-08-29-vehicles-multi.md` に貼ってある記録用の図で、
  同ドキュメントの本文の数字と対応している。**差し替えない。**
  再現できない値であることが図の中に注記として入る
- `vehicles_ladder_recheck / vehicles_recheck / vehicles_unseen_recheck /
  vehicles_vs_sienta_recheck` — **現分割**。いま人に見せる図はこちら。
  どの機械で測ったかを図の中に書く（node と mac では MAE が 0.079% ずれる）

シエンタ側（`leaderboard.csv`）はサンプリングせず 5,507 行を全部使うので
この問題の影響を受けない。したがって現分割の図でもそのまま引ける。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import ROOT, SIENTA, VEHICLES  # noqa: E402
from plot_style import use_japanese_font  # noqa: E402

use_japanese_font()

OUT = ROOT / "results"
RECHECK = OUT / "s1_recheck.csv"

RULE = "#4a7fb0"    # 人手のルール・文字ベースの表現
EMB = "#2e6b4f"     # 埋め込み
BOTH = "#1f4d5a"    # 埋め込み + 文字TF-IDF の併用
PLAIN = "#9aa0a6"
DELTA = "#b03030"

STALE = ("※ この図は 8/31 以前の6万行（並び固定前）の測定。現在のコードでは再現しない。"
         "いま見せる図は results/vehicles_*_recheck.png を使うこと")


def _stale_note(fig) -> None:
    """再現できない値の図であることを、図自身に持たせる。

    画像だけが資料に貼られたり投影されたりしても注意書きが一緒に動くように、
    ドキュメント側の脚注ではなく図の中に入れておく。
    """
    fig.text(0.008, 0.012, STALE, fontsize=7.5, color=DELTA)


def _last(path: Path, before: str | None = None,
          since: str | None = None) -> pd.DataFrame:
    """leaderboard を手法名で引ける形にする。

    `before` / `since` は `実行日時` で測定の系統を選ぶための境目。
    同じ手法名が複数の系統にあるので、**先に系統を絞ってから**
    `keep="last"` を効かせないと、図の中に別条件の値が1本だけ混ざる。
    """
    lb = pd.read_csv(path, encoding="utf-8-sig")
    if before is not None:
        lb = lb[lb["実行日時"] < before]
    if since is not None:
        lb = lb[lb["実行日時"] >= since]
    return lb.drop_duplicates("手法", keep="last").set_index("手法")


def _source_note(fig, text: str) -> None:
    """どの条件で測った図なのかを、図自身に持たせる。"""
    fig.text(0.008, 0.012, text, fontsize=7.5, color="#555555")


def _recheck() -> pd.DataFrame:
    """9/1 に測り直した現分割の値。手法名の空白ゆれを潰してから引く。"""
    d = pd.read_csv(RECHECK, encoding="utf-8-sig")
    d["手法"] = d["手法"].str.replace(r"\s+", " ", regex=True).str.strip()
    return d.set_index("手法")


# ----------------------------------------------------------------------
# 旧分割（8/29 の記録用。docs/2026-08-29-vehicles-multi.md と対応）
# ----------------------------------------------------------------------

def ladder(lb: pd.DataFrame) -> None:
    items = [
        ("A1 線形回帰\n構造化列", "A1 線形回帰・構造化列", PLAIN),
        ("A2 LightGBM\n構造化列", "A2 LightGBM・構造化列", PLAIN),
        ("B1 +model\nそのまま", "B1 + model そのまま", RULE),
        ("B2 +model\n手書きルール", "B2 + model を手書きルールで正規化", RULE),
        ("C1 +model\n単語TF-IDF", "C1 + model の単語TF-IDF", RULE),
        ("C2 +model\n文字TF-IDF", "C2 + model の文字TF-IDF", RULE),
        ("D1 +model\n埋め込み", "D1 構造化列+model の埋め込み", EMB),
        ("D4 埋め込み\n+文字TF-IDF", "D4 埋め込み+文字TF-IDF", BOTH),
    ]
    labels = [i[0] for i in items]
    vals = np.array([lb.loc[i[1], "MAE"] for i in items])
    errs = [lb.loc[i[1], "MAE_std"] for i in items]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=[i[2] for i in items])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,.0f}",
                ha="center", fontsize=10, fontweight="bold")
    for i in range(1, len(vals)):
        ax.annotate(f"−{vals[i-1]-vals[i]:,.0f}",
                    xy=(i - 0.5, (vals[i - 1] + vals[i]) / 2 - 260),
                    ha="center", fontsize=9, color=DELTA)

    ax.set_ylabel("MAE（USD・5-fold CV）")
    ax.set_title("【旧分割】複数車種のはしご — Craigslist 60,000件 / 価格中央値 $11,995"
                 "（緑系＝埋め込みを使った構成）", fontsize=12)
    ax.set_ylim(0, vals.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _stale_note(fig)
    fig.savefig(OUT / "vehicles_ladder.png", dpi=150)
    print("保存:", OUT / "vehicles_ladder.png")


def unseen(path: Path) -> None:
    """訓練データに無かった model の行だけを取り出した比較。"""
    d = pd.read_csv(path, encoding="utf-8-sig")
    labels = ["B2 手書きルール", "C2 文字TF-IDF", "D1 埋め込み", "D4 併用"]
    colors = [RULE, RULE, EMB, BOTH]
    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - w / 2, d["既知MAE"], w, color="#c9d6e3", label="既知の model")
    b2 = ax.bar(x + w / 2, d["未知MAE"], w, color=colors, label="未知の model")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 60,
                    f"{b.get_height():,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MAE（USD）")
    ax.set_title("【旧分割】訓練データに無い model が来たとき — 手法別の MAE",
                 fontsize=12)
    ax.legend()
    ax.set_ylim(0, d["未知MAE"].max() * 1.22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _stale_note(fig)
    fig.savefig(OUT / "vehicles_unseen.png", dpi=150)
    print("保存:", OUT / "vehicles_unseen.png")


def cross_dataset(v: pd.DataFrame, s: pd.DataFrame) -> None:
    """単一車種と複数車種で「人手のルール」と「埋め込み」の効き方を比べる。

    価格の単位が違う（万円 / USD）ので、絶対値ではなく
    「構造化列だけの LightGBM から何%縮んだか」で揃える。
    """
    sets = [
        ("シエンタ 5,507件\n（単一車種）",
         s.loc["A2 LightGBM・従来列", "MAE"],
         s.loc["Ab1 +グレード名", "MAE"],
         s.loc["D2 従来列+タイトル埋め込み(グレード名なし)", "MAE"]),
        ("Craigslist 60,000件\n（複数車種）",
         v.loc["A2 LightGBM・構造化列", "MAE"],
         v.loc["B2 + model を手書きルールで正規化", "MAE"],
         v.loc["D1 構造化列+model の埋め込み", "MAE"]),
    ]
    x = np.arange(len(sets))
    w = 0.35
    rule = [(a - r) / a * 100 for _, a, r, _ in sets]
    emb = [(a - e) / a * 100 for _, a, _, e in sets]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w / 2, rule, w, color=RULE, label="人手のルール（正規表現）")
    b2 = ax.bar(x + w / 2, emb, w, color=EMB, label="埋め込み（ルールなし）")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                    f"{b.get_height():.1f}%", ha="center", fontsize=10,
                    fontweight="bold")
    ax.set_xticks(x, [s[0] for s in sets])
    ax.set_ylabel("構造化列のみ（A2）からの MAE 改善率")
    ax.set_title("【旧分割】車種を増やすと、人手のルールだけが効かなくなる",
                 fontsize=12)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(rule + emb) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _stale_note(fig)
    fig.savefig(OUT / "vehicles_vs_sienta.png", dpi=150)
    print("保存:", OUT / "vehicles_vs_sienta.png")


# ----------------------------------------------------------------------
# 現分割（9/1 の測り直し。いま人に見せるのはこちら）
# ----------------------------------------------------------------------

def recheck_ladder(lb: pd.DataFrame) -> None:
    """現分割（9/1・node）で測り直したベースラインのはしご。

    D1 以降（埋め込み・機能A）はまだ node で測り直していないので、
    **この図は C2 までで止める。**足りない分を s1_recheck（mac 測定）から
    借りてくると機械をまたいだ値を1本の棒グラフに並べることになる。
    """
    items = [
        ("0 中央値\n（予測しない）", "0  中央値", PLAIN),
        ("A1 線形回帰\n構造化列", "A1 線形回帰・構造化列", PLAIN),
        ("A2 LightGBM\n構造化列", "A2 LightGBM・構造化列", PLAIN),
        ("B1 +model\nそのまま", "B1 + model そのまま", RULE),
        ("B2 +model\n手書きルール", "B2 + model を手書きルールで正規化", RULE),
        ("C1 +model\n単語TF-IDF", "C1 + model の単語TF-IDF", RULE),
        ("C2 +model\n文字TF-IDF", "C2 + model の文字TF-IDF", RULE),
    ]
    missing = [k for _, k, _ in items if k not in lb.index]
    if missing:
        print(f"※ 現分割に {missing} が無いので vehicles_ladder_recheck.png は作りません")
        return

    labels = [i[0] for i in items]
    vals = np.array([lb.loc[i[1], "MAE"] for i in items])
    errs = [lb.loc[i[1], "MAE_std"] for i in items]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4,
                  color=[i[2] for i in items])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 160, f"{v:,.0f}",
                ha="center", fontsize=10, fontweight="bold")
    for i in range(1, len(vals)):
        ax.annotate(f"−{vals[i-1]-vals[i]:,.0f}",
                    xy=(i - 0.5, (vals[i - 1] + vals[i]) / 2 - 300),
                    ha="center", fontsize=9, color=DELTA)

    ax.set_ylabel("MAE（USD・5-fold CV）")
    ax.set_title("model をどう扱うか — Craigslist 60,000件 / 価格中央値 $11,995"
                 "（2026-09-01 測定）", fontsize=12)
    ax.set_ylim(0, vals.max() * 1.2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _source_note(fig, "測定: 2026-09-01 / 計算ノード / 現在の6万行。"
                      "埋め込み以降（D1〜F+）はまだノードで測り直していないので"
                      "この図には含めていない → results/vehicles_recheck.png")
    fig.savefig(OUT / "vehicles_ladder_recheck.png", dpi=150)
    print("保存:", OUT / "vehicles_ladder_recheck.png")


def recheck_overall(r: pd.DataFrame) -> None:
    """model の扱いを変えた6手法の全体 MAE。

    値の幅が 2,577〜2,758 と狭い。0 から棒を描くと全部同じ高さに見え、
    軸を途中から始めると棒の長さが差を偽るので、**点で描いて誤差棒を添える**。
    点なら基準線が 0 でなくてよい。
    """
    items = [
        ("B2 手書きルール", "B2 手書きルール", RULE),
        ("C2 文字TF-IDF", "C2 文字TF-IDF", RULE),
        ("D1 e5 埋め込み", "D1 e5 埋め込み", EMB),
        ("D4 埋め込み+文字TF-IDF", "D4 埋め込み+文字TF-IDF", BOTH),
        ("F 機能A（埋め込み列）", "F 機能A（埋め込み列）", EMB),
        ("F+ 機能A+文字TF-IDF", "F+ 機能A+文字TF-IDF", BOTH),
    ]
    labels = [i[0] for i in items]
    vals = np.array([r.loc[i[1], "MAE"] for i in items])
    errs = np.array([r.loc[i[1], "MAE_std"] for i in items])
    y = np.arange(len(items))[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor="#9aa0a6",
                elinewidth=1.4, capsize=4, zorder=1)
    ax.scatter(vals, y, s=90, c=[i[2] for i in items], zorder=2)
    for v, yy in zip(vals, y):
        ax.text(v, yy + 0.26, f"{v:,.0f}", ha="center", fontsize=10,
                fontweight="bold")

    best = int(np.argmin(vals))
    ax.annotate("最良", xy=(vals[best], y[best]), xytext=(0, -22),
                textcoords="offset points", ha="center", fontsize=9,
                color=BOTH, fontweight="bold")

    ax.set_yticks(y, labels)
    ax.set_ylim(-0.9, len(items) - 0.4)
    ax.set_xlabel("MAE（USD・5-fold CV。誤差棒は fold 間の標準偏差）")
    ax.set_title("model の扱いを変えた比較 — Craigslist 60,000件（2026-09-01 測定）",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _source_note(fig, "測定: 2026-09-01 / 手元の Mac / 現在の6万行"
                      "（results/s1_recheck.csv）")
    fig.savefig(OUT / "vehicles_recheck.png", dpi=150)
    print("保存:", OUT / "vehicles_recheck.png")


def recheck_unseen(r: pd.DataFrame) -> None:
    """訓練データに無かった model の行だけを取り出した比較（現分割）。"""
    items = [
        ("B2\n手書きルール", "B2 手書きルール", RULE),
        ("C2\n文字TF-IDF", "C2 文字TF-IDF", RULE),
        ("D1\ne5 埋め込み", "D1 e5 埋め込み", EMB),
        ("D4\n埋め込み+TF-IDF", "D4 埋め込み+文字TF-IDF", BOTH),
        ("F\n機能A", "F 機能A（埋め込み列）", EMB),
        ("F+\n機能A+TF-IDF", "F+ 機能A+文字TF-IDF", BOTH),
    ]
    known = np.array([r.loc[i[1], "既知MAE"] for i in items])
    unk = np.array([r.loc[i[1], "未知MAE"] for i in items])
    x = np.arange(len(items))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(x - w / 2, known, w, color="#c9d6e3", label="既知の model")
    b2 = ax.bar(x + w / 2, unk, w, color=[i[2] for i in items],
                label="未知の model")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 50,
                    f"{b.get_height():,.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, [i[0] for i in items], fontsize=9)
    ax.set_ylabel("MAE（USD）")
    ax.set_title("訓練データに無い model が来たとき — 手法別の MAE"
                 "（Craigslist 60,000件・2026-09-01 測定）", fontsize=12)
    ax.legend()
    ax.set_ylim(0, unk.max() * 1.22)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _source_note(fig, "測定: 2026-09-01 / 手元の Mac / 現在の6万行"
                      "（results/s1_recheck.csv）")
    fig.savefig(OUT / "vehicles_unseen_recheck.png", dpi=150)
    print("保存:", OUT / "vehicles_unseen_recheck.png")


def recheck_cross(r: pd.DataFrame, s: pd.DataFrame) -> None:
    """単一車種と複数車種で「人手のルール」と「埋め込み」の効き方を比べる。

    旧分割版は「構造化列だけ（A2）から何%縮んだか」で揃えていたが、
    **現分割には A2 の測定が無い**（`run_s1_recheck.py` は比較線として
    B2 以降だけを引き直したため）。改善率を出すには A2 も測り直す必要があり、
    それは作図の仕事ではない。

    そこで**各データの中の絶対値を並べ、差を注記する**形にした。単位が違うので
    2枚に分け、軸も別々にしてある。伝えたいのは「各パネルの中の差」だけである。
    """
    def panel(title: str, unit: str, fmt: str, dfmt: str,
              rule_row: pd.Series, emb_row: pd.Series) -> dict:
        """1枚分の材料。fmt は値の書式、dfmt は符号つきの差の書式。"""
        return {"title": title, "unit": unit, "fmt": fmt, "dfmt": dfmt,
                "rule": rule_row["MAE"], "emb": emb_row["MAE"],
                "noise": max(rule_row["MAE_std"], emb_row["MAE_std"])}

    panels = [
        panel("シエンタ 5,507件（単一車種）", "万円", "{:.2f}", "{:+.2f}",
              s.loc["Ab1 +グレード名"],
              s.loc["D2 従来列+タイトル埋め込み(グレード名なし)"]),
        panel("Craigslist 60,000件（複数車種）", "USD", "{:,.0f}", "{:+,.0f}",
              r.loc["B2 手書きルール"],
              r.loc["F 機能A（埋め込み列）"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, p in zip(axes, panels):
        vals = [p["rule"], p["emb"]]
        y = [1, 0]
        ax.barh(y, vals, height=0.5, color=[RULE, EMB])
        for v, yy in zip(vals, y):
            ax.text(v * 1.02, yy, p["fmt"].format(v), va="center", fontsize=11,
                    fontweight="bold")
        ax.set_yticks(y, ["人手のルール\n（正規表現）", "機能A\n（ルールなし）"],
                      fontsize=10)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel(f"MAE（{p['unit']}）", fontsize=10)
        ax.set_title(p["title"], fontsize=11)
        ax.grid(axis="x", alpha=0.3)

        # 勝ち負けは fold 間の振れと比べて決める。振れより小さい差を
        # 「勝った」と読むのがこのプロジェクトで一番やりがちな間違いなので、
        # 図の側で機械的に判定させる。
        diff = p["emb"] - p["rule"]
        if abs(diff) <= p["noise"]:
            verdict = f"互角（fold の振れ ±{p['fmt'].format(p['noise'])} の内側）"
        else:
            verdict = "機能A の勝ち" if diff < 0 else "人手のルールの勝ち"
        ax.annotate(f"差 {p['dfmt'].format(diff)} {p['unit']} — {verdict}",
                    xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                    fontsize=10, color=DELTA, fontweight="bold")

    fig.suptitle("車種を増やすと、人手のルールが追いつかなくなる"
                 "（単位が違うので、見るのは各パネルの中の差だけ）", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    _source_note(fig, "測定: 2026-09-01 / 手元の Mac。"
                      "シエンタは leaderboard.csv、Craigslist は s1_recheck.csv")
    fig.savefig(OUT / "vehicles_vs_sienta_recheck.png", dpi=150)
    print("保存:", OUT / "vehicles_vs_sienta_recheck.png")


# 系統の境目。9/1 に clean_vehicles.py の並びを固定したので、
# その前後で load_dataset(sample=60_000) が引く行が違う。
SPLIT_FIX = "2026-08-31"


def main() -> None:
    sienta = _last(SIENTA.leaderboard)

    # 旧分割 — 8/29 のドキュメントに貼ってある記録用。
    # 同名の行が 9/1 にも入っているので、日付で 8/31 より前に限定する。
    old = _last(VEHICLES.leaderboard, before=SPLIT_FIX)
    ladder(old)
    breakdown = OUT / "vehicles_unseen_breakdown.csv"
    if breakdown.exists():
        unseen(breakdown)
    cross_dataset(old, sienta)

    # 現分割 — いま人に見せる図
    recheck_ladder(_last(VEHICLES.leaderboard, since=SPLIT_FIX))
    if RECHECK.exists():
        r = _recheck()
        recheck_overall(r)
        recheck_unseen(r)
        recheck_cross(r, sienta)
    else:
        print(f"※ {RECHECK.name} が無いので現分割の図は作りません "
              "（scripts/run_s1_recheck.py で作れます）")


if __name__ == "__main__":
    main()
