"""機能B が統計モデルに負けた原因を、行ごとの記録から切り分ける。

`run_llm_predictor.py` が出す `results/llm_predictor_rows.csv` を読むだけなので、
**LLM を呼ばない = 無料**。以下の4つを順に確かめる。

Q1 LLM はどちらに動いたか
    LightGBM の予測から離れた方向が、正解に近づく側だったか遠ざかる側だったか。
    「離れること自体が悪い」のか「離れ方が下手」なのかを分ける。

Q2 confidence は誤差を見分けられているか
    信頼度ルーティング（PRD「信頼度ルーティング」・P4）は「自信の無い行だけ LLM に回す」設計。
    それが成り立つのは confidence が実際の誤差と相関している場合だけ。

Q3 ルーティングすると何が起きるか
    confidence の高い行だけ LLM の答えを採り、残りは LightGBM に任せる。
    採用率を 0%→100% で振って MAE と費用の曲線を出す（P4 の中身）。

Q4 単純に混ぜたらどうか
    LLM と LightGBM の平均。アンサンブルの下限として置いておく。

実行:
    .venv/bin/python scripts/analyze_llm_predictor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "results" / "llm_predictor_rows.csv"
COST_PER_ROW = 0.0105   # 実測（300行で $3.157）


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    if not SRC.exists():
        print(f"{SRC} がありません。先に run_llm_predictor.py を実行してください。")
        raise SystemExit(1)
    d = pd.read_csv(SRC, encoding="utf-8-sig")
    y = d["実際"].to_numpy(float)
    llm = d["機能B"].to_numpy(float)
    lgbm = d["LightGBM"].to_numpy(float)
    conf = d["confidence"].to_numpy(float)
    n = len(d)

    print(f"{n} 行を分析します。\n")
    print(f"  機能B（LLM）   MAE {mae(y, llm):6.2f} 万円")
    print(f"  LightGBM       MAE {mae(y, lgbm):6.2f} 万円")

    # --- Q1 LLM はどちらに動いたか -----------------------------------
    print("\n" + "=" * 74)
    print("Q1 LLM は LightGBM からどちらに動いたか")
    print("=" * 74)
    move = llm - lgbm                       # LLM が動かした量と向き
    err_lgbm = np.abs(lgbm - y)
    err_llm = np.abs(llm - y)
    better = err_llm < err_lgbm
    moved = np.abs(move) > 1e-9

    print(f"  動かした行: {moved.sum()}/{n}（{moved.mean():.1%}）"
          f" / 平均の動き幅 {np.abs(move[moved]).mean():.2f} 万円")
    print(f"  動いて良くなった行: {better.sum()}（{better.mean():.1%}）")
    print(f"  動いて悪くなった行: {(~better & moved).sum()}"
          f"（{(~better & moved).mean():.1%}）")
    print(f"  良くなった行の改善幅: "
          f"{(err_lgbm - err_llm)[better].mean():.2f} 万円")
    print(f"  悪くなった行の悪化幅: "
          f"{(err_llm - err_lgbm)[~better & moved].mean():.2f} 万円")
    print("\n  → 当たり外れの回数はほぼ半々でも、外したときの傷が深ければ負ける。"
          "\n     上の2つの幅を比べるのがこの表の読みどころ。")

    # 動きの向きに偏りがあるか（強気/弱気）
    print(f"\n  上方修正した行: {(move > 0).mean():.1%}"
          f" / 下方修正した行: {(move < 0).mean():.1%}")
    print(f"  平均の修正量: {move.mean():+.2f} 万円"
          f"（正なら「LLM は統計モデルより強気」）")

    # --- Q2 confidence は誤差を見分けるか -----------------------------
    print("\n" + "=" * 74)
    print("Q2 confidence は誤差を見分けられているか")
    print("=" * 74)
    print(f"  confidence の分布: 平均 {conf.mean():.2f} / "
          f"最小 {conf.min():.2f} / 最大 {conf.max():.2f} / "
          f"水準数 {len(np.unique(conf))}")
    corr = float(np.corrcoef(conf, err_llm)[0, 1])
    print(f"  confidence と絶対誤差の相関: {corr:+.3f}"
          "（負なら「自信があるほど当たる」＝ルーティングに使える）")
    rows = []
    for lo, hi in [(0.0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        rows.append({"confidence": f"{lo:.2f}〜{hi:.2f}", "行数": int(m.sum()),
                     "機能B の MAE": round(mae(y[m], llm[m]), 2),
                     "LightGBM の MAE": round(mae(y[m], lgbm[m]), 2)})
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Q3 ルーティング曲線 -------------------------------------------
    print("\n" + "=" * 74)
    print("Q3 confidence の高い行だけ LLM を採用する（P4 のルーティング曲線）")
    print("=" * 74)
    print("  ※ ここでは「LLM を呼ぶ行を選ぶ」のではなく「呼んだあとに採用する行を"
          "\n     選ぶ」形で測っている。費用は全行に呼んだ前提の上限。")
    order = np.argsort(-conf, kind="stable")     # confidence の高い順
    rows = []
    for r in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        k = int(round(n * r))
        take = np.zeros(n, bool)
        take[order[:k]] = True
        blend = np.where(take, llm, lgbm)
        rows.append({"LLM を採用する割合": f"{r:.0%}", "行数": k,
                     "MAE": round(mae(y, blend), 2),
                     "費用_usd（全行に呼んだ場合）": round(n * COST_PER_ROW, 2)})
    curve = pd.DataFrame(rows)
    print()
    print(curve.to_string(index=False))
    best = curve.loc[curve["MAE"].idxmin()]
    print(f"\n  → 最良は採用率 {best['LLM を採用する割合']} で MAE {best['MAE']}。")

    # 上限の参考: 行ごとに良いほうを選べたら（オラクル）
    oracle = np.where(err_llm < err_lgbm, llm, lgbm)
    print(f"  参考（オラクル・実現不可）: 行ごとに良いほうを選べれば "
          f"MAE {mae(y, oracle):.2f}。"
          "\n     ここまで差があるということは、"
          "**どちらを使うべきかを見分ける信号があれば伸びしろは大きい**。")

    # --- Q4 単純な混合 -------------------------------------------------
    print("\n" + "=" * 74)
    print("Q4 単純に平均したら")
    print("=" * 74)
    for w in (0.25, 0.5, 0.75):
        mix = w * llm + (1 - w) * lgbm
        print(f"  LLM {w:.0%} + LightGBM {1 - w:.0%}: MAE {mae(y, mix):6.2f}")

    out = ROOT / "results" / "llm_routing_curve.csv"
    curve.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nルーティング曲線: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
