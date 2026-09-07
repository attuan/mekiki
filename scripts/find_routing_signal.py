"""ルーティングの信号を探す — LLM の自己申告 confidence 以外に何が使えるか。

## なぜこれを探すのか

機能B の初回実測（`docs/2026-09-01-llm-predictor.md`）で分かったのは次の2点。

- **オラクル**（行ごとに LLM と LightGBM の良いほうを選べた場合）は MAE 10.37。
  LightGBM 単体の 12.99 より 2.62 万円も良い。**伸びしろは大きい。**
- しかし **LLM の自己申告 confidence はその信号になっていない。**
  confidence の高い順に採用しても MAE は単調に動くだけで、途中に山がない。

そこで「どちらを使うべきか」を見分ける別の信号を探す。**信号として使えるのは、
推論時に手に入るものだけ**である（正解を見て決めるのは反則）。候補は3つ。

1. **統計モデル同士の食い違い**（LightGBM と XGBoost の予測差）
   食い違うほど「この行は難しい」ので、LLM の判断が効く余地があるかもしれない
2. **近傍の類似度の低さ**（1位の類似度）
   似た事例が見つからない行は、LLM に渡す証拠そのものが弱い
3. **LLM が動かした量**（|LLM − LightGBM|）
   大きく動かした行ほど、LLM が「何か見つけた」と主張している

各信号について「その値が高い/低い行だけ LLM を採用する」と MAE がどう動くかを見る。
**オラクルにどれだけ近づけるか**が評価軸。

## 実行

    .venv/bin/python scripts/find_routing_signal.py                       # シエンタ
    .venv/bin/python scripts/find_routing_signal.py --dataset vehicles

LLM を呼ばないので**無料**。行ごとの記録（`results/llm_predictor*_rows.csv`）を
読むだけである。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import N_SPLITS, SEED, load_dataset  # noqa: E402
from run_llm_predictor import DATASETS, SAMPLES, SPECS, build  # noqa: E402

from unfold.llm import ClaudeClient  # noqa: E402


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - y)))


def neighbour_similarity(name: str, n_eval: int) -> np.ndarray:
    """行ごとの「1位の近傍との類似度」を、測定と同じ分割で再現する。

    近傍索引は LLM を使わないので、ここだけ計算し直せば済む。
    `run_llm_predictor.run_eval` と同じ seed・同じ抽出手順を踏むこと。
    """
    from sklearn.model_selection import KFold

    ds = DATASETS[name]
    df = load_dataset(verbose=False, dataset=ds, sample=SAMPLES[name])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    client = ClaudeClient(api_key="")     # 呼ばないのでキーは不要

    out = []
    for tr_idx, te_idx in kf.split(df):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test_all = df.iloc[te_idx].reset_index(drop=True)
        pick = rng.choice(len(test_all), size=min(n_eval, len(test_all)),
                          replace=False)
        test = test_all.iloc[np.sort(pick)].reset_index(drop=True)
        model = build(client, 5, name, "trees").fit(train)
        _, sim = model.index_.query(test)
        out.append(sim[:, 0])             # 1位の類似度
    return np.concatenate(out)


def sweep(signal: np.ndarray, llm: np.ndarray, base: np.ndarray,
          y: np.ndarray, high_first: bool) -> pd.DataFrame:
    """信号の高い（または低い）順に r% だけ LLM を採用したときの MAE。"""
    order = np.argsort(-signal if high_first else signal, kind="stable")
    n = len(y)
    rows = []
    for r in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        k = int(round(n * r))
        take = np.zeros(n, bool)
        take[order[:k]] = True
        rows.append({"採用率": r, "MAE": mae(np.where(take, llm, base), y)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="ルーティング信号を探す")
    ap.add_argument("--dataset", default="sienta", choices=["sienta", "vehicles"])
    ap.add_argument("--n-eval", type=int, default=None,
                    help="測定時に使った --n-eval と同じ値（省略時は行数から推定）")
    args = ap.parse_args()

    suffix = "" if args.dataset == "sienta" else f"_{args.dataset}"
    src = ROOT / "results" / f"llm_predictor{suffix}_trees_rows.csv"
    if not src.exists():
        print(f"{src} がありません。先に次を実行してください:")
        print(f"  .venv/bin/python scripts/run_llm_predictor.py "
              f"--dataset {args.dataset} --evidence trees")
        raise SystemExit(1)

    d = pd.read_csv(src, encoding="utf-8-sig")
    y = d["実際"].to_numpy(float)
    llm = d["機能B"].to_numpy(float)
    lgbm = d["LightGBM"].to_numpy(float)
    xgb = d["XGBoost"].to_numpy(float)
    conf = d["confidence"].to_numpy(float)
    n = len(d)
    n_eval = args.n_eval or int(round(n / N_SPLITS))
    unit = DATASETS[args.dataset].unit

    base_mae = mae(lgbm, y)
    llm_mae = mae(llm, y)
    oracle = mae(np.where(np.abs(llm - y) < np.abs(lgbm - y), llm, lgbm), y)
    print(f"{n} 行 / {DATASETS[args.dataset].name}\n")
    print(f"  LightGBM             {base_mae:9,.2f} {unit}")
    print(f"  機能B（全行採用）      {llm_mae:9,.2f} {unit}")
    print(f"  オラクル（実現不可）    {oracle:9,.2f} {unit}"
          f"  ← ここに近づけるのが目標")
    print(f"  伸びしろ             {base_mae - oracle:9,.2f} {unit}\n")

    print("近傍の類似度を計算し直しています（LLM は呼びません）…")
    sim1 = neighbour_similarity(args.dataset, n_eval)
    if len(sim1) != n:
        print(f"  警告: 行数が合いません（{len(sim1)} != {n}）。"
              "--n-eval を測定時と同じ値にしてください。")
        sim1 = np.resize(sim1, n)

    signals = {
        "LLM の自己申告 confidence（高い順）": (conf, True),
        "木2つの食い違い |LGBM−XGB|（大きい順）": (np.abs(lgbm - xgb), True),
        "木2つの食い違い（小さい順）": (np.abs(lgbm - xgb), False),
        "近傍1位の類似度（低い順）": (sim1, False),
        "近傍1位の類似度（高い順）": (sim1, True),
        "LLM が動かした量（大きい順）": (np.abs(llm - lgbm), True),
        "LLM が動かした量（小さい順）": (np.abs(llm - lgbm), False),
    }

    print("\n" + "=" * 78)
    print("信号ごとに「上位 r% だけ LLM を採用」したときの MAE")
    print("=" * 78)
    table = {}
    for label, (sig, high) in signals.items():
        s = sweep(sig, llm, lgbm, y, high)
        table[label] = s.set_index("採用率")["MAE"]
    out = pd.DataFrame(table).T
    out.columns = [f"{c:.0%}" for c in out.columns]
    print(out.round(2).to_string())

    print(f"\n（採用率 0% = LightGBM のみ {base_mae:,.2f} / "
          f"100% = 機能B 全行 {llm_mae:,.2f}）")

    best_label, best_rate, best_mae = None, None, np.inf
    for label, row in out.iterrows():
        v = row.min()
        if v < best_mae:
            best_label, best_rate, best_mae = label, row.idxmin(), v
    print(f"\n最も良かった組み合わせ: {best_label} を {best_rate} 採用 → "
          f"MAE {best_mae:,.2f} {unit}")
    print(f"  LightGBM 比 {base_mae - best_mae:+,.2f} / "
          f"オラクルまでの残り {best_mae - oracle:,.2f}")
    print("\n※ 信号の選び方も採用率も**同じデータで選んでいる**ので、"
          "この数字は楽観側に偏る。\n"
          "   有望な信号が見つかったら、別の fold や別データで確かめること。")

    dst = ROOT / "results" / f"routing_signals{suffix}.csv"
    out.to_csv(dst, encoding="utf-8-sig")
    print(f"\n結果: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
