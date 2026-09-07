"""P4 の続き — 信頼度ルーティング（`AdaptivePredictor`）をライブラリとして実測する。

`find_routing_signal.py` は**測定済みの行データを後から並べ替えて**曲線を引いた。
こちらは同じことを **`unfold` の API だけで**やる。確かめたいのは2つ。

1. **ライブラリだけで同じ形の曲線が引けるか。**
   割合を上げるほど MAE が下がるという 8/31 までの傾向を、
   `results/routing_signals_vehicles.csv` と並べて確かめる
2. **運用の形（見積もり → 実行 → 承認）が一通り動くか。**
   `plan()` で呼ぶ前に費用を見て、`predict()` で一部だけ回し、
   `approve()` すると次回そのぶん呼ばずに済む、という流れ

## 実行

    .venv/bin/python scripts/run_adaptive.py --dataset vehicles          # 曲線＋運用デモ
    .venv/bin/python scripts/run_adaptive.py --dataset vehicles --plan-only  # 見積もりだけ（無料）

**LLM を呼ぶので課金される。**1行あたり約 $0.0086、`--n-eval 120`（600行）の
曲線で約 $5。同じ行を二度は課金しない（ディスクキャッシュ）ので、
実装をいじって測り直すぶんには無料。上限は `--max-cost`（既定 $1.00）で止まる。

**8/31 までの測定のキャッシュは使えない。** 当時の
`vehicles_multi_clean.parquet` は書き出し順が実行ごとに変わっており、
`sample(60_000)` が引く行が今と違うためである（9/1 に `ORDER BY id` で
固定した）。**以後は同じ行が出るので、再実行は無料になる。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import N_SPLITS, SEED, load_dataset  # noqa: E402
from run_llm_predictor import DATASETS, SAMPLES, build  # noqa: E402

from unfold import AdaptivePredictor  # noqa: E402
from unfold.llm import ClaudeClient  # noqa: E402

RATES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def folds(name: str, n_eval: int):
    """`run_llm_predictor.run_eval` とまったく同じ分割・同じ抽出を再現する。

    ここを揃えないとプロンプトが変わり、キャッシュが効かず課金される。
    """
    ds = DATASETS[name]
    df = load_dataset(dataset=ds, sample=SAMPLES[name])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test_all = df.iloc[te_idx].reset_index(drop=True)
        pick = rng.choice(len(test_all), size=min(n_eval, len(test_all)),
                          replace=False)
        test = test_all.iloc[np.sort(pick)].reset_index(drop=True)
        yield fold, train, test


def make(client: ClaudeClient, name: str, args, **kw) -> AdaptivePredictor:
    inner = build(client, args.n_examples, name, args.evidence)
    return AdaptivePredictor(predictor=inner, signal=args.signal, **kw)


# ---------------------------------------------------------------------
# 1. 見積もり — LLM を呼ぶ前に「何行・いくら・何秒」を見る
# ---------------------------------------------------------------------

def show_plan(client: ClaudeClient, name: str, args) -> None:
    ds = DATASETS[name]
    fold, train, test = next(iter(folds(name, args.n_eval)))
    print(f"\n[fold{fold}] 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
    m = make(client, name, args, escalate_rate=args.rate).fit(train)
    print(f"\n--- plan()（LLM は呼ばない）---")
    for k, v in m.plan(test).items():
        print(f"  {k}: {v}")
    print(f"\n  ※ 全 {len(train) + len(test):,} 行に広げると "
          f"${(len(train) + len(test)) * args.rate * 0.0086:,.2f}、"
          f"{(len(train) + len(test)) * args.rate * 1.03 / 3600:,.1f} 時間"
          f"（並列8の実測から）")


# ---------------------------------------------------------------------
# 2. 曲線 — 精度・費用・レイテンシを同時に出す
# ---------------------------------------------------------------------

def run_curve(client: ClaudeClient, name: str, args) -> pd.DataFrame:
    """全行に回した結果を集め、割合を振ったときの曲線をプールして描く。

    fold ごとに曲線を引くと 120 行ずつの順位になり、
    `find_routing_signal.py`（600行をまとめて並べ替え）と比べられない。
    そこで **fold をまたいで 600 行を一度に並べ替える**。
    """
    ds = DATASETS[name]
    parts = []
    for fold, train, test in folds(name, args.n_eval):
        print(f"\n[fold{fold}] 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
        # 曲線を引くには全行ぶんの「LLM ならこう答えた」が要るので rate=1.0
        m = make(client, name, args, escalate_rate=1.0).fit(train)
        m.predict(test, verbose=True)
        prov = m.provenance()
        base_col = [c for c in prov.columns if c.startswith("証拠_")][0]
        parts.append(pd.DataFrame({
            "fold": fold,
            "実際": test[ds.target].to_numpy(dtype=float),
            "機能B": prov["予測"].to_numpy(dtype=float),
            "統計モデル": prov[base_col].to_numpy(dtype=float),
            "信号": prov["信号"].to_numpy(dtype=float),
        }))
        if client.usage.cost > args.max_cost:
            raise SystemExit(
                f"費用が上限 ${args.max_cost} を超えました "
                f"(${client.usage.cost:.2f})。--max-cost で変えられます。")

    d = pd.concat(parts, ignore_index=True)
    y = d["実際"].to_numpy()
    llm = d["機能B"].to_numpy()
    base = d["統計モデル"].to_numpy()
    order = np.argsort(-d["信号"].to_numpy(), kind="stable")

    unit_cost = (client.usage.cost / max(client.usage.calls - client.usage.cache_hits, 1)
                 if client.usage.calls > client.usage.cache_hits else 0.0086)
    rows = []
    for r in RATES:
        k = int(round(len(d) * r))
        pred = base.copy()
        pred[order[:k]] = llm[order[:k]]
        rows.append({"割合": r, "行数": k, "MAE": mae(y, pred),
                     "費用_usd": round(k * unit_cost, 4),
                     "推定時間_秒": round(23.85 + np.ceil(k / 8) * 4.6, 1)})
    curve = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"ルーティング曲線（{len(d)} 行 / 信号: {args.signal}）")
    print("=" * 78)
    print(curve.round(2).to_string(index=False))
    print(f"\n  0% = 統計モデルだけ {mae(y, base):,.2f} {ds.unit} / "
          f"100% = 全行 LLM {mae(y, llm):,.2f} {ds.unit}")

    # 8/31 までの測定と並べる。**同じ数字にはならない。**
    # 当時の parquet は行の並びが実行ごとに変わっていたため、6万行の抽出が
    # 今と違う（`clean_vehicles.py` の並び順を固定したのは 9/1）。
    # 傾向（割合を上げるほど MAE が下がるか）が一致するかだけを見る。
    ref_path = ROOT / "results" / f"routing_signals_{name}.csv"
    if ref_path.exists() and args.signal == "disagreement":
        ref = pd.read_csv(ref_path, encoding="utf-8-sig", index_col=0)
        label = [i for i in ref.index if "食い違い" in i and "大きい順" in i]
        if label:
            got = curve.set_index("割合")["MAE"]
            print(f"\n--- 参考: 8/31 までの測定（{ref_path.name}）---")
            print("  ※ 抽出した 6 万行が当時と違うので、値そのものは比べられない")
            print(f"  {'割合':>6} {'今回':>10} {'当時':>10}")
            for r in RATES:
                print(f"  {r:>6.0%} {got[r]:>10,.2f} "
                      f"{float(ref.loc[label[0], f'{r:.0%}']):>10,.2f}")
    return curve


# ---------------------------------------------------------------------
# 3. 運用の形 — 一部だけ回し、承認して高速パスを広げる
# ---------------------------------------------------------------------

def run_operation(client: ClaudeClient, name: str, args) -> None:
    ds = DATASETS[name]
    fold, train, test = next(iter(folds(name, args.n_eval)))
    truth = test[ds.target].to_numpy(dtype=float)

    m = make(client, name, args, escalate_rate=args.rate).fit(train)
    pred = m.predict(test, verbose=True)

    print("\n" + "=" * 78)
    print(f"運用の形（fold{fold} / 上位 {args.rate:.0%} だけ LLM）")
    print("=" * 78)
    print(m.report())
    print(f"\n  この {len(test)} 行の MAE: {mae(truth, pred):,.2f} {ds.unit}")

    # **回した行で、LLM は統計モデルより良かったか。**
    # 「高速パスのほうが誤差が小さい」のは、難しい行を選んで回している以上
    # 当たり前であって、それでは何も確かめたことにならない。
    # 同じ行の上で LLM と統計モデルを突き合わせる。
    prov = m.provenance()
    base_col = [c for c in prov.columns if c.startswith("証拠_")][0]
    route = m.route()
    print()
    for path, g in route.groupby("経路"):
        rows = g["行"].to_numpy()
        err_used = np.abs(pred[rows] - truth[rows]).mean()
        err_base = np.abs(prov[base_col].to_numpy()[rows] - truth[rows]).mean()
        label = "LLM  " if path == "llm" else "統計値"
        print(f"  {path:>4}: {len(g):>4} 行 / 採用した値（{label}） {err_used:9,.2f}"
              f" / {base_col[3:]}単体 {err_base:9,.2f} {ds.unit}")
    base_all = float(np.abs(prov[base_col].to_numpy() - truth).mean())
    print(f"\n  1行も呼ばなかった場合（{base_col[3:]}だけ）: {base_all:,.2f} {ds.unit}"
          f" → 上位 {args.rate:.0%} を回して {mae(truth, pred):,.2f} "
          f"（{base_all - mae(truth, pred):+,.2f}）")
    print("  → llm の行で左が右より小さければ、そこに回した判断が当たっている。")

    print("\n--- explain()（LLM に回った行の1つめ）---")
    llm_rows = route.loc[route["経路"] == "llm", "行"].tolist()
    if llm_rows:
        print(m.explain(int(llm_rows[0]))[:1200])

    q = m.review_queue()
    print(f"\n--- 教師ラベル候補（review_queue）{len(q)} 件のうち先頭3件 ---")
    cols = [c for c in ["行", "LLMの答え", "confidence", "信号"] if c in q.columns]
    print(q[cols].head(3).round(3).to_string(index=False))

    n = m.approve()
    print(f"\n承認しました: {n} 件")
    before = client.usage.calls
    m.predict(test)
    print(f"  同じ {len(test)} 行をもう一度予測 → LLM 呼び出し "
          f"{client.usage.calls - before} 回（承認済みは呼ばない）")
    print(f"  由来の内訳: {m.provenance()['由来'].value_counts().to_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="信頼度ルーティングの実測")
    ap.add_argument("--dataset", default="vehicles",
                    choices=["sienta", "vehicles"])
    ap.add_argument("--n-eval", type=int, default=120,
                    help="各 fold の test から採点する行数（測定時と揃えること）")
    ap.add_argument("--n-examples", type=int, default=5)
    ap.add_argument("--evidence", default="trees", choices=["all", "trees"],
                    help="測定と同じ trees が既定。all にすると課金され直す")
    ap.add_argument("--signal", default="disagreement",
                    choices=["disagreement", "similarity", "unseen"])
    ap.add_argument("--rate", type=float, default=0.3,
                    help="運用デモで LLM に回す割合")
    ap.add_argument("--plan-only", action="store_true",
                    help="見積もりだけ出して終わる（LLM を呼ばない＝無料）")
    ap.add_argument("--no-curve", action="store_true",
                    help="曲線を飛ばして運用デモだけ")
    ap.add_argument("--max-cost", type=float, default=1.0,
                    help="これを超えたら止める（USD）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = ClaudeClient(model=args.model, effort=args.effort,
                          max_workers=args.workers)
    print(f"データ {args.dataset} / 信号 {args.signal} / 証拠 {args.evidence} / "
          f"モデル {args.model}")

    show_plan(client, args.dataset, args)
    if args.plan_only:
        print("\n--plan-only なのでここで終わります（LLM は呼んでいません）。")
        return

    if not client.available():
        print("\nANTHROPIC_API_KEY がありません。--plan-only なら無料で動きます。")
        raise SystemExit(1)

    if not args.no_curve:
        curve = run_curve(client, args.dataset, args)
        out = ROOT / "results" / f"adaptive_curve_{args.dataset}.csv"
        curve.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n曲線: {out.relative_to(ROOT)}")

    run_operation(client, args.dataset, args)

    print("\n--- 実測費用 ---")
    for k, v in client.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
