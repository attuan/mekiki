"""P4 の残り — 機能B のレイテンシ（応答時間）を測る。

## なぜ測るのか

PRD 信頼度ルーティングは「閾値を動かしたときに **精度・レイテンシ・費用の3つが同時に見える**こと」
を要件にしている。精度と費用は取れたが、レイテンシだけ未計測だった。

レイテンシは費用と違って**行数に比例しない**。並列で投げるので、
実質は「1リクエストの時間 × (行数 ÷ 並列数)」になる。つまり
**並列数を上げれば時間は縮むが、費用は変わらない。** そこを分けて測る。

## 測り方

キャッシュを通さず（`cache_dir=None`）に実際のリクエストを出し、
1件あたりの所要時間を記録する。統計モデルの fit や近傍索引の構築は
**LLM を呼ぶ前に1回だけ**やる処理なので、別に測って分けて報告する。

**この測定は必ず課金される**（キャッシュを切っているため）。
既定の 20 行 × 3 条件 = 60 回で、Craigslist なら約 $0.5。

## 実行

    .venv/bin/python scripts/measure_latency.py --dataset vehicles --n 20
    .venv/bin/python scripts/measure_latency.py --dataset vehicles --n 20 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import SEED, load_dataset  # noqa: E402
from run_llm_predictor import DATASETS, SAMPLES, build  # noqa: E402

from unfold.llm import ClaudeClient  # noqa: E402


def percentiles(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {"件数": len(a), "中央値": float(np.median(a)),
            "平均": float(a.mean()), "p90": float(np.percentile(a, 90)),
            "最小": float(a.min()), "最大": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description="機能B のレイテンシを測る")
    ap.add_argument("--dataset", default="vehicles",
                    choices=["sienta", "vehicles"])
    ap.add_argument("--n", type=int, default=20, help="1条件あたりの行数")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8],
                    help="試す並列数")
    ap.add_argument("--description", type=int, default=0,
                    help="自由記述を載せる文字数（0 なら使わない）")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばず、準備処理の時間だけ測る")
    args = ap.parse_args()

    name = args.dataset
    ds = DATASETS[name]

    print(f"データ {name} / 1条件 {args.n} 行 / 並列数 {args.workers}")
    if not args.dry_run:
        n_calls = args.n * len(args.workers)
        print(f"**キャッシュを切って実際に呼びます。合計 {n_calls} 回課金されます。**")

    # --- 準備処理（LLM を呼ぶ前に1回だけ走る部分）---------------------
    t0 = time.perf_counter()
    df = load_dataset(verbose=False, dataset=ds, sample=SAMPLES[name])
    t_load = time.perf_counter() - t0

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    test = df.iloc[idx[:args.n]].reset_index(drop=True)
    train = df.iloc[idx[args.n:]].reset_index(drop=True)

    probe = ClaudeClient(api_key="")   # fit の計測では呼ばない
    t0 = time.perf_counter()
    model = build(probe, 5, name, "all", args.description).fit(train)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.index_.query(test)
    t_query = time.perf_counter() - t0

    print("\n" + "=" * 74)
    print("準備処理（LLM を呼ぶ前。行数に対して1回だけ）")
    print("=" * 74)
    print(f"  データ読み込み（{len(df):,} 行）      {t_load:8.2f} 秒")
    print(f"  統計モデルの fit + 近傍索引の構築    {t_fit:8.2f} 秒")
    print(f"  近傍検索（{args.n} 行ぶん）           {t_query * 1000:8.1f} ミリ秒")
    print(f"  → 1行あたりの近傍検索              "
          f"{t_query / args.n * 1000:8.2f} ミリ秒")

    if args.dry_run:
        print("\n--dry-run のため LLM の計測は省略しました。")
        return

    if not ClaudeClient().available():
        print("\nANTHROPIC_API_KEY がありません。")
        raise SystemExit(1)

    # --- LLM 呼び出し ---------------------------------------------------
    print("\n" + "=" * 74)
    print("LLM 呼び出し（キャッシュを切って実測）")
    print("=" * 74)

    rows = []
    for w in args.workers:
        # cache_dir=None で毎回本物のリクエストを出す。
        # 条件ごとにクライアントを作り直して累計を分ける。
        client = ClaudeClient(cache_dir=None, max_workers=w)
        m = build(client, 5, name, "all", args.description).fit(train)

        # 1件ずつの時間を取るため ask を包む
        per_call: list[float] = []
        original = client.ask

        def timed(system, user, schema, _o=original, _p=per_call):
            s = time.perf_counter()
            out = _o(system, user, schema)
            _p.append(time.perf_counter() - s)
            return out

        client.ask = timed                      # type: ignore[method-assign]

        t0 = time.perf_counter()
        m.predict(test)
        wall = time.perf_counter() - t0

        p = percentiles(per_call)
        rows.append({
            "並列数": w,
            "全体_秒": round(wall, 2),
            "1行あたり_秒": round(wall / args.n, 3),
            "1件の中央値_秒": round(p["中央値"], 2),
            "1件のp90_秒": round(p["p90"], 2),
            "1件の最大_秒": round(p["最大"], 2),
            "費用_usd": round(client.usage.cost, 4),
            "エラー": client.usage.errors,
        })
        print(f"  並列 {w:2d}: 全体 {wall:6.1f} 秒 / 1行 {wall / args.n:5.2f} 秒 "
              f"/ 1件の中央値 {p['中央値']:5.2f} 秒 / p90 {p['p90']:5.2f} 秒")

    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))

    base = res.iloc[0]
    print(f"\n読み方:")
    print(f"  - **1件の中央値（{base['1件の中央値_秒']:.2f} 秒）は並列数を上げても"
          "変わらない。** これが API 側の応答時間そのもの。")
    print("  - 全体時間は並列数にほぼ反比例する。**費用は並列数を変えても同じ**なので、"
          "\n    急ぐなら並列数を上げるのが正解（レート制限に当たるまで）。")
    if args.description:
        print(f"  - 自由記述を {args.description} 文字まで載せた条件での計測。")

    slowest = res["1件の最大_秒"].max()
    print(f"  - 最も遅かった1件は {slowest:.1f} 秒。**バッチ処理なら問題ないが、"
          "\n    対話的な用途では信頼度ルーティングで LLM に回す行を絞る必要がある。**")

    out = ROOT / "results" / f"latency_{name}.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n結果: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
