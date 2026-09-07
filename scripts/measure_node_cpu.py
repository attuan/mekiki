"""LLM バッチが計算ノードの自動停止（CPU 5% 未満が1時間）に該当するかを測る。

## なぜ測るのか

`docs/2026-09-01-migration.md` は「LLM のバッチは 16 vCPU に対して 0.2 コア分も
使わないので自動停止に殺される」と書いているが、**この 0.2 コアは推定値で、
実測ではなかった。** 伊藤さんにしきい値引き上げを依頼するなら実測値を添えたい。

## 測り方

CloudWatch の `CPUUtilization` は**インスタンス全体**（16 vCPU を 100%）の値なので、
`/proc/stat` から同じ定義で計算する。あわせて自分のプロセスの CPU 時間
（`/proc/self/stat` の utime + stime、全スレッド分）も取り、
**「ジョブ単独では何コア使うか」**を他の負荷と切り離して出す。

閾値 5% は 16 vCPU 換算で **0.8 コア**にあたる。ここが判定の基準線。

3つの区間に分けて測る。

1. **待機** — 何もしない区間。ノードの地の負荷（VS Code サーバ等）を知る
2. **準備** — データ読み込みと統計モデルの fit。ここは CPU を使う
3. **LLM** — 実際に API を叩く区間。キャッシュを切るので**課金される**

## 実行

    .venv/bin/python scripts/measure_node_cpu.py --dry-run      # 課金なし
    .venv/bin/python scripts/measure_node_cpu.py --n 60         # 約 $0.5
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import SEED, load_dataset  # noqa: E402
from run_llm_predictor import DATASETS, SAMPLES, build  # noqa: E402

from unfold.llm import ClaudeClient  # noqa: E402

N_CPU = os.cpu_count() or 1
CLOCK = os.sysconf("SC_CLK_TCK")
THRESHOLD_PCT = 5.0                      # 自動停止アラームのしきい値


def _stat_totals() -> tuple[float, float]:
    """システム全体の (使用時間, 総時間) を秒で返す。"""
    with open("/proc/stat") as f:
        parts = [float(x) for x in f.readline().split()[1:]]
    idle = parts[3] + parts[4]           # idle + iowait
    return (sum(parts) - idle) / CLOCK, sum(parts) / CLOCK


def _self_cpu() -> float:
    """自プロセス（全スレッド）の CPU 時間を秒で返す。"""
    with open("/proc/self/stat") as f:
        fields = f.read().rsplit(") ", 1)[1].split()
    return (float(fields[11]) + float(fields[12])) / CLOCK


class Sampler(threading.Thread):
    """1秒ごとにシステム全体と自プロセスの CPU を記録する。"""

    def __init__(self, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.samples: list[tuple[float, float, float]] = []   # 時刻, 全体%, 自分のコア数

    def run(self) -> None:
        prev_busy, prev_total = _stat_totals()
        prev_self = _self_cpu()
        prev_t = time.perf_counter()
        while not self.stop_flag.wait(self.interval):
            busy, total = _stat_totals()
            mine = _self_cpu()
            now = time.perf_counter()
            d_total = total - prev_total
            sys_pct = 100.0 * (busy - prev_busy) / d_total if d_total > 0 else 0.0
            my_cores = (mine - prev_self) / (now - prev_t)
            self.samples.append((now, sys_pct, my_cores))
            prev_busy, prev_total, prev_self, prev_t = busy, total, mine, now

    def slice(self, t0: float, t1: float) -> list[tuple[float, float, float]]:
        return [s for s in self.samples if t0 <= s[0] <= t1]


def summarize(name: str, rows: list[tuple[float, float, float]],
              wall: float) -> dict[str, float]:
    if not rows:
        return {"区間": name, "秒": wall, "標本": 0}
    sys_pct = np.array([r[1] for r in rows])
    cores = np.array([r[2] for r in rows])
    return {"区間": name, "秒": wall, "標本": len(rows),
            "全体CPU%平均": float(sys_pct.mean()),
            "全体CPU%最大": float(sys_pct.max()),
            "自分コア平均": float(cores.mean()),
            "自分コア最大": float(cores.max()),
            "自分CPU%平均": float(cores.mean() / N_CPU * 100),
            "自分CPU%最大": float(cores.max() / N_CPU * 100)}


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM バッチの CPU 使用率を測る")
    ap.add_argument("--dataset", default="vehicles", choices=["sienta", "vehicles"])
    ap.add_argument("--n", type=int, default=60, help="LLM に投げる行数")
    ap.add_argument("--workers", type=int, default=8, help="並列数")
    ap.add_argument("--idle-sec", type=float, default=30.0, help="待機区間の秒数")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばない（課金なし。待機と準備だけ測る）")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    print(f"CPU {N_CPU} 論理コア / しきい値 {THRESHOLD_PCT}% "
          f"= {N_CPU * THRESHOLD_PCT / 100:.1f} コア相当")
    print(f"データ {args.dataset} / {args.n} 行 / 並列 {args.workers}")
    if not args.dry_run:
        print(f"**キャッシュを切るので {args.n} 回課金されます。**")

    sampler = Sampler()
    sampler.start()
    results = []

    # --- 1. 待機（地の負荷）---------------------------------------------
    t0 = time.perf_counter()
    time.sleep(args.idle_sec)
    t1 = time.perf_counter()
    results.append(summarize("待機（何もしない）", sampler.slice(t0, t1), t1 - t0))

    # --- 2. 準備（読み込み + fit）----------------------------------------
    t0 = time.perf_counter()
    df = load_dataset(verbose=False, dataset=ds, sample=SAMPLES[args.dataset])
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    test = df.iloc[idx[:args.n]].reset_index(drop=True)
    train = df.iloc[idx[args.n:]].reset_index(drop=True)
    probe = ClaudeClient(api_key="")
    build(probe, 5, args.dataset, "all", 0).fit(train)
    t1 = time.perf_counter()
    results.append(summarize("準備（読み込み+fit）", sampler.slice(t0, t1), t1 - t0))

    # --- 3. LLM 呼び出し -------------------------------------------------
    if not args.dry_run:
        if not ClaudeClient().available():
            print("\nANTHROPIC_API_KEY がありません。")
            raise SystemExit(1)
        client = ClaudeClient(cache_dir=None, max_workers=args.workers)
        model = build(client, 5, args.dataset, "all", 0).fit(train)
        t0 = time.perf_counter()
        model.predict(test)
        t1 = time.perf_counter()
        results.append(summarize(f"LLM（{args.n}行・並列{args.workers}）",
                                 sampler.slice(t0, t1), t1 - t0))
        usage = client.summary()
    else:
        usage = None

    sampler.stop_flag.set()

    # --- 報告 -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'区間':<24}{'秒':>7}{'全体CPU%':>10}{'他負荷%':>9}"
          f"{'自分コア':>9}{'自分CPU%':>10}{'(最大)':>8}")
    print("=" * 78)
    for r in results:
        if r.get("標本", 0) == 0:
            continue
        other = r['全体CPU%平均'] - r['自分CPU%平均']
        print(f"{r['区間']:<24}{r['秒']:>7.1f}{r['全体CPU%平均']:>10.2f}"
              f"{other:>9.2f}{r['自分コア平均']:>9.3f}"
              f"{r['自分CPU%平均']:>10.2f}{r['自分CPU%最大']:>8.2f}")

    llm = next((r for r in results if r["区間"].startswith("LLM")), None)
    if llm:
        print("\n" + "-" * 78)
        print(f"LLM 区間のジョブ単独: {llm['自分コア平均']:.3f} コア "
              f"= 全体の {llm['自分CPU%平均']:.2f}%")
        print(f"自動停止のしきい値  : {THRESHOLD_PCT}% "
              f"= {N_CPU * THRESHOLD_PCT / 100:.1f} コア")
        verdict = "下回る → 自動停止に該当する" if llm["自分CPU%平均"] < THRESHOLD_PCT \
            else "上回る → 自動停止には該当しない"
        print(f"判定                : {verdict}")
        if usage:
            print(f"費用                : ${usage.get('費用_usd', 0):.4f}")
    print("-" * 78)


if __name__ == "__main__":
    main()
