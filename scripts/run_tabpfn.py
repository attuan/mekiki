"""TabPFN — 「LLM を使わない側の到達点」を測る（PRD の P0.5 / R6 / S5）。

TabPFN は表形式データ用の基盤モデル。人工的に生成した大量の表データで
事前学習済みの Transformer で、**その場での学習をしない**。訓練データを
入力として丸ごと渡し、1回の推論で答えを出す（in-context learning）。
LightGBM のように木を育てる工程が無い。

なぜ測るか: これを超えられないと「LLM で精度を上げた」という主張が成立しない。
機能A を実装する前に線を引いておく。

環境の注意:
    このスクリプトだけ **.venv-embed** で動かす（torch が要るため）。
    主環境 .venv には torch が無い。逆に .venv-embed には lightgbm が無いので
    `features.py` は import できない。特徴量づくりをここに自前で持っているのは
    そのため（LightGBM 側と同じ列構成になるよう eval_protocol の定数を共有する）。

    版に注意。使うのは **tabpfn 2.2.1** で、これは PRD が R6 として引用している
    Nature 2025 の版そのものである。ただし 8/29 に 2.2.1 を選んだ理由（当時の
    測定機 Intel Mac の torch が 2.2.2 上限で、8.x の torch>=2.5 を満たせなかった）は
    ノードへの移行で消えた。いまの理由は別で、**7.1.0 以降は重みの取得に
    Prior Labs のライセンス承諾が要る**ためである
    （docs/2026-09-01-embed-env-rebuild.md）。版が変われば数字も変わるので、
    leaderboard の備考には tabpfn.__version__ を自動で書く。

実行:
    TABPFN_DISABLE_TELEMETRY=1 .venv-embed/bin/python scripts/run_tabpfn.py --probe
    TABPFN_DISABLE_TELEMETRY=1 .venv-embed/bin/python scripts/run_tabpfn.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")  # 外部への利用状況送信を止める
# TabPFN は CPU で 1,000行を超える学習を既定で拒否する（遅すぎるため）。
# このマシンには実用的な GPU が無い（Radeon Pro 560X で MPS は動くが CPU より速くならない）ので、
# 明示的に解除して CPU で回す。ignore_pretraining_limits ではなくこの環境変数を使うのは、
# 行数・特徴量数が事前学習の想定内かどうかの検査は残しておきたいため。
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, SEED, SIENTA,
    cross_validate, load_dataset,
)

NUM = LEGACY_NUM + EXTRA_NUM
BOO = LEGACY_BOOL
CAT = LEGACY_CAT + EXTRA_CAT


def build_xy(train: pd.DataFrame, test: pd.DataFrame, target: str):
    """LightGBM の「B 構造化列フル」と同じ列構成を numpy 行列にする。

    TabPFN は数値行列を受け取るので、カテゴリは train に出た水準だけで
    整数に振り直す（test にしか無い水準は NaN。LightGBM の as_category と同じ扱い）。
    どの列がカテゴリかは categorical_features_indices で明示する。
    """
    cols, cat_idx = [], []
    Xtr, Xte = {}, {}

    for c in NUM:
        Xtr[c] = train[c].astype("float64").to_numpy()
        Xte[c] = test[c].astype("float64").to_numpy()
        cols.append(c)
    for c in BOO:
        Xtr[c] = train[c].astype("float64").to_numpy()
        Xte[c] = test[c].astype("float64").to_numpy()
        cols.append(c)
    for c in CAT:
        cats = pd.Index(train[c].dropna().unique())
        Xtr[c] = pd.Categorical(train[c], categories=cats).codes.astype("float64")
        te = pd.Categorical(test[c].where(test[c].isin(cats)), categories=cats)
        Xte[c] = te.codes.astype("float64")
        # pandas の codes は未知/欠損を -1 にする。TabPFN には NaN として渡す
        Xtr[c][Xtr[c] < 0] = np.nan
        Xte[c][Xte[c] < 0] = np.nan
        cat_idx.append(len(cols))
        cols.append(c)

    A = np.column_stack([Xtr[c] for c in cols])
    B = np.column_stack([Xte[c] for c in cols])
    return A, B, train[target].to_numpy(dtype=float), cat_idx


def make_tabpfn(target: str, n_estimators: int = 8):
    """cross_validate に渡す fit_predict を作る。"""
    from tabpfn import TabPFNRegressor

    def fit_predict(train, test):
        A, B, y, cat_idx = build_xy(train, test, target)
        model = TabPFNRegressor(
            device="cpu",              # MPS は実測で速くならないので使わない
            n_estimators=n_estimators,
            categorical_features_indices=cat_idx,
            random_state=SEED,
        )
        model.fit(A, y)
        return model.predict(B)
    return fit_predict


def probe(df: pd.DataFrame, target: str) -> None:
    """本番前に所要時間を測る。全5foldを回す前に見積もりを立てるため。

    TabPFN は訓練データ全体を毎回 Transformer に通すので、行数に対して
    素直に重くなる。5-fold の1 fold は訓練 4,405行 / 予測 1,102行。
    """
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    n_test = 200
    test = df.iloc[idx[:n_test]].reset_index(drop=True)
    pool = df.iloc[idx[n_test:]].reset_index(drop=True)

    print(f"{'訓練行数':>8} {'n_est':>6} {'fit':>7} {'predict':>8} {'MAE':>7}   "
          f"1fold(訓練4,405/予測1,102)の見積もり")
    print("-" * 84)
    for n_train in [500, 1000, 2000, 4405]:
        for n_est in ([8] if n_train < 4405 else [8, 1]):
            tr = pool.iloc[:n_train]
            A, B, y, cat_idx = build_xy(tr, test, target)
            m = TabPFNRegressor(device="cpu", n_estimators=n_est,
                                categorical_features_indices=cat_idx,
                                random_state=SEED)
            t0 = time.time(); m.fit(A, y); t_fit = time.time() - t0
            t0 = time.time(); p = m.predict(B); t_pred = time.time() - t0
            mae = float(np.abs(p - test[target].to_numpy(dtype=float)).mean())
            # 予測は行数に比例するとみて 1,102行ぶんに引き延ばす
            est = t_fit + t_pred * (1102 / n_test)
            print(f"{n_train:>8,} {n_est:>6} {t_fit:>6.1f}s {t_pred:>7.1f}s "
                  f"{mae:>7.2f}   約 {est:5.1f}秒 → 5fold {est*5/60:4.1f}分")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="所要時間だけ測る")
    ap.add_argument("--n-estimators", type=int, default=8,
                    help="内部アンサンブルの本数。減らすと速いが精度は落ちうる")
    ap.add_argument("--no-record", action="store_true",
                    help="leaderboard.csv に書かない")
    args = ap.parse_args()

    df = load_dataset(dataset=SIENTA)
    target = SIENTA.target
    print()

    if args.probe:
        probe(df, target)
        return 0

    import tabpfn  # 版は記録に残す。2.2.1 と 8.5.0 では数字が違う
    name = f"F  TabPFN・構造化列フル(n_est={args.n_estimators})"
    note = ("表形式の基盤モデル。LightGBM の B と同じ列構成。"
            f"tabpfn {tabpfn.__version__} / CPU / n_estimators={args.n_estimators}。"
            "LLMを使わない側の到達点（PRD の R6・S5）")
    print(f"[{name}]")
    t0 = time.time()
    cross_validate(name, make_tabpfn(target, args.n_estimators), df,
                   note=note, dataset=SIENTA, record=not args.no_record)
    print(f"  所要 {time.time()-t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
