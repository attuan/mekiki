"""S1・S2 を今の 6 万行で測り直す（LLM を呼ばない＝無料）。

## なぜ測り直すか

S1 の目標「2,596 未満」も S2 の目標「3,715 未満」も、**8/31 以前の 6 万行**で出した
値である。ところがそのときの `vehicles_multi_clean.parquet` は書き出し順が
実行ごとに変わっており、`sample(60_000)` が引く行が今と違う
（`docs/2026-09-01-adaptive.md`。9/1 に `ORDER BY id` で固定した）。

**当時の parquet は残っていないので、比較線ごと引き直すしかない。**
片方だけ新しい分割で測って「勝った」と言うと、分割の当たり外れを
実力と読み違える（実際 9/1 の測定は全体が 20〜25 USD 下振れしていた）。

## 何を並べるか

    C2  文字TF-IDF                      ← S2 の比較線（未知の model に強い）
    D1  e5 埋め込み
    D4  埋め込み＋文字TF-IDF             ← S1 の比較線（従来の最良）
    F   機能A（埋め込み列・既定エンコーダ）  ← 判定したいもの
    F+  機能A ＋文字TF-IDF               ← 機能A 側でも併用したらどうなるか

既知（train に同じ model があった行）と未知に分けた MAE も出す。S2 は未知の列で見る。

## 実行

    .venv/bin/python scripts/run_s1_recheck.py

LLM を呼ばないので無料。10 分ほどかかる。**リーダーボードには記録しない**
（`leaderboard_vehicles.csv` は旧い分割の値なので、混ぜると比べられなくなる）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import SEED, VEHICLES, cross_validate, load_dataset  # noqa: E402
from features import (  # noqa: E402
    LGBM_PARAMS, as_category, make_lgbm, numeric_frame, tfidf_frames,
)
from run_baselines_vehicles import (  # noqa: E402
    BOOL, CAT, N_SAMPLE, NUM, RULE_COL, TEXT, normalize_model,
)
from run_embedding_vehicles import (  # noqa: E402
    attach_embeddings, emb_extra, unseen_mask,
)

from unfold import Feature  # noqa: E402

TARGET = VEHICLES.target
PARTIAL = ROOT / "results" / "s1_recheck_partial.csv"


def feature_run(with_tfidf: bool = False):
    """機能A の埋め込み列で学習する。with_tfidf なら文字TF-IDF も足す。"""
    def fit_predict(train, test):
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        f = Feature(source=TEXT, type="embedding", name="memb")
        f.fit(train)
        Etr, Ete = f.transform(train), f.transform(test)
        Xtr, Xte = numeric_frame(train, NUM, BOOL), numeric_frame(test, NUM, BOOL)
        Ctr, Cte = as_category(train, test, CAT)
        Xtr = pd.concat([Xtr, Ctr, Etr.set_index(Xtr.index)], axis=1)
        Xte = pd.concat([Xte, Cte, Ete.set_index(Xte.index)], axis=1)
        if with_tfidf:
            Ttr, Tte = tfidf_frames(train, test, TEXT, "char")
            Xtr = pd.concat([Xtr, Ttr.set_index(Xtr.index)], axis=1)
            Xte = pd.concat([Xte, Tte.set_index(Xte.index)], axis=1)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, train[TARGET], categorical_feature=CAT)
        return model.predict(Xte)
    return fit_predict


def main() -> None:
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[TEXT].map(normalize_model)
    df = attach_embeddings(df)
    print("  埋め込み: vehicles_emb_model_e5small.parquet を model で結合（384次元）")

    runs = [
        ("B2 手書きルール", make_lgbm(TARGET, NUM, BOOL, CAT + [RULE_COL])),
        ("C2 文字TF-IDF", make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char")),
        ("D1 e5 埋め込み", make_lgbm(TARGET, NUM, BOOL, CAT, extra=emb_extra())),
        ("D4 埋め込み+文字TF-IDF",
         make_lgbm(TARGET, NUM, BOOL, CAT, TEXT, "char", extra=emb_extra())),
        ("F  機能A（埋め込み列）", feature_run()),
        ("F+ 機能A+文字TF-IDF", feature_run(with_tfidf=True)),
    ]

    oof: dict[str, dict] = {}
    rows = []
    print()
    for name, fn in runs:
        t0 = time.time()
        out: dict = {}
        # **記録しない。** leaderboard は旧い分割の値なので混ぜられない
        m = cross_validate(name, fn, df, record=False, verbose=False,
                           dataset=VEHICLES, oof_out=out)
        oof[name] = out
        rows.append({"手法": name, "MAE": m["MAE"], "MAE_std": m["MAE_std"]})
        print(f"  {name:24} MAE {m['MAE']:9,.2f} ± {m['MAE_std']:6,.2f}"
              f"  ({time.time() - t0:.0f}秒)")
        # **1本ごとに書き出す。** 全部で 20 分近くかかるので、
        # 途中で落ちたときに最初からやり直しにならないようにする
        pd.DataFrame(rows).to_csv(PARTIAL, index=False, encoding="utf-8-sig")

    # 既知／未知に分ける
    y = df[TARGET].to_numpy(dtype=float)
    fold = next(iter(oof.values()))["fold"]
    unseen = unseen_mask(df, fold, TEXT)
    for r in rows:
        err = np.abs(oof[r["手法"]]["pred"] - y)
        r["既知MAE"] = err[~unseen].mean()
        r["未知MAE"] = err[unseen].mean()
    res = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"今の 6 万行・5-fold・seed {SEED}"
          f"（未知の model: {unseen.sum():,} 行 / {len(df):,} = {unseen.mean():.1%}）")
    print("=" * 78)
    print(res.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    f = res.set_index("手法")

    def verdict(diff: float, spread: float) -> str:
        """PRD の「有意に」の定義に従う: 5-fold の振れ幅を超える差があること。

        単に小さいかどうかで判定すると、分割の当たり外れを実力と読み違える。
        """
        if diff >= 0:
            return "未達"
        return "達成" if abs(diff) > spread else f"同着（差が振れ幅 ±{spread:,.0f} 以内）"

    print("\n--- 判定 ---")
    s1_line = f.loc["D4 埋め込み+文字TF-IDF", "MAE"]
    s1_now = f.loc["F  機能A（埋め込み列）", "MAE"]
    s1_plus = f.loc["F+ 機能A+文字TF-IDF", "MAE"]
    print(f"S1（全体 MAE / 従来の最良 D4 を下回るか）")
    spread = max(f.loc["D4 埋め込み+文字TF-IDF", "MAE_std"],
                 f.loc["F  機能A（埋め込み列）", "MAE_std"])
    print(f"   D4 {s1_line:,.2f} 対 機能A {s1_now:,.2f}"
          f"（差 {s1_now - s1_line:+,.2f}）→ "
          f"{verdict(s1_now - s1_line, spread)}")
    print(f"   参考: 機能A+文字TF-IDF {s1_plus:,.2f}"
          f"（差 {s1_plus - s1_line:+,.2f}）→ {verdict(s1_plus - s1_line, spread)}")
    s2_line = f.loc["C2 文字TF-IDF", "未知MAE"]
    s2_now = f.loc["F  機能A（埋め込み列）", "未知MAE"]
    s2_plus = f.loc["F+ 機能A+文字TF-IDF", "未知MAE"]
    print(f"S2（未知の model / 文字TF-IDF 単体を下回るか）")
    print(f"   C2 {s2_line:,.2f} 対 機能A {s2_now:,.2f}"
          f"（差 {s2_now - s2_line:+,.2f}）→ "
          f"{verdict(s2_now - s2_line, spread)}")
    print(f"   参考: 機能A+文字TF-IDF {s2_plus:,.2f}"
          f"（差 {s2_plus - s2_line:+,.2f}）→ {verdict(s2_plus - s2_line, spread)}")

    out_path = ROOT / "results" / "s1_recheck.csv"
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n結果: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
