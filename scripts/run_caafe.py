"""S4 — 機能A と CAAFE を同じ土俵で比べる（R2。記録は `docs/2026-09-01-caafe.md`）。

**問い**: 非構造テキストの扱い方として、
「LLM に前処理コードを書かせる」（CAAFE）と
「コードを書かず埋め込みで持つ」（機能A）のどちらが良いか。

受け入れ基準 S4 は「CAAFE 以下の MAE」。ここで負けたら、機能A の実装方針を
コード生成側に寄せる判断もありうる、と PRD に書いてある。**唯一残っていた
未測定の受け入れ基準**である。

## 揃えたもの

既存の測定（`docs/2026-08-29-feature-vehicles.md`）と同じ土俵に乗せる。

- データ: Craigslist 60,000 行（seed 42 で抽出）
- 分割: 5-fold・seed 42
- 下流モデル: LightGBM（`features.LGBM_PARAMS`）
- 指標: MAE（USD）

並べるのは4つ。

    A2' 構造化列のみ                    ← 何もしない線
    B2' ＋model を手書きルールで正規化    ← 人手ルールの線
    F(c)' ＋機能A（埋め込み列）          ← こちらの手法
    CAAFE ＋LLM が書いた特徴量コード      ← 比べたい相手

## CAAFE 側の手順（fold ごとに独立）

1. その fold の train をさらに 8:2 に割る（内側の train / 検証）
2. LLM に列の様子を見せてコードを提案させる → 実行 → 検証 MAE が下がれば採用
3. これを `--n-iterations` 回くり返す
4. **採用したコードをその fold の train と test に適用**し、外側の MAE を測る

**コード生成には test を一切見せていない。** fold ごとに作り直すので、
「たまたま良いコードが1本できた」ではなく再現性のある比較になる。

## 実行

    .venv/bin/python scripts/run_caafe.py                    # 本番（LLM を呼ぶ）
    .venv/bin/python scripts/run_caafe.py --folds 1 --n-iterations 3   # 小さく試す
    .venv/bin/python scripts/run_caafe.py --skip-caafe       # 比較線だけ（無料）

**LLM が書いたコードをその場で実行する。**危ない語は弾いているが、
任意コード実行であることに変わりはない（`scripts/caafe.py` の冒頭を読むこと）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import caafe  # noqa: E402
from eval_protocol import N_SPLITS, SEED, VEHICLES, load_dataset  # noqa: E402
from features import LGBM_PARAMS, as_category, numeric_frame  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    BOOL, CAT, N_SAMPLE, NUM, RULE_COL, TEXT, normalize_model,
)

from unfold import Feature  # noqa: E402
from unfold.llm import ClaudeClient  # noqa: E402

TARGET = VEHICLES.target

#: LLM に渡すデータセットの説明。CAAFE の肝は「意味情報を使うこと」なので、
#: 列名だけでなく、どういう素性のデータかを書いて渡す
DESCRIPTION = """\
アメリカの Craigslist（個人売買の掲示板）に出稿された中古車の一覧です。
1行が1台で、出品者が入力した項目が並んでいます。目的は車両価格（USD）の予測です。

- `model` は出品者が自由に書いた文字列で、表記がまったく揃っていません
  （例: "f-150 xlt supercrew", "f150 lariat 4x4", "silverado 1500 crew cab lt"）。
  メーカー名・グレード・駆動方式・キャブ形状などが混ざって入っています
- `age` は 2021 年基準の経過年数、`odometer` は走行距離（マイル）
- 選択式の項目（condition, cylinders, fuel, title_status, transmission,
  drive, size, type, paint_color, state）は未入力が多く、欠損が3〜6割ある列もあります
- 価格帯は $1,000〜$100,000、中央値は $12,000 前後です
"""


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def build_xy(train: pd.DataFrame, test: pd.DataFrame, extra_cat: list[str],
             extra_num: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """生成列を含めて学習用の行列を組む。

    生成列は数値にも文字列にもなりうるので、**dtype を見て振り分ける**。
    どちらに入れるかで結果が変わるため、ここは推測せず型で決める。
    """
    num, cat = list(NUM) + extra_num, list(CAT) + extra_cat
    Xtr, Xte = numeric_frame(train, num, BOOL), numeric_frame(test, num, BOOL)
    Ctr, Cte = as_category(train, test, cat)
    return (pd.concat([Xtr, Ctr], axis=1), pd.concat([Xte, Cte], axis=1), cat)


def split_generated(df: pd.DataFrame, before: set[str]) -> tuple[list[str], list[str]]:
    """生成された列を、数値列と カテゴリ列に振り分ける。"""
    new = [c for c in df.columns if c not in before]
    num = [c for c in new if pd.api.types.is_numeric_dtype(df[c])
           or pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in new if c not in num]
    return num, cat


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                extra_cat: list[str], extra_num: list[str]) -> np.ndarray:
    Xtr, Xte, cat = build_xy(train, test, extra_cat, extra_num)
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(Xtr, train[TARGET], categorical_feature=cat)
    return model.predict(Xte)


# ---------------------------------------------------------------------
# 比較線
# ---------------------------------------------------------------------

def run_plain(train, test, mode: str) -> np.ndarray:
    """A2'（構造化列のみ）と B2'（＋手書きルール）。"""
    if mode == "rule":
        train = train.assign(**{RULE_COL: train[TEXT].map(normalize_model)})
        test = test.assign(**{RULE_COL: test[TEXT].map(normalize_model)})
        return fit_predict(train, test, [RULE_COL], [])
    return fit_predict(train, test, [], [])


def run_feature(train, test) -> np.ndarray:
    """F(c)' 機能A の埋め込み列（既定エンコーダ）。"""
    f = Feature(source=TEXT, type="embedding", name="memb")
    f.fit(train)
    Etr, Ete = f.transform(train), f.transform(test)
    Xtr, Xte, cat = build_xy(train, test, [], [])
    Xtr = pd.concat([Xtr, Etr.set_index(Xtr.index)], axis=1)
    Xte = pd.concat([Xte, Ete.set_index(Xte.index)], axis=1)
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(Xtr, train[TARGET], categorical_feature=cat)
    return model.predict(Xte)


# ---------------------------------------------------------------------
# CAAFE
# ---------------------------------------------------------------------

def make_score_fn(inner_train_idx: np.ndarray, inner_val_idx: np.ndarray):
    """検証 MAE を返す関数を作る。CAAFE の採否判定はこれだけを見る。"""
    base_cols = None

    def score(df: pd.DataFrame) -> float:
        nonlocal base_cols
        if base_cols is None:
            base_cols = set(df.columns)
        extra_num, extra_cat = split_generated(df, base_cols)
        tr = df.iloc[inner_train_idx].reset_index(drop=True)
        va = df.iloc[inner_val_idx].reset_index(drop=True)
        pred = fit_predict(tr, va, extra_cat, extra_num)
        return mae(va[TARGET], pred)

    return score


#: CAAFE に見せる列。**他の手法と完全に同じ**にする。
#: 特に `description` は渡さない。平均 2,972 字あってプロンプトが桁で膨らむうえ、
#: 43.7% の行に売り値が書いてあるので、そこから数字を拾うコードを書かれると
#: 「予測」ではなく「答えの読み取り」になる（docs/2026-09-01-description-leak.md）。
USED_COLS = list(NUM) + list(BOOL) + list(CAT) + [TEXT]


def run_caafe_fold(train: pd.DataFrame, test: pd.DataFrame, client: ClaudeClient,
                   n_iterations: int, inner_rows: int, verbose: bool = True,
                   extra_cols: list[str] | None = None,
                   apply_together: bool = False):
    """1 fold ぶんの CAAFE。生成 → 採用コードを train/test に適用 → 予測。"""
    cols = USED_COLS + list(extra_cols or []) + [TARGET]
    train, test = train[cols], test[cols]
    rng = np.random.default_rng(SEED)
    # 生成の反復を現実的な時間に収めるため、内側は間引く。
    # **間引くのは生成のときだけ**で、最終評価は fold の全行で行う
    pick = rng.choice(len(train), size=min(inner_rows, len(train)), replace=False)
    dev = train.iloc[np.sort(pick)].reset_index(drop=True)
    n_val = max(int(len(dev) * 0.2), 1)
    perm = rng.permutation(len(dev))
    score_fn = make_score_fn(perm[n_val:], perm[:n_val])

    result = caafe.generate(dev, TARGET, score_fn, client,
                            description=DESCRIPTION, n_iterations=n_iterations,
                            verbose=verbose)

    if not result.code.strip():
        return run_plain(train, test, "none"), result

    before = set(train.columns)
    if apply_together:
        # **train と test をつないでから1回だけ当てる。**
        # 生成コードには `pd.factorize` や `value_counts` のように
        # 「そのフレームの中身」に依存する処理が混ざる。別々に当てると
        # 同じ model に違う番号が付き、test で意味が変わってしまう。
        # ただしこれは test の行を見て統計を取ることでもある（transductive）。
        both = caafe.run_code(pd.concat([train, test], ignore_index=True),
                              result.code)
        tr2 = both.iloc[:len(train)].reset_index(drop=True)
        te2 = both.iloc[len(train):].reset_index(drop=True)
    else:
        tr2 = caafe.run_code(train, result.code)
        te2 = caafe.run_code(test, result.code)
    extra_num, extra_cat = split_generated(tr2, before)
    # test 側に作れなかった列があれば、そろえて落とす（片方だけの列は使えない）
    keep = [c for c in extra_num + extra_cat if c in te2.columns]
    extra_num = [c for c in extra_num if c in keep]
    extra_cat = [c for c in extra_cat if c in keep]
    return fit_predict(tr2, te2, extra_cat, extra_num), result


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="機能A と CAAFE を同じ土俵で比べる")
    ap.add_argument("--n-iterations", type=int, default=8,
                    help="CAAFE の提案回数（fold ごと）")
    ap.add_argument("--inner-rows", type=int, default=20_000,
                    help="生成の反復に使う行数（間引く。最終評価は全行）")
    ap.add_argument("--folds", type=int, default=N_SPLITS,
                    help="何 fold まで回すか（試すときは 1）")
    ap.add_argument("--apply-together", action="store_true",
                    help="採用コードを train と test をつないだ状態で当てる。"
                         "pd.factorize などの状態を持つ処理が壊れるのを防ぐが、"
                         "test の行を見て統計を取ることにもなる")
    ap.add_argument("--with-description", action="store_true",
                    help="自由記述（description）も CAAFE に見せる。**土俵が変わる**うえ "
                         "43.7%% の行に売り値が書いてあるのでリークしうる")
    ap.add_argument("--skip-caafe", action="store_true",
                    help="比較線だけ測る（LLM を呼ばない＝無料）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-cost", type=float, default=5.0)
    args = ap.parse_args()

    client = ClaudeClient(model=args.model, effort=args.effort, max_tokens=4000)
    if not args.skip_caafe and not client.available():
        raise SystemExit("ANTHROPIC_API_KEY がありません（--skip-caafe なら無料で動きます）")

    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows, steps_rows = [], []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        if fold > args.folds:
            break
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)
        truth = test[TARGET].to_numpy(dtype=float)
        print(f"\n[fold{fold}] 訓練 {len(train):,} 行 → 評価 {len(test):,} 行")

        rec = {"fold": fold}
        for name, fn in [("A2' 構造化列のみ", lambda: run_plain(train, test, "none")),
                         ("B2' ＋手書きルール", lambda: run_plain(train, test, "rule")),
                         ("F(c)' ＋機能A（埋め込み列）", lambda: run_feature(train, test))]:
            t0 = time.time()
            rec[f"MAE_{name}"] = mae(truth, fn())
            print(f"    {name:28} MAE {rec[f'MAE_{name}']:9,.2f}"
                  f"  ({time.time() - t0:.0f}秒)")

        if not args.skip_caafe:
            t0 = time.time()
            pred, result = run_caafe_fold(
                train, test, client, args.n_iterations, args.inner_rows,
                extra_cols=["description"] if args.with_description else None,
                apply_together=args.apply_together)
            rec["MAE_CAAFE"] = mae(truth, pred)
            rec["CAAFE_採用数"] = result.n_accepted
            rec["CAAFE_検証MAE_前"] = result.base_score
            rec["CAAFE_検証MAE_後"] = result.best_score
            print(f"    {'CAAFE ＋生成した特徴量':28} MAE {rec['MAE_CAAFE']:9,.2f}"
                  f"  ({time.time() - t0:.0f}秒 / 採用 {result.n_accepted} 件)")
            for s in result.steps:
                steps_rows.append({"fold": fold, "回": s.iteration, "名前": s.name,
                                   "採用": s.accepted, "検証MAE": s.score,
                                   "差": s.note, "理由": s.reason, "コード": s.code})
            if client.usage.cost > args.max_cost:
                print(f"\n費用が上限 ${args.max_cost} を超えました。ここで止めます。")
                rows.append(rec)
                break
        rows.append(rec)

    res = pd.DataFrame(rows)
    cols = [c for c in res.columns if c.startswith("MAE_")]
    print("\n" + "=" * 78)
    print(f"{len(res)}-fold 平均（60,000 行・seed {SEED}）")
    print("=" * 78)
    summary = res[cols].mean().sort_values()
    for name, v in summary.items():
        sd = res[name].std()
        mark = ""
        if name == "MAE_CAAFE":
            mark = " ← 比べたい相手"
        elif "機能A" in name:
            mark = " ★機能A"
        print(f"  {name[4:]:30} MAE {v:9,.2f} ± {sd:6,.2f}{mark}")

    if "MAE_CAAFE" in summary.index:
        f = summary[[c for c in summary.index if "機能A" in c][0]]
        c = summary["MAE_CAAFE"]
        verdict = "機能A の勝ち（S4 を満たす）" if f <= c else "**CAAFE の勝ち（S4 未達）**"
        print(f"\n  → 機能A {f:,.2f} 対 CAAFE {c:,.2f}（差 {c - f:+,.2f}）。{verdict}")
        print(f"  ※ 5-fold の振れ幅を超える差かどうかで判断すること")

    suffix = "_together" if args.apply_together else ""
    out = ROOT / "results" / f"caafe{suffix}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nfold ごとの結果: {out.relative_to(ROOT)}")
    if steps_rows:
        sp = ROOT / "results" / f"caafe_steps{suffix}.csv"
        pd.DataFrame(steps_rows).to_csv(sp, index=False, encoding="utf-8-sig")
        print(f"提案の記録:     {sp.relative_to(ROOT)}")
        print("\n--- 実測費用 ---")
        for k, v in client.summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
