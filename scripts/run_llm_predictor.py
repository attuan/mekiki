"""P3 — 機能B（LLMPredictor）を実測する。

問いは1つだけ。**証拠として渡した統計モデル単体より、LLM の最終判断は良いか。**
良くなければ機能B は価値を出していない（PRD「機能B」/ 受け入れ基準 S6）。

## なぜ普通の 5-fold CV をそのまま使わないか

`eval_protocol.cross_validate` は全 5,507 行を予測する。機能B は1行1回
LLM を呼ぶので、それだと 5,507 リクエストかかる。そこで

- 学習（統計モデルの fit・近傍索引の構築）は **各 fold の train 全体**で行い、
- 採点は **test からランダム抽出した n 行**だけで行う

という形にした。統計モデルと LLM をまったく同じ行で採点するので、
両者の比較は成り立つ。ただし行数が違うので、リーダーボードの
全行 12.21 とは直接比べられない。**同じ抽出行での統計モデルの MAE も
必ず一緒に出す**のはそのため。

## 実行

    .venv/bin/python scripts/run_llm_predictor.py --demo          # 3行だけ見る
    .venv/bin/python scripts/run_llm_predictor.py --n-eval 60     # 5foldで300行
    .venv/bin/python scripts/run_llm_predictor.py --n-eval 60 --dry-run  # 費用見積り

`--dry-run` は LLM を呼ばずにプロンプトだけ組み、トークン数を数えて
費用を見積もる。**本番前に必ず一度これを通すこと。**
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

from dataclasses import replace  # noqa: E402

from eval_protocol import (  # noqa: E402
    EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, N_SPLITS, SEED,
    SIENTA, VEHICLES, VEHICLES_BOOL, VEHICLES_CAT, VEHICLES_LONG_TEXT,
    VEHICLES_NUM, VEHICLES_TEXT, Dataset, load_dataset,
)
from features import make_lgbm  # noqa: E402
from unfold import ColumnSpec, LLMPredictor, TreeModel  # noqa: E402
from unfold.llm import PRICING, ClaudeClient  # noqa: E402

# 証拠を作る統計モデルに渡す列。run_baselines*.py の「構造化列フル」と同じ。
# **テキストは統計モデルには入れず、LLM 側にだけ渡す。**
# こうすると「テキストを LLM が読めた効果」だけを取り出せる。
SPECS: dict[str, ColumnSpec] = {
    "sienta": ColumnSpec(
        numeric=LEGACY_NUM + EXTRA_NUM,
        boolean=LEGACY_BOOL,
        categorical=LEGACY_CAT + EXTRA_CAT,
        text="装備テキスト",
    ),
    # Craigslist 側は eval_protocol.VEHICLES_* に揃える（他の手法と同じ列）。
    # description（平均2,972文字）は入れない。5事例ぶん貼るとプロンプトが
    # 桁で膨らみ、費用も比較可能性も壊れるため（PRD「信頼度ルーティング」の指摘）。
    "vehicles": ColumnSpec(
        numeric=VEHICLES_NUM,
        boolean=VEHICLES_BOOL,
        categorical=VEHICLES_CAT,
        text=VEHICLES_TEXT,
    ),
}

DATASETS: dict[str, Dataset] = {"sienta": SIENTA, "vehicles": VEHICLES}

# Craigslist を全 200,374 行で回すと近傍索引の構築が重い。
# **既存のリーダーボードと同じ 60,000 行**に揃えて比較可能にする
# （CLAUDE.md「S1・S2 の線は行数とセットで固定する」）。
SAMPLES: dict[str, int | None] = {"sienta": None, "vehicles": 60_000}

# 自由記述の本文。--description で使うときだけ ColumnSpec に足す。
# シエンタ側には自由記述にあたる列が無い（タイトルしか取れていない）。
LONG_TEXT: dict[str, str | None] = {"sienta": None, "vehicles": VEHICLES_LONG_TEXT}


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def build(client: ClaudeClient, n_examples: int, name: str,
          evidence: str = "all", description: int = 0,
          description_examples: int = 0) -> LLMPredictor:
    """機能B を組む。`evidence` で証拠1 に何を並べるかを変える。

    "all"    … LightGBM・XGBoost・近傍中央値（既定。設計書どおり全部渡す）
    "trees"  … 木モデルだけ。近傍は証拠2（類似事例）としてだけ渡す

    "trees" を用意したのは、近傍中央値が単体で MAE 30.86 と弱く、
    LLM の修正方向がそれと相関 +0.62 で引きずられていたため
    （`analyze_llm_predictor.py`）。**弱い証拠を渡さないほうが良いのか**を
    切り分ける。近傍そのものは類似事例として渡り続けるので、
    フューショットの仕組みは変わらない。
    """
    ds, spec = DATASETS[name], SPECS[name]
    if description:
        # 自由記述を使う版。元の SPECS は書き換えず複製する
        # （同じプロセスで両方の構成を測ることがあるため）。
        col = LONG_TEXT.get(name)
        if col is None:
            raise SystemExit(f"{name} には自由記述の列がありません")
        spec = replace(spec, long_text=col, long_text_chars=description,
                       long_text_example_chars=description_examples)
    models = None
    if evidence == "trees":
        models = [TreeModel("LightGBM", spec, kind="lgbm"),
                  TreeModel("XGBoost", spec, kind="xgb")]
    elif evidence != "all":
        raise SystemExit(f"--evidence は all か trees です: {evidence}")
    return LLMPredictor(target=ds.target, unit=ds.unit, spec=spec,
                        n_examples=n_examples, client=client, models=models)


# ---------------------------------------------------------------------
# デモ — 1例つくって挙動を見る
# ---------------------------------------------------------------------

def run_demo(client: ClaudeClient, n_rows: int, n_examples: int, name: str,
             evidence: str = "all", description: int = 0,
             description_examples: int = 0) -> None:
    ds, spec = DATASETS[name], SPECS[name]
    df = load_dataset(dataset=ds, sample=SAMPLES[name])
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    train = df.iloc[idx[n_rows:]].reset_index(drop=True)
    test = df.iloc[idx[:n_rows]].reset_index(drop=True)

    print(f"\n訓練 {len(train):,} 行で学習し、{len(test)} 行を予測します。")
    model = build(client, n_examples, name, evidence, description,
                  description_examples).fit(train)
    pred = model.predict(test, verbose=True)
    truth = test[ds.target].to_numpy(dtype=float)

    print("\n" + "=" * 78)
    for i in range(len(test)):
        print(model.explain(i))
        print(f"  → 実際の価格: {truth[i]:,.1f} {ds.unit}"
              f"（誤差 {abs(pred[i] - truth[i]):.1f}）")
        print("-" * 78)

    print(f"\nこの {len(test)} 行の MAE: {mae(truth, pred):,.2f} {ds.unit}")
    print("\n費用:")
    for k, v in model.cost().items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------
# 本測定 — 5-fold、各 fold の test から n 行だけ採点
# ---------------------------------------------------------------------

def run_eval(client: ClaudeClient, n_eval: int, n_examples: int,
             dry_run: bool, name: str, evidence: str = "all",
             description: int = 0, description_examples: int = 0) -> None:
    ds, spec = DATASETS[name], SPECS[name]
    df = load_dataset(dataset=ds, sample=SAMPLES[name])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    per_row: list[pd.DataFrame] = []
    prompt_chars = 0

    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test_all = df.iloc[te_idx].reset_index(drop=True)
        pick = rng.choice(len(test_all), size=min(n_eval, len(test_all)),
                          replace=False)
        test = test_all.iloc[np.sort(pick)].reset_index(drop=True)
        truth = test[ds.target].to_numpy(dtype=float)

        print(f"\n[fold{fold}] 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
        model = build(client, n_examples, name, evidence, description,
                      description_examples).fit(train)

        # 統計モデル単体の予測（＝機能B が超えるべき線）
        base = {m.name: np.asarray(m.predict(test), dtype=float)
                for m in model.models_}


        if dry_run:
            # LLM を呼ばずにプロンプトだけ組んで、長さを測る
            idx, sim = model.index_.query(test)
            for i in range(len(test)):
                nbs = [{"訓練行": int(j), "類似度": float(s),
                        "価格": float(model.index_.y_[j]),
                        "属性": model.train_.iloc[j][
                            model.spec.all_columns()].to_dict()}
                       for j, s in zip(idx[i], sim[i])]
                ev = {k: float(v[i]) for k, v in base.items()}
                prompt_chars += len(model._build_prompt(test.iloc[i], ev, nbs))
            for mname, p in base.items():
                print(f"    {mname:24} MAE {mae(truth, p):9,.2f}")
            rows.append({"fold": fold, "n": len(test), **{
                f"MAE_{k}": mae(truth, v) for k, v in base.items()}})
            continue


        # **公平性のための比較線。** 証拠にした木モデルにはテキストを渡して
        # いないので、機能B が勝っても「LLM が賢い」のか「テキストが効く」のか
        # 区別できない。そこで**テキストを文字TF-IDF で入れた LightGBM**を
        # 並べる（リーダーボードの最良構成にあたる）。これは証拠には入れず、
        # 採点だけする。
        ref = make_lgbm(ds.target, spec.numeric, spec.boolean,
                        spec.categorical, spec.text, "char")
        # 参考線は spec.text（短いテキスト）だけを使う。自由記述は
        # 数値化していないので木には渡せず、そこが機能B の土俵になる。
        base["参考: LightGBM+文字TF-IDF"] = np.asarray(
            ref(train, test), dtype=float)

        pred = model.predict(test, verbose=True)

        # 行ごとの記録。あとで「confidence が誤差を見分けられるか」（P4 の
        # ルーティング曲線）を調べるのに使う。キャッシュが効くので再実行は無料。
        per_row.append(pd.DataFrame({
            "fold": fold, "実際": truth, "機能B": pred,
            "confidence": [p.confidence for p in model.predictions_],
            "由来": [p.origin for p in model.predictions_],
            "理由": [p.reason for p in model.predictions_],
            **{mname: v for mname, v in base.items()},
        }))

        rec = {"fold": fold, "n": len(test), "MAE_機能B": mae(truth, pred)}
        for mname, p in base.items():
            rec[f"MAE_{mname}"] = mae(truth, p)
        rec["LLMが答えた割合"] = float(np.mean(
            [p.origin == "llm" for p in model.predictions_]))
        rec["平均confidence"] = float(np.mean(
            [p.confidence for p in model.predictions_]))
        rows.append(rec)
        for k, v in rec.items():
            if k.startswith("MAE_"):
                print(f"    {k[4:]:24} {v:9,.2f}")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"5-fold 平均（各 fold {n_eval} 行 = 合計 {int(res['n'].sum())} 行で採点）")
    print("=" * 78)
    summary = res[[c for c in res.columns if c.startswith("MAE_")]].mean()
    for mname, v in summary.sort_values().items():
        mark = " ★機能B" if mname == "MAE_機能B" else ""
        print(f"  {mname[4:]:26} MAE {v:9,.2f} {ds.unit}{mark}")

    if not dry_run and "MAE_機能B" in summary:
        best_base = summary.drop("MAE_機能B").min()
        best_name = summary.drop("MAE_機能B").idxmin()[4:]
        diff = best_base - summary["MAE_機能B"]
        verdict = "上回った" if diff > 0 else "届かなかった"
        print(f"\n  → 比較線の最良は {best_name}（{best_base:,.2f}）。"
              f"機能B は {abs(diff):,.2f} {ds.unit} {verdict}。")
        print("     ※「参考:」が付いた行は証拠には渡していない比較専用のモデル。"
              "\n        テキストを木に入れた版で、これに勝てるかが本当の関門。")
        print(f"  → LLM が答えた割合 {res['LLMが答えた割合'].mean():.1%} / "
              f"平均 confidence {res['平均confidence'].mean():.2f}")

    suffix = ("" if name == "sienta" else f"_{name}")
    suffix += "" if evidence == "all" else f"_{evidence}"
    suffix += f"_desc{description}" if description else ""
    out = ROOT / "results" / f"llm_predictor{suffix}.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nfold ごとの結果: {out.relative_to(ROOT)}")
    if per_row:
        rout = ROOT / "results" / f"llm_predictor{suffix}_rows.csv"
        pd.concat(per_row, ignore_index=True).to_csv(
            rout, index=False, encoding="utf-8-sig")
        print(f"行ごとの結果: {rout.relative_to(ROOT)}")

    if dry_run:
        n_rows_total = int(res["n"].sum())
        # 1トークンあたりの文字数はデータの言語で違う（日本語 1.6 / 英語 3.5）。
        # system は毎回 +約400トークン
        chars_per_token = 1.6 if name == "sienta" else 3.5
        tok = (prompt_chars / chars_per_token) + n_rows_total * 400
        pin, pout = PRICING.get(client.model, PRICING["claude-opus-5"])
        est = (tok * pin + n_rows_total * 150 * pout) / 1_000_000
        print(f"\n--- 費用の見積もり（{client.model}）---")
        print(f"  行数 {n_rows_total:,} / プロンプト平均 "
              f"{prompt_chars / n_rows_total:,.0f} 文字")
        print(f"  推定入力トークン {tok:,.0f} → 概算 ${est:.2f}")
        print(f"  全 {len(df):,} 行に広げた場合の概算: "
              f"${est * len(df) / n_rows_total:.2f}")
        print("  ※ 実測は初回実行の cost() を見ること。ここは桁を確かめるだけ。")
    else:
        print("\n--- 実測費用 ---")
        for k, v in client.summary().items():
            print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="機能B（LLMPredictor）の実測")
    ap.add_argument("--demo", action="store_true",
                    help="数行だけ予測して explain() を表示する")
    ap.add_argument("--demo-rows", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=60,
                    help="各 fold の test から採点する行数")
    ap.add_argument("--n-examples", type=int, default=5,
                    help="LLM に見せる類似事例の件数（few-shot の shot 数）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dataset", default="sienta",
                    choices=["sienta", "vehicles"],
                    help="sienta=単一車種・日本語 / vehicles=複数車種・英語")
    ap.add_argument("--evidence", default="all", choices=["all", "trees"],
                    help="証拠1 に並べるモデル。trees は近傍中央値を外す")
    ap.add_argument("--description", type=int, default=0, metavar="文字数",
                    help="自由記述を査定対象に載せる上限。0 なら使わない")
    ap.add_argument("--description-examples", type=int, default=0,
                    metavar="文字数",
                    help="自由記述を類似事例にも載せる上限。0 なら載せない")
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばずにプロンプトだけ組んで費用を見積もる")
    args = ap.parse_args()

    client = ClaudeClient(model=args.model, effort=args.effort,
                          max_workers=args.workers)
    if not args.dry_run and not client.available():
        print("ANTHROPIC_API_KEY がありません。.env にキーを書いてください。")
        print("（--dry-run なら LLM を呼ばずに費用の見積もりだけできます）")
        raise SystemExit(1)

    print(f"データ {args.dataset} / モデル {args.model} / "
          f"effort {args.effort} / 類似事例 {args.n_examples} 件 / "
          f"証拠 {args.evidence}"
          + ("  [DRY RUN: LLM を呼びません]" if args.dry_run else ""))

    if args.demo:
        run_demo(client, args.demo_rows, args.n_examples, args.dataset,
                 args.evidence, args.description, args.description_examples)
    else:
        run_eval(client, args.n_eval, args.n_examples, args.dry_run,
                 args.dataset, args.evidence, args.description,
                 args.description_examples)


if __name__ == "__main__":
    main()
