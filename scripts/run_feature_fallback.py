"""機能A の LLM フォールバック（設計書 05）を実測する。

## 何を確かめたいか

機能A は「埋め込み → 近傍で分類 → confidence が閾値未満なら LLM へ」という
造りになっている（設計書 01〜05）。04 までは APIキー無しで実装・測定済みで、
05 だけが `QueueOnlyFallback`（答えずにレビュー待ちへ積むだけ）だった。

そこにキーが来たので、実物の `ClaudeFallback` を差して問う:

> **近傍分類が自信を持てなかった行を LLM に回すと、実際に正しくなるのか。
> 1行いくらか。**

これが成り立たないなら、信頼度ルーティング（設計書の AdaptivePredictor、
PRD「信頼度ルーティング」）の前提そのものが崩れる。

## 3つの課題

    sienta        … タイトル → グレード名。宣言値10個。正解は正規表現版のグレード名
    vehicles      … model → 車種の芯。宣言値60個。正解は手書きルールの正規化結果
    vehicles-type … model → ボディ形状。宣言値13個。正解はデータが持つ type 列

**vehicles は測定として成立しなかった**（`docs/2026-09-02-feature-fallback-vehicles.md`）。
正解を `normalize_model()`（先頭2語）に取ったため正解が入力の部分文字列になり、
LLM を呼ばない 0% の時点で宣言値内 accuracy が 1.000 に達してしまった。
再現のために残してあるが、新しく測るなら vehicles-type を使うこと。

**vehicles-type はそのやり直し。** 正解を「テキストから機械的に決まらない列」に
替えてある。`type` は出品者が選んだボディ形状で、`f-150 xlt supercrew` → `pickup`
のように世界知識が要る。宣言値13個で正解を100%覆うので、
「宣言値の外にある裾は当てようがない」という前回の頭打ちも起きない。

## 測る前に必ず見る3つ（前回の反省を手続きにしたもの）

`--dry-run` は LLM を呼ばずに次を出す。**ここを通らない課題に課金しない。**

1. 宣言値のカバー率 — 低いとその分が上限になる（前回はこれが 37.0% だった）
2. 0%（LLM 未使用）の accuracy と、この課題の上限・下限 — 余地が無いなら測る意味がない
   - 上限は「同じ入力テキストの多数決」。`type` は出品者入力なので `f-150` が
     truck と pickup に割れており、テキストだけを見る手法はこれを超えられない
   - 下限は「最頻クラスを常に答える」
3. 確信度の下位の accuracy が、残りより実際に低いか — ここが同じなら
   確信度は「LLM に回すべき行」を選べておらず、回しても直らない

## 測り方

エスカレーション率を 0% → 30% と上げながら、生成した列が正解と
どれだけ一致するか（中間ラベル accuracy）を見る。
`escalate_rate` は「confidence の低い順に何割を回すか」なので、
5% に回る行は 15% に回る行の部分集合になる。**プロンプトのキャッシュが
効くので、率を上げても追加ぶんしか課金されない。**

費用を抑えるため 1 fold・test を抽出して測る。
**下流の MAE はここでは測らない。** 下流まで見るには訓練側の行も
同じ率でエスカレーションする必要があり、桁が変わるため。

## 測る軸（vehicles-type では2×2）

教師ラベルの起点（PRD 機能A の 01）とエンコーダを掛け合わせて測れる。

    --label-source labelname … 値の名前だけを参照事例にする（案b・ゼロショット）
    --label-source human     … 訓練データに人手ラベルを n 件だけ与える（案a）
    --encoder default        … 既定の文字TF-IDF（追加依存なし）
    --encoder e5             … 計算済みの e5-small 埋め込み（意味的に近い車種を拾う）

## 実行

    .venv/bin/python scripts/run_feature_fallback.py
    .venv/bin/python scripts/run_feature_fallback.py --dataset vehicles-type \\
        --label-source both --encoder both --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_feature import COL, DECLARED_VALUES, GENERATED  # noqa: E402
from demo_feature_vehicles import (  # noqa: E402
    GENERATED as V_GENERATED, declared_values,
)
from eval_protocol import N_SPLITS, SEED, VEHICLES, load_dataset  # noqa: E402
from run_baselines_vehicles import (  # noqa: E402
    N_SAMPLE, RULE_COL, TEXT as V_TEXT, normalize_model,
)
from type_values import TYPE_VALUES  # noqa: E402

from unfold import Feature, PrecomputedEncoder, QueueOnlyFallback  # noqa: E402
from unfold.fallback import ClaudeFallback  # noqa: E402
from unfold.llm import ClaudeClient  # noqa: E402

RATES = (0.0, 0.05, 0.15, 0.30)

# 計算済み埋め込み。model の19,739種類ぶん（scripts/embed_vehicles.py が作る）と、
# 案b で参照事例になる値の名前ぶん（scripts/embed_type_values.py が作る）。
# PrecomputedEncoder は未知の文字列を扱えないので、案b では後者が要る。
EMB_MODEL = ROOT / "sampledata" / "processed" / "vehicles_emb_model_e5small.parquet"
EMB_VALUES = ROOT / "sampledata" / "processed" / "vehicles_emb_typevalues_e5small.parquet"

# 宣言値（`type` の13水準）は scripts/type_values.py に置いてある。
# 埋め込みを作る側（.venv-embed）と綴りを共有する必要があるため。
TYPE_COL = "type"
TYPE_GENERATED = "body_type_generated"
N_SEED_LABELS = 200      # 案a で与える人手ラベルの件数（仕様書が前提にしている件数）


# --- 測定対象 ---------------------------------------------------------


@dataclass(frozen=True)
class FallbackTask:
    """1つの課題について、この測定が知っている必要のあること。

    以降の処理はここに書かれた列名しか見ない。データセット固有の知識
    （元データの列名・宣言する値の作り方）は `build` の中に閉じ込める。
    """

    name: str
    build: Callable[[], tuple[pd.DataFrame, list[str]]]  # → (データ, 宣言値)
    source: str          # 分類の入力にする非構造列
    truth: str           # 正解とみなす列
    generated: str       # 機能A が作る列の名前
    out: Path            # 結果の書き出し先
    note: str            # 画面に出す1行説明
    emb: Path | None = None    # --encoder e5 で使う計算済み埋め込み
    emb_values: Path | None = None   # 案b の参照事例（値の名前）ぶんの埋め込み


@dataclass(frozen=True)
class Variant:
    """測定の1条件。ラベルの起点 × エンコーダ。"""

    label_source: str        # "labelname"（案b）/ "human"（案a）
    encoder: str             # "default" / "e5"
    n_labels: int | None = None

    @property
    def label_name(self) -> str:
        return ("値の名前だけ（案b）" if self.label_source == "labelname"
                else f"人手ラベル{self.n_labels}件（案a）")

    @property
    def encoder_name(self) -> str:
        return "文字TF-IDF（既定）" if self.encoder == "default" else "e5-small"

    def __str__(self) -> str:
        return f"{self.label_name} × {self.encoder_name}"


def _build_sienta() -> tuple[pd.DataFrame, list[str]]:
    """シエンタ 5,507行。宣言値はカタログを見れば書ける10グレード。"""
    return load_dataset(), list(DECLARED_VALUES)


def _build_vehicles() -> tuple[pd.DataFrame, list[str]]:
    """Craigslist 60,000行の抽出。宣言値は「よく出る model を60個」。

    正解は手書きルール（区切り記号以降を捨てて先頭2語）の正規化結果。
    **この作りでは測定が成立しない**ことが分かっている
    （`docs/2026-09-02-feature-fallback-vehicles.md`）。再現のために残してある。
    """
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df[RULE_COL] = df[V_TEXT].map(normalize_model)
    return df, declared_values(df)


def _build_vehicles_type() -> tuple[pd.DataFrame, list[str]]:
    """Craigslist のうち `type` が入っている行。宣言値は type の13水準。

    `type` は出品者が選んだボディ形状で、**入力の `model` から機械的には
    決まらない**。だから近傍分類が迷う行が生まれ、LLM に回す余地がある。
    24.5% が欠損なので、値のある行だけで採点する（欠損を当てにいく課題ではない）。
    """
    df = load_dataset(dataset=VEHICLES, sample=N_SAMPLE)
    df = df[df[TYPE_COL].notna()].reset_index(drop=True)
    return df, list(TYPE_VALUES)


TASKS = {
    "sienta": FallbackTask(
        name="シエンタ（単一車種・日本語）",
        build=_build_sienta,
        source=COL,
        truth="グレード名",
        generated=GENERATED,
        out=ROOT / "results" / "feature_fallback.csv",
        note="タイトルからグレード名を作る。グレード名は文字どおり書かれている",
    ),
    "vehicles": FallbackTask(
        name="Craigslist（複数車種・英語）— 成立しないことが分かっている作り",
        build=_build_vehicles,
        source=V_TEXT,
        truth=RULE_COL,
        generated=V_GENERATED,
        out=ROOT / "results" / "feature_fallback_vehicles_rulecol.csv",
        note="model から車種の芯を作る。正解が入力の部分文字列になるため自明化する",
    ),
    "vehicles-type": FallbackTask(
        name="Craigslist（複数車種・英語）",
        build=_build_vehicles_type,
        source=V_TEXT,
        truth=TYPE_COL,
        generated=TYPE_GENERATED,
        out=ROOT / "results" / "feature_fallback_vehicles.csv",
        note="model（自由記述）からボディ形状を作る。正解はデータが持つ type 列",
        emb=EMB_MODEL,
        emb_values=EMB_VALUES,
    ),
}


# --- 測定 -------------------------------------------------------------


def seed_labels(train: pd.DataFrame, truth: str, n_labels: int) -> pd.Series:
    """訓練データのうち n_labels 件だけラベルを残す（人手ラベルの再現）。"""
    y = train[truth].astype("object").reset_index(drop=True)
    if n_labels >= len(y):
        return y
    rng = np.random.default_rng(SEED)
    keep = rng.choice(len(y), size=n_labels, replace=False)
    masked = pd.Series([np.nan] * len(y), dtype="object")
    masked.iloc[keep] = y.iloc[keep].to_numpy()
    return masked


def make_encoder(task: FallbackTask, variant: Variant):
    """(encoder, preprocess) を返す。既定は None（Feature 側の文字TF-IDF）。"""
    if variant.encoder == "default":
        return None, True
    if task.emb is None:
        raise SystemExit(f"{task.name} には計算済み埋め込みの設定がありません。")
    if not task.emb.exists():
        raise SystemExit(
            f"{task.emb.name} がありません。"
            "先に .venv-embed/bin/python scripts/embed_vehicles.py を実行してください。")
    tables = [pd.read_parquet(task.emb)]
    if variant.label_source == "labelname":
        # 案b は値の名前そのものが参照事例になる。PrecomputedEncoder は
        # 未知の文字列を扱えないので、その13語ぶんの埋め込みが要る
        if task.emb_values is None or not task.emb_values.exists():
            raise SystemExit(
                f"{task.emb_values.name if task.emb_values else '値の名前の埋め込み'}"
                " がありません。先に .venv-embed/bin/python "
                "scripts/embed_type_values.py を実行してください。")
        tables.append(pd.read_parquet(task.emb_values))
    table = pd.concat(tables, ignore_index=True).drop_duplicates(subset=[task.source])
    # 前処理をかけると文字列が変わり、対応表を引けなくなるので切る
    return PrecomputedEncoder(table, task.source, name="e5small"), False


def one_rate(task: FallbackTask, variant: Variant, train: pd.DataFrame,
             test: pd.DataFrame, values: list[str], rate: float,
             client: ClaudeClient, dry_run: bool) -> dict:
    """ある条件・ある率で1回だけ測る。"""
    fb = (QueueOnlyFallback() if (rate == 0.0 or dry_run)
          else ClaudeFallback(client=client))
    encoder, preprocess = make_encoder(task, variant)
    f = Feature(source=task.source, type="category", values=values,
                k="auto", escalate_rate=rate if rate > 0 else None,
                threshold=0.9, fallback=fb, name=task.generated,
                encoder=encoder, preprocess=preprocess)
    y = (seed_labels(train, task.truth, variant.n_labels)
         if variant.label_source == "human" else None)
    f.fit(train, y)
    pred = f.transform(test).astype(str).to_numpy()
    truth = test[task.truth].astype(str).to_numpy()

    prov = f._prov()
    llm_rows = (prov["由来"] == "llm").to_numpy()
    ok = pred == truth
    # 宣言した値に無い正解は、そもそも当てようがない。
    # LLM の効果は「宣言値に含まれる行」で見ないと過小評価になる
    in_scope = np.isin(truth, values)

    row = {
        "ラベル起点": variant.label_name,
        "エンコーダ": variant.encoder_name,
        "エスカレーション率": rate,
        "実際に回した行": int(llm_rows.sum()),
        "accuracy": float(ok.mean()),
        "宣言値に限った accuracy": float(ok[in_scope].mean()),
        "回した行の accuracy": (float(ok[llm_rows].mean())
                                if llm_rows.any() else float("nan")),
        "回した行の宣言値内 accuracy": (
            float(ok[llm_rows & in_scope].mean())
            if (llm_rows & in_scope).any() else float("nan")),
        "費用_usd": float(prov["コスト"].sum()),
    }
    if rate == 0.0:
        # 事前チェック3: 確信度の下位は本当に当たっていないのか。
        # 下位30%と残り70%で accuracy を割る（0% のときだけ意味がある）
        conf = prov["confidence"].to_numpy()
        n_low = int(round(len(conf) * 0.30))
        order = np.argsort(conf, kind="stable")
        low = np.zeros(len(conf), dtype=bool)
        low[order[:n_low]] = True
        row["確信度下位30%の accuracy"] = float(ok[low].mean())
        row["残り70%の accuracy"] = float(ok[~low].mean())
    return row


def ceiling(pop: pd.DataFrame, scored: pd.DataFrame, source: str,
            truth: str) -> float:
    """同じ入力テキストの多数決で取れる accuracy の上限。

    正解が入力テキストの関数として一意に決まらないとき（`type` は出品者が
    各出品で選ぶので `f-150` が truck と pickup に割れる）、
    **テキストだけを見るどんな手法もこの値を超えられない**。

    多数決は母集団 `pop` 全体で取り、それを `scored`（採点する行）に当てる。
    採点する400行の中だけで多数決を取ると、同じ文字列がほとんど重複しないので
    上限が実際より高く出る（実測で 0.827 → 0.925）。
    """
    maj = pop.groupby(source)[truth].agg(lambda s: s.value_counts().index[0])
    best = scored[source].map(maj)
    return float((best.to_numpy() == scored[truth].to_numpy()).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="機能A の LLM フォールバックを実測")
    ap.add_argument("--dataset", default="sienta", choices=sorted(TASKS),
                    help="測定する課題（既定 sienta）")
    ap.add_argument("--label-source", default="labelname",
                    choices=("labelname", "human", "both"),
                    help="教師ラベルの起点（既定 labelname＝値の名前だけ）")
    ap.add_argument("--n-labels", type=int, default=N_SEED_LABELS,
                    help="--label-source human のときに与える人手ラベルの件数")
    ap.add_argument("--encoder", default="default",
                    choices=("default", "e5", "both"),
                    help="近傍分類のエンコーダ（既定 default＝文字TF-IDF）")
    ap.add_argument("--n-test", type=int, default=400,
                    help="採点に使う test の行数（費用を抑えるため抽出する）")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM を呼ばず、事前チェックと 0% の結果だけ出す")
    args = ap.parse_args()

    task = TASKS[args.dataset]
    client = ClaudeClient(model=args.model, effort=args.effort,
                          max_workers=args.workers)
    if not args.dry_run and not client.available():
        print("APIキーが読めません。scripts/check_api_key.py で確認してください。")
        raise SystemExit(1)

    sources = (("labelname", "human") if args.label_source == "both"
               else (args.label_source,))
    encoders = (("default", "e5") if args.encoder == "both" else (args.encoder,))
    variants = [Variant(s, e, args.n_labels if s == "human" else None)
                for s in sources for e in encoders]

    df, values = task.build()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(iter(kf.split(df)))       # fold1 だけ使う
    train = df.iloc[tr_idx].reset_index(drop=True)
    test_all = df.iloc[te_idx].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(test_all),
                              size=min(args.n_test, len(test_all)),
                              replace=False))
    test = test_all.iloc[pick].reset_index(drop=True)

    truth_all = test[task.truth].astype(str)
    covered = float(np.isin(truth_all.to_numpy(), values).mean())
    n_levels = int(df[task.truth].astype(str).nunique())
    floor = float(test[task.truth].astype(str).value_counts(normalize=True).iloc[0])
    print(f"\n=== {task.name} ===")
    print(task.note)
    print(f"fold1 のみ / 訓練 {len(train):,} 行 → 採点 {len(test)} 行")
    print(f"宣言した{len(values)}値がカバーするのは test の {covered:.1%}"
          f"（正解は全体で {n_levels:,} 水準）")
    print(f"目盛り: 下限（最頻クラスを常に答える） {floor:.3f} / "
          f"上限（同じ入力テキストの多数決） 全体 "
          f"{ceiling(df, df, task.source, task.truth):.3f} ・採点する"
          f"{len(test)}行 {ceiling(df, test, task.source, task.truth):.3f}")
    print(f"測る条件: {len(variants)} 通り")
    for v in variants:
        print(f"  - {v}")
    print()

    rows = [one_rate(task, v, train, test, values, r, client, args.dry_run)
            for v in variants for r in RATES]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    show = [c for c in res.columns if c not in
            ("確信度下位30%の accuracy", "残り70%の accuracy")]
    print(res[show].round(4).to_string(index=False))

    print("\n--- 事前チェック（0% の行から）---")
    for v in variants:
        b = res[(res["ラベル起点"] == v.label_name)
                & (res["エンコーダ"] == v.encoder_name)
                & (res["エスカレーション率"] == 0.0)].iloc[0]
        print(f"  {v}")
        print(f"    0% の accuracy {b['accuracy']:.3f}"
              f"（宣言値内 {b['宣言値に限った accuracy']:.3f}）")
        print(f"    確信度 下位30% {b['確信度下位30%の accuracy']:.3f}"
              f" / 残り70% {b['残り70%の accuracy']:.3f}"
              f" … 差 {b['残り70%の accuracy'] - b['確信度下位30%の accuracy']:+.3f}")

    if not args.dry_run:
        print("\n--- LLM を回した効果 ---")
        for v in variants:
            sub = res[(res["ラベル起点"] == v.label_name)
                      & (res["エンコーダ"] == v.encoder_name)]
            base = sub.iloc[0]
            print(f"  {v}: 0% で {base['宣言値に限った accuracy']:.3f}")
            for _, r in sub.iloc[1:].iterrows():
                d = r["宣言値に限った accuracy"] - base["宣言値に限った accuracy"]
                n = int(r["実際に回した行"])
                per = r["費用_usd"] / n if n else float("nan")
                print(f"    {r['エスカレーション率']:.0%} 回すと "
                      f"{r['宣言値に限った accuracy']:.3f}（{d:+.3f}） / "
                      f"回した行の accuracy {r['回した行の accuracy']:.3f} / "
                      f"{n} 行 / ${r['費用_usd']:.3f}（1行 ${per:.4f}）")

    task.out.parent.mkdir(exist_ok=True)
    res.to_csv(task.out, index=False, encoding="utf-8-sig")
    print(f"\n結果: {task.out.relative_to(ROOT)}")
    if not args.dry_run:
        print("\n--- 実測費用（累計）---")
        for k, v in client.summary().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
