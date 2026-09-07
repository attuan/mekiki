"""`python -m unfold.demo` — ライブラリ全体を一度に動かして見せる入口。

**APIキーが無くても最後まで走る。**既定では LLM を1回も呼ばないので無料。
既定のデータは git に入っている 500 行の抜粋
（`sampledata/sample/vehicles_sample500.csv`）なので、
**clone 直後・データ取得なしで動く。**

    .venv/bin/python -m unfold.demo                    # 無料。LLM は呼ばない
    .venv/bin/python -m unfold.demo --run 20           # 20 行を予測する（うち3割が LLM に回る）
    .venv/bin/python -m unfold.demo --dataset vehicles # データセットを選ぶ

順番は実際の使い方どおりにしてある。

1. **リーク検査** — 同じ車が train と test に入っていないか（`check_duplicates`）
2. **スクリーニング** — そのデータでテキストが効くか（`screen`。LLM を呼ばない）
3. **機能A** — 非構造列を型付き列にする（`Feature`）
4. **見積もり** — LLM に何行回すと、いくら・何秒か（`AdaptivePredictor.plan`）
5. **実行** — `--run N` を付けたときだけ。予測と来歴（`explain` / `report` / `cost`）

## 新しいデータセットを足すには

**手順1〜5のコードは触らない。**触るのは下の「データセット定義」の節だけで、
やることは2つ。

1. 生の CSV を「学習に使える形」にする関数を1つ書く（`_load_vehicles` が見本）。
   目的変数・特徴量・識別子の列を、そのデータセット自身の名前で作る。
   列名を他のデータセットに合わせる必要はない（合わせない方がよい）。
2. `DemoDataset(...)` を1つ書いて `DATASETS` に登録する。

手順1〜5 はここで宣言された列名しか見ないので、データセットが増えても
分岐は増えない。テキスト列が無いデータ（例: シエンタの数値のみの表）は
機能A も機能B も対象外なので、このデモには載せない。
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from unfold import (
    AdaptivePredictor,
    Feature,
    UnfoldWarning,
    check_duplicates,
    check_overlap,
    screen,
)
from unfold.llm import ClaudeClient

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


# =====================================================================
# データセット定義 — 新しいデータを足すときに触るのはこの節だけ
# =====================================================================


@dataclass(frozen=True)
class FeatureSpec:
    """手順3（機能A）で作る列の宣言。`Feature` にそのまま渡す。"""

    source: str                 # 入力にする非構造列
    values: list[str]           # 取りうる値
    column: str                 # 作った結果を入れる列の名前
    type: str = "category"
    escalate_rate: float = 0.15
    description: str = ""       # 画面に出す1行説明（何から何を作るのか）


@dataclass(frozen=True)
class DemoDataset:
    """デモが1つのデータセットについて知っていることのすべて。

    手順1〜5 はここに書かれた列名だけを見る。データセット固有の知識
    （元データの列名・単位・外れ値の切り方）は `load` の中に閉じ込める。
    """

    name: str                              # 画面に出す名前
    path: Path                             # 元データ
    load: Callable[[Path], pd.DataFrame]   # 元データ → 学習に使える形
    target: str                            # 目的変数の列
    unit: str                              # 目的変数の単位（USD / 万円 など）
    numeric: list[str]
    categorical: list[str]
    text: str                              # 機能B に渡すテキスト列
    id_col: str                            # 行の識別子。照合からは外す
    dedup_key: str | None = None           # 1台1行に潰すための識別子（VIN など）
    dedup_label: str = ""                  # その識別子の呼び名（画面表示用）
    feature: FeatureSpec | None = None     # 手順3。無ければ手順3を飛ばす
    n_test: int = 60


def _load_vehicles(path: Path) -> pd.DataFrame:
    """Craigslist の抜粋を、そのまま学習に使える形にする。

    `scripts/clean_vehicles.py` の軽い版。**価格の外れ値は必ず切る**
    （元データは最大 $3,736,928,711 で、平均を取ると壊れる）。
    **列名は Kaggle の `vehicles.csv` のまま**にする。日本語に訳したり、
    他のデータセットの列名に寄せたりしない。
    """
    base_year = 2021          # データの掲載時期（posting_date は 2021-04〜05）
    df = pd.read_csv(path)
    df = df[df["price"].between(1_000, 100_000)]
    df = df[df["year"].notna() & df["manufacturer"].notna()
            & df["odometer"].notna()]
    out = pd.DataFrame({
        "id": df["id"],
        "price": df["price"].astype(float),
        "age": base_year - df["year"],
        "odometer": df["odometer"],
        "manufacturer": df["manufacturer"].str.lower().str.strip(),
        "model": df["model"].fillna("").str.lower().str.strip(),
        "fuel": df["fuel"].fillna("unknown"),
        "transmission": df["transmission"].fillna("unknown"),
        "state": df["state"],
        "VIN": df["VIN"],
    })
    return out.reset_index(drop=True)


VEHICLES = DemoDataset(
    name="Craigslist 中古車（複数車種・英語）500行の抜粋",
    path=ROOT / "sampledata" / "sample" / "vehicles_sample500.csv",
    load=_load_vehicles,
    target="price",
    unit="USD",
    numeric=["age", "odometer"],
    categorical=["manufacturer", "state", "fuel", "transmission"],
    text="model",
    id_col="id",
    dedup_key="VIN",
    dedup_label="VIN（車台番号）",
    feature=FeatureSpec(
        source="model",
        values=["pickup truck", "sedan", "suv", "van", "coupe"],
        column="body_type",
        description="model（自由な文字列）から、車体の種類を表す列を作る。",
    ),
    n_test=60,
)

DATASETS: dict[str, DemoDataset] = {"vehicles": VEHICLES}
DEFAULT_DATASET = "vehicles"


# =====================================================================
# 手順1〜5 — ここから下はデータセットに依存しない
# =====================================================================


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def step1_leak(ds: DemoDataset, df: pd.DataFrame) -> pd.DataFrame:
    hr("1. リーク検査 — 同じ行が二重に入っていないか")
    print("同じ車が複数の地域に出稿されているデータでは、ランダムに分割すると")
    print("同じ車が train と test の両方に入り、答えを見て答えることになる。\n")

    print("[全列で照合]")
    print(check_duplicates(df, ignore=[ds.id_col]))

    if ds.dedup_key is None:
        print("\n（このデータには1行1件を保証する識別子が無いので、"
              "全列照合だけで判断する）")
        return df.reset_index(drop=True)

    label = ds.dedup_label or ds.dedup_key
    print(f"\n[{label} で照合]")
    keyed = df[df[ds.dedup_key].notna()]
    print(check_duplicates(keyed, keys=[ds.dedup_key]))

    before = len(df)
    df = df[~df[ds.dedup_key].notna() | ~df[ds.dedup_key].duplicated()]
    df = df.reset_index(drop=True)
    print(f"\n→ 1件1行に潰した: {before} 行 → {len(df)} 行")
    return df


def split(ds: DemoDataset, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    test = df.iloc[idx[:ds.n_test]].reset_index(drop=True)
    train = df.iloc[idx[ds.n_test:]].reset_index(drop=True)
    ignore = [ds.id_col] + ([ds.dedup_key] if ds.dedup_key else [])
    print(f"\n訓練 {len(train)} 行 / 評価 {len(test)} 行に分けた。")
    print(check_overlap(train, test, ignore=ignore))
    return train, test


def step2_screen(ds: DemoDataset, df: pd.DataFrame) -> None:
    hr("2. スクリーニング — このデータでテキストは効くか（LLM を呼ばない）")
    print("機能B はテキストが目的変数を左右するデータでしか効かない。")
    print("先に無料で判定しておくと、効かないデータに課金しなくて済む。\n")
    print(screen(df, target=ds.target, text=ds.text, unit=ds.unit,
                 numeric=ds.numeric, categorical=ds.categorical))


def step3_feature(ds: DemoDataset, train: pd.DataFrame) -> None:
    hr("3. 機能A（Feature）— 非構造列を型のついた列にする")
    if ds.feature is None:
        print("このデータには機能A の対象列を定義していないので飛ばす。")
        return

    spec = ds.feature
    if spec.description:
        print(spec.description)
    print("中身は「埋め込み → 既存の値の名前の近傍で分類」で、")
    print("**確信度が低い行だけ LLM に回る**（キーが無ければ人手レビュー待ちに積む）。\n")

    f = Feature(source=spec.source, type=spec.type, values=list(spec.values),
                escalate_rate=spec.escalate_rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        train = train.copy()
        train[spec.column] = f.fit_transform(train)

    print("状態:")
    for k, v in f.status().items():
        print(f"  {k}: {v}")
    print("\n割り当てられた値:")
    print(train[spec.column].value_counts().to_string())
    q = f.review_queue()
    if len(q):
        print(f"\nレビュー待ち {len(q)} 件のうち先頭3件"
              "（確信できなかった行。承認すると次回の教師ラベルになる）:")
        cols = [c for c in ["テキスト", "LLMの答え", "推測"] if c in q.columns]
        print(q[cols].head(3).to_string(index=False))


def step4_plan(ds: DemoDataset, train: pd.DataFrame, test: pd.DataFrame,
               rate: float, client: ClaudeClient) -> tuple[AdaptivePredictor, dict]:
    hr("4. 見積もり — 何行を LLM に回すと、いくら・何秒か（LLM を呼ばない）")
    print("機能B は1行につき1回 LLM を呼ぶので、行数がそのまま費用と時間になる。")
    print("信頼度ルーティングは「統計モデルが迷っている行」だけを回す。\n")

    model = AdaptivePredictor(
        target=ds.target, unit=ds.unit, numeric=ds.numeric,
        categorical=ds.categorical, text=ds.text,
        n_examples=3, escalate_rate=rate, client=client)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        model.fit(train)
    plan = model.plan(test)
    for k, v in plan.items():
        print(f"  {k}: {v}")
    return model, plan


def step5_run(ds: DemoDataset, model: AdaptivePredictor, test: pd.DataFrame,
              n_rows: int) -> None:
    hr(f"5. 実行 — 先頭 {n_rows} 行を予測する")
    sub = test.iloc[:n_rows].reset_index(drop=True)
    truth = sub[ds.target].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnfoldWarning)
        pred = model.predict(sub, verbose=True)

    print("\n" + model.report())
    print(f"\n  この {len(sub)} 行の MAE: "
          f"{np.mean(np.abs(pred - truth)):,.2f} {ds.unit}")

    prov = model.provenance()
    base_col = [c for c in prov.columns if c.startswith("証拠_")][0]
    base = prov[base_col].to_numpy(dtype=float)
    print(f"  1行も呼ばなかった場合（{base_col[3:]}だけ）: "
          f"{np.mean(np.abs(base - truth)):,.2f} {ds.unit}")

    print("\n--- 行ごとの経路 ---")
    print(model.route().round(2).head(10).to_string(index=False))

    llm_rows = model.route().query("経路 == 'llm'")["行"].tolist()
    if llm_rows:
        print("\n--- explain()（LLM に回った行の1つめ）---")
        print(model.explain(int(llm_rows[0]))[:1500])

    print("\n--- cost() ---")
    for k, v in model.cost().items():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="unfold を一通り動かす（既定では LLM を呼ばない）")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    choices=sorted(DATASETS),
                    help=f"使うデータセット（既定 {DEFAULT_DATASET}）")
    ap.add_argument("--run", type=int, default=0, metavar="行数",
                    help="実際に予測する行数。このうち --rate の割合だけが "
                         "LLM に回る。0 なら1回も呼ばない（既定）")
    ap.add_argument("--rate", type=float, default=0.3,
                    help="LLM に回す割合（信頼度ルーティングの閾値）")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    if not ds.path.exists():
        raise SystemExit(f"データがありません: {ds.path}")

    print(f"データ: {ds.name}")
    print(f"  {ds.path.relative_to(ROOT)}")
    df = ds.load(ds.path)
    print(f"  {len(df)} 行 / {ds.target} 中央値 "
          f"{df[ds.target].median():,.0f} {ds.unit}")

    df = step1_leak(ds, df)
    train, test = split(ds, df)
    step2_screen(ds, df)
    step3_feature(ds, train)

    client = ClaudeClient()
    model, plan = step4_plan(ds, train, test, args.rate, client)

    if args.run <= 0:
        hr("ここまで LLM を1回も呼んでいない（費用 $0）")
        n_hint = 20
        n_llm = max(1, round(n_hint * args.rate))
        unit_cost = float(plan["1行あたり_usd"])
        print("実際に投げるには:")
        print(f"    .venv/bin/python -m unfold.demo --dataset {args.dataset} "
              f"--run {n_hint}")
        print(f"  {n_hint} 行のうち {n_llm} 行が LLM に回る。"
              f"1行あたり約 ${unit_cost:.4f} なので約 ${n_llm * unit_cost:.2f}。")
        return
    if not client.available():
        hr("APIキーがありません")
        print(".env のキーを設定すると、ここから先が動く。")
        print("  確認: .venv/bin/python scripts/check_api_key.py")
        return
    step5_run(ds, model, test, args.run)


if __name__ == "__main__":
    main()
