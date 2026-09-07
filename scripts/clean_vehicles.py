"""Kaggle Craigslist 中古車データのクレンジング。

アメリカの個人売買掲示板 Craigslist に 2021年4〜5月に出稿された中古車 42万件。
1行が1件の出稿で、列はすべて出品者の自己申告である。ここでやるのは
「価格 price（USD）を予測する」ための学習用データを作ること。

**列名は元データの英語のまま扱う。** 日本語に訳したり、別のデータセット
（シエンタ）の列に読み替えたりしない。このデータで何が価格を動かすかは
このデータの言葉で考える。

出力: sampledata/processed/vehicles_multi_clean.parquet

再現:
    .venv/bin/python scripts/clean_vehicles.py

## どの列をなぜ残すか

目的変数

    price           予測したい値。USD。出品者の希望価格であって成約価格ではない

車の素性を直接決める、欠損の無い4列。ここだけで MAE の大半が決まる

    year            年式。1990〜2022 に限定
    age             2021 - year。年式と完全に従属だが、木モデルに
                    「経過年数」という単調な軸を直接渡したいので両方持つ
    odometer        走行距離（マイル）。価格との相関が最も強い数値列
    manufacturer    メーカー。40社。欠損なし

**本実験の主役**。ここをどう扱うかが unfold 機能A の問いそのもの

    model           出品者が自由に入力した車種の記述。19,739 種類あり、
                    6割は1回しか出てこない。"f-150 xlt supercrew" のように
                    車種・グレード・キャブ形状・駆動方式・宣伝文が
                    ひと続きに書かれていて、区切りも語順も揃っていない。
                    カテゴリとして持つ／ルールで正規化する／TF-IDF にする／
                    埋め込みにする、のどれが良いかをこの列で比較する

出品フォームの選択式項目。欠損は多いが、木モデルは欠損のまま食えるので落とさない。
未入力であること自体が出品者の手抜き具合を表していて情報になる

    condition       状態（欠損 39.1%）
    cylinders       気筒数（34.5%）。排気量の代理になる
    fuel            燃料（0.6%）
    title_status    名義状態（1.3%）。事故車・抹消登録かどうか
    transmission    変速機（0.4%）
    drive           駆動方式（26.9%）
    size            車体サイズ（62.8%）。欠損が最も多い
    type            ボディ形状（24.5%）
    paint_color     色（27.6%）
    state           州。51水準。地域相場を表す最も粗い地理情報

自由記述。機能B（LLMPredictor）に渡す候補

    description     出品者が書いた説明文。中央値 1,075 字。
                    **43.7% の行に価格そのものが書かれている**ので、
                    そのまま特徴量にすると答えを読んでいるだけになる
    description_length  説明文の長さ。本文を使わずに「どれだけ丁寧な出稿か」
                    だけを取り出したいときに使う

予測には使わないが、評価の正しさのために必要な列

    id              出稿ID。並び順の固定と、リーク検査の除外指定に使う
    VIN             車台番号（欠損 52.1%）。同じ車の重複出稿を突き止める鍵
    region          出稿地域（404水準）。同じ車が複数地域に出ているので、
                    予測に使うと重複を通じて相場ではなく個体を当てにいく

落とした列とその理由

    lat / long / posting_date / url / image_url / county / region_url
                    位置の細かい座標と URL 類。個体の識別子に近く、
                    価格の説明にならないか、リークの経路になる

## 行のフィルタ

    price 1,000〜100,000 USD  中央値 13,950 に対し最大 37億。0円や桁違いの
                             入力ミスを落とす
    year 1990〜2022          1900年代の入力ミスとクラシックカーを落とす
    odometer 100〜400,000    最大 1,000万マイルという異常値がある
    model / manufacturer     本実験の主役の列なので欠損は使えない

## 重複出稿

同じ車が複数の region に出稿されている（同一 VIN が最大 261 件、価格は同一）。
ランダム分割の CV では同じ車が train と test の両方に入りリークするので、
1台1行に潰す。実害は scripts/check_duplicate_leak.py で測ってある。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampledata" / "raw" / "vehicles.csv"
OUT = ROOT / "sampledata" / "processed" / "vehicles_multi_clean.parquet"
# 重複を残した版。「重複排除をサボると精度がどれだけ水増しされるか」を
# 測るためだけに使う（scripts/check_duplicate_leak.py）。
OUT_DUP = ROOT / "sampledata" / "processed" / "vehicles_multi_withdup.parquet"

# 掲載期間は 2021-04〜05 なので、車齢はこの年を基準にする。
BASE_YEAR = 2021

# 除外条件の理由は冒頭の docstring「行のフィルタ」に書いてある。
FILTERS = """
    price BETWEEN 1000 AND 100000
    AND year BETWEEN 1990 AND 2022
    AND odometer BETWEEN 100 AND 400000
    AND model IS NOT NULL AND manufacturer IS NOT NULL
"""


def main(dedup: bool = True) -> None:
    if not SRC.exists():
        raise SystemExit(
            f"{SRC} がありません。先に `python3 scripts/download_data.py` を実行してください。"
        )

    src = f"read_csv_auto('{SRC}', ignore_errors=true)"
    con = duckdb.connect()

    n_all = con.sql(f"SELECT count(*) FROM {src}").fetchone()[0]

    # --- 1. 行のフィルタ -------------------------------------------------
    con.sql(f"CREATE TABLE filtered AS SELECT * FROM {src} WHERE {FILTERS}")
    n_filtered = con.sql("SELECT count(*) FROM filtered").fetchone()[0]

    # --- 2. 重複排除 ------------------------------------------------------
    # 同じ車が複数の region に出稿されている（同一 VIN が最大 261 件、価格は同一）。
    # ランダム分割の CV では同じ車が train と test の両方に入りリークするので、
    # ここで必ず1台1行にする。VIN が無い行は内容の一致で潰す。
    con.sql(
        """
        CREATE TABLE dedup AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY coalesce(
                    VIN,
                    concat_ws('|', manufacturer, model, year, odometer, price,
                              coalesce(description, ''))
                )
                ORDER BY id
            ) AS rn
            FROM filtered
        ) WHERE rn = 1
        """
    )
    n_dedup = con.sql("SELECT count(*) FROM dedup").fetchone()[0]

    # 重複を残した版を作るときは、この段階の表を差し替えるだけ
    if not dedup:
        con.sql("DROP TABLE dedup")
        con.sql("CREATE TABLE dedup AS SELECT * FROM filtered")

    # --- 3. 列の整形 ------------------------------------------------------
    # 元データの列名をそのまま使う。手を入れるのは、
    #   - 文字列の小文字化・空白正規化（表記ゆれのうち、意味を持たないぶんだけ）
    #   - year から age を作る
    #   - description の長さを別列にする
    # の3つだけ。model の表記ゆれをここで潰すと「非構造テキストをどう扱うか」
    # という実験の問い自体が消えてしまうので、model は正規化しない。
    con.sql(
        f"""
        CREATE TABLE clean AS
        SELECT
            id,
            price,
            {BASE_YEAR} - year                          AS age,
            year,
            odometer,
            lower(trim(manufacturer))                   AS manufacturer,
            regexp_replace(lower(trim(model)), '\\s+', ' ', 'g') AS model,
            condition,
            cylinders,
            fuel,
            title_status,
            transmission,
            drive,
            size,
            type,
            paint_color,
            state,
            region,
            length(description)                         AS description_length,
            description,
            VIN
        FROM dedup
        """
    )

    out = OUT if dedup else OUT_DUP
    out.parent.mkdir(parents=True, exist_ok=True)
    # **必ず id 順で書き出す。** DuckDB は並列に読むので、指定しないと
    # 実行のたびに行の並びが変わる。並びが変われば
    # `load_dataset(sample=60_000)` が引く 6 万行も変わり、
    # 同じ seed でも別のデータで測ることになる（LLM のキャッシュも全部外れる）。
    con.sql(f"COPY (SELECT * FROM clean ORDER BY id) "
            f"TO '{out}' (FORMAT PARQUET)")

    # --- 4. 要約 ----------------------------------------------------------
    print(f"元データ            {n_all:,} 行")
    print(f"フィルタ後          {n_filtered:,} 行  ({n_all - n_filtered:,} 行を除外)")
    print(f"重複排除後          {n_dedup:,} 行  ({n_filtered - n_dedup:,} 行が重複出稿)")
    if not dedup:
        print("※ --no-dedup 指定のため重複はそのまま残している")
    print(f"→ {out.relative_to(ROOT)}")
    print()
    print(con.sql(
        """
        SELECT count(*) AS 行数,
               count(DISTINCT manufacturer) AS メーカー数,
               count(DISTINCT model) AS "model の種類",
               round(median(price)) AS 価格中央値,
               round(avg(price)) AS 価格平均,
               round(median(odometer)) AS 走行距離中央値,
               round(median(age), 1) AS 車齢中央値
        FROM clean
        """
    ).df().T.to_string())
    print()
    print("欠損率（%）")
    cols = [r[0] for r in con.sql("DESCRIBE clean").fetchall()]
    miss = con.sql(
        "SELECT " + ", ".join(
            f'round(100.0 * sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) / count(*), 1) AS "{c}"'
            for c in cols
        ) + " FROM clean"
    ).df().T
    print(miss.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-dedup", action="store_true",
                    help="重複出稿を残したまま出力する（リークの実害を測る用）")
    main(dedup=not ap.parse_args().no_dedup)
