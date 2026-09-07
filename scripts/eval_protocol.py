"""評価プロトコル — すべての実験をここに通して同じ土俵に乗せる。

CLAUDE.md の「同じ指標で精度を記録する」を機械的に守るための共通基盤。
以降 unfold の機能A（特徴量生成）・機能B（LLMPredictor）を試すときも、
必ず `cross_validate()` を経由させて `results/leaderboard.csv` に積む。

データセットは2つあり、CV の手続きは共通で置き場所と目的変数だけが違う。
`Dataset` として定義してあり、リーダーボードも別ファイルに分けている
（価格の単位が 万円 / USD で違うため、同じ表に混ぜると比較できるように見えてしまう）。

    SIENTA   … シエンタ 5,507行・日本語・単一車種  → results/leaderboard.csv
    VEHICLES … Craigslist 20万行・英語・複数車種   → results/leaderboard_vehicles.csv

    cross_validate("手法名", fit_predict, df, dataset=VEHICLES)

固定しているもの（一度決めたら変えない。変えるなら過去の行も引き直す）:

- データ    : sampledata/processed/usedsienta_clean.parquet（シエンタ 5,513行）
              sampledata/processed/vehicles_multi_clean.parquet（Craigslist 200,374行）
- 目的変数  : 車両本体価格_万円（シエンタ）/ price（Craigslist）
              ※ 支払総額ではない。支払総額には店舗ごとの諸費用が乗っていて、
                 車両そのものの価値以外の分散が混ざるため。
- 分割      : KFold(n_splits=5, shuffle=True, random_state=42)
- 指標      : MAE / RMSE / MAPE / R2（すべて万円スケールで計算）
              主指標は MAE。中央値185万円に対して外れ値が穏やかなので、
              RMSE より解釈しやすい（MAE 10 = 平均10万円ずれる）。

固定できないもの（記録して区別する）:

- 実行環境  : 機械が違うと同じコード・同じデータでも MAE が僅かに動く。
              固定できないので `実行環境` 列に測定機を記録し、
              混在時は `show_leaderboard()` が警告する。

使い方:

    from eval_protocol import load_dataset, cross_validate

    df = load_dataset()
    cross_validate("手法名", fit_predict, df, note="何をしたか")

`fit_predict(train_df, test_df) -> np.ndarray` は fold ごとに呼ばれる。
**前処理も含めて train だけで fit すること。** TF-IDF や埋め込みを
全データで fit すると test の情報が漏れる。
"""

from __future__ import annotations

import csv
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
N_SPLITS = 5

# --- 測定機の識別 -----------------------------------------------------
# 同じコード・同じデータでも、機械が違うと MAE が僅かに動く。LightGBM の
# バイナリ差（AVX2/clang と AVX-512/gcc で和の順序が変わる）によるもので、
# 60,000行・1,013列で 0.079% を実測した（docs/2026-09-01-migration.md）。
# ズレは fold 間のばらつきより1桁小さいが、どの機械で測ったか分からないと
# 手法による差なのか機械による差なのかを切り分けられない。だから1列足す。
_KNOWN_HOSTS = {"attuan-compute": "node"}


def runtime_tag() -> str:
    """測定機を表す短い札。

    node = 計算ノード（Ubuntu / m7i.4xlarge）… 測定はここで行う
    mac  = 手元の Mac（macOS / i7-8850H）  … 8/31 までの記録はすべてこれ

    知らない機械では hostname をそのまま使う。環境変数 CARPRICE_RUNTIME で
    上書きできる（機械を作り直して hostname が変わったときの逃げ道）。
    """
    tag = os.environ.get("CARPRICE_RUNTIME")
    if tag:
        return tag
    host = platform.node().split(".")[0]
    if host in _KNOWN_HOSTS:
        return _KNOWN_HOSTS[host]
    if platform.system() == "Darwin":
        return "mac"
    return host or "unknown"



@dataclass(frozen=True)
class Dataset:
    """実験対象のデータセット。CV の手続きは共通、置き場所と目的変数だけが違う。

    リーダーボードを分けているのは、価格の単位（万円 / USD）も行数も違うため。
    同じ表に混ぜると MAE の数字が比較できるように見えてしまう。
    """
    name: str
    path: Path
    target: str
    leaderboard: Path
    unit: str


SIENTA = Dataset(
    name="シエンタ（単一車種・日本語）",
    path=ROOT / "sampledata" / "processed" / "usedsienta_clean.parquet",
    target="車両本体価格_万円",
    leaderboard=ROOT / "results" / "leaderboard.csv",
    unit="万円",
)

VEHICLES = Dataset(
    name="Craigslist（複数車種・英語）",
    path=ROOT / "sampledata" / "processed" / "vehicles_multi_clean.parquet",
    target="price",
    leaderboard=ROOT / "results" / "leaderboard_vehicles.csv",
    unit="USD",
)

DEFAULT = SIENTA

# 後方互換（シエンタ用スクリプトが直接参照している）
DATA = SIENTA.path
LEADERBOARD = SIENTA.leaderboard
TARGET = SIENTA.target

# --- 特徴量の区分（シエンタ）-------------------------------------------
# 列名は日本語。これはスクレイピング元が日本語のサイトだからで、
# Craigslist 側とは無関係。両データセットで揃えているのは CV の手続きだけで、
# 列名も列の構成も揃えない。
#
# 「従来手法」の列。スクレイピング当時に回帰分析へ入れていたものと同じ構成。
# ここが出発点で、これを超えられるかが本プロジェクトの問い。
LEGACY_NUM = ["車齢", "走行距離_km"]
LEGACY_BOOL = ["修復歴あり", "保証付", "ハイブリッド"]
LEGACY_CAT = ["車検区分", "都道府県"]

# クレンジングで新たに取り出せた構造化列。
# 車検残月数は「期限指定」の車にしか無く欠損2,926件だが、欠損自体が
# 「車検整備付」を意味するので落とさず欠損のまま木に渡す。
EXTRA_NUM = ["車検残月数", "装備数"]
EXTRA_CAT = ["グレード名", "色_基本"]

# 非構造テキスト。unfold 機能A の入力になる列。
TEXT_COL = "装備テキスト"

# 除外する列とその理由:
#   車名(シエンタのみ)・排気量(1.5のみ)     … 定数
#   年式                                    … 車齢と完全に従属
#   支払総額_万円                            … 目的変数のリーク
#   車検満了                                 … 車検残月数として数値化済み
#   物件ID・店舗・タイトル・グレード_原文     … ID的／テキスト。別途扱う
#   色                                       … 色_基本に集約済み
#   走行距離_異常・装備記載あり・タイトル切れ疑い・タイトル文字数
#                                            … クレンジングの品質フラグ。
#                                               予測に使うのは筋が悪いので外す


# --- 特徴量の区分（Craigslist）-----------------------------------------
# 列名は元データ（Kaggle の vehicles.csv）の英語のまま。各列を採る理由は
# scripts/clean_vehicles.py の docstring に書いてある。ここでは
# 「予測に入れる列」と「入れない列」の線引きだけを持つ。

# 出品フォームの数値項目。欠損が無く、価格との関係も単調で、
# これだけで MAE の大半が決まる。age は year と従属だが、木に
# 「経過年数」という軸を直接渡したいので数値側に置く。
VEHICLES_NUM = ["age", "odometer"]
VEHICLES_BOOL: list[str] = []

# 出品フォームの選択式項目。欠損は 0.4%〜62.8% とばらつくが、
# LightGBM は欠損のまま食えるうえ、未入力であること自体が情報なので落とさない。
# state は地域相場を表す最も粗い地理情報。region（404水準）は入れない
# ——同じ車が複数地域に出稿されているので、相場ではなく個体を当てにいく。
VEHICLES_CAT = ["manufacturer", "condition", "cylinders", "fuel", "title_status",
                "transmission", "drive", "size", "type", "paint_color", "state"]

# 本実験の主役。出品者が自由に書いた車種の記述（19,739種類・6割が1回きり）。
# ここをどう表現するかを比較するのが Craigslist を使う目的。
VEHICLES_TEXT = "model"

# 自由記述の本文。中央値 1,075 字で、43.7% の行に価格そのものが書かれている。
# 使うときは必ず金額を伏せる（unfold.predictor の mask）。
VEHICLES_LONG_TEXT = "description"


def load_dataset(verbose: bool = True, dataset: Dataset = DEFAULT,
                 sample: int | None = None) -> pd.DataFrame:
    """学習用データを読む。目的変数が欠損の行だけ落とす。

    sample を渡すと seed 固定でランダムに間引く（Craigslist 20万行を
    そのまま埋め込み比較まで回すと現実的な時間に収まらないため）。
    """
    df = pd.read_parquet(dataset.path)
    n_before = len(df)
    df = df[df[dataset.target].notna()].reset_index(drop=True)
    n_notna = len(df)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=SEED).reset_index(drop=True)
    y = df[dataset.target]
    if verbose:
        print(f"データ: {dataset.path.relative_to(ROOT)}（{dataset.name}）")
        print(f"  {n_before:,} 行 → {n_notna:,} 行"
              f"（価格欠損 {n_before - n_notna} 行を除外）"
              + (f" → 抽出 {len(df):,} 行" if len(df) != n_notna else ""))
        print(f"  価格: 中央値 {y.median():,.1f} {dataset.unit} / "
              f"平均 {y.mean():,.1f} {dataset.unit} / "
              f"範囲 {y.min():,.0f}〜{y.max():,.0f} {dataset.unit}")
    return df


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / y_true)) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
    }


def cross_validate(
    name: str,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    df: pd.DataFrame | None = None,
    note: str = "",
    record: bool = True,
    verbose: bool = True,
    dataset: Dataset = DEFAULT,
    oof_out: dict | None = None,
) -> dict[str, float]:
    """5-fold CV を回し、結果を leaderboard.csv に1行追記する。

    fit_predict は (train_df, test_df) を受け取り test_df の予測値
    （万円スケール）を返す。中で log 変換するのは自由だが、返す時点で
    万円に戻しておくこと。指標は必ず万円で計算する。
    """
    if df is None:
        df = load_dataset(verbose=False, dataset=dataset)
    target = dataset.target

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    per_fold: list[dict[str, float]] = []
    # oof_out を渡すと out-of-fold 予測を受け取れる。
    # 「train に無かった model の行だけで MAE を測る」ような事後分析に使う。
    oof = np.full(len(df), np.nan)
    fold_id = np.zeros(len(df), dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        y_pred = np.asarray(fit_predict(train_df, test_df), dtype=float)
        y_true = test_df[target].to_numpy(dtype=float)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"{name}: 予測の形が合いません {y_pred.shape} != {y_true.shape}"
            )
        oof[te_idx] = y_pred
        fold_id[te_idx] = fold
        m = _metrics(y_true, y_pred)
        per_fold.append(m)
        if verbose:
            print(f"  fold{fold}  MAE {m['MAE']:6.2f}  RMSE {m['RMSE']:6.2f}"
                  f"  MAPE {m['MAPE']:5.1f}%  R2 {m['R2']:.3f}")

    agg = {k: float(np.mean([f[k] for f in per_fold])) for k in per_fold[0]}
    agg["MAE_std"] = float(np.std([f["MAE"] for f in per_fold]))

    if verbose:
        print(f"  {'平均':6}  MAE {agg['MAE']:6.2f} (±{agg['MAE_std']:.2f})"
              f"  RMSE {agg['RMSE']:6.2f}  MAPE {agg['MAPE']:5.1f}%"
              f"  R2 {agg['R2']:.3f}")

    if oof_out is not None:
        oof_out["pred"] = oof
        oof_out["fold"] = fold_id
    if record:
        _append(name, agg, len(df), note, dataset)
    return agg


def _append(name: str, agg: dict[str, float], n_rows: int, note: str,
            dataset: Dataset = DEFAULT) -> None:
    board = dataset.leaderboard
    board.parent.mkdir(exist_ok=True)
    header = ["実行日時", "実行環境", "手法", "MAE", "MAE_std", "RMSE", "MAPE",
              "R2", "行数", "fold数", "seed", "備考"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        runtime_tag(),
        name,
        f"{agg['MAE']:.3f}", f"{agg['MAE_std']:.3f}", f"{agg['RMSE']:.3f}",
        f"{agg['MAPE']:.2f}", f"{agg['R2']:.4f}",
        n_rows, N_SPLITS, SEED, note,
    ]
    is_new = not board.exists()
    with board.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(header)
        w.writerow(row)


def show_leaderboard(dataset: Dataset = DEFAULT) -> pd.DataFrame:
    """これまでの全実験を MAE 昇順で表示する。

    複数の測定機の行が混ざっているときは警告する。並べて見ること自体は
    構わないが、環境をまたいだ差を手法の差として読んではいけない。
    """
    if not dataset.leaderboard.exists():
        print("まだ結果がありません。")
        return pd.DataFrame()
    lb = pd.read_csv(dataset.leaderboard, encoding="utf-8-sig")
    envs = sorted(set(lb.get("実行環境", pd.Series(dtype=str)).dropna()))
    if len(envs) > 1:
        print(f"注意: 実行環境が混ざっています（{' / '.join(envs)}）。"
              "機械が違うと MAE が僅かに動くので、環境をまたいだ行の差は"
              "手法の差として読まないこと（docs/2026-09-01-migration.md）。")
    return lb.sort_values("MAE").reset_index(drop=True)
