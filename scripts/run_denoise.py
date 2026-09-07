"""入力テキストのノイズ除去が、埋め込みの「似ている」を価格に近づけるかを測る。

    .venv/bin/python scripts/run_denoise.py

前提: .venv-embed/bin/python scripts/embed_denoise.py を先に実行しておくこと。

測る問いは3つ。1と2が今回の本題（近傍の選び方・機能B の前提）、3は副作用の確認。

  Q1 グレードを分離できるか
     同じグレード同士の類似度が、違うグレード同士より高くなっているか。
     生テキストでは 0.915 対 0.912 でほぼ差がなかった（機能B の証拠設計に直結）。
     単純な平均の差は文の長さに左右されるので、順位で見る AUC も出す。
     AUC 0.5 = まったく区別できない、1.0 = 完全に分離できている。

  Q2 近傍が価格の証拠になるか【機能B の前提そのもの】
     テスト行ごとに訓練データから意味的に近い k 件を引き、その価格の中央値を
     予測値とする（＝機能B が「証拠」として渡す事例で価格を当てにいく）。
     これを 5-fold CV に載せれば、他の手法とまったく同じ土俵の MAE になる。

  Q3 教師あり学習の精度は落ちないか
     ノイズ除去はテキストを削る操作なので、消しすぎれば情報も落ちる。
     LightGBM に埋め込みを渡した D2 相当（グレード名の列は使わない）で確認する。

版（text_variants.py）:
  V0 生 / V1 既出語の除去 / V2 定数語の除去 / V3 先頭区切りまで / V23 V3+V2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import (  # noqa: E402
    DATA, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, ROOT, SEED, TARGET,
    cross_validate, load_dataset,
)
from features import LGBM_PARAMS, as_category as _as_category, numeric_frame as _numeric_frame  # noqa: E402

EMB_DIR = ROOT / "sampledata" / "processed"
VARIANTS = {
    "V0 生（現状）": EMB_DIR / "usedsienta_emb_title_e5small.parquet",
    "V1 既出語の除去": EMB_DIR / "usedsienta_emb_titlev1_e5small.parquet",
    "V2 定数語の除去": EMB_DIR / "usedsienta_emb_titlev2_e5small.parquet",
    "V3 先頭区切りまで": EMB_DIR / "usedsienta_emb_titlev3_e5small.parquet",
    "V23 V3+定数語除去": EMB_DIR / "usedsienta_emb_titlev23_e5small.parquet",
}
K = 5  # 近傍として引く件数。機能B が LLM に渡す事例数の想定


def load_embeddings() -> dict[str, np.ndarray]:
    """埋め込みを読み、load_dataset() と同じ行に揃える（価格欠損6行を落とす）。"""
    keep = pd.read_parquet(DATA, columns=[TARGET])[TARGET].notna().to_numpy()
    out = {}
    for name, path in VARIANTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} がありません。先に埋め込みを作ってください:\n"
                f"  .venv-embed/bin/python scripts/embed_denoise.py")
        V = pd.read_parquet(path).drop(columns=["行番号"]).to_numpy(dtype="float32")
        out[name] = V[keep]
    return out


# --- Q1 グレードの分離 -------------------------------------------------

def grade_separation(V: np.ndarray, grade: np.ndarray, n_pairs: int = 200_000,
                     seed: int = SEED) -> dict[str, float]:
    """ランダムなペアを引き、同一グレードか否かで類似度を比べる。

    平均の差だけでなく AUC も出す。AUC は「同一グレードのペアの方が
    異グレードのペアより類似度が高い確率」で、値の絶対水準や文長の影響を受けない。
    """
    rng = np.random.default_rng(seed)
    n = len(V)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    ok = i != j
    i, j = i[ok], j[ok]
    sim = np.einsum("ij,ij->i", V[i], V[j])  # 正規化済みなので内積＝コサイン類似度
    same = grade[i] == grade[j]

    s_pos, s_neg = sim[same], sim[~same]
    # AUC は順位で計算する（Mann-Whitney U）
    order = np.argsort(np.concatenate([s_pos, s_neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    n1, n2 = len(s_pos), len(s_neg)
    auc = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return {"同一グレード": float(s_pos.mean()), "異グレード": float(s_neg.mean()),
            "差": float(s_pos.mean() - s_neg.mean()), "AUC": float(auc)}


# --- Q2 近傍を証拠にした価格予測 ---------------------------------------

def make_knn(V: np.ndarray, k: int = K, numeric: np.ndarray | None = None,
             w: float = 0.0):
    """意味的近傍 k 件の価格中央値を予測値として返す fit_predict。

    numeric と w を渡すと、距離に数値列（車齢・走行距離）の差を混ぜる
    （近傍の対策案「数値列を含めた距離にする」の実装）。
    数値列は訓練データの標準偏差で割って尺度を揃える。
    """
    def fit_predict(train, test):
        tr = train["_行番号"].to_numpy()
        te = test["_行番号"].to_numpy()
        sim = V[te] @ V[tr].T                      # 高いほど近い
        if numeric is not None and w > 0:
            Ntr, Nte = numeric[tr], numeric[te]
            sd = Ntr.std(axis=0)
            sd[sd == 0] = 1.0
            # 数値の距離（大きいほど遠い）を類似度から引く
            d = np.zeros_like(sim)
            for c in range(Ntr.shape[1]):
                d += np.abs(Nte[:, [c]] - Ntr[:, c][None, :]) / sd[c]
            sim = sim - w * d / Ntr.shape[1]
        idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
        y_tr = train[TARGET].to_numpy(dtype=float)
        return np.median(y_tr[idx], axis=1)
    return fit_predict


def make_knn_banded(V: np.ndarray, k: int = K, band: float = 0.2):
    """近傍の対策案b「価格帯で絞ってから意味的に並べる」の実装。

    推論時に本当の価格は使えないので、**構造化列だけの LightGBM で粗く予測し、
    その ±band の価格帯にある訓練事例だけを候補にする。**そのうえで意味的に
    近い順に k 件を選ぶ。機能B に置き換えると「統計モデルが当たりをつけ、
    埋め込みが事例を選ぶ」という二段構えになる。

    候補が k 件に満たない行は帯を広げる（最大3回。それでも足りなければ全体）。
    """
    def fit_predict(train, test):
        # 粗い予測器。テキストは一切使わない（帯を決めるためだけのもの）
        Xtr = _numeric_frame(train, LEGACY_NUM, LEGACY_BOOL)
        Xte = _numeric_frame(test, LEGACY_NUM, LEGACY_BOOL)
        Ctr, Cte = _as_category(train, test, LEGACY_CAT)
        rough = LGBMRegressor(**LGBM_PARAMS)
        rough.fit(pd.concat([Xtr, Ctr], axis=1), train[TARGET],
                  categorical_feature=LEGACY_CAT)
        guess = rough.predict(pd.concat([Xte, Cte], axis=1))

        y_tr = train[TARGET].to_numpy(dtype=float)
        sim = V[test["_行番号"].to_numpy()] @ V[train["_行番号"].to_numpy()].T
        pred = np.empty(len(test))
        for r in range(len(test)):
            w = band
            for _ in range(3):
                m = np.abs(y_tr - guess[r]) <= w * max(guess[r], 1e-6)
                if m.sum() >= k:
                    break
                w *= 2
            cand = np.flatnonzero(m) if m.sum() >= k else np.arange(len(y_tr))
            top = cand[np.argsort(-sim[r, cand])[:k]]
            pred[r] = np.median(y_tr[top])
        return pred
    return fit_predict


# --- Q3 教師あり学習に渡したときの精度 ---------------------------------

def make_lgbm(V: np.ndarray):
    """従来列 + 埋め込み384次元（グレード名の列は使わない）＝ D2 相当。"""
    def fit_predict(train, test):
        Xtr = _numeric_frame(train, LEGACY_NUM, LEGACY_BOOL)
        Xte = _numeric_frame(test, LEGACY_NUM, LEGACY_BOOL)
        Ctr, Cte = _as_category(train, test, LEGACY_CAT)
        Xtr = pd.concat([Xtr, Ctr], axis=1)
        Xte = pd.concat([Xte, Cte], axis=1)
        names = [f"emb_{i}" for i in range(V.shape[1])]
        Etr = pd.DataFrame(V[train["_行番号"].to_numpy()], columns=names, index=Xtr.index)
        Ete = pd.DataFrame(V[test["_行番号"].to_numpy()], columns=names, index=Xte.index)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(pd.concat([Xtr, Etr], axis=1), train[TARGET],
                  categorical_feature=LEGACY_CAT)
        return model.predict(pd.concat([Xte, Ete], axis=1))
    return fit_predict


def show_neighbours(df: pd.DataFrame, embs: dict[str, np.ndarray],
                    texts: pd.DataFrame, n_show: int = 2, k: int = 3) -> None:
    """同じ基準車に対して、版ごとに引かれる近傍がどう変わるかを並べる。"""
    rng = np.random.default_rng(0)
    picks = rng.choice(len(df), n_show, replace=False)
    print("\n" + "=" * 78)
    print("同じ車の近傍が、版によってどう変わるか（定性確認）")
    print("=" * 78)
    for i in picks:
        r = df.iloc[i]
        print(f"\n■ 基準: {r['グレード名']:<10} {r[TARGET]:6.1f}万円  "
              f"{r['車齢']}年 {r['走行距離_km']:,.0f}km")
        print(f"   {r['タイトル'][:60]}")
        for name, V in embs.items():
            sim = V @ V[i]
            sim[i] = -1
            top = np.argsort(-sim)[:k]
            gap = float(np.mean(np.abs(df[TARGET].to_numpy()[top] - r[TARGET])))
            prices = " / ".join(f"{p:.0f}" for p in df[TARGET].to_numpy()[top])
            print(f"   [{name}] 近傍3台 {prices} 万円  平均価格差 {gap:5.1f} 万円")
    del texts


def main() -> None:
    df = load_dataset()
    df["_行番号"] = np.arange(len(df))
    embs = load_embeddings()
    grade = df["グレード名"].fillna("(不明)").to_numpy()
    numeric = df[["車齢", "走行距離_km"]].to_numpy(dtype=float)

    print("\n" + "=" * 78)
    print("Q1 グレードを分離できるか（同一グレードのペアと異グレードのペアの類似度）")
    print("=" * 78)
    rows = []
    for name, V in embs.items():
        m = grade_separation(V, grade)
        rows.append({"版": name, **m})
        print(f"  {name:<18} 同一 {m['同一グレード']:.4f}  異 {m['異グレード']:.4f}  "
              f"差 {m['差']:+.4f}  AUC {m['AUC']:.3f}")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "denoise_grade_separation.csv",
                              index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"Q2 近傍{K}件の価格中央値で予測する（機能B が渡す証拠の質）")
    print("=" * 78)
    for name, V in embs.items():
        label = f"kNN{K} タイトル埋め込み {name}"
        print(f"[{label}]")
        cross_validate(label, make_knn(V), df,
                       note="意味的近傍k件の価格中央値。機能B の証拠の質を測る")
        print()

    print("[kNN 距離に車齢・走行距離を混ぜる（対策案a）]")
    best = max(embs, key=lambda n: grade_separation(embs[n], grade)["AUC"])
    print(f"  分離が最も良かった版を使う: {best}")
    for w in (0.05, 0.15, 0.3):
        label = f"kNN{K} {best} + 数値距離 w={w}"
        cross_validate(label, make_knn(embs[best], numeric=numeric, w=w), df,
                       note="コサイン類似度から車齢・走行距離の標準化距離を引く")
        print()

    print("[kNN 価格帯で絞ってから意味的に並べる（対策案b）]")
    for name in ("V0 生（現状）", best):
        label = f"kNN{K} {name} + 価格帯±20%で事前絞り込み"
        print(f"  {label}")
        cross_validate(label, make_knn_banded(embs[name]), df,
                       note="構造化列だけの LightGBM で粗く予測し、その価格帯の事例から意味的に選ぶ")
        print()

    print("\n" + "=" * 78)
    print("Q3 教師あり学習（LightGBM）に渡したときの精度 — D2 相当・削りすぎの確認")
    print("=" * 78)
    for name, V in embs.items():
        label = f"D2' 従来列+タイトル埋め込み {name}"
        print(f"[{label}]")
        cross_validate(label, make_lgbm(V), df,
                       note="グレード名の列を使わない。生タイトル版 13.68 が基準")
        print()

    show_neighbours(df, embs, pd.DataFrame())


if __name__ == "__main__":
    main()
