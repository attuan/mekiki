"""TabPFN（run_tabpfn_emb.py）と同じ入力で LightGBM を測る対照実験。

    .venv/bin/python scripts/run_lgbm_emb_match.py

**なぜ必要か。** 既存の D3（LightGBM・構造化列フル+タイトル埋め込み 12.378）は
**生のタイトル埋め込み384次元**を渡していた。一方 run_tabpfn_emb.py は
定数語を除いた版の埋め込みを PCA で圧縮して渡している。入力が違うので
そのままでは「TabPFN が勝った/負けた」と言えない。ここで LightGBM 側を
**まったく同じ入力**で引き直し、器の差だけを取り出す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_protocol import (  # noqa: E402
    DATA, EXTRA_CAT, EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT, LEGACY_NUM, SEED,
    TARGET, cross_validate, load_dataset,
)
from features import LGBM_PARAMS, as_category, numeric_frame  # noqa: E402

NUM, BOO, CAT = LEGACY_NUM + EXTRA_NUM, LEGACY_BOOL, LEGACY_CAT + EXTRA_CAT
EMB = ROOT / "sampledata" / "processed" / "usedsienta_emb_titlev2_e5small.parquet"
DIMS = (32, 64, 384)


def make(V: np.ndarray, dim: int):
    def fit_predict(train, test):
        Xtr, Xte = numeric_frame(train, NUM, BOO), numeric_frame(test, NUM, BOO)
        Ctr, Cte = as_category(train, test, CAT)
        Xtr, Xte = pd.concat([Xtr, Ctr], axis=1), pd.concat([Xte, Cte], axis=1)
        Etr, Ete = V[train["_行番号"].to_numpy()], V[test["_行番号"].to_numpy()]
        if dim < Etr.shape[1]:
            # PCA は訓練データの分布を使うので必ず fold の中で fit する
            pca = PCA(n_components=dim, random_state=SEED)
            Etr, Ete = pca.fit_transform(Etr), pca.transform(Ete)
        names = [f"pc{i}" for i in range(Etr.shape[1])]
        Xtr = pd.concat([Xtr, pd.DataFrame(Etr, columns=names, index=Xtr.index)], axis=1)
        Xte = pd.concat([Xte, pd.DataFrame(Ete, columns=names, index=Xte.index)], axis=1)
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(Xtr, train[TARGET], categorical_feature=CAT)
        return model.predict(Xte)
    return fit_predict


def main() -> None:
    keep = pd.read_parquet(DATA, columns=[TARGET])[TARGET].notna().to_numpy()
    V = pd.read_parquet(EMB).drop(columns=["行番号"]).to_numpy(dtype="float32")[keep]
    df = load_dataset()
    df["_行番号"] = np.arange(len(df))
    for dim in DIMS:
        name = f"D3' LightGBM・構造化列フル+タイトル埋め込み{dim}次元"
        print(f"[{name}]")
        cross_validate(name, make(V, dim), df,
                       note="TabPFN の G と同じ入力（定数語除去版の埋め込みを PCA 圧縮）")
        print()


if __name__ == "__main__":
    main()
