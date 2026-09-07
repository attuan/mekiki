"""機能A の宣言値（`type` の13水準）を埋め込みベクトルに変換する。

**このスクリプトは隔離環境 .venv-embed で動かす。**（主環境には torch を入れない）

    .venv-embed/bin/python scripts/embed_type_values.py

なぜ要るのか。機能A の案b（値の名前だけを起点にする）は、`values=[...]` に
書いた値そのものを参照事例として埋め込む。`--encoder e5` で測るときの
`PrecomputedEncoder` は**計算済みの対応表にある文字列しか扱えない**ので、
model の19,739種類ぶん（`embed_vehicles.py`）とは別に、この13語ぶんが要る。

model 側と同じモデル・同じ接頭辞で計算する（同じ空間に載らないと近傍を取れない）。
列名も `model` に揃えてあるのは、`run_feature_fallback.py` が2つの表を
そのまま縦に連結して1つの対応表として引くため。

出力: sampledata/processed/vehicles_emb_typevalues_e5small.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from type_values import TYPE_VALUES  # noqa: E402

OUT = ROOT / "sampledata" / "processed" / "vehicles_emb_typevalues_e5small.parquet"

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "        # e5 系は接頭辞込みで学習されている。model 側と揃える
COL = "model"             # 連結先（vehicles_emb_model_e5small.parquet）に合わせる


def main() -> None:
    print(f"宣言値 {len(TYPE_VALUES)} 語: {TYPE_VALUES}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128
    vecs = model.encode([PREFIX + v for v in TYPE_VALUES],
                        batch_size=16, normalize_embeddings=True)
    out = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
    out.insert(0, COL, TYPE_VALUES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"出力: {OUT.relative_to(ROOT)}  形 {vecs.shape}")


if __name__ == "__main__":
    main()
