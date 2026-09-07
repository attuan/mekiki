"""Craigslist（複数車種）の model を埋め込みベクトルに変換する。

**このスクリプトは隔離環境 .venv-embed で動かす。**（主環境には torch を入れない）

    bash scripts/setup_embed_env.sh                  # 初回だけ
    .venv-embed/bin/python scripts/embed_vehicles.py

シエンタの embed_text.py と同じモデル・同じ設定。違うのは対象列だけで、
シエンタの「タイトル（＝グレード名を含む生文）」にあたるのが
Craigslist の `model`（model 列）になる。

    'f-150 raptor arizona raptor*rust free*icon level kit*tech pkg*pano roof'
    '2500 slt / quad cab / 4x4 / leather / 5.9 l high output / cummins diesel'

19,739 種類の自由記述で、6割が1回しか出てこない。正規表現で芯を抜くのが
難しいのはここ。埋め込みならルールなしで扱えるかを測るための入力を作る。

**行ではなく「文字列の種類」ごとに埋め込む。** 20万行のうち model の種類は
19,739 しかないので、重複を潰してから計算すれば10分の1以下の時間で済む。
突き合わせは文字列そのものをキーにする。

出力: sampledata/processed/vehicles_emb_model_e5small.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "sampledata" / "processed" / "vehicles_multi_clean.parquet"
OUT = ROOT / "sampledata" / "processed" / "vehicles_emb_model_e5small.parquet"

MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "        # e5 系は接頭辞込みで学習されている
COL = "model"


def main() -> None:
    df = pd.read_parquet(SRC, columns=[COL])
    uniq = pd.Index(df[COL].fillna("").unique())
    print(f"入力: {SRC.name}  {len(df):,} 行 → 重複を潰して {len(uniq):,} 種類")

    print(f"モデル読み込み: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128

    t0 = time.time()
    vecs = model.encode(
        (PREFIX + uniq.to_series()).tolist(),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  {time.time() - t0:.1f} 秒 / 形 {vecs.shape}")

    out = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
    out.insert(0, COL, uniq)
    out.to_parquet(OUT, index=False)
    print(f"保存: {OUT.relative_to(ROOT)}（{OUT.stat().st_size / 1024**2:.1f} MB）")


if __name__ == "__main__":
    main()
