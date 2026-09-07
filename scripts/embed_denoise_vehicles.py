"""Craigslist の model から、ノイズを除いた版を作って埋め込む（.venv-embed で実行）。

    .venv-embed/bin/python scripts/embed_denoise_vehicles.py

シエンタでやった埋め込み前のノイズ除去（embed_denoise.py）を、複数車種データに
そのまま持ち込めるかを見る。**同じ3つの汎用ルールを、区切り記号だけ書式に
合わせて差し替える。**車種固有の知識を足さずに済むなら、機能A の前処理として
ライブラリに入れられる、という判断材料になる。

  V1 既出語の除去   … メーカー・州・燃料・駆動・変速機など、別の列にすでにある値を消す
                      （'ford f-150 4x4 diesel' の ford / 4x4 / diesel は列にもある）
  V3 先頭区切りまで … 'f-150 raptor arizona raptor*rust free*icon level kit' のように
                      "*" や "/" 以降は宣伝文なので落とす
  V13 両方

V2（定数語の除去）は入れない。model は 19,739 種類の自由記述で、
出現率 90% を超えるトークンが存在しないため空振りになる（実行前に確認済み）。

行ではなく文字列の種類ごとに埋め込む点は embed_vehicles.py と同じ。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import text_variants as tv  # noqa: E402

SRC = ROOT / "sampledata" / "processed" / "vehicles_multi_clean.parquet"
OUT_DIR = ROOT / "sampledata" / "processed"
MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "
COL = "model"
# 別の構造化列にすでに入っている情報。ここに挙げた列の値は model から消す
KNOWN = ["manufacturer", "state", "fuel", "drive", "transmission", "size", "type", "paint_color", "cylinders"]
# Craigslist の model は "*" と "/" で宣伝文をつなぐ
BREAK_CHARS = "*/|,;+"


def main() -> None:
    df = pd.read_parquet(SRC, columns=[COL] + KNOWN)
    # 「model × 他の列の値」の組み合わせ単位で作る必要があるので、
    # 重複潰しは (model, 既出語) の組で行う
    key = df[[COL] + KNOWN].fillna("").astype(str)
    uniq = key.drop_duplicates().reset_index(drop=True)
    print(f"入力: {SRC.name}  {len(df):,} 行 → 組み合わせ {len(uniq):,} 種類")

    raw = uniq[COL]
    known_cols = [uniq[c] for c in KNOWN]
    v1 = tv.v1_drop_known(raw, known_cols)
    v3 = tv.v3_head_segment(raw, break_chars=BREAK_CHARS)
    v13 = tv.v1_drop_known(v3, known_cols)
    variants = {"v1": v1, "v3": v3, "v13": v13}

    for tag, s in variants.items():
        print(f"  {tag}: 平均 {s.str.len().mean():5.1f} 文字（生 {raw.str.len().mean():.1f}）"
              f"  空文字 {(s.str.len() == 0).sum():,}  種類 {s.nunique():,}")

    print(f"\nモデル読み込み: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128

    for tag, s in variants.items():
        # 版ごとに文字列がさらに重複するので、もう一度潰してから埋め込む
        texts = pd.Index(s.unique())
        print(f"\n[{tag}] {len(texts):,} 種類を埋め込み中")
        t0 = time.time()
        vecs = model.encode((PREFIX + texts.to_series()).tolist(), batch_size=64,
                            normalize_embeddings=True, show_progress_bar=True)
        print(f"  {time.time() - t0:.1f} 秒 / 形 {vecs.shape}")
        out = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
        out.insert(0, "テキスト", texts)
        dst = OUT_DIR / f"vehicles_emb_model{tag}_e5small.parquet"
        out.to_parquet(dst, index=False)
        print(f"  保存: {dst.relative_to(ROOT)}")

    # 行 → 版ごとのテキスト の対応表。評価側はこれを使って埋め込みを引く
    uniq_out = uniq.copy()
    for tag, s in variants.items():
        uniq_out[tag] = s.to_numpy()
    uniq_out.to_parquet(OUT_DIR / "vehicles_model_variants.parquet", index=False)
    print("\n保存: sampledata/processed/vehicles_model_variants.parquet")


if __name__ == "__main__":
    main()
