"""ノイズを除いたタイトルを埋め込む（隔離環境 .venv-embed で実行）。

    .venv-embed/bin/python scripts/embed_denoise.py

背景: タイトルの生文字列をそのまま埋め込むと、類似度が「価格の近さ」ではなく
「書式の一致」に反応する（`docs/2026-08-29-denoise.md`）。原因はコサイン類似度が文全体の
向きを見るため、価格を決める短い部分（グレード名）が装備の羅列に埋もれること。
そこで**入力テキストを削るだけで改善するか**を測る。

版の作り方は text_variants.py に置いてある（車種に依存しない汎用ルールのみ）。
V0（生）は既存の usedsienta_emb_title_e5small.parquet をそのまま使うので、
ここでは V1〜V23 の4本だけを作る。モデル・接頭辞・max_seq_length は
embed_text.py と完全に同じにする（比較の条件を揃えるため）。
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

SRC = ROOT / "sampledata" / "processed" / "usedsienta_clean.parquet"
OUT_DIR = ROOT / "sampledata" / "processed"
MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "
DIM_TAG = "e5small"


def build_variants(df: pd.DataFrame) -> dict[str, pd.Series]:
    """4つの版を作る。V0（生）は既存ファイルを使うのでここには含めない。"""
    raw = df["タイトル"].fillna("")
    known = [df["店舗"], df["都道府県"], df["車名"]]

    v1 = tv.v1_drop_known(raw, known)
    v2, stop = tv.v2_drop_constant(raw)
    # V3 は「最初の区切りまで」を先に取る。V1 を先にかけると全角スペースが
    # 半角に潰れて区切りが消えてしまうので順番が重要
    v3 = tv.v1_drop_known(tv.v3_head_segment(raw), known)
    v23, _ = tv.v2_drop_constant(v3)
    print(f"V2 が消した定数語（出現率 90% 以上）: {stop}")
    return {"v1": v1, "v2": v2, "v3": v3, "v23": v23}


def main() -> None:
    df = pd.read_parquet(SRC)
    print(f"入力: {SRC.name}  {len(df):,} 行")
    variants = build_variants(df)
    for tag, s in variants.items():
        print(f"  {tag}: 平均 {s.str.len().mean():5.1f} 文字  例 {s.iloc[0][:50]!r}")

    print(f"\nモデル読み込み: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 128

    for tag, s in variants.items():
        texts = (PREFIX + s).tolist()
        print(f"\n[{tag}] {len(texts):,} 件を埋め込み中")
        t0 = time.time()
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                            show_progress_bar=True)
        print(f"  {time.time() - t0:.1f} 秒 / 形 {vecs.shape}")
        out = pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(vecs.shape[1])])
        out.insert(0, "行番号", range(len(out)))
        dst = OUT_DIR / f"usedsienta_emb_title{tag}_{DIM_TAG}.parquet"
        out.to_parquet(dst, index=False)
        print(f"  保存: {dst.relative_to(ROOT)}")

    # 版ごとのテキストも保存する（近傍を目で確認するときに使う）
    txt = pd.DataFrame({"行番号": range(len(df)), "v0": df["タイトル"].fillna("")})
    for tag, s in variants.items():
        txt[tag] = s.to_numpy()
    txt.to_parquet(OUT_DIR / "usedsienta_title_variants.parquet", index=False)
    print(f"\n保存: sampledata/processed/usedsienta_title_variants.parquet")


if __name__ == "__main__":
    main()
