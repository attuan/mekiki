"""機能A の既定エンコーダの次元数を振る（ライブラリの既定値を決めるため）。

    .venv/bin/python scripts/run_encoder_sweep.py [--sample 60000]

`CharTfidfEncoder(n_components=256)` の 256 は、e5-small の 384 次元に合わせた
だけの暫定値で、根拠が無い。次元を落とせば速くなり、上げれば表現力が増えるが、
下流の MAE がどう動くかは測っていない。**ライブラリの既定値なので、
使う人全員に効く。**ここで決めておく。

条件は P1・P2 と同じ 60,000 行・5-fold。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from eval_protocol import VEHICLES, cross_validate, load_dataset  # noqa: E402
from features import make_lgbm  # noqa: E402
from run_baselines_vehicles import BOOL, CAT, N_SAMPLE, NUM, TEXT  # noqa: E402
from run_embedding_vehicles import breakdown  # noqa: E402

from unfold import CharTfidfEncoder, Feature  # noqa: E402

TARGET = VEHICLES.target
DIMS = [64, 128, 256, 512]


def extra_for(dim: int):
    def extra(train: pd.DataFrame, test: pd.DataFrame):
        f = Feature(source=TEXT, type="embedding", name=f"uf{dim}",
                    encoder=CharTfidfEncoder(n_components=dim))
        f.fit(train)
        return f.transform(train), f.transform(test)
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    args = ap.parse_args()

    df = load_dataset(dataset=VEHICLES, sample=args.sample)
    oof: dict[str, dict] = {}
    for dim in DIMS:
        name = f"F(c)-{dim} 機能A・埋め込み列（SVD{dim}次元）"
        print(f"[{name}]", flush=True)
        t0 = time.time()
        out: dict = {}
        cross_validate(name, make_lgbm(TARGET, NUM, BOOL, CAT, extra=extra_for(dim)),
                       df, note=f"既定エンコーダの次元数を振る（{len(df):,}行）",
                       dataset=VEHICLES, oof_out=out)
        oof[name] = out
        print(f"  ({time.time() - t0:.0f} 秒)\n", flush=True)

    tbl = breakdown(df, oof)
    print("=" * 78)
    print("次元数ごとの MAE（USD）")
    print("=" * 78)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    dst = ROOT / "results" / "encoder_dim_sweep.csv"
    tbl.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n保存: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
