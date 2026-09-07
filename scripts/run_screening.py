"""事前スクリーニング（`unfold.screen`）を3つの列に対して回し、CSV に落とす。

**LLM を呼ばないので無料。** ただし文字 n-gram TF-IDF を 20,000 行で回すので
数分かかる（Craigslist の description が特に重い）。結果は `results/screening.csv` に入り、
`notebooks/04_llm_results.ipynb` がそれを読む。

問いは1つ。**そのデータでテキストが価格を説明しているか。**
機能B は「文字 TF-IDF より LLM のほうがうまくテキストを読める」ぶんだけ
上積みする仕組みなので、TF-IDF で効いていない場所では上積みも起きない、
という前提を確かめている。

実行:
    .venv/bin/python scripts/run_screening.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import (  # noqa: E402
    SIENTA, VEHICLES, VEHICLES_CAT, VEHICLES_LONG_TEXT, VEHICLES_NUM,
    VEHICLES_TEXT, load_dataset,
)

from unfold import screen  # noqa: E402

warnings.filterwarnings("ignore")




def main() -> None:
    rows = []

    print("[1/3] シエンタ・装備テキスト")
    d = load_dataset(verbose=False, dataset=SIENTA)
    r = screen(d, target="車両本体価格_万円", text="装備テキスト", unit="万円",
               numeric=["車齢", "走行距離_km", "車検残月数", "装備数"],
               boolean=["修復歴あり", "保証付", "ハイブリッド"],
               categorical=["車検区分", "都道府県", "グレード名", "色_基本"])
    print(r, "\n")
    rows.append(("シエンタ", "装備テキスト", r))

    v = load_dataset(verbose=False, dataset=VEHICLES, sample=20_000)
    for i, col in enumerate([VEHICLES_TEXT, VEHICLES_LONG_TEXT], start=2):
        print(f"[{i}/3] Craigslist・{col}")
        r = screen(v, target=VEHICLES.target, text=col, unit="USD", sample=None,
                   numeric=VEHICLES_NUM, categorical=VEHICLES_CAT)
        print(r, "\n")
        rows.append(("Craigslist", col, r))

    out = pd.DataFrame([{
        "データ": ds, "列": col, "行数": r.n_rows, "単位": r.unit,
        "値の種類": r.n_unique_text, "平均文字数": round(r.mean_text_chars, 1),
        "テキスト無しMAE": round(r.mae_without_text, 2),
        "テキスト有りMAE": round(r.mae_with_text, 2),
        "テキスト寄与率": round(r.text_contribution, 4),
        "判定": r.verdict,
    } for ds, col, r in rows])

    dst = ROOT / "results" / "screening.csv"
    dst.parent.mkdir(exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n結果: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
