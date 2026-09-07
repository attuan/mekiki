"""自由記述に正解（価格）が漏れていないかを検査する。

## やり方

**採点した行を「description に価格そのものが書いてある行」と「書いていない行」に
分けて MAE を出す。** 片方だけ極端に当たっていたら、予測ではなく読み取りである。

これは中古車に限らず使える一般の手順である。目的変数が漏れていそうな行と
そうでない行に分け、改善幅が一様かどうかを見るだけでよい。

  一様に改善  → 本物の効果
  片方だけ改善 → リーク

## 実行

    .venv/bin/python scripts/check_description_leak.py

`results/description_leak.csv` に落ちる。
`notebooks/04_llm_results.ipynb` がそれを読む。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_protocol import N_SPLITS, SEED, VEHICLES, load_dataset  # noqa: E402

N_EVAL = 120

# 比較する3構成。ファイル名と表示名。
CONFIGS = [
    ("description なし", "llm_predictor_vehicles_rows.csv"),
    ("description あり・伏字なし【無効】", "llm_predictor_vehicles_desc2000_LEAKED_rows.csv"),
    ("description あり・伏字あり", "llm_predictor_vehicles_desc2000_rows.csv"),
]


def selected_rows() -> pd.DataFrame:
    """測定と同じ分割・同じ抽出で、採点した600行を復元する。

    `run_llm_predictor.run_eval` と同じ seed・同じ手順を踏むこと。
    ここがずれると別の行の description を見ることになる。
    """
    df = load_dataset(verbose=False, dataset=VEHICLES, sample=60_000)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rng = np.random.default_rng(SEED)
    picked = []
    for _, te in kf.split(df):
        ta = df.iloc[te].reset_index(drop=True)
        pick = np.sort(rng.choice(len(ta), size=N_EVAL, replace=False))
        picked.append(ta.iloc[pick].reset_index(drop=True))
    return pd.concat(picked, ignore_index=True)


def price_is_written(text: str, price: float) -> bool:
    """description に、その車の価格と完全一致する数字が書かれているか。"""
    if price <= 0:
        return False
    for n in re.findall(r"[\d][\d,]{2,}", str(text)):
        try:
            if int(n.replace(",", "")) == int(price):
                return True
        except ValueError:
            pass
    return False


def main() -> None:
    sel = selected_rows()
    y = sel["price"].to_numpy(dtype=float)
    leak = np.array([price_is_written(t, p) for t, p in zip(sel["description"], y)])
    print(f"採点した {len(y)} 行のうち、description に価格そのものが書かれている行: "
          f"{leak.sum()} 行（{leak.mean():.1%}）\n")

    rows = []
    for label, fname in CONFIGS:
        path = ROOT / "results" / fname
        if not path.exists():
            print(f"  （{fname} が無いので飛ばします）")
            continue
        d = pd.read_csv(path, encoding="utf-8-sig")
        if not np.allclose(d["実際"].to_numpy(float), y):
            raise SystemExit(f"{fname} の行が対応していません。"
                             "--n-eval や seed が測定時と違う可能性があります。")
        err = np.abs(d["機能B"].to_numpy(float) - y)
        rows.append({
            "構成": label,
            "全600行": round(float(err.mean()), 2),
            "価格記載あり": round(float(err[leak].mean()), 2),
            "記載なし": round(float(err[~leak].mean()), 2),
            "記載あり行数": int(leak.sum()),
            "記載なし行数": int((~leak).sum()),
        })

    res = pd.DataFrame(rows)
    base = res.iloc[0]
    res["記載ありの改善率"] = (base["価格記載あり"] - res["価格記載あり"]) / base["価格記載あり"]
    res["記載なしの改善率"] = (base["記載なし"] - res["記載なし"]) / base["記載なし"]
    print(res.round(3).to_string(index=False))

    print("\n読み方:")
    print("  改善率が **記載あり側だけ極端に大きい** → 読み取り（リーク）")
    print("  改善率が **両側でほぼ同じ**           → 本物の効果")

    dst = ROOT / "results" / "description_leak.csv"
    dst.parent.mkdir(exist_ok=True)
    res.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n結果: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
