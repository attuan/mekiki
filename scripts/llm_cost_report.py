"""これまでの Claude API 費用を、応答キャッシュから集計して CSV に落とす。

## 何をしているか

`unfold/llm.py` は API の応答を `sampledata/processed/llm_cache/` に
1呼び出し＝1ファイルで保存している。そこにトークン数が入っているので、
**課金額は後から再計算できる**。このスクリプトはそれを1行1呼び出しの
CSV にまとめ、合計を表示する。

## この数字の限界（読むときに必ず意識すること）

1. **請求額そのものではなく再計算値。** 単価は `unfold/llm.py` の
   `PRICING` を使う。単価が改定されると過去ぶんの金額も変わってしまう。
2. **キャッシュ作成ぶんが記録に無い。** キャッシュ書き込み時の
   `cache_creation_input_tokens` を保存していないので、実測はここより
   わずかに高い（数セント）。
3. **モデル名を記録していない。** どのモデルで呼んだかがキャッシュに
   残らないので `--model` で指定する（既定は `claude-opus-5`）。
   複数モデルを混ぜて使った場合は分離できない。
4. **日時はファイルの更新時刻。** キャッシュを別マシンにコピーしたり
   バックアップから戻したりすると失われる。
5. **キャッシュを消すと履歴も消える。** `sampledata/processed/` は
   git 管理外で「再生成できる中間データ」の置き場だが、ここだけは
   再生成に**実費がかかる**。消す前にこの CSV を出しておくこと。

## 実行

    .venv/bin/python scripts/llm_cost_report.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unfold.llm import DEFAULT_CACHE_DIR, DEFAULT_MODEL, PRICING  # noqa: E402

#: キャッシュにはモデル名も呼び出し元も残らないので、返ってきた JSON の
#: キーの顔ぶれで用途を見分ける。機能を足したらここに1行足す。
KINDS: dict[frozenset[str], str] = {
    frozenset({"price", "confidence", "reason"}): "機能B（価格予測）",
    frozenset({"value", "confidence", "reason"}): "機能A（フォールバック）",
}


def classify(data: dict) -> str:
    return KINDS.get(frozenset(data.keys()), "不明")


def collect(cache_dir: Path, model: str) -> pd.DataFrame:
    pin, pout = PRICING.get(model, PRICING[DEFAULT_MODEL])
    rows = []
    for path in sorted(cache_dir.rglob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # 書き込み中に落ちた残骸は数えない
        i = int(rec.get("input_tokens", 0))
        o = int(rec.get("output_tokens", 0))
        c = int(rec.get("cache_read_tokens", 0))
        rows.append({
            "日時": dt.datetime.fromtimestamp(path.stat().st_mtime)
                    .strftime("%Y-%m-%d %H:%M:%S"),
            "用途": classify(rec.get("data", {})),
            "モデル": model,
            "入力トークン": i,
            "出力トークン": o,
            "キャッシュ読出トークン": c,
            # キャッシュ読み出しは通常の 1/10 単価
            "費用_usd": round((i * pin + c * pin * 0.1 + o * pout) / 1_000_000, 6),
            "キー": path.stem[:12],
        })
    return pd.DataFrame(rows).sort_values("日時", ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM API 費用の集計")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="キャッシュに残らないので手で指定する")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_cost.csv"))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"キャッシュがありません: {cache_dir}")
        raise SystemExit(1)

    df = collect(cache_dir, args.model)
    if df.empty:
        print("呼び出しの記録がありません。")
        raise SystemExit(0)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"呼び出し {len(df):,} 件 / 合計 ${df['費用_usd'].sum():.4f}"
          f" / 1件あたり ${df['費用_usd'].mean():.5f}")
    print("\n--- 用途別 ---")
    by_kind = df.groupby("用途").agg(件数=("費用_usd", "size"),
                                     費用_usd=("費用_usd", "sum"))
    print(by_kind.round(4).to_string())
    print("\n--- 日別 ---")
    by_day = df.assign(日=df["日時"].str[:10]).groupby("日").agg(
        件数=("費用_usd", "size"), 費用_usd=("費用_usd", "sum"))
    print(by_day.round(4).to_string())
    print(f"\n明細: {out.relative_to(ROOT)}")
    print("※ 請求額ではなく再計算値。詳しくは冒頭の docstring を読むこと。")


if __name__ == "__main__":
    main()
