"""キャッシュに保存されたプロンプトと応答を、人が読める形で表示する。

## なぜ要るか

`sampledata/processed/llm_cache/` のファイル名は
`sha256(model, effort, system, user, schema)` なので**逆に戻せない**。
そのため中身を見ないと「どの行の、何を聞いた結果か」が分からない。
`unfold/llm.py` が user プロンプトを一緒に保存するようにしてあるので、
ここではそれを整形して出すだけ。**API は呼ばない（無料）。**

`system_sha` は system プロンプト本文の指紋（sha256 の先頭8桁）。
本文はソース側（`unfold/predictor.py` の `SYSTEM_PROMPT`、
`unfold/fallback.py` の `CLASSIFY_SYSTEM`）にあるので、突き合わせて
「どの版のプロンプトで得た答えか」を判定する。

## 実行

    .venv/bin/python scripts/show_prompt.py                # 最新のものを1件
    .venv/bin/python scripts/show_prompt.py --n 3          # 最新3件
    .venv/bin/python scripts/show_prompt.py <ファイル名かキーの先頭>
    .venv/bin/python scripts/show_prompt.py --list         # プロンプト入りの件数だけ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "sampledata" / "processed" / "llm_cache"


def known_systems() -> dict[str, str]:
    """ソースにある system プロンプトの指紋 → 名前。"""
    from unfold.fallback import CLASSIFY_SYSTEM
    from unfold.predictor import SYSTEM_PROMPT

    out = {}
    for name, text in [("機能B（価格予測）", SYSTEM_PROMPT),
                       ("機能A（分類フォールバック）", CLASSIFY_SYSTEM)]:
        out[hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]] = name
    return out


def show(path: Path, systems: dict[str, str]) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    sha = rec.get("system_sha")
    print("=" * 78)
    print(f"ファイル   {path.relative_to(ROOT)}")
    print(f"system     {sha or '（未保存）'}"
          f"  {systems.get(sha, '— ソースのどれとも一致しません（版が変わった）')}")
    print(f"トークン   入力 {rec.get('input_tokens', 0):,}"
          f" / 出力 {rec.get('output_tokens', 0):,}"
          f" / キャッシュ読出 {rec.get('cache_read_tokens', 0):,}")
    print("-" * 78)
    user = rec.get("user")
    if user is None:
        print("user プロンプトは保存されていません（この機能より前に作られたファイル）。")
    else:
        print("【投げた user プロンプト】\n")
        print(user)
    print("-" * 78)
    print("【返ってきた応答】\n")
    print(json.dumps(rec.get("data", {}), ensure_ascii=False, indent=2))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="キャッシュの中身を見る")
    ap.add_argument("key", nargs="?", help="ファイル名またはキーの先頭数文字")
    ap.add_argument("--n", type=int, default=1, help="新しい順に何件出すか")
    ap.add_argument("--list", action="store_true",
                    help="件数の内訳だけ出す")
    args = ap.parse_args()

    files = sorted(CACHE.rglob("*.json"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not files:
        raise SystemExit(f"{CACHE} が空です。")

    if args.list:
        withp = [f for f in files
                 if "user" in json.loads(f.read_text(encoding="utf-8"))]
        print(f"キャッシュ全体          {len(files):,} 件")
        print(f"プロンプト保存あり      {len(withp):,} 件")
        print(f"プロンプト保存なし（旧） {len(files) - len(withp):,} 件")
        return

    systems = known_systems()
    if args.key:
        hit = [f for f in files if f.stem.startswith(args.key.split(".")[0])]
        if not hit:
            raise SystemExit(f"{args.key} で始まるキャッシュがありません。")
        show(hit[0], systems)
        return

    for f in files[:args.n]:
        show(f, systems)


if __name__ == "__main__":
    main()
