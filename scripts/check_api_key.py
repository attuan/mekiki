"""`.env` に書いた APIキーが本当に使えるかを確かめる。

キーを貼ったら最初にこれを実行する。3段階で切り分ける:

1. キーが読めているか（どこから読めたか、末尾4文字だけ表示）
2. よくある書き間違いがないか（クォート付き・空白混入・雛形のまま等）
3. 実際に Claude を1回呼べるか（数円未満。キャッシュを通さず必ず本物を叩く）

使い方:
    .venv/bin/python scripts/check_api_key.py          # 3 まで実行
    .venv/bin/python scripts/check_api_key.py --no-call # 2 まで（課金なし）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unfold.llm import DEFAULT_MODEL, ClaudeClient, load_api_key  # noqa: E402

ENV_PATH = ROOT / ".env"


def mask(key: str) -> str:
    """ログに出しても安全な形にする。末尾4文字だけ残す。"""
    return f"{key[:10]}…{key[-4:]}（{len(key)}文字）" if len(key) > 18 else "（短すぎます）"


def raw_env_line() -> str | None:
    """`.env` に書かれた ANTHROPIC_API_KEY の行を、加工せずそのまま返す。"""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY"):
            return line
    return None


def check_format(key: str, line: str | None) -> list[str]:
    """よくある書き間違いを列挙する。空リストなら問題なし。"""
    problems: list[str] = []
    if line is not None:
        value = line.partition("=")[2]
        if value != value.strip():
            problems.append("= の直後や行末に空白があります。キーだけを詰めて書いてください")
        if value.strip()[:1] in {'"', "'"}:
            problems.append("クォートで囲む必要はありません（囲んでも読めますが不要）")
        if "sk-ant-api03-AbCd" in value:
            problems.append("雛形の例のままです。自分のキーに置き換えてください")
    if not key.startswith("sk-ant-"):
        problems.append("キーが sk-ant- で始まっていません。別のサービスのキーかもしれません")
    return problems


def main() -> int:
    print("1) キーを読めるか")
    key = load_api_key()
    if not key:
        print("   ✗ 見つかりません。")
        print(f"   {ENV_PATH} を開き、ANTHROPIC_API_KEY= の右にキーを貼って保存してください。")
        print("   キーの発行: https://console.anthropic.com/settings/keys")
        return 1

    source = "環境変数 ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY") else ".env"
    print(f"   ✓ {source} から読めました: {mask(key)}")
    if source.startswith("環境変数") and raw_env_line():
        print("   注意: 環境変数が .env より優先されます。.env を直しても効かない場合は")
        print("         `unset ANTHROPIC_API_KEY` してから実行してください。")

    print("2) 書き方の確認")
    problems = check_format(key, raw_env_line())
    for p in problems:
        print(f"   ! {p}")
    if not problems:
        print("   ✓ 問題なし")

    if "--no-call" in sys.argv:
        print("3) 実際の呼び出しは --no-call のため省略しました")
        return 0

    print(f"3) 実際に呼べるか（{DEFAULT_MODEL} に1回だけ質問します）")
    # cache_dir=None: ディスクキャッシュを通さず、必ず本物のリクエストを出す
    client = ClaudeClient(cache_dir=None, max_tokens=64)
    answer = client.ask(
        system="あなたは疎通確認用の応答器です。",
        user="疎通確認です。ok に true を入れて返してください。",
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"], "additionalProperties": False},
    )
    if not answer.ok:
        print(f"   ✗ 失敗: {answer.error}")
        if "authentication" in (answer.error or "").lower():
            print("   → キーが無効か、コピーが途中で切れています。作り直して貼り直してください。")
        return 1

    print(f"   ✓ 応答: {answer.data}")
    print(f"   今回の費用: ${answer.cost:.5f}（入力 {answer.input_tokens} / "
          f"出力 {answer.output_tokens} トークン）")
    print("\n準備完了です。unfold から Claude を呼べます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
