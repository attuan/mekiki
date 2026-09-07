#!/usr/bin/env python3
"""`.env` の中身が Claude の文脈に入るのを止める PreToolUse フック。

APIキーを一度でも読み取ると、その値は会話に取り込まれて Anthropic の API に送られ、
ローカルの会話ログ（~/.claude/projects/）にも平文で残り、以降の往復でも
再送され続ける。キーの管理コストが跳ね上がるので、そもそも読ませない。

`permissions.deny` の `Read(./.env)` が Read ツール経由を塞ぐので、
こちらは Bash 経由（`cat .env` など）を塞ぐ担当。次の2つを拒否する。

  1. `.env` を指したうえで、中身を出力する系のコマンドを使っている
     （cat / grep / python など。`.env.example` は雛形なので対象外）
  2. キーそのものを表示しようとしている（`echo $ANTHROPIC_API_KEY` など）

`cp .env.example .env` や `git check-ignore .env` のように、中身を出さない
操作は通す。どうしても必要なときは CARPRICE_ALLOW_ENV_READ=1 を付けて実行する。

判定に迷ったら「通す」に倒す（フックの不具合で作業が止まるほうが困るため）。
標準ライブラリのみ / Python 3.8 以降で動く。
"""
import json
import os
import re
import sys

# 中身を標準出力に出しうるコマンド。ここに無い cp/mv/ls/touch などは通す
READERS = (
    "cat|bat|tac|nl|head|tail|less|more|strings|xxd|od|hexdump"
    "|grep|egrep|fgrep|rg|ag|ack|awk|sed|cut|tr|sort|uniq|paste|join"
    "|python|python3|node|ruby|perl|php|jq|source"
)
# `.env` そのものを指しているか。`.env.example` や `.venv` は含まない
ENV_FILE = re.compile(r"(?<![\w.\-])\.env(?![\w.\-])")
READER = re.compile(r"(?<![\w.\-])(" + READERS + r")(?![\w\-])")
# 値を直接表示しようとしている場合（ファイルを経由しない漏れ方）
KEY_VALUE = re.compile(
    r"\$\{?ANTHROPIC_API_KEY|\$\{?KAGGLE_API_TOKEN"
    r"|printenv\b[^|;&]*(ANTHROPIC|KAGGLE)"
    r"|\benv\b[^|;&]*\|[^|;&]*(ANTHROPIC|KAGGLE)")

GUIDANCE = (
    "\n\nAPIキーを読み取ると、その値は会話に取り込まれて API に送られ、"
    "\nローカルの会話ログにも平文で残ります。"
    "\n\n代わりに使えるもの:"
    "\n  - キーが使えるかの確認 -> .venv/bin/python scripts/check_api_key.py"
    "\n    （末尾4文字だけ表示し、書き間違いも指摘します）"
    "\n  - 雛形を見たいだけ -> cat .env.example"
    "\n  - それでも本当に中身を読む必要があるなら、"
    "\n    CARPRICE_ALLOW_ENV_READ=1 を付けて実行してください。"
)


def deny(reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    # JSON を解釈しないハーネスでも止まるよう、終了コード 2 でも拒否する
    sys.stderr.write(reason + "\n")
    sys.exit(2)


def main():
    if os.environ.get("CARPRICE_ALLOW_ENV_READ") == "1":
        sys.exit(0)
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    command = (payload.get("tool_input") or {}).get("command") or ""

    if KEY_VALUE.search(command):
        deny("APIキーの値を表示しようとしています。中止しました。" + GUIDANCE)
    if ENV_FILE.search(command) and READER.search(command):
        deny(".env の中身を読み出そうとしています。中止しました。" + GUIDANCE)
    sys.exit(0)


if __name__ == "__main__":
    main()
