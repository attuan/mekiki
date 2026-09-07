#!/usr/bin/env python3
"""測定や実装だけをコミットしようとしたときに、通しドキュメントの更新を促す PreToolUse フック。

`results/` に数字を足したり `unfold/` を直したりしたのに、
`docs/progress-log.md` などを直さないままコミットすると、記録と実物が静かにズレる。
CLAUDE.md の「作業が一段落するたびに progress-log に足す」という運用を、
お願いではなく機械的に思い出させる。

Bash ツールの実行直前に呼ばれ、コマンドが `git commit` なら
入りうるファイルの内訳を見て判定する。

  素材（results/ など）が入っていて、通しドキュメントが1本も入っていない -> 確認を求める
  それ以外                                                                -> 何もしない

**拒否はしない。** 測定を先にコミットして記録を後で書く進め方は普通にあるので、
止めるのではなく `/update-docs` を思い出させるだけにとどめる。

判定に失敗したときは必ず「通す」側に倒す。フックの不具合で作業が止まるほうが困るため。
コマンドの解析は隣の block_large_git_add.py を使い回す（同じ解析を2つ持たない）。
標準ライブラリのみ / Python 3.8 以降で動く。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from block_large_git_add import has_flag, lines, parse_git, run, split_segments
except Exception:  # 解析部品が読めないなら黙って通す
    sys.exit(0)

# 状況が動いたら書き直す必要がある「通し」のドキュメント。
# docs/PRD.md は入れない。人間が自分の手で書く文書なので、
# 「PRD を直したから測定の反映も済んでいる」とは言えない。
STANDING = {"docs/progress-log.md", "docs/related-work.md",
            "docs/README.md", "README.md"}

# これらが動いたなら、通しドキュメントのどこかに反映されるはず
SOURCE_PREFIXES = ("results/", "unfold/", "scripts/")


def is_source(path):
    if path.startswith(SOURCE_PREFIXES):
        return True
    # 日付つきの測定記録も素材。単体では索引・進捗ログへの反映が要る
    return path.startswith("docs/2026-") and path.endswith(".md")


def committed_paths(command, cwd):
    """このコマンドでコミットされうるパス。git commit が含まれないなら空。"""
    found = []
    for tokens in split_segments(command):
        parsed = parse_git(tokens)
        if parsed is None or parsed[0] != "commit":
            continue
        args = parsed[1]
        found += lines(run(["git", "diff", "--cached", "--name-only"], cwd))
        if has_flag(args, "--all", "a"):
            found += lines(run(["git", "ls-files", "-m", "--exclude-standard"], cwd))
    return sorted(set(found))


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if payload.get("tool_name") != "Bash":
        sys.exit(0)
    command = (payload.get("tool_input") or {}).get("command") or ""
    if "commit" not in command:
        sys.exit(0)

    cwd = payload.get("cwd") or os.getcwd()
    try:
        paths = committed_paths(command, cwd)
    except Exception:
        sys.exit(0)
    if not paths:
        sys.exit(0)

    sources = [p for p in paths if is_source(p)]
    if not sources or any(p in STANDING for p in paths):
        sys.exit(0)

    listed = "、".join(sources[:5]) + (" ほか" if len(sources) > 5 else "")
    reason = (
        f"測定・実装だけをコミットしようとしている（{listed}）。\n"
        "通しドキュメント（docs/progress-log.md・docs/related-work.md・"
        "README.md）は1本も含まれていない。\n"
        "反映が要るなら /update-docs で先に直す。要らない（作業途中・記録は後でまとめる）なら"
        "このまま進めてよい。"
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
