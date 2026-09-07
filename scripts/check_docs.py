#!/usr/bin/env python3
"""通しドキュメント3本（progress-log / related-work / README）の更新漏れを機械的に探す。

「内容が正しいか」は人（または LLM）にしか判定できないが、
**内容以前のズレ**は文字列の突き合わせで見つかる。ここで見るのはその4つだけ。

  A. 本数        docs/README.md と README.md に書いた「NN本」が実ファイル数と合っているか
  B. 索引漏れ    docs/2026-*.md が docs/README.md の表に載っているか（逆に、消えた行がないか）
  C. 記号        本文で使っている P / S / R が docs/README.md の表に定義されているか
  D. 鮮度        results/ や docs/2026-*.md より通しドキュメントのほうが古いコミットのままでないか

D だけは git の履歴を見る。測定を足したのに progress-log を直していない、が典型的な取りこぼしで、
これは日付の大小だけで検出できる。**内容の正しさは見ていない**ので、
ここが通っても「更新済み」の証明にはならない。判断の入口として使う。

使い方:

    python3 scripts/check_docs.py            # 指摘があれば表示して終了コード 1
    python3 scripts/check_docs.py --quiet    # 終了コードだけ使う（フックから呼ぶとき）

標準ライブラリのみ / Python 3.8 以降。venv の有効化に依存させない。
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 「通し」のドキュメント。日付がつかず、状況が動いたら直す必要があるもの。
# docs/PRD.md は入れない。人間が自分の手で書く文書で、更新の頻度も理由も別なので、
# ここに入れると「PRD だけ直した」ことで鮮度の検査（D）が満足してしまう。
STANDING = ["docs/progress-log.md", "docs/related-work.md",
            "docs/README.md", "README.md"]

# 通しドキュメントより新しくなっていたら「反映漏れかもしれない」と疑う対象
SOURCES = ["results/", "unfold/", "scripts/"]

# 記号（P / S / R）の定義元は docs/README.md の表。日付つきドキュメントが
# 本文で使っている番号が、そこに定義されているかを見る。
#
# 2026-09-01 に PRD を書き換えるまでは PRD が定義元だったが、書き換えで
# P / S / R の表そのものが PRD から無くなったため、docs/README.md に一本化した。
# なお旧実装は R の定義元を related-work.md にしていたが、あのファイルに
# R 番号は1つも出てこないため、この検査は書かれてから一度も動いていなかった。
SYMBOL_DEFS = {
    "P": r"^\| P(\d+(?:\.\d+)?) \|",
    "S": r"^\| S(\d) \|",
    "R": r"^\| R(\d) \|",
}
SYMBOL_USES = {
    "P": r"\bP(\d+(?:\.\d+)?)\b",
    "S": r"\bS(\d)\b",
    "R": r"\bR(\d)\b",
}


def git(*args):
    """git を読み取り専用で呼ぶ。失敗したら None（呼び出し側は「指摘しない」に倒す）。"""
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return None
    return p.stdout.decode("utf-8", "replace") if p.returncode == 0 else None


def last_commit_epoch(path):
    """path を最後に触ったコミットの時刻（epoch 秒）。履歴になければ None。"""
    out = git("log", "-1", "--format=%ct", "--", path)
    out = (out or "").strip()
    return int(out) if out.isdigit() else None


def dated_docs():
    return sorted(p.name for p in DOCS.glob("2026-*.md"))


def read(rel):
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def check_count(issues):
    """docs/*.md の本数（docs/README.md 自身は数えない）が、文中の「NN本」と合っているか。"""
    actual = len([p for p in DOCS.glob("*.md") if p.name != "README.md"])
    for rel in ("docs/README.md", "README.md"):
        for written in re.findall(r"(\d+)\s*本のドキュメント|ドキュメント[はが]?\s*(\d+)\s*本|"
                                  r"`docs/`\s*にあります（(\d+)本）", read(rel)):
            n = next((x for x in written if x), None)
            if n and int(n) != actual:
                issues.append(f"{rel}: ドキュメントの本数が {n} 本と書いてあるが、実際は {actual} 本")


def check_index(issues):
    """日付つきドキュメントが docs/README.md の表から漏れていないか（両方向）。"""
    index = read("docs/README.md")
    listed = set(re.findall(r"`(2026-[\w.-]+\.md)`", index))
    actual = set(dated_docs())
    for name in sorted(actual - listed):
        issues.append(f"docs/README.md: `{name}` が索引に載っていない")
    for name in sorted(listed - actual):
        issues.append(f"docs/README.md: `{name}` を索引に載せているが、ファイルが存在しない")


def check_symbols(issues):
    """本文で使っている P / S / R が docs/README.md の表に定義されているか。

    逆（定義したが誰も使っていない）は指摘しない。番号を先に決めてから
    後で測るのが通常の順序なので、未使用は正常な状態でありうる。
    """
    index = read("docs/README.md")
    users = ["docs/progress-log.md"] + [f"docs/{n}" for n in dated_docs()]
    for kind, def_re in SYMBOL_DEFS.items():
        defined = set(re.findall(def_re, index, re.MULTILINE))
        if not defined:
            issues.append(f"docs/README.md: {kind} の定義表が見つからない"
                          "（書式を変えたなら check_docs.py の SYMBOL_DEFS も直す）")
            continue
        for rel in users:
            used = set(re.findall(SYMBOL_USES[kind], read(rel)))
            for n in sorted(used - defined):
                issues.append(f"{rel}: {kind}{n} を使っているが、"
                              "docs/README.md に定義がない")


def check_freshness(issues):
    """通しドキュメントを最後に直して以降、素材だけが動いたコミットが積もっていないか。

    4本を1つのまとまりとして「最後にどれかを直した時刻」を基準にする。
    ドキュメントごとに個別に見ると、related-work.md のように
    めったに動かないファイルが毎回引っかかって警報が意味を失うため。
    """
    times = [t for t in (last_commit_epoch(rel) for rel in STANDING) if t is not None]
    if not times:
        return
    base = max(times)

    paths = SOURCES + [f"docs/{n}" for n in dated_docs()]
    out = git("log", "--format=%ct\t%h\t%s", "--", *paths)
    if out is None:
        return
    pending = []
    for line in out.strip().split("\n"):
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) > base:
            pending.append(f"{parts[1]} {parts[2]}")
    if pending:
        issues.append("通しドキュメントを最後に直してから、素材だけが動いたコミットが "
                      f"{len(pending)} 件ある: " + " / ".join(pending[:5]) +
                      (" ..." if len(pending) > 5 else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="終了コードだけ返す")
    args = ap.parse_args()

    issues = []
    for check in (check_count, check_index, check_symbols, check_freshness):
        try:
            check(issues)
        except Exception as e:  # 検査の不具合で作業を止めない
            if not args.quiet:
                print(f"（{check.__name__} は実行できなかった: {e}）", file=sys.stderr)

    if not args.quiet:
        if issues:
            print("ドキュメントの更新漏れの疑い:")
            for m in issues:
                print(f"  - {m}")
            print("\n直すときは /update-docs を使う（判断が要る書き換えはスキル側の手順に従う）。")
        else:
            print("通しドキュメントに、機械的に見つかるズレはない。")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
