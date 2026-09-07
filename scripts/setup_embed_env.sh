#!/usr/bin/env bash
# 埋め込み計算と TabPFN 専用の隔離環境を作る。
#
# なぜ分けるのか（2026/9/1 に理由が入れ替わった）:
#   もとは依存の衝突を避けるためだった。Intel Mac の torch は 2.2.2 が上限で、
#   numpy 1.x ビルドだったので主環境の numpy 2.x と共存できず、入れると
#   pandas / LightGBM ごと壊れた。計算ノード（Ubuntu / x86_64）ではこの制約が無い。
#
#   それでも分けたままにしているのは、**unfold が torch 無しで動くことの番人**だから。
#   主環境に torch を入れると、ライブラリが誤って torch に依存してもテストが通ってしまう。
#   詳しくは docs/2026-09-01-embed-env-rebuild.md。
#
# 使い方:
#   bash scripts/setup_embed_env.sh                     # 環境を作る
#   .venv-embed/bin/python scripts/embed_text.py        # 埋め込みを計算する
#   .venv/bin/python scripts/run_embedding.py           # 主環境で精度を測る
#
# 埋め込みは parquet に落として受け渡すので、主環境は torch に一切触らない。
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -d .venv-embed ]; then
  echo ".venv-embed は既にあります。作り直すなら rm -rf .venv-embed してから実行してください。"
  exit 0
fi

PY312=$(command -v python3.12 || true)
[ -x "$PY312" ] || PY312=/usr/local/opt/python@3.12/bin/python3.12   # Homebrew（macOS）
[ -x "$PY312" ] || { echo "python3.12 が見つかりません"; exit 1; }

"$PY312" -m venv .venv-embed
.venv-embed/bin/pip install --upgrade pip -q

# torch は CPU 版インデックスから入れる。ノードに GPU は無いので、既定の PyPI から
# 入れると使わない CUDA の wheel が19個（数GB）ぶら下がってくるだけ損になる。
.venv-embed/bin/pip install -q \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements-embed.txt

echo "--- 確認 ---"
.venv-embed/bin/python - <<'PY'
import numpy, torch, transformers, sentence_transformers as st, tabpfn
# transformers / sentence-transformers は固定していない（tabpfn 2.2.1 の
# huggingface-hub<1 制約に合わせて resolver が選ぶ）。選ばれた版は記録に残すこと。
print("numpy       :", numpy.__version__)
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("st          :", st.__version__)
print("tabpfn      :", tabpfn.__version__)
PY
