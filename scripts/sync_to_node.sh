#!/usr/bin/env bash
# 計算ノードへデータを移送する（移行時に一度だけ使う）。
#
#   bash scripts/sync_to_node.sh            # 実際に転送する
#   bash scripts/sync_to_node.sh --dry-run  # 何が送られるか見るだけ
#
# 前提: ~/.ssh/config に Host attuan が書かれていて `ssh attuan` で入れること。
#       ノード側に /work/carprice が clone されていること。
#
# なぜスクリプトにしたか。git 管理外のデータには、失うと困るものが2種類ある。
#
#   1. llm_cache — 置いていくと、同じ質問をもう一度 API から買うことになる。
#      6万行のバッチは $516 なので、消し忘れの代償が大きい。
#   2. *.parquet — 中間データは「コードで再生成できる」建前だが、
#      行の並びが1つでもずれると位置で引く load_dataset() が別の行を返し、
#      これまでの測定値と比較できなくなる（CLAUDE.md「中間データの並び順を変えない」）。
#      再生成に賭けるより、転送してチェックサムで一致を確かめる方が確実。
#
# raw/vehicles.csv は送らない。1.4GB あるうえ Kaggle から直接取り直せるので、
# ノード側で scripts/download_data.py を叩く方が速い。
set -euo pipefail

HOST="attuan"
REMOTE="/work/carprice"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

cd "$LOCAL"

echo "移送元: $LOCAL"
echo "移送先: $HOST:$REMOTE"
[[ -n "$DRY" ]] && echo "（--dry-run: 実際には転送しません）"
echo

# 相手側にディレクトリを用意する
if [[ -z "$DRY" ]]; then
  ssh "$HOST" "mkdir -p $REMOTE/sampledata/processed $REMOTE/sampledata/raw"
fi

# rsync の共通指定。-a で属性を保つ。
# macOS 標準の rsync は openrsync（protocol 29 / "2.6.9 compatible"）で、
# GNU rsync 3.x の --info や --partial を受け付けない。
# 両方で動くよう、使うフラグは -a と -e だけに絞る。
RS=(rsync -a -e ssh)
if rsync --version 2>&1 | head -1 | grep -q "^rsync  *version 3"; then
  RS+=(-z --partial --info=progress2)   # GNU rsync 3.x なら進捗と圧縮を使う
fi
[[ -n "$DRY" ]] && RS+=(--dry-run)

echo "=== 1/3 LLM キャッシュ（これを失うと課金され直す） ==="
"${RS[@]}" sampledata/processed/llm_cache/ \
           "$HOST:$REMOTE/sampledata/processed/llm_cache/"

echo
echo "=== 2/3 学習に使う中間データ（並び順を保つため転送する） ==="
# openrsync は --include/--exclude の重ね掛けが不安定なので、ファイルを直接並べる
PARQUETS=$(ls sampledata/processed/*_clean.parquet \
              sampledata/processed/*_emb_*.parquet \
              sampledata/processed/*_variants.parquet 2>/dev/null)
if [[ -n "$PARQUETS" ]]; then
  # shellcheck disable=SC2086
  "${RS[@]}" $PARQUETS "$HOST:$REMOTE/sampledata/processed/"
else
  echo "  対象の parquet がありません"
fi

echo
echo "=== 3/3 重複入りの版（再生成できるので任意・324MB） ==="
read -r -p "vehicles_multi_withdup.parquet も送りますか？ [y/N] " ans
if [[ "${ans:-N}" =~ ^[Yy]$ ]]; then
  "${RS[@]}" sampledata/processed/vehicles_multi_withdup.parquet \
             "$HOST:$REMOTE/sampledata/processed/"
else
  echo "  送りませんでした。必要になったら scripts/clean_vehicles.py で作れます。"
fi

if [[ -n "$DRY" ]]; then
  echo; echo "--dry-run のためチェックサム照合は省略します。"
  exit 0
fi

echo
echo "=== 照合（両側でハッシュを取って突き合わせる） ==="
LOCAL_SUM=$(find sampledata/processed -name '*.parquet' -o -path '*llm_cache*' -type f \
            | sort | xargs shasum -a 256 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
REMOTE_SUM=$(ssh "$HOST" "cd $REMOTE && find sampledata/processed -name '*.parquet' -o -path '*llm_cache*' -type f | sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1")
echo "  手元    : $LOCAL_SUM"
echo "  ノード  : $REMOTE_SUM"
echo
echo "※ 片方だけにファイルがあると当然ずれます。個別に比べるには:"
echo "   ssh $HOST 'cd $REMOTE && ls -la sampledata/processed/'"
