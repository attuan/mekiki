# carprice

株式会社INDX のインターン（2026/8/28〜）で作成する、中古車価格の自動予測プログラム。
LLM を使った AutoMLライブラリを構成し、特徴量生成と予測を自動化することが目標。

## 開発アプローチ（構想の変遷）

「何を作るか」は3段階で変わってきた。**現在地は 3.（伊藤さんの仕様書）**。
今後のミーティングにより方針が変わる可能性があるが、古い資料を読むときはどの時点のものかを意識すること。

### 1. エントリーシート時点 — 中古車推薦ツール

カーセンサーからスクレイピング → 回帰分析 → オススメ中古車のランキングを出力する、
学生向けの「アプリ」構想。エンドユーザーがいる完成品を想定していて、
LLM はまだ主役ではなかった（→ `dialogs/entrysheet.md`）。

### 2. 8/12 ミーティング時点 — LLM 活用の3方式

中古車価格予測という題材は据え置きで、LLM をどこに差すかで3方式を並べ、
まず全部試して比較する方針になった（→ `dialogs/2026...ミーティング.md`）。

1. LLM で直接価格を予測する
2. LLM でモデル選択を自動化する（統計モデルは従来通り）
3. LLM で特徴量生成を自動化する

背景: 従来の回帰分析では定性情報をダミー変数でしか扱えず精度に限界があった。
そこを LLM で埋められるかが本プロジェクトの主題。この時点ではどれをメインにするか未決定で、
**同じ指標で精度を記録して比較する**ことが作業の中心だった。

### 3. 伊藤さんの仕様書（unfold）時点 — 2機能に統合【現在地】

3方式が **機能A（特徴量生成）** と **機能B（LLM Predict）** の2つに畳まれ、
scikit-learn 互換の API（`fit` / `transform` / `predict` / `predict_proba`）を持つ
Python ライブラリ `unfold` として設計し直された（→ `dialogs/unfold-landing.html`）。

3方式からの変更点は次のとおり。

- **方式3 → 機能A（`Feature`）。** 画像・自由記述などの非構造列を
  `Feature(source=..., type=..., values=[...])` と宣言するだけで型付き列にする。
  内部は「埋め込み → 既存の教師ラベルの近傍で分類」で、LLM に毎回書かせるわけではない。
- **方式1 → 機能B（`LLMPredictor`）。** LLM に生レコードを渡して価格を当てさせるのではなく、
  XGBoost・LightGBM・semantic k-NN に先に解かせ、その予測値と類似事例を「証拠」として
  まとめて渡し、最終判断だけさせる。直接予測は残ったが、**統計モデルの上に乗る層**になった。
- **方式2（モデル選択の自動化）は消えた。** LLM が1つのモデルを選ぶのではなく、
  複数モデルを全部走らせて結果を並べ、LLM が重み付けする形に置き換わった。
  「選択」という工程自体がなくなっている。
- **新しく入った軸: 信頼度ルーティング（adaptive）。** 3方式にはなかった観点。
  全行を LLM に投げず、埋め込み分類の confidence が閾値を下回った行だけエスカレーションする。
  LLM の答えは教師ラベル候補としてキューされ、承認すると次回は高速パスが広がる（能動学習）。
  ミーティングで課題に挙がっていた「キャッシュ活用による LLM 予測の軽量化」への回答にあたる。
- **来歴（provenance）が必須要件になった。** 各セルが human / model / llm のどれ由来か、
  confidence、参照した事例、コストを保持し、`model.explain()` で辿れること。
- **題材が中古車に限定されなくなった。** 仕様書は汎用ライブラリとして書かれており
  （churn 予測の例も載っている）、中古車価格はその代表ユースケースという位置づけ。

つまり方向性は **「3方式を比較する」→「組み合わせて1本のパイプラインにする」** に変わった。
ただし各方式を同じ指標で測ること自体はまだ有効で、機能Bにどの証拠を入れるかの判断材料になる。
やろうとしていることは、各データの分析ではなく、ライブラリの作成であることに留意する。

## ディレクトリ

- `dialogs/` — 議事録・設計資料。**編集しない**（記録なので）
  - `entrysheet.md` — 最初の企画案
  - `2026...ミーティング.md` — 方針が端的にまとまっている。迷ったらまずここ
  - `unfold-landing.html` — 伊藤さんによる設計書
  - `2026-08-31-transcript.md` — 8/31 の生の文字起こし。**約18万文字あるので全文を読まない。**
    `grep -n` で当たりを付けてから `sed -n 'A,Bp'` で該当箇所だけ開くこと。
    要点は同日の Gemini メモ側にまとまっているので、まずそちらを見る
- `unfold/` — **ライブラリ本体**。仕様書 `dialogs/unfold-landing.html` の実装
  - `feature.py` — 機能A（`Feature`）。埋め込み → 近傍分類 → 確信度の低い行だけ LLM へ
  - `predictor.py` — 機能B（`LLMPredictor`）。統計モデルと類似事例を証拠に LLM が最終判断
  - `adaptive.py` — 信頼度ルーティング（`AdaptivePredictor`）。機能B を包み、呼ぶ前に手に入る信号で LLM に回す行を絞る
  - `leakage.py` — 重複レコードの検知。fit / predict のときに自動で警告する
  - `demo.py` — `python -m unfold.demo` で全体を一度に動かす入口。既定では LLM を呼ばない
  - `llm.py` — **唯一 Claude API を呼ぶ場所**。ディスクキャッシュ・費用計上・並列実行
- `tests/` — `unfold` のテスト。`.venv/bin/python -m pytest tests -q`
- `scripts/` — データ取得・測定などの補助スクリプト
  - `caafe.py` は比較対象（CAAFE 相当）の実装。**LLM が書いたコードを実行する**ので、
    自分のデータ・自分の環境でだけ動かすこと
  - `check_docs.py` は通しドキュメント（progress-log・related-work・README）の
    更新漏れを機械的に探す。本数・索引漏れ・P/S/R の不整合と、素材だけが動いたコミットを見る
- `sampledata/` — データ。用途ごとに4つに分かれている（下記）
- `docs/` -資料置き場。`PRD.md`は編集しないこと。他資料は編集して良いが、基本的にREADMEにのっとり、これ以上通しドキュメントをつくらないこと。(人間が追えなくなる)
## データ

置き場所のルールはこの4つ。新しいデータを足すときは必ずどれかに分類する。

| ディレクトリ | git管理 | 中身 |
|---|---|---|
| `sampledata/raw/` | ✕ | 外部から再取得できる巨大な生データ |
| `sampledata/scraped/` | ○ | 自分たちで集めた、再取得できないデータ |
| `sampledata/processed/` | ✕ | 分析途中の中間データ。コードで再生成できる |
| `sampledata/sample/` | ○ | 動作確認用の小さな抜粋 |

**判断基準は「再現できるか」。** 再取得・再生成できるものは git に入れず、
手順（スクリプト）の方を git に入れる。git は全履歴を全員に配るので、
一度でも巨大ファイルを入れると削除しても永久に残る。

### `sampledata/scraped/usedsientaL.edit.omit.csv`（203KB, 日本語）

2026年3月にカーセンサーから Octoparse でスクレイピングしたトヨタ シエンタのデータ。
列: `index, 車両本体価格(万円), 走行距離(km), 車歴, 修復歴, 保証, ハイブリッド, 車検, 都道府県`
数値列はすでにダミー変数化済み（0/1）。都道府県だけが文字列。
**再取得できないので git 管理下に置いている。**

### `sampledata/raw/vehicles.csv`（1.4GB, 英語, git管理外）

Kaggle の Craigslist 中古車データ。clone 直後は存在しないので、まず取得する:

```bash
python3 scripts/download_data.py
```

**巨大なので絶対に全体を読み込まないこと。** まずはサンプルで組み立てる:

```bash
head -20 sampledata/sample/vehicles_sample500.csv
```

`description` 列がフィールド内に改行を含むため、**`head` で行数を切ると
CSV レコードが壊れる。** 抜粋を作るときは必ず csv パーサを使うこと
（`sampledata/sample/vehicles_sample500.csv` はその方法で生成済み・500行）。

pandas で本番データを読むときも `nrows=` か `chunksize=` を必ず指定する。
欠損が多い（1行目から `year`, `manufacturer` などが空）ので、まず欠損率の確認から入る。

## 環境

Python 3.12 + venv。**システム標準の Python（macOS の 3.9）は使わない。**

実行環境は2つある。**測定は計算ノードで行う**（`docs/2026-09-01-migration.md`）。

| | 用途 | 備考 |
|---|---|---|
| 計算ノード（Ubuntu 24.04 / 16 vCPU / 64GB） | 測定・学習・LLM バッチ・ノートブック | データとキャッシュの正本はここ |
| 手元の Mac | コード編集、オフライン時の予備 | `results/` に数字を書く実行はしない |

```bash
source .venv/bin/activate     # 有効化。以後 python / pip は venv のものになる
```

有効化せずに単発で動かす場合は `.venv/bin/python スクリプト.py` と直接叩く。
依存は `requirements.txt`。追加したら `pip install -r requirements.txt` を再実行。

未セットアップの環境での構築手順:

```bash
# Ubuntu（計算ノード）— libomp 相当は wheel に同梱されているので不要
sudo apt update && sudo apt install -y python3.12-venv fonts-noto-cjk
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# macOS
brew install python@3.12 libomp          # libomp は xgboost/lightgbm に必要
/usr/local/opt/python@3.12/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`fonts-noto-cjk` はグラフの日本語のために要る（下記「ノートブック」参照）。

APIキーは `.env`（git管理外）に置く。雛形は `.env.example`。

### 巨大データの扱い方

`sampledata/raw/vehicles.csv` は 426,880行 / 1.4GB。

実測値（このマシン: Intel i7-8850H / 16GB）:

| 方法 | 全件集計の所要 | ピークメモリ |
|---|---|---|
| DuckDB（ファイルを直接SQL） | 1.3秒 | 0.22 GB |
| pandas 全件読み込み | 55.9秒 | 6.80 GB |

**探索・集計は DuckDB を使うこと。** 40倍速く、メモリは30分の1。
pandas 全件読みは 6.8GB 使うので、16GB 機ではブラウザ等と併用すると苦しい。
さらに `.copy()` や merge をすると倍近くまで膨らむので、
pandas に渡すのは必要な列・行に絞り込んだ後にする。

```python
import duckdb
duckdb.sql("SELECT manufacturer, count(*) FROM "
           "read_csv_auto('sampledata/raw/vehicles.csv', ignore_errors=true) "
           "GROUP BY 1 ORDER BY 2 DESC").show()
```

モデリングで部分集合を DataFrame にする段階になったら pandas に渡す。
中間データは `sampledata/processed/` に **parquet** で保存する（CSVより小さく速い）。

学習用に整形したものは `scripts/clean_vehicles.py` が作る
（`sampledata/processed/vehicles_multi_clean.parquet`, 200,374行）。
複数車種での検証結果は `docs/2026-08-29-vehicles-multi.md`。

### 中間データの並び順を変えない

`sampledata/processed/` は「コードで再生成できる」建前だが、**再生成した
parquet の行の並びが変わると測定が再現できなくなる。** DuckDB は並列に読むので、
`ORDER BY` を書かないと同じ SQL でも並びが実行ごとに変わり、
位置で引く `load_dataset(sample=60_000)` が別の6万行を返す。
プロンプトが変われば LLM のキャッシュも全部外れて課金され直す。

`clean_vehicles.py` は `ORDER BY id` で固定済み（9/1 に修正）。
**新しく中間データを作るスクリプトを書くときも、必ず書き出しの並びを固定すること。**
なお 8/31 までの Craigslist の測定値は修正前のもので、そのままでは再現できない。

### 既知のデータ品質問題

`vehicles.csv` の `price` は外れ値が激しい。平均 $75,199 に対し中央値 $13,950、
最大は $3,736,928,711。**平均を使う前に必ず外れ値処理をすること。**

**同じ車が複数の地域に重複出稿されている。** 同一 VIN が最大 261 件あり、
価格まで同一。フィルタ後 346,371 行のうち 145,997 行（42%）が重複で、
これを残したままランダム分割の交差検証をすると同じ車が train と test に入る。
**必ず1台1行に潰してから評価すること**（`clean_vehicles.py` が実施済み）。
手を抜くと MAE が 4.2%・R² が 0.03 だけ良く見える（実測は
`scripts/check_duplicate_leak.py`）。テキスト特徴量を使うほど影響が大きい。

## ノートブック

`notebooks/` に連番で置く（`01_explore_vehicles.ipynb` など）。
カーネルは必ず `.venv` のものを使う。ノートブック内では `!pip install` を使わず、
`requirements.txt` に追記して `pip install -r requirements.txt` する。

グラフに日本語を使う場合は先頭で以下を呼ぶ（未設定だと豆腐□になる）:

```python
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import use_japanese_font
use_japanese_font()
```

**フォント名を直書きしないこと。** macOS は Hiragino Sans、Ubuntu は Noto Sans CJK JP と
名前が違うので、決め打ちするともう一方の環境で豆腐になる。
`scripts/plot_style.py` が実際に入っているフォントを探して選ぶ。

出力込みでコミットしている（結果を共有するため）。差分が読みにくくなってきたら
`nbstripout` の導入を検討する。

## ルール

- 説明は日本語で書く
- 巨大ファイル（100MB 超）を git にコミットしない。`.gitignore` を確認してから `git add` する
  - このルールは `.claude/hooks/block_large_git_add.py` が機械的に強制している。
    `git add` / `git commit` の直前にサイズを検査し、100MB 以上は拒否、25MB 以上は確認を求める。
    閾値は環境変数 `CARPRICE_GIT_DENY_MB` / `CARPRICE_GIT_ASK_MB` で変えられる。
- APIキー（`.env`）の中身を読まない。読むと値が会話に取り込まれて API に送られ、
  ローカルの会話ログにも平文で残るため
  - このルールも機械的に強制している。`permissions.deny` の `Read(./.env)` が Read ツールを、
    `.claude/hooks/block_env_read.py` が Bash 経由（`cat .env` など）を塞ぐ。
    `.env.example` の閲覧や `cp .env.example .env` は通る。
    キーが有効かの確認は `scripts/check_api_key.py`（末尾4文字しか表示しない）を使う。
  - 判定は保守的で、キーについて**書く**だけの操作（このルールの説明を Bash の
    ヒアドキュメントで書くなど）も巻き込む。その場合は Edit ツールを使うか、
    `CARPRICE_ALLOW_ENV_READ=1` を付けて実行する
- 生成した中間ファイルは `sampledata/processed/` に置く（git 管理外）
- 作業が一段落したら通しドキュメント（`docs/progress-log.md`・
  `docs/related-work.md`・`README.md`・`docs/README.md`）を現状に合わせる。手順は
  `/update-docs`（`.claude/skills/update-docs/`）にある
  - `.claude/hooks/check_docs_updated.py` が `git commit` の直前に見ていて、
    `results/` や `unfold/` だけをコミットしようとして通しドキュメントが1本も入っていないと
    確認を求める。**拒否はしない**ので、記録を後でまとめる進め方はそのまま通せる
- コミットメッセージは日本語で可
