# 埋め込み環境（`.venv-embed`）の作り直し

**日付**: 2026-09-01
**対象**: `requirements-embed.txt` / `scripts/setup_embed_env.sh`
**前提**: `docs/2026-09-01-migration.md`（計算ノードへの全面移行）
**結論**: 依存は現行版に更新した。**TabPFN だけは 2.2.1 に据え置き**（理由は入れ替わった）

---

## なぜ作り直したか

`.venv-embed` は **Intel Mac の制約から生まれた環境**だった。当時の測定機に入る
torch は 2.2.2 が上限（それ以降は Apple Silicon 向けしかビルドされていない）で、
2.2.2 は numpy 1.x でビルドされていたため、主環境の numpy 2.x と共存できなかった。
主環境に入れると pandas と LightGBM ごと壊れるので、別の venv に隔離するしかなかった。

計算ノード（Ubuntu / x86_64）に移って**この制約は消えた**。実際、主環境への
ドライランでは torch 2.13.0・transformers 5.16.1・sentence-transformers 6.0.1・
tabpfn 8.5.0 が、numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0 / lightgbm 4.7.0 を
**1つも降格させずに**解決した。技術的には主環境に統合できる。

**それでも隔離は残した。理由が入れ替わっただけである。**

- 旧: 依存が衝突するから分ける（もう成り立たない）
- 新: **`unfold` が torch 無しで動くことの番人**として分ける

既定エンコーダを文字 TF-IDF にして torch を必須にしない、というのは
`unfold/encoders.py` と PRD §GPU 非前提 で明示している設計方針である。主環境に
torch を入れてしまうと、ライブラリのどこかが誤って torch に依存してもテストが
通ってしまい、配布して初めて気づくことになる。「主環境に torch が無い」ことが
その検査を無料でやってくれている。

---

## 入れ替わった版

旧 `requirements-embed.txt` は `numpy<2` / `torch==2.2.2` / `transformers==4.40.2` /
`tokenizers<0.20` / `huggingface_hub==0.23.5` / `sentence-transformers==2.7.0` と、
torch 2.2.2 に引きずられた固定が連鎖していた。上限が消えたので外した。

| | 旧（Mac） | 新（ノード） |
|---|---|---|
| numpy | 1.x（`<2`） | 2.5.2 |
| torch | 2.2.2 | **2.13.0+cpu** |
| transformers | 4.40.2 | 4.57.6 |
| sentence-transformers | 2.7.0 | 5.7.0 |
| tokenizers | `<0.20` | 0.22.2 |
| pandas | （未固定） | 2.3.3 |
| scikit-learn | （未固定） | 1.6.1 |
| tabpfn | 2.2.1 | **2.2.1（据え置き）** |

**torch は CPU 版インデックスから入れる。** ノードに GPU は無いので、既定の PyPI から
入れると使わない CUDA の wheel が19個（数GB）ぶら下がってくる。
`--index-url https://download.pytorch.org/whl/cpu` を `setup_embed_env.sh` に入れた。
これで 41 パッケージ → 22 パッケージになる。

**transformers が 4.x 系なのは tabpfn 2.2.1 の巻き添えである。** 下記のとおり
tabpfn 2.2.1 が `huggingface-hub<1` と `scikit-learn<1.7` を要求するため、
transformers 5.x（`huggingface-hub>=1.5` が必要）とは同居できない。いったん
resolver に解かせた組み合わせを、そのまま `==` で固定してある。**tabpfn の版を
動かすときは、この2つの固定も解き直すこと。**

**対象は Linux / x86_64 に絞った。** Intel Mac ではこの組み合わせは解決しない。
測定はノードで行うと決めた以上それでよく、旧ピンは git 履歴に残っている。

---

## TabPFN 8.5.0 を試して、やめた

移行の目的の1つは「Intel Mac に入らなかった TabPFN 8.x を使えるようにする」ことだった。
実際 8.5.0 は問題なく入る。しかし**別の壁が2つあった**。

### 壁1: 7.1.0 以降はライセンス承諾が要る

8.5.0 を入れて `run_tabpfn.py --probe` を回すと、重みのダウンロードで止まる。

```
tabpfn.errors.TabPFNLicenseError: TabPFN requires a one-time license acceptance
to download model weights for local inference, but no interactive terminal is available.
```

Prior Labs のアカウントを作り、ブラウザでライセンスに同意し、`TABPFN_TOKEN` を
環境変数に置くことを要求される。**2.2.1 には無かった関門である。**
どの版から入ったのかを、wheel に `browser_auth.py` が含まれるかで切り分けた。

| 版 | ライセンス関門 | torch 要件 |
|---|---|---|
| 2.2.1 | 無し | `>=2.1,<3` |
| 6.0.0 / 6.4.1 / 7.0.0 | 無し | `>=2.5`（7.0.0） |
| **7.1.0 以降**（7.1.1 / 8.0.0 / 8.5.0） | **有り** | `>=2.5` |

つまり **7.0.0 が「関門なしで入る最も新しい版」**である。

### 壁2: 2.2.1 は現行の transformers と同居できない

8.5.0 を諦めて 2.2.1 に戻したところ、今度は別の衝突が出た。

```
transformers 5.16.1 requires huggingface-hub<2.0,>=1.5.0,
but you have huggingface-hub 0.36.2 which is incompatible.
```

tabpfn 2.2.1 は `huggingface-hub<1` を要求する。transformers 5.x は `>=1.5` を
要求するので、**両立しない**。そこで transformers / sentence-transformers の固定を外し、
tabpfn 2.2.1 の制約に合う組み合わせ（transformers 4.57.6 / st 5.7.0）を選ばせた。
埋め込みの品質は下がらない（下記の検証を参照）。

### 決めたこと

**2.2.1 のまま進める。** 測定のためだけに外部サービスへアカウントを作るのは
割に合わないと判断した。副次的な利点として、2.2.1 は PRD が R6 として引用している
Nature 2025 の版そのものであり、8/29 の測定と直接比較できる。

**8/29 に 2.2.1 だった理由と、いま 2.2.1 である理由は違う。** ここを混同すると
「Mac をやめたのに版が上がっていないのはなぜか」が分からなくなるので、
leaderboard の備考にも両方を書いた。将来 8.x で測りたくなったら、
必要なのは新しいマシンではなく `TABPFN_TOKEN` である。

---

## 検証

### 転送済みの埋め込みは作り直さなくてよい

心配だったのは、**torch 2.2.2 / sentence-transformers 2.7.0 で計算した
`*_emb_*.parquet`（97MB）が、新しい版では再現しないのではないか**という点だった。
移行のときに LightGBM でバイナリ差を踏んでいるので、同じことを疑った。

シエンタのタイトル 200 行を新環境で埋め込み直し、転送済みの
`usedsienta_emb_title_e5small.parquet` と突き合わせた。

| | 結果 |
|---|---|
| 最大絶対差 | 1.341e-07 |
| コサイン類似度の最小値 | 0.99999994 |

**float32 の丸め誤差の範囲で一致した。** 埋め込みは版を上げても作り直す必要がない。
LightGBM のときと違って差が出ないのは、推論が行ごとに独立していて、
列方向の集約（＝和の順序が効く場所）が無いためと考えられる。

なお、この確認は sentence-transformers 6.0.1（8.5.0 を試していたとき）と
5.7.0（確定した構成）の両方で行い、どちらも同じ値だった。

### TabPFN 2.2.1 は torch 2.13 で動く

2.2.1 の要件は `torch<3,>=2.1` なので形式上は通るが、Mac 時代は 2.2.2 でしか
動かしていない組み合わせなので実際に回した。

```
    訓練行数  n_est     fit  predict     MAE   1fold(訓練4,405/予測1,102)の見積もり
     4,405      8    0.2s    24.1s   13.86   約 133.0秒 → 5fold 11.1分
     4,405      1    0.1s     3.8s   14.03   約  21.3秒 → 5fold  1.8分
```

問題なく動作した。ライセンス関門も出ない。

---

## 記録への反映

**leaderboard の TabPFN 5 行に版と事情を書いた。** 8/29〜8/30 に測った
F 2行・G 3行の備考に `tabpfn 2.2.1` と、なぜ 2.2.1 なのか（当時は Mac の torch 上限、
いまはライセンス関門）を明記した。G の3行はもともと版が抜けていたので補った。

**以後は版を自動で書く。** `run_tabpfn.py` と `run_tabpfn_emb.py` が
`tabpfn.__version__` を読んで備考に入れるようにした。手で書くと今回のように抜ける。

**版をまたいだ行を比較しないこと。** `実行環境` 列（mac / node）と合わせて、
比較する前に条件が揃っているかを見ること。

---

## 再現手順

```bash
rm -rf .venv-embed
bash scripts/setup_embed_env.sh
# --- 確認 ---
# numpy       : 2.5.2
# torch       : 2.13.0+cpu
# transformers: 4.57.6
# st          : 5.7.0
# tabpfn      : 2.2.1

TABPFN_DISABLE_TELEMETRY=1 .venv-embed/bin/python -u scripts/run_tabpfn.py --probe
```
