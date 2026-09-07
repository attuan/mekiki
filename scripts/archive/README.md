# scripts/archive — 結論が出て終わった測定

ここにあるのは、**一度きり走らせて結論が出た（あるいは棄却された）測定スクリプト**です。
消していないのは、日付つきドキュメントの「再現」コマンドがこれらを指しているためです。
`scripts/` 直下には、いまも使うもの（データ整形・現役の測定・作図・検査ツール）だけを残しています。

**新しく測るときにここのスクリプトを使わないこと。** 前提が古いか、結論が覆っています。

## パスが変わりました

日付つきドキュメントは記録なので書き換えていません。**そこに書かれた再現コマンドは、
`scripts/` を `scripts/archive/` に読み替えてください。**

```bash
# ドキュメントの記載
.venv/bin/python scripts/run_encoder_sweep.py
# いまの場所
.venv/bin/python scripts/archive/run_encoder_sweep.py
```

リポジトリ直下から実行するのは変わりません（移動に合わせて `ROOT` の階層は直してあります）。

## 中身

| ファイル | 何を測ったか | なぜ終わったか | 記録 |
|---|---|---|---|
| `run_encoder_sweep.py` | エンコーダの次元スイープ | 8/30 の夜間バッチで結論が出た | `docs/2026-08-30-overnight.md` |
| `run_fullscale.py` | 全行スケールでの精度 | 同上 | `docs/2026-08-30-overnight.md` |
| `run_lgbm_emb_match.py` | LightGBM + 埋め込みの条件合わせ | 同上 | `docs/2026-08-30-overnight.md` |
| `run_tabpfn_emb.py` | TabPFN + 埋め込み | 同上。TabPFN 単体の測定 `scripts/run_tabpfn.py` は現役なので残してある | `docs/2026-08-30-overnight.md` / `docs/2026-09-01-embed-env-rebuild.md` |
| `run_ablation.py` | 特徴量のアブレーション | **結論が覆った。** 複数車種で再現せずシエンタ固有だったと訂正済み（commit `1593c09`） | `docs/2026-08-29-baseline.md` |
| `run_s1_recheck.py` | S1・S2 を 6万行で測り直す | 測り直しが完了し、判定（S1 同着 / S2 未達）が確定した | `docs/2026-09-01-s1-s2-recheck.md` |
| `check_image_urls.py` | Craigslist の画像URLの生存率 | 100件中100件が 404 と判明し、**P5（画像モダリティ）が実施不能で確定**した | `docs/2026-08-29-image-urls.md` |
| `measure_node_cpu.py` | 計算ノードの自動停止しきい値に対する CPU 使用率 | 移行が完了し、役目を終えた | `docs/2026-09-01-migration.md` |

`run_ablation.py` だけは性格が違います。他は「終わった」ですが、これは**結論が間違っていた**ものです。
数字を引くときは訂正後の `docs/2026-08-29-vehicles-multi.md` を見てください。
