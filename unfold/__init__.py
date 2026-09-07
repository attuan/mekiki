"""unfold — 非構造データのための機械学習ライブラリ。

伊藤さんの設計書（dialogs/unfold-landing.html）の実装。
機能は2つ（`Feature` / `LLMPredictor`）で、そこに「どの行を LLM に回すか」を
決める信頼度ルーティング（`AdaptivePredictor`）が乗る。
いずれも scikit-learn 互換の API を持つ。

**機能A — `Feature`（特徴量生成）。** 画像や自由記述などの非構造列を、
宣言するだけで型付き列にする。中身は「埋め込み → 既存の教師ラベルの近傍で
分類」であって、LLM は confidence が閾値を下回った行にしか出てこない。

    from unfold import Feature

    df["グレード"] = Feature(
        source="タイトル",
        type="category",
        values=["G", "Z", "X", "G クエロ"],
    ).fit_transform(df)

**機能B — `LLMPredictor`（LLM Predict）。** LLM に生レコードを渡して当てさせる
のではなく、LightGBM・XGBoost・近傍検索に先に解かせ、その予測値と
「実際の価格が分かっている似た事例」を証拠としてまとめ、最終判断だけさせる。
類似事例は訓練データからその行ごとに引き直す（few-shot）。

    from unfold import LLMPredictor

    model = LLMPredictor(target="車両本体価格_万円", unit="万円",
                         numeric=["車齢", "走行距離_km"], text="装備テキスト")
    pred = model.fit(train_df).predict(test_df)
    model.explain(0)   # その行がなぜその値になったか

**信頼度ルーティング — `AdaptivePredictor`。** 機能B は1行ごとに LLM を呼ぶので、
行数がそのまま費用と時間になる（6万行なら約 $516・約17時間）。そこで
「LLM を呼ぶ前に手に入る信号」だけで呼ぶ行を選び、残りは統計モデルに任せる。
閾値ひとつで「全行呼ぶ」と「1行も呼ばない」の間を連続に動かせる。

    from unfold import AdaptivePredictor

    model = AdaptivePredictor(target="price", unit="USD", ...,
                              escalate_rate=0.3)   # 上位3割だけ LLM に回す
    model.plan(test)      # 呼ぶ前に「何行・いくら・何秒」（LLM を呼ばないので無料）
    model.curve(test)     # 割合を振ったときの精度・費用・レイテンシ

どちらも来歴（provenance）を持つ。`explain` / `confidence` / `examples` /
`cost` で、各セルが human / model / llm のどれ由来か、何を参照したか、
いくらかかったかを辿れる（設計書の Provenance 要件）。

**LLM を呼ぶのは2か所だけ**で、どちらも `unfold.llm.ClaudeClient` を通る。
機能B の最終判断と、機能A の `ClaudeFallback`（設計書 05）である。
APIキーが無い環境では `ClaudeClient.available()` が False を返し、
機能A は `QueueOnlyFallback`（答えずにレビュー待ちへ積むだけ）で動き続ける。
"""

from unfold.adaptive import AdaptivePredictor
from unfold.encoders import (
    CharTfidfEncoder,
    Encoder,
    PrecomputedEncoder,
    SentenceTransformerEncoder,
)
from unfold.errors import UnfoldError, UnfoldWarning
from unfold.fallback import ClaudeFallback, LLMFallback, QueueOnlyFallback
from unfold.feature import Feature
from unfold.leakage import (
    DuplicateReport,
    OverlapReport,
    check_duplicates,
    check_overlap,
)
from unfold.llm import ClaudeClient
from unfold.predictor import (
    ColumnSpec,
    LLMPredictor,
    NeighbourIndex,
    NeighbourModel,
    TreeModel,
)
from unfold.preprocess import drop_constant_tokens
from unfold.screening import ScreeningReport, screen

__all__ = [
    "Feature",
    "Encoder",
    "CharTfidfEncoder",
    "SentenceTransformerEncoder",
    "PrecomputedEncoder",
    "LLMFallback",
    "QueueOnlyFallback",
    "ClaudeFallback",
    "ClaudeClient",
    "LLMPredictor",
    "AdaptivePredictor",
    "ColumnSpec",
    "TreeModel",
    "NeighbourModel",
    "NeighbourIndex",
    "UnfoldError",
    "UnfoldWarning",
    "check_duplicates",
    "check_overlap",
    "DuplicateReport",
    "OverlapReport",
    "drop_constant_tokens",
    "screen",
    "ScreeningReport",
]

__version__ = "0.1.0"
