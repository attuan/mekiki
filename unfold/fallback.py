"""confidence が低い行の逃がし先（設計書 05「Uncertain? Escalate to the LLM.」）。

**既定は「答えずにレビュー待ちとして積むだけ」の `QueueOnlyFallback`** で、
LLM を呼ぶのは `ClaudeFallback` を明示的に渡したときだけ。
呼ばない側を既定にしてあるのは、費用が発生する処理を暗黙に走らせないため。

これは手抜きではなく、設計書の能動学習ループの片側そのものでもある。
仕様書には "Everything the fallback answers is queued as a labeling candidate.
Accept it and it joins the ground truth" とあり、**LLM の答えも人の承認も
同じキューを通る**。キューを先に作っておけば、LLM 実装は後から差し込める。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    """フォールバックが返す1行ぶんの答え。"""
    value: str | None
    confidence: float
    cost: float = 0.0
    origin: str = "llm"


@runtime_checkable
class LLMFallback(Protocol):
    """LLM 実装が満たすべき形。APIキーが決まったらこれを実装して差し込む。"""

    #: 1行あたりの推定費用（PRD S7 の「1行あたりの単価」に対応）
    cost_per_call: float

    def can_answer(self) -> bool:
        """いま実際に呼べるか（キー未設定なら False）。"""

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        """texts の各行に対して値を返す。context には近傍事例が入る。"""


@dataclass
class QueueOnlyFallback:
    """LLM を呼ばず、レビュー待ちのキューに積むだけの既定実装。

    APIキーが無い状態でも機能A の 01〜04（埋め込み → 近傍分類 → 確信度判定）を
    最後まで動かし、**05 に落ちた行が何%あるか**を測れるようにするためのもの。
    その割合がそのまま「LLM を呼ぶ必要がある行の割合」＝費用の見積もりになる。
    """

    cost_per_call: float = 0.0
    queued: list[dict] = field(default_factory=list)

    def can_answer(self) -> bool:
        return False

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        raise NotImplementedError(
            "QueueOnlyFallback は答えません。LLM を使うには LLMFallback を"
            "実装して Feature(fallback=...) に渡してください。")

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"行番号": row, "テキスト": text,
                            "分類器の推測": guess, "confidence": confidence})


# ---------------------------------------------------------------------
# 実物の LLM フォールバック（設計書 05）
# ---------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "候補のいずれか1つ"},
        "confidence": {"type": "number", "description": "0.0〜1.0"},
        "reason": {"type": "string", "description": "日本語で1文"},
    },
    "required": ["value", "confidence", "reason"],
    "additionalProperties": False,
}

CLASSIFY_SYSTEM = """\
あなたはテキストを決められた候補のどれか1つに割り当てる分類器です。

近傍分類が自信を持てなかった行だけがここに来ます。参考として、埋め込みで
近いと判定された既存の事例（テキストとそのラベル）が渡されます。

- value は必ず候補のいずれか1つと完全に一致する文字列にしてください。
- テキストから判断できないときは、最も近い候補を選んだうえで
  confidence を低くしてください。勝手に新しい値を作らないでください。
"""


class ClaudeFallback:
    """confidence が閾値を下回った行を Claude に投げるフォールバック。

    `QueueOnlyFallback` と同じ形をしているので `Feature(fallback=...)` に
    そのまま差し替えられる。**答えたものは同時にレビュー待ちキューにも積む**
    （設計書の "Everything the fallback answers is queued as a labeling
    candidate" ）。承認されれば次回から高速パスに乗る、という能動学習の片側。
    """

    def __init__(self, client=None, cost_per_call: float = 0.0) -> None:
        from unfold.llm import ClaudeClient
        self.client = client or ClaudeClient()
        self.cost_per_call = cost_per_call
        self.queued: list[dict] = []

    def can_answer(self) -> bool:
        return self.client.available()

    def _prompt(self, text: str, values: list[str] | None,
                ctx: dict) -> str:
        parts = [f"## 分類したいテキスト\n\n{text}\n"]
        if values:
            parts.append("## 候補\n")
            parts.extend(f"- {v}" for v in values)
            parts.append("")
        # Feature._escalate が渡す形。近傍分類が参照した事例そのもの
        near = ctx.get("examples") or []
        if near:
            parts.append("## 埋め込みで近いと判定された既存の事例（近い順）\n")
            for n in near:
                parts.append(f"- 「{n.get('テキスト', '')}」 → {n.get('値', '')}"
                             f"（類似度 {float(n.get('類似度', float('nan'))):.3f}）")
            parts.append("")
        parts.append("このテキストがどの候補にあたるか答えてください。")
        return "\n".join(parts)

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        if not self.can_answer():
            raise NotImplementedError(
                "ANTHROPIC_API_KEY がありません。QueueOnlyFallback を使うか、"
                ".env にキーを置いてください。")
        prompts = [self._prompt(t, values, c if isinstance(c, dict) else {})
                   for t, c in zip(texts, context or [{}] * len(texts))]
        answers = self.client.ask_many(CLASSIFY_SYSTEM, prompts, CLASSIFY_SCHEMA)
        out = []
        for text, a in zip(texts, answers):
            if a.ok and "value" in a.data:
                v = str(a.data["value"])
                # 候補外を返してきたら採用しない。勝手な値が特徴量に混ざるのを防ぐ
                if values and v not in values:
                    out.append(Answer(value=None, confidence=0.0, cost=a.cost,
                                      origin="llm"))
                    continue
                conf = float(a.data.get("confidence", 0.0))
                out.append(Answer(value=v, confidence=conf, cost=a.cost,
                                  origin="llm"))
                self.queued.append({"テキスト": text, "LLMの答え": v,
                                    "confidence": conf,
                                    "理由": a.data.get("reason", ""),
                                    "状態": "レビュー待ち"})
            else:
                out.append(Answer(value=None, confidence=0.0, cost=a.cost,
                                  origin="llm"))
        return out

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"行番号": row, "テキスト": text,
                            "分類器の推測": guess, "confidence": confidence,
                            "状態": "レビュー待ち"})
