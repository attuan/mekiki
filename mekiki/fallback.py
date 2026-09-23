"""Where low-confidence rows escape to (step 05 of `SemanticEncoder`: "Uncertain? Escalate.").

**The default is `QueueOnlyFallback`, which answers nothing and only queues rows for review**;
a model is called only when a fallback is passed explicitly.
The non-calling side is the default so that no cost-incurring step runs implicitly.

Fallbacks can be chained: `SemanticEncoder(fallback=[JevFallback(), LLMFallback()])` sends the
uncertain rows to Jev first (fast, cheap, calibrated probabilities) and only the rows Jev is
not confident about on to the frontier LLM. That gives the three stages "embedding classifier,
then Jev, then LLM". A stage that cannot be called (no key) is skipped silently.

This is not a shortcut: it is one half of the active-learning loop. Everything the fallback
answers is queued as a labeling candidate, and once accepted it joins the ground truth, i.e.
**LLM answers and human approvals go through the same queue**. With the queue built first,
the LLM implementation can be plugged in later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Answer:
    """One row's answer returned by a fallback. `origin` names the stage ("jev" / "llm")."""
    value: str | None
    confidence: float
    cost: float = 0.0
    origin: str = "llm"


@runtime_checkable
class Fallback(Protocol):
    """The shape a fallback stage must satisfy. Implement and plug it in once an API
    key exists.

    A stage may also carry a `threshold` attribute (as `JevFallback` does): answers whose
    confidence falls below it are handed to the next stage of the chain instead of being
    accepted. Without the attribute every answer with a value is accepted.
    """

    #: Estimated cost per row (USD)
    cost_per_call: float

    def can_answer(self) -> bool:
        """Whether it can actually be called right now (False if no key is set)."""

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        """Return a value for each row of texts. context holds the neighbouring examples."""


@dataclass
class QueueOnlyFallback:
    """Default implementation that never calls the LLM and only queues rows for review.

    Lets `SemanticEncoder` steps 01-04 (embedding -> nearest-neighbour classification -> confidence
    check) run to the end without an API key, so that **the share of rows falling to 05**
    can be measured. That share is directly "the share of rows that need the LLM",
    i.e. the cost estimate.
    """

    cost_per_call: float = 0.0
    queued: list[dict] = field(default_factory=list)

    def can_answer(self) -> bool:
        return False

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        raise NotImplementedError(
            "QueueOnlyFallback does not answer. To use an LLM, implement Fallback "
            "and pass it as SemanticEncoder(fallback=...).")

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"row_id": row, "text": text,
                            "classifier_guess": guess, "confidence": confidence})


# ---------------------------------------------------------------------
# The real LLM fallback (step 05)
# ---------------------------------------------------------------------

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "string", "description": "exactly one of the candidates"},
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "reason": {"type": "string", "description": "one sentence"},
    },
    "required": ["value", "confidence", "reason"],
    "additionalProperties": False,
}

CLASSIFY_SYSTEM = """\
You are a classifier that assigns a text to exactly one of a fixed set of candidates.

Only rows where nearest-neighbour classification was not confident reach you. As a
reference, you are given existing examples (text and label) judged close by embedding.

- value must be a string that exactly matches one of the candidates.
- When the text does not allow a judgement, pick the closest candidate and
  give a low confidence. Never invent a new value.
"""


class LLMFallback:
    """Fallback that sends rows whose confidence fell below the threshold to Claude.

    Same shape as `QueueOnlyFallback`, so it drops straight into `SemanticEncoder(fallback=...)`.
    **Everything it answers is also queued for review** as a labeling candidate. Once approved,
    the row rides the fast path next time: one half of the active-learning loop.
    """

    def __init__(self, client=None, cost_per_call: float = 0.0) -> None:
        from mekiki.llm import ClaudeClient
        self.client = client or ClaudeClient()
        self.cost_per_call = cost_per_call
        self.queued: list[dict] = []

    def can_answer(self) -> bool:
        return self.client.available()

    def _prompt(self, text: str, values: list[str] | None,
                ctx: dict) -> str:
        parts = [f"## Text to classify\n\n{text}\n"]
        if values:
            parts.append("## Candidates\n")
            parts.extend(f"- {v}" for v in values)
            parts.append("")
        # The shape SemanticEncoder._escalate passes: the very examples the classifier referenced
        near = ctx.get("examples") or []
        if near:
            parts.append("## Existing examples judged close by embedding (closest first)\n")
            for n in near:
                parts.append(f"- \"{n.get('text', '')}\" -> {n.get('value', '')}"
                             f" (similarity {float(n.get('similarity', float('nan'))):.3f})")
            parts.append("")
        parts.append("Answer which candidate this text corresponds to.")
        return "\n".join(parts)

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        if not self.can_answer():
            raise NotImplementedError(
                f"{self.client.why_unavailable()} Or use QueueOnlyFallback.")
        prompts = [self._prompt(t, values, c if isinstance(c, dict) else {})
                   for t, c in zip(texts, context or [{}] * len(texts), strict=True)]
        answers = self.client.ask_many(CLASSIFY_SYSTEM, prompts, CLASSIFY_SCHEMA)
        out = []
        for text, a in zip(texts, answers, strict=True):
            if a.ok and "value" in a.data:
                v = str(a.data["value"])
                # Reject answers outside the candidates, so arbitrary values never enter the feature
                if values and v not in values:
                    out.append(Answer(value=None, confidence=0.0, cost=a.cost,
                                      origin="llm"))
                    continue
                conf = float(a.data.get("confidence", 0.0))
                out.append(Answer(value=v, confidence=conf, cost=a.cost,
                                  origin="llm"))
                self.queued.append({"text": text, "llm_answer": v,
                                    "confidence": conf,
                                    "reason": a.data.get("reason", ""),
                                    "status": "pending_review"})
            else:
                out.append(Answer(value=None, confidence=0.0, cost=a.cost,
                                  origin="llm"))
        return out

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"row_id": row, "text": text,
                            "classifier_guess": guess, "confidence": confidence,
                            "status": "pending_review"})


# ---------------------------------------------------------------------
# The Jev fallback: the middle stage between the classifier and the LLM
# ---------------------------------------------------------------------

class JevFallback:
    """Fallback that asks Jev (TypeSafe AI's System One model) one `choice` question per row.

    Same shape as `LLMFallback`, so it drops into `SemanticEncoder(fallback=...)` on its own
    or as the first stage of a chain (`fallback=[JevFallback(), LLMFallback()]`). Jev sees
    the same evidence the LLM would: the text and the nearest labelled examples, as a JSON
    state. Its answer comes with a calibrated confidence; rows below `threshold` are
    handed on to the next stage (or queued for review when there is none).

    Everything it accepts is queued as a labeling candidate, like the LLM's answers.
    """

    def __init__(self, client=None, threshold: float = 0.7,
                 cost_per_call: float | None = None) -> None:
        from mekiki.jev import JevClient
        self.client = client or JevClient()
        self.threshold = threshold
        self.cost_per_call = (cost_per_call if cost_per_call is not None
                              else self.client.estimated_cost_per_call())
        self.queued: list[dict] = []

    def can_answer(self) -> bool:
        return self.client.available()

    @staticmethod
    def _state(text: str, ctx: dict) -> dict:
        near = ctx.get("examples") or []
        return {
            "text": text,
            "similar_labelled_examples": [
                {"text": str(n.get("text", "")), "value": str(n.get("value", "")),
                 "similarity": round(float(n.get("similarity", 0.0)), 3)}
                for n in near],
        }

    @staticmethod
    def _instructions(values: list[str]) -> str:
        return ("Which candidate does the text correspond to? The candidates are: "
                + ", ".join(values) + ". The similar labelled examples are existing rows "
                "judged close by embedding (closest first); use them as reference, but "
                "answer for the text itself.")

    def answer(self, texts: list[str], values: list[str] | None,
               context: list[dict]) -> list[Answer]:
        if not self.can_answer():
            raise NotImplementedError(
                f"{self.client.why_unavailable()} Or use QueueOnlyFallback.")
        if not values:
            # Jev only picks from a closed set, and SemanticEncoder always passes one
            raise NotImplementedError(
                "JevFallback needs candidate values (a closed set). Pass values=[...] to "
                "SemanticEncoder or fit it with labels.")
        values = [str(v) for v in values]
        states = [self._state(t, c if isinstance(c, dict) else {})
                  for t, c in zip(texts, context or [{}] * len(texts), strict=True)]
        results = self.client.choose_many(states, values, self._instructions(values))
        out = []
        for text, r in zip(texts, results, strict=True):
            if r.choice is None:
                out.append(Answer(value=None, confidence=0.0, cost=r.cost, origin="jev"))
                continue
            out.append(Answer(value=r.choice, confidence=r.confidence, cost=r.cost,
                              origin="jev"))
            if r.confidence >= self.threshold:
                self.queued.append({"text": text, "jev_answer": r.choice,
                                    "confidence": r.confidence,
                                    "probabilities": r.probabilities,
                                    "status": "pending_review"})
        return out

    def enqueue(self, row: int, text: str, guess: str | None,
                confidence: float) -> None:
        self.queued.append({"row_id": row, "text": text,
                            "classifier_guess": guess, "confidence": confidence,
                            "status": "pending_review"})
