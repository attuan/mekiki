"""Claude API を呼ぶための土台（機能B と機能A のフォールバックが共有する）。

**ここが唯一 HTTP を出す場所**。上の層（`LLMPredictor` / `ClaudeFallback`）は
「JSON を投げると JSON が返る」としか知らない。

設計上ゆずれない点が3つある。

1. **ディスクキャッシュを必ず通す。** 5-fold の交差検証は同じ行を何度も
   引き当てるし、実装をいじって測り直すたびに全行が再課金される。
   プロンプトのハッシュをキーにして結果を `sampledata/processed/llm_cache/`
   に置き、2度目以降は課金なしで返す。8/12 ミーティングで課題に挙がった
   「キャッシュ活用による LLM 予測の軽量化」の最小実装でもある。
   **保存するのは応答だけでなく user プロンプトも。** キーは sha256 なので
   逆に戻せず、本文を持たないと「何を聞いた結果か」を後から辿れない。
   system は全行同一なので指紋（`system_sha`）だけ持つ。中身を読むには
   `.venv/bin/python scripts/show_prompt.py` を使う。
2. **費用を必ず数える。** PRD S7 の「1行あたりの単価」は実測しないと
   出せない。呼ぶたびにトークンと金額を積む。
3. **並列で投げる。** 1行1リクエストなので直列だと 300行で数十分かかり、
   その日のうちに測り直せなくなる。

APIキーは `.env` の `ANTHROPIC_API_KEY` から読む。キーが無くても import は
通り、`available()` が False を返すだけにしてある（テストを回すため）。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = ROOT / "sampledata" / "processed" / "llm_cache"

#: 既定モデル。100万トークンあたりの入力/出力ドル単価つき。
#: 単価は費用の見積もりにしか使わないので、変わったらここだけ直す。
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_MODEL = "claude-opus-5"


@dataclass
class LLMAnswer:
    """1リクエストぶんの答えと、その代金。

    `from_cache` が True の行は今回課金されていない（`cost` は 0.0）。
    ただし `input_tokens` などは初回の値をそのまま持っているので、
    「キャッシュが無かったら何ドルだったか」も後から出せる。
    """

    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0
    from_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Usage:
    """呼び出しの累計。`LLMPredictor.cost()` がそのまま返す中身。"""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "呼び出し回数": self.calls,
            "キャッシュ命中": self.cache_hits,
            "エラー": self.errors,
            "入力トークン": self.input_tokens,
            "出力トークン": self.output_tokens,
            "キャッシュ読み出しトークン": self.cache_read_tokens,
            "費用_usd": round(self.cost, 4),
        }


def load_api_key() -> str | None:
    """`.env` と環境変数から APIキーを探す。無ければ None。

    python-dotenv が入っていない環境でも動くよう、自前でも `.env` を読む。
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip() or None
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == "ANTHROPIC_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None


def _sha8(text: str) -> str:
    """プロンプト本文の指紋（先頭8桁）。どの版で得た答えかを見分けるため。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


class ClaudeClient:
    """Claude に JSON を書かせるだけの薄いクライアント。

    `output_config.format` に JSON Schema を渡しているので、返り値が
    スキーマどおりであることは API 側が保証する。こちらで正規表現を書いて
    数字を拾う必要はない。

    思考（thinking）は既定のまま（adaptive）にして、代わりに `effort="low"`
    で深さを抑えている。証拠を渡したうえでの価格判断に長考は要らず、
    thinking を切ると別の不具合（ツール呼び出しやタグの漏れ）を招くため。
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low",
                 max_tokens: int = 1024, cache_dir: str | Path | None = "default",
                 api_key: str | None = None, max_workers: int = 8,
                 timeout: float = 120.0) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else load_api_key()
        if cache_dir == "default":
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self._client = None
        self._lock = threading.Lock()

    # --- 状態 ---------------------------------------------------------

    def available(self) -> bool:
        """いま実際に呼べるか（キーが無ければ False）。"""
        return bool(self.api_key)

    def _ensure_client(self):
        if self._client is None:
            if not self.available():
                raise RuntimeError(
                    "ANTHROPIC_API_KEY がありません。`cp .env.example .env` して "
                    "キーを書いてください。")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key,
                                               timeout=self.timeout)
        return self._client

    # --- キャッシュ ---------------------------------------------------

    def _key(self, system: str, user: str, schema: dict) -> str:
        payload = json.dumps(
            {"model": self.model, "effort": self.effort, "system": system,
             "user": user, "schema": schema},
            sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # 1階層掘るのは、1ディレクトリに数万ファイルを置かないため
        d = self.cache_dir / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        """キャッシュを読む。壊れていたら None を返して呼び直させる。

        壊れる経路は現実にある（途中で電源が落ちる、別プロセスと衝突する、
        古い版が書いた形式が残っている）。**ここで例外を投げると、
        1ファイルの破損で数百行の測定が丸ごと落ちる。**
        """
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(record, dict) or not isinstance(record.get("data"), dict):
            return None
        return record

    def _write_cache(self, key: str, record: dict) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # 途中で落ちても壊れたJSONを残さない

    # --- 本体 ---------------------------------------------------------

    def _price(self, input_tokens: int, output_tokens: int,
               cache_read: int) -> float:
        pin, pout = PRICING.get(self.model, PRICING[DEFAULT_MODEL])
        # キャッシュ読み出しは通常の 1/10。system をキャッシュしているぶん。
        return ((input_tokens * pin + cache_read * pin * 0.1
                 + output_tokens * pout) / 1_000_000)

    def ask(self, system: str, user: str, schema: dict) -> LLMAnswer:
        """system + user を投げて、schema に従う JSON を1つ受け取る。"""
        key = self._key(system, user, schema)
        cached = self._read_cache(key)
        if cached is not None:
            ans = LLMAnswer(data=cached["data"],
                            input_tokens=cached.get("input_tokens", 0),
                            output_tokens=cached.get("output_tokens", 0),
                            cache_read_tokens=cached.get("cache_read_tokens", 0),
                            cost=0.0, from_cache=True)
            with self._lock:
                self.usage.calls += 1
                self.usage.cache_hits += 1
            return ans

        try:
            client = self._ensure_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # system をプロンプトキャッシュに載せる。全行で共通なので
                # 2行目以降は system ぶんの入力が 1/10 の単価になる。
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": schema}},
            )
        except Exception as exc:  # ネットワーク・レート制限・スキーマ違反すべて
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return LLMAnswer(data={}, error=f"{type(exc).__name__}: {exc}")

        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            with self._lock:
                self.usage.calls += 1
                self.usage.errors += 1
            return LLMAnswer(data={}, error=f"JSON を読めませんでした: {exc}")

        u = resp.usage
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cr_tok = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cw_tok = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        cost = self._price(in_tok + int(cw_tok * 1.25), out_tok, cr_tok)

        # **user プロンプトも一緒に保存する。** キーは sha256 なので、
        # これが無いと「何を聞いた結果なのか」を後から辿れない
        # （挙動の説明・不審な答えの追跡ができなくなる）。
        # system は全行で同一なので本文は持たず、指紋だけ持つ。
        # 本文は unfold/predictor.py の SYSTEM_PROMPT などソース側にある。
        record = {"data": data, "input_tokens": in_tok, "output_tokens": out_tok,
                  "cache_read_tokens": cr_tok,
                  "user": user, "system_sha": _sha8(system)}
        self._write_cache(key, record)

        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            self.usage.cache_read_tokens += cr_tok
            self.usage.cost += cost

        return LLMAnswer(data=data, input_tokens=in_tok, output_tokens=out_tok,
                         cache_read_tokens=cr_tok, cost=cost)

    def ask_many(self, system: str, users: Sequence[str], schema: dict,
                 progress: Callable[[int, int], None] | None = None,
                 ) -> list[LLMAnswer]:
        """複数行をまとめて投げる。並列だが、返り値の順は入力どおり。

        1行目だけ先に直列で投げているのは **プロンプトキャッシュのため**。
        いきなり8本同時に出すと全部が「キャッシュ未作成」の状態で走り、
        共通の system ぶんが8回とも満額で課金される。
        """
        if not users:
            return []

        out: list[LLMAnswer | None] = [None] * len(users)
        out[0] = self.ask(system, users[0], schema)
        if progress:
            progress(1, len(users))

        rest = list(range(1, len(users)))
        if rest:
            done = 1
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(self.ask, system, users[i], schema): i
                           for i in rest}
                for fut in as_completed(futures):
                    out[futures[fut]] = fut.result()
                    done += 1
                    if progress:
                        progress(done, len(users))
        return [a for a in out if a is not None]

    def summary(self) -> dict[str, Any]:
        return {"モデル": self.model, "effort": self.effort, **self.usage.as_dict()}
