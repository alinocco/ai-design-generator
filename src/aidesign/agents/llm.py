"""Клиенты LLM: локальная Ollama и любой OpenAI-совместимый API.

Выбор — переменными окружения (или файлом .env в текущей папке):

    LLM_PROVIDER=ollama              # ollama | openai
    LLM_MODEL=qwen2.5:14b-instruct
    LLM_BASE_URL=http://localhost:11434
    LLM_API_KEY=                     # для API
    LLM_JSON_MODE=schema             # schema | object | none — как просить JSON у API
    LLM_TEMPERATURE=0.1
    LLM_NUM_CTX=8192                 # только Ollama: окно контекста (по умолчанию у неё меньше,
                                     # и длинный промпт молча обрезается)

Для Ollama используется родной /api/chat с format=<JSON-схема>: модель физически не может
выдать невалидный JSON (декодирование ограничено грамматикой). Для API — /chat/completions
с response_format; если провайдер не понимает json_schema, ставьте LLM_JSON_MODE=object
(DeepSeek) или none — тогда схему держит только промпт, а ответ всё равно проверяет Pydantic.

Зависимостей нет: HTTP через стандартную библиотеку.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:14b-instruct"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    json_mode: str = "schema"
    temperature: float = 0.1
    timeout: float = 180.0
    max_tokens: int = 2000
    num_ctx: int = 8192

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> "LLMConfig":
        env = dict(_read_env_file(env_file)) if env_file else {}
        env.update({k: v for k, v in os.environ.items() if k.startswith("LLM_")})
        cfg = cls()
        cfg.provider = env.get("LLM_PROVIDER", cfg.provider).strip().lower()
        if cfg.provider not in ("ollama", "openai"):
            raise LLMError(f"LLM_PROVIDER={cfg.provider!r}: ожидается ollama или openai")
        default_url = "http://localhost:11434" if cfg.provider == "ollama" else "https://api.openai.com/v1"
        cfg.base_url = env.get("LLM_BASE_URL", default_url).rstrip("/")
        cfg.model = env.get("LLM_MODEL", cfg.model)
        cfg.api_key = env.get("LLM_API_KEY", "")
        cfg.json_mode = env.get("LLM_JSON_MODE", cfg.json_mode).strip().lower()
        cfg.temperature = float(env.get("LLM_TEMPERATURE", cfg.temperature))
        cfg.timeout = float(env.get("LLM_TIMEOUT", cfg.timeout))
        cfg.max_tokens = int(env.get("LLM_MAX_TOKENS", cfg.max_tokens))
        cfg.num_ctx = int(env.get("LLM_NUM_CTX", cfg.num_ctx))
        return cfg

    def describe(self) -> str:
        return f"{self.provider}:{self.model} @ {self.base_url}"


def _read_env_file(path: str | Path):
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        yield k.strip(), v.strip().strip('"').strip("'")


@dataclass
class LLMCall:
    """Запись одного вызова — для журнала и отчёта."""

    agent: str
    model: str
    messages: list[Message]
    response: str
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class LLMClient(Protocol):
    config: LLMConfig
    log: list[LLMCall]

    def chat(self, messages: list[Message], schema: dict | None = None, *,
             agent: str = "") -> str: ...


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    last: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                last = LLMError(f"HTTP {e.code}: {body}")
                time.sleep(2 ** attempt)
                continue
            raise LLMError(f"HTTP {e.code} от {url}: {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"нет связи с {url}: {e.reason}. Запущена ли Ollama (`ollama serve`)?") from e
    raise LLMError(str(last))


class _Base:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.log: list[LLMCall] = []

    def chat(self, messages: list[Message], schema: dict | None = None, *, agent: str = "") -> str:
        t0 = time.perf_counter()
        try:
            text, pt, ct = self._chat(messages, schema)
        except Exception as e:
            self.log.append(LLMCall(agent, self.config.model, messages, "", time.perf_counter() - t0,
                                    error=str(e)))
            raise
        self.log.append(LLMCall(agent, self.config.model, messages, text, time.perf_counter() - t0, pt, ct))
        return text

    def _chat(self, messages, schema) -> tuple[str, int | None, int | None]:
        raise NotImplementedError


class OllamaClient(_Base):
    """Родной API Ollama: /api/chat, format = JSON-схема."""

    def _chat(self, messages, schema):
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens,
                        "num_ctx": self.config.num_ctx},
        }
        if schema is not None:
            payload["format"] = schema
        out = _post(f"{self.config.base_url}/api/chat", payload, {}, self.config.timeout)
        return out["message"]["content"], out.get("prompt_eval_count"), out.get("eval_count")


class OpenAICompatClient(_Base):
    """/chat/completions: OpenAI, DeepSeek, OpenRouter, vLLM, LM Studio, Ollama /v1 и т. п."""

    def _chat(self, messages, schema):
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if schema is not None and self.config.json_mode == "schema":
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "answer", "schema": schema}}
        elif schema is not None and self.config.json_mode == "object":
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
        out = _post(f"{self.config.base_url}/chat/completions", payload, headers, self.config.timeout)
        usage = out.get("usage") or {}
        return (out["choices"][0]["message"]["content"] or "", usage.get("prompt_tokens"),
                usage.get("completion_tokens"))


class FakeLLM(_Base):
    """Для тестов: отвечает заготовками по очереди или функцией от сообщений."""

    def __init__(self, replies: list[str] | Callable[[list[Message]], str]):
        super().__init__(LLMConfig(provider="ollama", model="fake"))
        self._replies = replies
        self.calls: list[list[Message]] = []

    def _chat(self, messages, schema):
        self.calls.append(messages)
        if callable(self._replies):
            return self._replies(messages), None, None
        if not self._replies:
            raise LLMError("FakeLLM: заготовленные ответы закончились")
        return self._replies.pop(0), None, None


def make_client(config: LLMConfig | None = None) -> LLMClient:
    config = config or LLMConfig.from_env()
    if config.provider == "ollama":
        return OllamaClient(config)
    return OpenAICompatClient(config)


@dataclass
class Usage:
    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    @classmethod
    def of(cls, log: list[LLMCall]) -> "Usage":
        u = cls()
        for c in log:
            u.calls += 1
            u.seconds += c.seconds
            u.prompt_tokens += c.prompt_tokens or 0
            u.completion_tokens += c.completion_tokens or 0
            u.errors += bool(c.error)
            u.by_agent[c.agent] = u.by_agent.get(c.agent, 0) + 1
        return u
