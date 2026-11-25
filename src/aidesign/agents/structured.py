"""Структурированный ответ LLM: JSON → Pydantic с повтором при ошибке.

Модель получает JSON-схему (в промпте и, если провайдер умеет, в параметре запроса).
Ответ проверяет Pydantic. Если проверка не прошла, модель видит свою ошибку и пробует ещё раз.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLMClient, LLMError, Message

T = TypeVar("T", bound=BaseModel)


def inline_schema(model: type[BaseModel]) -> dict:
    """JSON-схема без $ref/$defs: так её понимают и Ollama, и строгие API."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> str:
    """Достать JSON из ответа: модели любят обрамлять его ```json ... ``` и пояснениями."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("в ответе нет JSON-объекта")
    return text[start:end + 1]


@dataclass
class Structured:
    value: BaseModel
    attempts: int
    raw: str


def ask_structured(client: LLMClient, model: type[T], system: str, user: str, *,
                   agent: str, retries: int = 2) -> Structured:
    schema = inline_schema(model)
    messages: list[Message] = [
        {"role": "system", "content": system + "\n\nОтвечай ТОЛЬКО JSON-объектом по схеме:\n"
         + json.dumps(schema, ensure_ascii=False)},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for attempt in range(1, retries + 2):
        raw = client.chat(messages, schema, agent=agent)
        try:
            value = model.model_validate_json(extract_json(raw))
            return Structured(value, attempt, raw)
        except (ValueError, ValidationError) as e:
            last_error = _short_error(e)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Ответ не прошёл проверку: {last_error}\n"
                                            "Верни исправленный JSON целиком, без пояснений."},
            ]
    raise LLMError(f"{agent}: модель {retries + 1} раза вернула ответ не по схеме: {last_error}")


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        parts = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5]]
        return "; ".join(parts)
    return str(e)[:300]
