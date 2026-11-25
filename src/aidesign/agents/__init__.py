"""LLM-агенты: Аналитик брифа, Агент правок и Оркестратор.

LLM переводит текст в ограничения по узкому словарю; координаты считает солвер,
проверяет Критик (код). Модель выбирается в .env: локальная Ollama или любой
OpenAI-совместимый API (см. agents/llm.py).
"""

from .brief_analyst import BriefResult, analyze_brief
from .edit_agent import EditResult, parse_edit
from .llm import (FakeLLM, LLMCall, LLMClient, LLMConfig, LLMError, OllamaClient,
                  OpenAICompatClient, Usage, make_client)
from .orchestrator import Orchestrator, Turn

__all__ = [
    "BriefResult", "analyze_brief", "EditResult", "parse_edit",
    "FakeLLM", "LLMCall", "LLMClient", "LLMConfig", "LLMError", "OllamaClient",
    "OpenAICompatClient", "Usage", "make_client", "Orchestrator", "Turn",
]
