"""Структурированный бриф: результат работы Аналитика брифа."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Occupants(BaseModel):
    adults: int = Field(1, ge=0)
    children: int = Field(0, ge=0)
    remote_work: bool = False
    pets: bool = False


class Brief(BaseModel):
    raw_text: str = ""
    occupants: Occupants = Occupants()
    style: list[str] = []
    palette: list[str] = Field([], description="hex-цвета, если извлечены из референсов")
    must_have: list[str] = []
    nice_to_have: list[str] = []
    avoid: list[str] = []
    budget_rub: int | None = None
    notes: str | None = None
