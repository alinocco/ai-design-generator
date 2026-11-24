"""Расстановка мебели и сантехники: результат работы солвера."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    """Предмет. (x, y) — центр, w — вдоль, d — вглубь до поворота, мм.

    rotation — по часовой стрелке (ось y направлена вниз); при 90 и 270
    габариты по осям меняются местами.
    """

    id: str
    type: str = Field(description="тип из каталога: bed_double, desk, wardrobe, ...")
    room: str
    x: float
    y: float
    w: float = Field(gt=0)
    d: float = Field(gt=0)
    rotation: Literal[0, 90, 180, 270] = 0


class Layout(BaseModel):
    variant: str = "A"
    items: list[Item] = []
    score: float | None = None
