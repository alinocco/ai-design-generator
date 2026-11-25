"""Шаг 3: генерация планировки — расстановка мебели и сантехники солвером CP-SAT."""

from .layout import LayoutResult, generate_layout
from .room import RoomResult, solve_room

__all__ = ["LayoutResult", "generate_layout", "RoomResult", "solve_room"]
