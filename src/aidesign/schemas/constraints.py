"""Ограничения: язык, на котором агенты договариваются друг с другом и с солвером.

У каждого ограничения есть источник. По нему Оркестратор понимает, что снять,
если пользователь отменил правку, и как разрешать конфликты
(безопасность и конструктив → нормы → явные правки → эргономика → бриф и стиль).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ConstraintSource(str, Enum):
    norm = "norm"  # Нормоконтролёр
    brief = "brief"  # Аналитик брифа из брифа
    user = "user"  # правка пользователя (см. comment_id)
    system = "system"  # значения по умолчанию, эргономика


class Rule(str, Enum):
    against_wall = "against_wall"
    near = "near"  # subject рядом с target (window, riser, door, item id)
    facing = "facing"
    not_in_swing = "not_in_swing"  # не в зоне открывания двери
    min_clearance = "min_clearance"  # params.mm
    in_room = "in_room"
    exclude = "exclude"  # subject не ставить
    require = "require"  # subject обязательно поставить (hard) или по возможности (soft)
    keep = "keep"  # зафиксировать текущее положение
    no_demolition = "no_demolition"  # для стен
    within_riser_radius = "within_riser_radius"  # params.mm


PRIORITY: dict[ConstraintSource, int] = {
    ConstraintSource.norm: 0,
    ConstraintSource.user: 1,
    ConstraintSource.system: 2,
    ConstraintSource.brief: 3,
}


class Constraint(BaseModel):
    id: str
    kind: Literal["hard", "soft"]
    rule: Rule
    subject: str = Field(description="тип предмета (desk) или id объекта (w5, item_3)")
    target: str | None = None
    room: str | None = None
    params: dict[str, Any] = {}
    weight: float | None = Field(None, gt=0, description="только для soft")
    source: ConstraintSource
    comment_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "soft" and self.weight is None:
            self.weight = 1.0
        if self.kind == "hard" and self.weight is not None:
            raise ValueError("у hard-ограничения не бывает веса")
        if self.source == ConstraintSource.user and not self.comment_id:
            raise ValueError("ограничение из правки должно ссылаться на comment_id")
        return self
