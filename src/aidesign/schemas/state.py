"""ProjectState — единственный источник истины, общая доска (blackboard) агентов."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from .brief import Brief
from .constraints import Constraint
from .layout import Layout
from .plan import Plan


class Stage(str, Enum):
    recognition = "recognition"  # шаг 1
    plan_confirmed = "plan_confirmed"
    replanning = "replanning"  # шаг 2
    layout = "layout"  # шаг 3
    done = "done"


class Comment(BaseModel):
    id: str
    stage: Stage
    text: str
    target: str | None = Field(None, description="id объекта на плане, к которому правка")
    parsed_to: list[str] = Field([], description="id созданных ограничений")


class ProjectState(BaseModel):
    project_id: str
    version: int = 1
    parent_version: int | None = None
    stage: Stage = Stage.recognition
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str | None = Field(None, description="что изменилось в этой версии")

    plan: Plan
    brief: Brief | None = None
    constraints: list[Constraint] = []
    layout: Layout | None = None
    comments: list[Comment] = []

    def next_version(self, note: str) -> "ProjectState":
        """Копия состояния как новая версия; исходная остаётся неизменной."""
        return self.model_copy(
            deep=True,
            update={
                "version": self.version + 1,
                "parent_version": self.version,
                "created_at": datetime.now(timezone.utc),
                "note": note,
            },
        )
