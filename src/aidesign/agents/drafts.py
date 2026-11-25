"""Что возвращает LLM (черновики) и как черновик превращается в ограничения схемы.

LLM работает с узким словарём: типы предметов из каталога, id комнат и предметов из
текущего состояния, несколько правил. Всё, что не проходит проверку по плану и каталогу,
отбрасывается с предупреждением — в ProjectState попадают только осмысленные ограничения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..catalog import CATALOG
from ..geometry.core import room_area_m2
from ..schemas import (Constraint, ConstraintSource, Layout, OpeningKind, Plan, Rule)
from ..schemas.plan import ROOM_NAMES_RU

RuleName = Literal["exclude", "require", "keep", "near", "against_wall", "min_clearance",
                   "not_in_swing", "facing"]


class ConstraintDraft(BaseModel):
    rule: RuleName
    subject: str = Field(description="тип предмета из каталога (desk) или id предмета (r2_desk)")
    room: str | None = Field(None, description="id комнаты, если правило относится к одной комнате")
    target: str | None = Field(None, description="для near/facing: window или id объекта")
    mm: int | None = Field(None, description="для min_clearance: свободное место в мм")
    hard: bool = Field(True, description="true — обязательно, false — по возможности")


class BriefDraft(BaseModel):
    adults: int = Field(1, ge=0, le=10)
    children: int = Field(0, ge=0, le=10)
    remote_work: bool = False
    pets: bool = False
    style: list[str] = []
    must_have: list[str] = []
    nice_to_have: list[str] = []
    avoid: list[str] = []
    budget_rub: int | None = None
    constraints: list[ConstraintDraft] = []


class EditDraft(BaseModel):
    intent: Literal["change", "question", "unsupported"]
    constraints: list[ConstraintDraft] = []
    move: list[str] = Field([], description="id предметов, которые надо переставить в другое место")
    rooms: list[str] = Field([], description="id комнат, которые надо пересчитать")
    reply: str = Field("", description="короткий ответ пользователю по-русски")


# --- контекст для промпта -----------------------------------------------------------

def catalog_text() -> str:
    return "\n".join(f"- {k} — {v.name_ru}" for k, v in CATALOG.items())


def plan_text(plan: Plan) -> str:
    lines = []
    for r in plan.rooms:
        if r.type.value in ("balcony",):
            continue
        lines.append(f"- {r.id} — {r.display_name}, {room_area_m2(r):.1f} м²")
    return "\n".join(lines)


def layout_text(layout: Layout | None, plan: Plan | None = None) -> str:
    """Предметы по комнатам: так модели проще связать «стол в спальне» с r2_desk."""
    if not layout or not layout.items:
        return "(мебель ещё не расставлена)"
    names = {r.id: r.display_name for r in plan.rooms} if plan else {}
    by_room: dict[str, list] = {}
    for i in layout.items:
        by_room.setdefault(i.room, []).append(i)
    lines = []
    for room_id, items in by_room.items():
        lines.append(f"{room_id} — {names.get(room_id, room_id)}:")
        lines += [f"  - {i.id} — {CATALOG[i.type].name_ru if i.type in CATALOG else i.type} ({i.type})"
                  for i in items]
    return "\n".join(lines)


RULES_TEXT = """\
- exclude — не ставить предмет (subject — тип; room — если только в одной комнате)
- require — обязательно поставить предмет (subject — тип; room — если важна комната)
- keep — оставить предмет на месте (subject — id предмета)
- near — поставить рядом с target (target: window — окно, или id предмета)
- against_wall — поставить к стене
- min_clearance — оставить перед предметом свободное место mm (мм)
- not_in_swing — не ставить в зону открывания двери
- facing — повернуть лицом к target"""


# --- черновик → ограничения ----------------------------------------------------------

@dataclass
class Converted:
    constraints: list[Constraint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_RU_TO_TYPE = {v.name_ru.lower(): k for k, v in CATALOG.items()}
_ROOM_WORDS = {name.lower(): rt for rt, name in ROOM_NAMES_RU.items()}


def _room_id(plan: Plan, value: str | None, warnings: list[str]) -> str | None:
    if not value:
        return None
    ids = {r.id for r in plan.rooms}
    if value in ids:
        return value
    # модель могла написать тип или название комнаты вместо id
    rt = _ROOM_WORDS.get(value.lower())
    cands = [r for r in plan.rooms if r.type.value == value or (rt is not None and r.type == rt)]
    if len(cands) == 1:
        return cands[0].id
    warnings.append(f"комната «{value}» не найдена — правило применено ко всем комнатам")
    return None


def to_constraints(drafts: list[ConstraintDraft], plan: Plan, layout: Layout | None, *,
                   source: ConstraintSource, id_prefix: str,
                   comment_id: str | None = None) -> Converted:
    out = Converted()
    items = {i.id: i for i in (layout.items if layout else [])}
    targets_ok = ({"window"} | {f.id for f in plan.fixtures} | set(items)
                  | {o.id for o in plan.openings if o.kind != OpeningKind.window})
    n = 0
    for d in drafts:
        room = _room_id(plan, d.room, out.warnings)
        subject = d.subject.strip()
        subject = _RU_TO_TYPE.get(subject.lower(), subject)

        if d.rule == "keep":
            if subject in items:
                subjects = [subject]
            elif subject in CATALOG:  # «оставь кровать» → все кровати (в комнате)
                subjects = [i.id for i in items.values()
                            if i.type == subject and (room is None or i.room == room)]
            else:
                subjects = []
            if not subjects:
                out.warnings.append(f"keep: предмет «{d.subject}» не найден в расстановке")
                continue
        else:
            if subject in items:  # id предмета → его тип и комната
                room = room or items[subject].room
                subject = items[subject].type
            if subject not in CATALOG:
                out.warnings.append(f"{d.rule}: «{d.subject}» нет в каталоге — пропущено")
                continue
            subjects = [subject]

        params = {}
        target = d.target
        if d.rule in ("near", "facing"):
            if not target or target not in targets_ok:
                out.warnings.append(f"{d.rule}: цель «{d.target}» не найдена — пропущено")
                continue
        else:
            target = None
        if d.rule == "min_clearance":
            params["mm"] = int(d.mm) if d.mm and 300 <= d.mm <= 2000 else 900

        for s in subjects:
            n += 1
            hard = d.hard or d.rule in ("keep", "exclude")
            out.constraints.append(Constraint(
                id=f"{id_prefix}{n}", kind="hard" if hard else "soft", rule=Rule(d.rule),
                subject=s, target=target, room=room, params=params,
                weight=None if hard else 1.0, source=source,
                comment_id=comment_id if source == ConstraintSource.user else None))
    return out


_CONFLICT = {Rule.require: {Rule.exclude, Rule.require}, Rule.exclude: {Rule.require, Rule.exclude},
             Rule.min_clearance: {Rule.min_clearance}, Rule.near: {Rule.near},
             Rule.keep: {Rule.keep}, Rule.against_wall: {Rule.against_wall},
             Rule.facing: {Rule.facing}, Rule.not_in_swing: {Rule.not_in_swing}}


def merge_constraints(old: list[Constraint], new: list[Constraint]) -> tuple[list[Constraint], list[str]]:
    """Новые правила вытесняют старые о том же предмете: «убери стол» отменяет «нужен стол».

    Нормы (source=norm) правкой не вытесняются."""
    kept, dropped = [], []
    for c in old:
        clash = next((n for n in new if n.subject == c.subject
                      and c.rule in _CONFLICT.get(n.rule, set())
                      and (n.room is None or c.room in (None, n.room))), None)
        if clash is not None and c.source != ConstraintSource.norm:
            dropped.append(c.id)
        else:
            kept.append(c)
    return kept + new, dropped
