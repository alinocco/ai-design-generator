"""Агент правок: комментарий пользователя к текущей версии → ограничения и комнаты на пересчёт."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Constraint, ConstraintSource, ProjectState, Rule
from .drafts import (RULES_TEXT, EditDraft, catalog_text, layout_text, plan_text,
                     to_constraints)
from .llm import LLMClient
from .structured import ask_structured

SYSTEM = """Ты — Агент правок в системе, которая расставляет мебель в квартире.
Пользователь смотрит на текущую расстановку и пишет комментарий. Переведи его в изменения.

intent:
- change — пользователь просит что-то изменить в расстановке;
- question — спрашивает или просто комментирует («зачем…?», «сколько…?», «всё нравится»): ничего не меняем;
- unsupported — просьба, которую система не умеет: снести или передвинуть стену, цвет и отделка,
  сдвиг на точное расстояние («на 20 см левее»).

constraints — правила по словарю ниже; subject — тип из каталога или id предмета из расстановки,
room — id комнаты. Как выбрать предмет: сначала найди комнату, о которой говорит пользователь,
потом предмет в ЭТОЙ комнате по списку расстановки. «Стол» в спальне — это рабочий стол из спальни.

move — id предметов, которые пользователь просит переставить/перенести/передвинуть, НЕ говоря куда.
move пустой, если предмет убирают (exclude), заменяют, ставят у окна (near) или если пользователь
ничего не просил переставлять.

rooms — id комнат, где что-то меняется. reply — одна-две фразы по-русски.

Соответствия:
- «убери X» → exclude; «добавь X», «поставь X» (которого нет) → require;
- «X не трогай», «X оставь на месте» → keep с id предмета;
- «X мешает открыть дверь» → move X и not_in_swing для X;
- «больше места перед X» → min_clearance для X (mm ≈ 900, если число не названо);
- «X у окна» → near, target "window";
- «вместо X поставь Y» → exclude X и require Y.
Не добавляй правил, о которых пользователь не говорил.

Каталог предметов:
{catalog}

Правила (rule):
{rules}

Примеры для ДРУГОЙ квартиры (r5 — Гостиная: r5_sofa, r5_tv_stand, r5_armchair;
r7 — Спальня: r7_bed, r7_wardrobe, r7_desk). Пустые поля опущены.
«Шкаф в спальне мешает двери, перенеси его. Кровать оставь.» →
{{"intent": "change", "constraints": [{{"rule": "not_in_swing", "subject": "wardrobe", "room": "r7"}}, {{"rule": "keep", "subject": "r7_bed"}}], "move": ["r7_wardrobe"], "rooms": ["r7"], "reply": "Переставлю шкаф, кровать останется."}}
«Убери кресло из гостиной.» →
{{"intent": "change", "constraints": [{{"rule": "exclude", "subject": "armchair", "room": "r5"}}], "move": [], "rooms": ["r5"], "reply": "Уберу кресло."}}
«Поставь стол у окна.» →
{{"intent": "change", "constraints": [{{"rule": "near", "subject": "desk", "room": "r7", "target": "window"}}], "move": [], "rooms": ["r7"], "reply": "Поставлю стол у окна."}}
«Вместо дивана хочу два кресла.» →
{{"intent": "change", "constraints": [{{"rule": "exclude", "subject": "sofa", "room": "r5"}}, {{"rule": "require", "subject": "armchair", "room": "r5"}}], "move": [], "rooms": ["r5"], "reply": "Уберу диван и поставлю кресло."}}
«Зачем в гостиной тумба под телевизор?» →
{{"intent": "question", "constraints": [], "move": [], "rooms": [], "reply": "Тумба под ТВ входит в стандартный набор гостиной. Если она не нужна, напишите «убери тумбу»."}}
«Подвинь диван на 30 см вправо.» →
{{"intent": "unsupported", "constraints": [], "move": [], "rooms": [], "reply": "Точные сдвиги пока не поддерживаются. Могу переставить диван в другое место."}}"""


@dataclass
class EditResult:
    intent: str
    constraints: list[Constraint]
    move: list[str]
    rooms: list[str]
    reply: str
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1


def parse_edit(llm: LLMClient, state: ProjectState, text: str, *, comment_id: str,
               target: str | None = None) -> EditResult:
    system = SYSTEM.format(catalog=catalog_text(), rules=RULES_TEXT)
    user = (f"План квартиры:\n{plan_text(state.plan)}\n\n"
            f"Текущая расстановка:\n{layout_text(state.layout, state.plan)}\n\n"
            + (f"Комментарий оставлен к объекту: {target}\n" if target else "")
            + f"Комментарий пользователя:\n{text.strip()}")
    res = ask_structured(llm, EditDraft, system, user, agent="edit_agent")
    d: EditDraft = res.value  # type: ignore[assignment]
    conv = to_constraints(d.constraints, state.plan, state.layout, source=ConstraintSource.user,
                          id_prefix=f"{comment_id}_c", comment_id=comment_id)
    items = {i.id: i for i in (state.layout.items if state.layout else [])}
    move = [m for m in d.move if m in items]
    for m in d.move:
        if m not in items:
            conv.warnings.append(f"move: предмет «{m}» не найден")
    cons, move, notes = reconcile(conv.constraints, move, items)
    conv.warnings += notes
    intent = d.intent
    if intent != "change":  # вопрос или «не умею» — правила модели игнорируем
        cons, move = [], []
    elif _EXACT_SHIFT.search(text):
        intent, cons, move = "unsupported", [], []
        d.reply = "Точные сдвиги на заданное расстояние пока не поддерживаются. Могу переставить предмет в другое место."
    rooms = affected_rooms(state, cons, move, d.rooms) if (cons or move) else []
    if intent == "change" and not (cons or move):
        intent = "question"
    return EditResult(intent, cons, move, rooms, d.reply, conv.warnings, res.attempts)


# «сдвинь на 20 см», «подвинь кровать на полметра левее» — точный сдвиг солвер не умеет
_EXACT_SHIFT = re.compile(
    r"(сдвин|подвин|передвин|отодвин|придвин)\w*[^.!?]*?\bна\s+(\d+[.,]?\d*|пол|четверть)\s*"
    r"(см|сантиметр|мм|миллиметр|м\b|метр)", re.I)


def reconcile(cons: list[Constraint], move: list[str], items: dict) -> tuple[list[Constraint], list[str], list[str]]:
    """Убрать противоречия, которые часто делают небольшие модели.

    - переставлять предмет, который убирают, заменяют или ставят у окна, незачем;
    - require на тип, который и так стоит и переставляется, — лишний;
    - keep и move одного предмета противоречат друг другу: побеждает move."""
    notes = []
    removed = {(c.subject, c.room) for c in cons if c.rule in (Rule.exclude, Rule.near)}
    keep_move = []
    for m in move:
        it = items[m]
        if (it.type, it.room) in removed or (it.type, None) in removed:
            notes.append(f"move {m} не нужен: для него уже есть правило")
        else:
            keep_move.append(m)
    moved_types = {(items[m].type, items[m].room) for m in keep_move}
    out = []
    for c in cons:
        if c.rule == Rule.require and ((c.subject, c.room) in moved_types
                                       or any(t == c.subject for t, _ in moved_types) and c.room is None):
            notes.append(f"require {c.subject} лишний: предмет уже есть и переставляется")
            continue
        if c.rule == Rule.keep and c.subject in keep_move:
            notes.append(f"keep {c.subject} противоречит move — оставлен move")
            continue
        out.append(c)
    return out, keep_move, notes


def affected_rooms(state: ProjectState, constraints: list[Constraint], move: list[str],
                   asked: list[str]) -> list[str]:
    """Какие комнаты пересчитать. Модель может ошибиться, поэтому список выводится и из правил."""
    room_ids = {r.id for r in state.plan.rooms}
    items = state.layout.items if state.layout else []
    by_id = {i.id: i for i in items}
    out = {r for r in asked if r in room_ids}
    out |= {by_id[m].room for m in move}
    for c in constraints:
        if c.rule == Rule.keep:
            continue  # keep ничего не меняет сам по себе
        if c.room:
            out.add(c.room)
        elif c.rule == Rule.require:
            from ..solver.layout import _requirements
            out |= set(_requirements(state.plan, [c]))
        else:
            out |= {i.room for i in items if i.type == c.subject}
            if not out and c.rule != Rule.exclude:
                out |= room_ids
    return sorted(out)
