"""Агент правок: комментарий пользователя к текущей версии → ограничения и комнаты на пересчёт."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Constraint, ConstraintSource, ProjectState, Rule
from .drafts import EditActionsDraft, layout_text, plan_text
from .resolve import actions_to_constraints, resolve_items
from .llm import LLMClient
from .structured import ask_structured

SYSTEM = """Ты разбираешь комментарий пользователя к расстановке мебели в квартире.

intent:
- change — просит что-то изменить в мебели;
- question — спрашивает или просто комментирует («зачем…?», «почему…?», «сколько…?», «всё нравится»);
- unsupported — просит то, что не касается расстановки мебели или требует точности:
  стены, ремонт, отделка, цвет, свет; сдвиг на точное расстояние («на 20 см левее»).

actions — при change КАЖДОЕ действие отдельно:
- «убрать» — убери, не нужен, вместо X (убрать X);
- «добавить» — добавь, поставь (новый предмет), вместо X поставь Y (добавить Y);
- «оставить» — не трогай, оставь как есть, пусть остаётся;
- «переставить» — переставь, перенеси, передвинь (без точного расстояния); если названа комната,
  куда перенести, — укажи её в room: «перенеси кресло в гостиную» → room «гостиная»;
- «к окну» — поставь у окна, ближе к окну;
- «больше места» — больше места перед предметом (size_mm, если названо: «метр» = 1000);
- «не мешать двери» — предмет мешает открыть дверь;
- «к стене».
object — предмет в именительном падеже, как его назвал пользователь: «стол», «стиральная машина».
room — комната словами, если названа: «спальня», «гостиная», «ванная».
При question и unsupported actions пустой. reply — одна фраза по-русски.

Пример: «Шкаф мешает открыть дверь, перенеси его. Кровать не трогай.»
{"intent": "change", "actions": [{"action": "не мешать двери", "object": "шкаф"},
{"action": "оставить", "object": "кровать"}], "reply": "Переставлю шкаф, кровать останется."}
Пример: «Переставь диван куда-нибудь.»
{"intent": "change", "actions": [{"action": "переставить", "object": "диван"}], "reply": "Найду дивану другое место."}
Пример: «Вместо дивана в гостиной хочу кресло.»
{"intent": "change", "actions": [{"action": "убрать", "object": "диван", "room": "гостиная"},
{"action": "добавить", "object": "кресло", "room": "гостиная"}], "reply": "Уберу диван и поставлю кресло."}
Пример: «Зачем в гостиной тумба под телевизор?»
{"intent": "question", "actions": [], "reply": "Тумба входит в стандартный набор гостиной; если не нужна, напишите «убери тумбу»."}
Пример: «Поклей обои в спальне.»
{"intent": "unsupported", "actions": [], "reply": "Отделку я пока не подбираю, только расстановку мебели."}"""


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
    user = (f"Комнаты:\n{plan_text(state.plan)}\n\n"
            f"Расстановка:\n{layout_text(state.layout, state.plan)}\n\n"
            + (f"Комментарий оставлен к предмету: {target}\n" if target else "")
            + f"Комментарий пользователя:\n{text.strip()}")
    res = ask_structured(llm, EditActionsDraft, SYSTEM, user, agent="edit_agent")
    d: EditActionsDraft = res.value  # type: ignore[assignment]
    items = {i.id: i for i in (state.layout.items if state.layout else [])}
    actions = list(d.actions)
    # «поставь X у окна» модели часто размечают как «переставить»: окно в тексте — значит, к окну
    if _WINDOW.search(text):
        for a in actions:
            if a.action == "переставить":
                a.action = "к окну"
    if target in items:  # «этот стол убери» — предмет, к которому оставлен комментарий
        t_item = items[target]
        for a in actions:
            hits = resolve_items(a.object, None, state.layout)
            if any(h.type == t_item.type for h in hits) or not hits:
                if not a.room:
                    a.room = t_item.room
    conv = actions_to_constraints(actions, state.plan, state.layout, source=ConstraintSource.user,
                                  id_prefix=f"{comment_id}_c", comment_id=comment_id)
    cons, move, notes = reconcile(conv.constraints, conv.move, items)
    warnings = conv.warnings + notes
    intent, reply = d.intent, d.reply
    if intent != "change":  # вопрос или «не умею» — действия модели игнорируем
        cons, move = [], []
    elif _EXACT_SHIFT.search(text):
        intent, cons, move = "unsupported", [], []
        reply = ("Точные сдвиги на заданное расстояние пока не поддерживаются. "
                 "Могу переставить предмет в другое место.")
    rooms = affected_rooms(state, cons, move, []) if (cons or move) else []
    if intent == "change" and not (cons or move):
        intent = "question"
    return EditResult(intent, cons, move, rooms, reply, warnings, res.attempts)


_WINDOW = re.compile(r"\bокн|\bокош", re.I)

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
