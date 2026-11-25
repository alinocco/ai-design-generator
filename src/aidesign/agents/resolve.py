"""Русские слова → наши идентификаторы. Работает код, а не LLM.

Модель разбирает фразу на действия обычными словами («убрать — стол — спальня»).
Здесь это превращается в тип каталога, id предмета, id комнаты и правило.
Неоднозначность («стол», «тумба», «шкаф») решается по комнате и по тому, что в ней стоит.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..catalog import CATALOG, HOME_ROOMS
from ..schemas import (Constraint, ConstraintSource, Item, Layout, Plan, RoomType, Rule)

# Синонимы: список вариантов, вариант — основы слов, каждая должна начинать какое-то слово фразы.
# Чем больше основ совпало, тем сильнее вариант («рабочий стол» сильнее просто «стола»).
SYNONYMS: dict[str, list[list[str]]] = {
    "bed_double": [["кроват"], ["двуспальн"], ["двуспальн", "кроват"], ["постел"]],
    "bed_single": [["кроват"], ["односпальн"], ["односпальн", "кроват"], ["детск", "кроват"]],
    "nightstand": [["тумб"], ["тумбочк"], ["прикроват"], ["тумб", "прикроват"], ["тумбочк", "прикроват"]],
    "wardrobe": [["шкаф"], ["гардероб"], ["шкаф", "купе"], ["шкаф", "одежд"], ["платян", "шкаф"]],
    "hall_wardrobe": [["шкаф"], ["вешалк"], ["шкаф", "прихож"], ["шкаф", "верхн"]],
    "desk": [["стол"], ["письмен"], ["компьютерн"], ["рабоч", "стол"], ["рабоч", "мест"],
             ["письмен", "стол"], ["компьютерн", "стол"], ["стол", "работ"], ["стол", "урок"]],
    "dining_table": [["стол"], ["обеден", "стол"], ["обеден"], ["стол", "обед"], ["стол", "еды"],
                     ["кухонн", "стол"]],
    "sofa": [["диван"]],
    "armchair": [["кресл"]],
    "tv_stand": [["тумб"], ["телевиз"], ["тв"], ["tv"], ["тумб", "тв"], ["тумб", "телевиз"]],
    "bookcase": [["стеллаж"], ["книжн"], ["полк"], ["стеллаж", "книг"], ["книжн", "стеллаж"]],
    "kitchen_run": [["гарнитур"], ["кухонн", "гарнитур"], ["плит"]],
    "fridge": [["холодильн"]],
    "bathtub": [["ванн"], ["джакуз"]],
    "shower": [["душ"], ["душев"], ["душев", "кабин"]],
    "toilet": [["унитаз"]],
    "sink": [["раковин"], ["умывальн"]],
    "washer": [["стирал"], ["стиралк"], ["стирал", "машин"]],
}

ROOM_STEMS: list[tuple[str, tuple[RoomType, ...]]] = [
    ("спальн", (RoomType.bedroom,)),
    ("детск", (RoomType.bedroom,)),
    ("кабинет", (RoomType.bedroom, RoomType.living)),
    ("гостин", (RoomType.living, RoomType.kitchen_living)),
    ("зал", (RoomType.living, RoomType.kitchen_living)),
    ("кухн", (RoomType.kitchen, RoomType.kitchen_living)),
    ("столов", (RoomType.kitchen, RoomType.kitchen_living)),
    ("ванн", (RoomType.bathroom,)),
    ("санузел", (RoomType.bathroom, RoomType.toilet)),
    ("санузл", (RoomType.bathroom, RoomType.toilet)),
    ("туалет", (RoomType.toilet, RoomType.bathroom)),
    ("прихож", (RoomType.hall, RoomType.corridor)),
    ("коридор", (RoomType.corridor, RoomType.hall)),
]

_WORD = re.compile(r"[a-zа-я0-9_]+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("ё", "е"))


def candidate_types(phrase: str) -> list[tuple[str, int]]:
    """Типы каталога, подходящие к фразе, с силой совпадения (по убыванию)."""
    p = phrase.strip()
    if p in CATALOG:
        return [(p, 99)]
    toks = tokens(p)
    out = []
    for t, variants in SYNONYMS.items():
        best = 0
        for v in variants:
            if all(any(tok.startswith(stem) for tok in toks) for stem in v):
                best = max(best, len(v))
        if best:
            out.append((t, best))
    name_hits = [t for t, s in CATALOG.items() if s.name_ru.lower() == p.lower()]
    out += [(t, 50) for t in name_hits]
    out.sort(key=lambda x: -x[1])
    return out


def resolve_room(plan: Plan, phrase: str | None) -> tuple[str | None, str | None]:
    """(id комнаты, предупреждение). Принимает id, название или тип комнаты словами."""
    if not phrase:
        return None, None
    ids = {r.id for r in plan.rooms}
    if phrase in ids:
        return phrase, None
    toks = tokens(phrase)
    for stem, types in ROOM_STEMS:
        if any(tok.startswith(stem) for tok in toks):
            for rt in types:
                cands = [r for r in plan.rooms if r.type == rt]
                if len(cands) == 1:
                    return cands[0].id, None
                if len(cands) > 1:
                    return None, f"комнат «{phrase}» несколько — уточните, какая"
    for r in plan.rooms:
        if r.name and r.name.lower() == phrase.lower():
            return r.id, None
    return None, f"комната «{phrase}» не найдена"


def resolve_items(phrase: str, room: str | None, layout: Layout | None) -> list[Item]:
    """Предметы расстановки, о которых говорит фраза («стол» в спальне → r2_desk).

    Идём от самых сильных совпадений к слабым и берём первый тип, который реально стоит
    в комнате (или в квартире, если комната не названа)."""
    items = layout.items if layout else []
    by_id = {i.id: i for i in items}
    if phrase.strip() in by_id:
        return [by_id[phrase.strip()]]
    cands = candidate_types(phrase)
    pool = [i for i in items if room is None or i.room == room]
    for t, _ in cands:
        hits = [i for i in pool if i.type == t]
        if hits:
            return hits
    return []


def resolve_type_for_new(phrase: str, room: str | None, plan: Plan) -> str | None:
    """Тип для «добавить X»: предпочтение тому, что уместно в этой комнате."""
    cands = candidate_types(phrase)
    if not cands:
        return None
    top = cands[0][1]
    best = [t for t, s in cands if s == top]
    if room is not None:
        rtype = next(r.type for r in plan.rooms if r.id == room)
        fitting = [t for t in best if rtype in HOME_ROOMS.get(t, ())]
        if fitting:
            return fitting[0]
    return best[0]


# --- действия → ограничения -----------------------------------------------------------

# В программе санузла ванна и душ — альтернативы одного места
ALTERNATIVES = {"shower": "bathtub", "bathtub": "shower"}

ACTIONS = ("убрать", "добавить", "оставить", "переставить", "к окну", "больше места",
           "не мешать двери", "к стене")


@dataclass
class Resolved:
    constraints: list[Constraint] = field(default_factory=list)
    move: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def actions_to_constraints(actions, plan: Plan, layout: Layout | None, *, source: ConstraintSource,
                           id_prefix: str, comment_id: str | None = None,
                           brief: bool = False) -> Resolved:
    """actions — объекты с полями action, object, room, size_mm, hard (см. drafts.ActionDraft)."""
    out = Resolved()
    n = 0

    def add(rule: Rule, subject: str, room: str | None, hard: bool, target: str | None = None,
            params: dict | None = None):
        nonlocal n
        n += 1
        hard = hard or rule in (Rule.keep, Rule.exclude)
        out.constraints.append(Constraint(
            id=f"{id_prefix}{n}", kind="hard" if hard else "soft", rule=rule, subject=subject,
            target=target, room=room, params=params or {}, weight=None if hard else 1.0,
            source=source, comment_id=comment_id if source == ConstraintSource.user else None))

    for a in actions:
        room, warn = resolve_room(plan, a.room)
        if warn:
            out.warnings.append(warn)
        act = a.action
        if act == "добавить":
            t = resolve_type_for_new(a.object, room, plan)
            if t is None:
                out.warnings.append(f"«{a.object}» нет в каталоге — пропущено")
                continue
            add(Rule.require, t, room, a.hard)
            continue

        if act == "переставить" and room is not None and layout is not None:
            # «перенеси стиралку на кухню»: предмет стоит в другой комнате → убрать там, добавить здесь
            anywhere = resolve_items(a.object, None, layout)
            elsewhere = [i for i in anywhere if i.room != room]
            if elsewhere and not [i for i in anywhere if i.room == room]:
                for t in sorted({i.type for i in elsewhere}):
                    for r in sorted({i.room for i in elsewhere if i.type == t}):
                        add(Rule.exclude, t, r, True)
                    add(Rule.require, t, room, a.hard)
                continue

        items = resolve_items(a.object, room, layout) if layout else []
        if items:
            types = sorted({i.type for i in items})
            room = room or (items[0].room if len({i.room for i in items}) == 1 else None)
        else:
            cands = candidate_types(a.object)
            if room is not None and len(cands) > 1:
                rtype = next(r.type for r in plan.rooms if r.id == room)
                fit = [t for t, s in cands if s == cands[0][1] and rtype in HOME_ROOMS.get(t, ())]
                types = fit[:1] or [cands[0][0]]
            else:
                types = [cands[0][0]] if cands else []
        if not types:
            out.warnings.append(f"«{a.object}» не найдено — пропущено")
            continue

        if act == "убрать":
            for t in types:
                add(Rule.exclude, t, room, True)
        elif act == "оставить":
            if not items:
                out.warnings.append(f"оставить: «{a.object}» нет в расстановке")
            for i in items:
                add(Rule.keep, i.id, None, True)
        elif act == "переставить":
            if not items:
                out.warnings.append(f"переставить: «{a.object}» нет в расстановке")
            out.move += [i.id for i in items if i.id not in out.move]
        elif act == "к окну":
            for t in types:
                add(Rule.near, t, room, a.hard, target="window")
                if brief:
                    add(Rule.require, t, room, a.hard)
        elif act == "больше места":
            mm = a.size_mm if a.size_mm and 300 <= a.size_mm <= 2000 else 900
            for t in types:
                add(Rule.min_clearance, t, room, a.hard, params={"mm": int(mm)})
        elif act == "не мешать двери":
            for t in types:
                add(Rule.not_in_swing, t, room, True)
            if not brief:
                out.move += [i.id for i in items if i.id not in out.move]
        elif act == "к стене":
            for t in types:
                add(Rule.against_wall, t, room, a.hard)
    # взаимозаменяемые предметы: «добавить душ» значит «вместо ванны», если про ванну не сказано
    said = {c.subject for c in out.constraints if c.rule in (Rule.require, Rule.exclude)}
    for c in list(out.constraints):
        alt = ALTERNATIVES.get(c.subject)
        if c.rule == Rule.require and alt and alt not in said:
            add(Rule.exclude, alt, c.room, True)
            said.add(alt)
    # одно и то же правило дважды («добавь кресло» + «кресло к окну» в брифе) — оставляем первое;
    # require без комнаты лишний, если тот же предмет уже требуется в конкретной комнате
    roomed = {(c.rule, c.subject) for c in out.constraints if c.room is not None}
    seen, uniq = set(), []
    for c in out.constraints:
        key = (c.rule, c.subject, c.room, c.target)
        if key in seen or (c.rule == Rule.require and c.room is None and (c.rule, c.subject) in roomed):
            continue
        seen.add(key)
        uniq.append(c)
    out.constraints = uniq
    return out

