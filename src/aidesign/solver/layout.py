"""Расстановка по всей квартире: программа каждой комнаты → солвер → проверка Критиком."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..catalog import program_for
from ..geometry.clearance import check_layout
from ..geometry.core import room_area_m2
from ..geometry.validate import Issue
from ..catalog import CATALOG, HOME_ROOMS, Request
from ..geometry.clearance import MIN_PASSAGE_MM, WET_PASSAGE_MM, WET_ROOMS
from ..schemas import Brief, Constraint, ConstraintSource, Item, Layout, Plan, Rule
from .room import RoomResult, solve_room


@dataclass
class LayoutResult:
    layout: Layout
    rooms: list[RoomResult] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)  # от Критика (геометрия) и от солвера

    @property
    def unplaced(self) -> dict[str, list[str]]:
        return {r.room_id: r.unplaced for r in self.rooms if r.unplaced}


def _room_constraints(constraints: list[Constraint], room_id: str) -> list[Constraint]:
    return [c for c in constraints if c.room in (None, room_id)]


def _requirements(plan: Plan, constraints: list[Constraint]) -> dict[str, list[Constraint]]:
    """Правила require → комната, куда добавить предмет.

    Без указанной комнаты берём самую большую комнату подходящего типа (HOME_ROOMS)."""
    out: dict[str, list[Constraint]] = {}
    for c in constraints:
        if c.rule != Rule.require:
            continue
        room_id = c.room
        if room_id is None:
            for rt in HOME_ROOMS.get(c.subject, ()):
                cands = [r for r in plan.rooms if r.type == rt]
                if cands:
                    room_id = max(cands, key=room_area_m2).id
                    break
        if room_id is not None:
            out.setdefault(room_id, []).append(c)
    return out


def _apply_requirements(requests: list[Request], reqs: list[Constraint]) -> list[Request]:
    requests = list(requests)
    for c in reqs:
        if c.subject not in CATALOG:
            continue
        hit = next((r for r in requests if c.subject in r.alternatives), None)
        if hit is None:
            requests.append(Request((c.subject,), required=c.kind == "hard",
                                    weight=int(300 * (c.weight or 1)), key=c.subject))
        else:
            # «нужна именно ванна» — выбранный тип становится первым и единственным
            if len(hit.alternatives) > 1:
                hit.alternatives = (c.subject,)
            if c.kind == "hard":
                hit.required = True
            else:
                hit.weight = max(hit.weight, int(300 * (c.weight or 1)))
    return requests


def generate_layout(
    plan: Plan,
    brief: Brief | None = None,
    constraints: list[Constraint] | None = None,
    previous: Layout | None = None,
    rooms: list[str] | None = None,
    variant: str = "A",
    seed: int = 0,
    time_limit: float = 5.0,
    critic_rounds: int = 2,
    move: list[str] | None = None,
) -> LayoutResult:
    """Расставить мебель.

    rooms — пересчитать только эти комнаты (частичная перегенерация после правки);
    остальные берутся из previous без изменений. Ограничение keep фиксирует предмет на месте.
    move — id предметов из previous, которые надо поставить на новое место.
    """
    constraints = constraints or []
    remote = bool(brief and brief.occupants.remote_work)
    adults = brief.occupants.adults if brief else 2
    keep_ids = {c.subject for c in constraints if c.rule == Rule.keep}

    required_by_room = _requirements(plan, constraints)
    prev_by_id = {i.id: i for i in (previous.items if previous else [])}
    avoid_by_room: dict[str, list[Item]] = {}
    for mid in move or []:
        if mid in prev_by_id:
            avoid_by_room.setdefault(prev_by_id[mid].room, []).append(prev_by_id[mid])

    items: list[Item] = []
    results: list[RoomResult] = []
    jobs = {}  # room_id → (requests, constraints, fixed)
    for room in plan.rooms:
        prev_items = [i for i in (previous.items if previous else []) if i.room == room.id]
        if rooms is not None and room.id not in rooms:
            items.extend(prev_items)
            continue
        # вместе с предметом остаются его пристройки (кровать → тумбочки)
        fixed = [i for i in prev_items
                 if i.id in keep_ids or any(i.id.startswith(k + "_") for k in keep_ids)]
        fixed_keys = {i.id.removeprefix(f"{room.id}_") for i in fixed}
        requests = _apply_requirements(
            program_for(room.type, room_area_m2(room), remote, adults),
            required_by_room.get(room.id, []))
        requests = [r for r in requests if r.key not in fixed_keys]
        if not requests:
            items.extend(fixed)
            continue
        jobs[room.id] = (requests, _room_constraints(constraints, room.id), fixed)
        rr = solve_room(plan, room.id, requests, jobs[room.id][1], fixed=fixed,
                        time_limit=time_limit, seed=seed, avoid=avoid_by_room.get(room.id))
        results.append(rr)
        items.extend(fixed + rr.items)

    layout = Layout(variant=variant, items=items)
    issues = check_layout(plan, layout)

    # Критик → Планировщик: где к предмету нет прохода, пересчитываем комнату с увеличенной
    # зоной обслуживания этого типа. Новый вариант берём, только если он не хуже.
    for _ in range(critic_rounds):
        flagged: dict[str, set[str]] = {}
        by_id = {i.id: i for i in layout.items}
        for iss in issues:
            if iss.code == "no_passage" and iss.object_id in by_id:
                it = by_id[iss.object_id]
                if it.room in jobs and it.type in CATALOG:
                    flagged.setdefault(it.room, set()).add(it.type)
        if not flagged:
            break
        changed = False
        for room_id, types in flagged.items():
            requests, cons, fixed = jobs[room_id]
            passage = WET_PASSAGE_MM if plan.room(room_id).type in WET_ROOMS else MIN_PASSAGE_MM
            have = {c.subject: c.params.get("mm", 0) for c in cons if c.rule == Rule.min_clearance}
            extra = [Constraint(id=f"critic_{room_id}_{t}", kind="hard", rule=Rule.min_clearance,
                                subject=t, room=room_id, source=ConstraintSource.system,
                                params={"mm": int(max(have.get(t, CATALOG[t].clear_front), 0)
                                                  + passage)})
                     for t in sorted(types)]
            new_cons = [c for c in cons if not (c.rule == Rule.min_clearance and c.subject in types)]
            new_cons += extra
            old = next(r for r in results if r.room_id == room_id)
            rr = solve_room(plan, room_id, requests, new_cons, fixed=fixed,
                            time_limit=time_limit, seed=seed, avoid=avoid_by_room.get(room_id))
            if len(rr.unplaced) > len(old.unplaced):
                continue
            trial = Layout(variant=variant, items=[i for i in layout.items if i.room != room_id]
                           + fixed + rr.items)
            trial_issues = check_layout(plan, trial)
            n_old = sum(1 for i in issues if i.object_id in {x.id for x in old.items})
            n_new = sum(1 for i in trial_issues if i.object_id in {x.id for x in rr.items})
            if n_new < n_old:
                rr.notes.append("пересчитано после замечания Критика: " + ", ".join(sorted(types)))
                results[results.index(old)] = rr
                jobs[room_id] = (requests, new_cons, fixed)
                layout, issues, changed = trial, trial_issues, True
        if not changed:
            break

    for rr in results:
        for key in rr.unplaced:
            issues.append(Issue(code="not_placed", level="warning", object_id=rr.room_id,
                                message=f"В комнате {rr.room_id} не поместилось: {key}."))
    return LayoutResult(layout=layout, rooms=results, issues=issues)
