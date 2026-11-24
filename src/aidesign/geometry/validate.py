"""Проверки плана — основа агента-Валидатора (шаг 1) и части Критика.

Каждая проблема — Issue с кодом, уровнем, объектом и понятным текстом:
этот текст агент возвращает Обмерщику или показывает пользователю.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import BaseModel
from shapely.geometry import Point as SPoint
from shapely.ops import unary_union

from ..schemas import OpeningKind, Plan, RoomType
from .core import host_wall, opening_polygon, room_area_m2, room_polygon, wall_polygon


class Issue(BaseModel):
    code: str
    level: Literal["error", "warning"]
    object_id: str | None = None
    message: str


AREA_TOLERANCE = 0.05  # 5% расхождения с подписанной площадью
OVERLAP_MM2 = 0.05e6  # 0,05 м² — допуск на перекрытие полигонов


def validate_plan(plan: Plan) -> list[Issue]:
    issues: list[Issue] = []
    issues += _check_rooms(plan)
    issues += _check_openings(plan)
    issues += _check_fixtures(plan)
    issues += _check_connectivity(plan)
    return issues


def _check_rooms(plan: Plan) -> list[Issue]:
    out: list[Issue] = []
    polys = {}
    for r in plan.rooms:
        p = room_polygon(r)
        if not p.is_valid:
            out.append(Issue(code="room_invalid", level="error", object_id=r.id,
                             message=f"Полигон помещения «{r.display_name}» самопересекается."))
            continue
        polys[r.id] = p
        if r.label_area_m2:
            a = room_area_m2(r)
            diff = abs(a - r.label_area_m2) / r.label_area_m2
            if diff > AREA_TOLERANCE:
                out.append(Issue(
                    code="area_mismatch", level="warning", object_id=r.id,
                    message=(f"«{r.display_name}»: по геометрии {a:.1f} м², на плане "
                             f"{r.label_area_m2:.1f} м² (расхождение {diff:.0%}). "
                             "Проверьте масштаб или границы помещения."),
                ))
    ids = list(polys)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            inter = polys[a].intersection(polys[b]).area
            if inter > OVERLAP_MM2:
                out.append(Issue(code="rooms_overlap", level="error", object_id=a,
                                 message=f"Помещения {a} и {b} перекрываются на {inter / 1e6:.2f} м²."))
    if plan.walls:
        walls = unary_union([wall_polygon(w) for w in plan.walls])
        for rid, p in polys.items():
            inter = p.intersection(walls).area
            if inter > max(OVERLAP_MM2, 0.03 * p.area):
                out.append(Issue(code="room_into_walls", level="warning", object_id=rid,
                                 message=(f"Помещение {rid} заходит в стены на {inter / 1e6:.2f} м²: "
                                          "полигон должен идти по чистовым граням стен.")))
    return out


def _check_openings(plan: Plan) -> list[Issue]:
    out: list[Issue] = []
    for o in plan.openings:
        if host_wall(plan, o) is None:
            out.append(Issue(code="opening_no_wall", level="error", object_id=o.id,
                             message=f"Проём {o.id} ({o.kind.value}) не стоит ни на одной стене."))
        if o.kind in (OpeningKind.door, OpeningKind.entrance) and o.swing is None:
            out.append(Issue(code="door_no_swing", level="warning", object_id=o.id,
                             message=f"У двери {o.id} не указано открывание: зона открывания не учитывается."))
    return out


def _check_fixtures(plan: Plan) -> list[Issue]:
    out: list[Issue] = []
    polys = [room_polygon(r).buffer(50) for r in plan.rooms]
    for f in plan.fixtures:
        if not any(p.contains(SPoint(f.at)) for p in polys):
            out.append(Issue(code="fixture_outside", level="warning", object_id=f.id,
                             message=f"{f.kind.value} {f.id} не попадает ни в одно помещение."))
    has_wet = any(r.type in (RoomType.bathroom, RoomType.toilet) for r in plan.rooms)
    if has_wet and not any(f.kind.value.endswith("riser") for f in plan.fixtures):
        out.append(Issue(code="no_riser", level="warning",
                         message="На плане есть санузел, но не отмечен ни один стояк."))
    return out


OPEN_CONTACT_MM = 600.0  # общая граница без стены длиннее этого — помещения связаны


def rooms_graph(plan: Plan) -> dict[str, set[str]]:
    """Какие помещения связаны: дверью, проёмом без двери или общей границей без стены
    (открытая планировка: кухня и гостиная в ResPlan — разные полигоны без стены между ними)."""
    graph: dict[str, set[str]] = {r.id: set() for r in plan.rooms}
    polys = {r.id: room_polygon(r) for r in plan.rooms if room_polygon(r).is_valid}
    walls = unary_union([wall_polygon(w) for w in plan.walls]).buffer(20)
    ids = list(polys)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            contact = polys[a].buffer(40).intersection(polys[b].buffer(40))
            if contact.is_empty:
                continue
            if contact.difference(walls).area > OPEN_CONTACT_MM * 80:
                graph[a].add(b)
                graph[b].add(a)
    for o in plan.openings:
        if o.kind == OpeningKind.window:
            continue
        zone = opening_polygon(plan, o).buffer(80)
        touching = [rid for rid, p in polys.items() if p.intersects(zone)]
        for a in touching:
            for b in touching:
                if a != b:
                    graph[a].add(b)
    return graph


def entrance_room(plan: Plan) -> str | None:
    polys = {r.id: room_polygon(r) for r in plan.rooms}
    for o in plan.openings:
        if o.kind == OpeningKind.entrance:
            zone = opening_polygon(plan, o).buffer(80)
            for rid, p in polys.items():
                if p.intersects(zone):
                    return rid
    return None


def _check_connectivity(plan: Plan) -> list[Issue]:
    start = entrance_room(plan)
    if start is None:
        return [Issue(code="no_entrance", level="warning",
                      message="Не отмечена входная дверь: нельзя проверить, что все помещения доступны.")]
    graph = rooms_graph(plan)
    seen, queue = {start}, deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur] - seen:
            seen.add(nxt)
            queue.append(nxt)
    return [
        Issue(code="room_unreachable", level="error", object_id=r.id,
              message=f"В «{r.display_name}» ({r.id}) нельзя попасть от входа: нет двери или проёма.")
        for r in plan.rooms
        if r.id not in seen and r.type != RoomType.balcony
    ]

