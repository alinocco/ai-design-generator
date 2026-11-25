"""Проверки расстановки: пересечения, двери, проходы. Основа геометрической части Критика.

Ширина прохода проверяется морфологически: свободную площадь комнаты «сжимаем» на
половину минимальной ширины. Если после сжатия от двери можно добраться до точки
подхода к предмету, значит, к нему ведёт проход не уже min_width.
"""

from __future__ import annotations

from shapely.geometry import Point as SPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..schemas import Item, Layout, OpeningKind, Plan, RoomType
from .core import door_swing, item_polygon, opening_polygon, room_polygon
from .validate import Issue

MIN_PASSAGE_MM = 700.0
WET_PASSAGE_MM = 600.0  # в санузлах норма — 600 мм свободного места перед прибором
WET_ROOMS = {RoomType.bathroom, RoomType.toilet}


def free_space(plan: Plan, room_id: str, items: list[Item]) -> BaseGeometry:
    room = room_polygon(plan.room(room_id))
    taken = unary_union([item_polygon(i) for i in items if i.room == room_id])
    return room.difference(taken)


def room_doors(plan: Plan, room_id: str):
    room = room_polygon(plan.room(room_id))
    return [
        o for o in plan.openings
        if o.kind != OpeningKind.window and room.intersects(opening_polygon(plan, o).buffer(80))
    ]


def _door_entry_point(plan: Plan, room: Polygon, o) -> SPoint:
    """Точка сразу за дверью внутри комнаты."""
    zone = opening_polygon(plan, o).buffer(250)
    inside = zone.intersection(room)
    return inside.representative_point() if not inside.is_empty else SPoint(o.a)


def check_layout(plan: Plan, layout: Layout, min_passage: float | None = None) -> list[Issue]:
    issues: list[Issue] = []
    by_room: dict[str, list[Item]] = {}
    for it in layout.items:
        by_room.setdefault(it.room, []).append(it)

    for room_id, items in by_room.items():
        room = room_polygon(plan.room(room_id))
        polys = {it.id: item_polygon(it) for it in items}
        passage = min_passage or (WET_PASSAGE_MM if plan.room(room_id).type in WET_ROOMS
                                  else MIN_PASSAGE_MM)

        for it in items:
            if not room.buffer(1).contains(polys[it.id]):
                issues.append(Issue(code="item_outside_room", level="error", object_id=it.id,
                                    message=f"{it.type} ({it.id}) выходит за пределы комнаты {room_id}."))
        ids = list(polys)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if polys[a].intersection(polys[b]).area > 1e3:
                    issues.append(Issue(code="items_overlap", level="error", object_id=a,
                                        message=f"{a} и {b} пересекаются."))

        doors = room_doors(plan, room_id)
        for o in doors:
            swing = door_swing(plan, o)
            if swing is None:
                continue
            for it in items:
                if polys[it.id].intersection(swing).area > 1e3:
                    issues.append(Issue(code="blocks_door", level="error", object_id=it.id,
                                        message=f"{it.type} ({it.id}) мешает открыть дверь {o.id}."))

        if doors:
            free = room.difference(unary_union(list(polys.values())))
            core = free.buffer(-passage / 2)
            starts = [_door_entry_point(plan, room, o) for o in doors]
            reach = _components_touching(core, starts, passage)
            # человек в проходе дотягивается ещё на ~15 см: раковина в нише рядом с унитазом доступна
            reach_zone = reach.buffer(passage / 2 + 150) if not reach.is_empty else reach
            for it in items:
                # пристройку (тумбочку у кровати) достаточно достать через её основной предмет
                group = [polys[j] for j in polys if j == it.id or it.id.startswith(j + "_")]
                if reach.is_empty or not any(reach_zone.intersects(g.buffer(50)) for g in group):
                    issues.append(Issue(
                        code="no_passage", level="warning", object_id=it.id,
                        message=(f"К {it.type} ({it.id}) нет прохода шириной {passage:.0f} мм "
                                 "от двери."),
                    ))
    return issues


def _components_touching(core: BaseGeometry, points: list[SPoint], tol: float) -> BaseGeometry:
    parts = list(getattr(core, "geoms", [core])) if not core.is_empty else []
    hit = [p for p in parts if any(p.distance(pt) <= tol for pt in points)]
    return unary_union(hit) if hit else Polygon()
