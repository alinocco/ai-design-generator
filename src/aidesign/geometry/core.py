"""Базовая геометрия плана на Shapely. Все величины — в миллиметрах."""

from __future__ import annotations

import math

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..schemas import Item, Opening, OpeningKind, Plan, Room, Wall

DEFAULT_WALL_MM = 200.0


def seg_len(a, b) -> float:
    return math.dist(a, b)


def rect_along(a, b, width: float) -> Polygon:
    """Прямоугольник шириной width, натянутый на отрезок a→b (торцы плоские)."""
    (x1, y1), (x2, y2) = a, b
    length = math.dist(a, b)
    if length == 0:
        return Polygon()
    nx, ny = -(y2 - y1) / length * width / 2, (x2 - x1) / length * width / 2
    return Polygon([(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)])


def wall_polygon(w: Wall) -> Polygon:
    return rect_along(w.a, w.b, w.thickness)


def host_wall(plan: Plan, o: Opening) -> Wall | None:
    """Стена, на которой стоит проём: явно указанная или ближайшая параллельная."""
    if o.wall:
        try:
            return plan.wall(o.wall)
        except StopIteration:
            pass
    # проём может лежать на цельной стене или в разрыве между двумя кусками стены:
    # ищем ближайшую параллельную стену, до оси которой от отрезка проёма — меньше полутолщины
    seg = LineString([o.a, o.b])
    ang = math.atan2(o.b[1] - o.a[1], o.b[0] - o.a[0])
    best, best_d = None, float("inf")
    for w in plan.walls:
        wang = math.atan2(w.b[1] - w.a[1], w.b[0] - w.a[0])
        diff = abs((ang - wang + math.pi / 2) % math.pi - math.pi / 2)
        if diff > math.radians(10):
            continue
        d = LineString([w.a, w.b]).distance(seg)
        if d <= w.thickness / 2 + 50 and d < best_d:
            best, best_d = w, d
    return best


def opening_thickness(plan: Plan, o: Opening) -> float:
    w = host_wall(plan, o)
    return w.thickness if w else DEFAULT_WALL_MM


def opening_polygon(plan: Plan, o: Opening, extra: float = 2.0) -> Polygon:
    """Прямоугольник проёма в толще стены (чуть шире стены, чтобы вырезать её начисто)."""
    return rect_along(o.a, o.b, opening_thickness(plan, o) + extra)


def walls_geometry(plan: Plan) -> BaseGeometry:
    """Все стены одним массивом, с вырезанными проёмами."""
    walls = unary_union([wall_polygon(w) for w in plan.walls])
    holes = unary_union([opening_polygon(plan, o) for o in plan.openings])
    return walls.difference(holes)


def room_polygon(r: Room) -> Polygon:
    return Polygon(r.polygon)


def area_m2(geom: BaseGeometry) -> float:
    return geom.area / 1e6


def room_area_m2(r: Room) -> float:
    return area_m2(room_polygon(r))


def total_area_m2(plan: Plan, include_balcony: bool = False) -> float:
    return sum(
        room_area_m2(r) for r in plan.rooms if include_balcony or r.type.value != "balcony"
    )


def door_swing(plan: Plan, o: Opening) -> Polygon | None:
    """Зона открывания двери: четверть круга радиусом в ширину проёма."""
    if o.kind not in (OpeningKind.door, OpeningKind.entrance) or o.swing is None:
        return None
    hinge, free = (o.a, o.b) if o.swing.hinge == "a" else (o.b, o.a)
    r = seg_len(o.a, o.b)
    if r == 0:
        return None
    # нормаль к a→b: «left» — слева по ходу a→b, как видно на экране (ось y вниз)
    (ax, ay), (bx, by) = o.a, o.b
    nx, ny = (by - ay) / r, (ax - bx) / r
    if o.swing.side == "right":
        nx, ny = -nx, -ny
    t = opening_thickness(plan, o) / 2
    hx, hy = hinge[0] + nx * t, hinge[1] + ny * t  # петля — на грани стены
    fx, fy = free[0] + nx * t, free[1] + ny * t
    start = math.atan2(fy - hy, fx - hx)
    end = math.atan2(ny, nx)
    delta = (end - start + math.pi) % (2 * math.pi) - math.pi
    pts = [(hx, hy)]
    for i in range(17):
        ang = start + delta * i / 16
        pts.append((hx + r * math.cos(ang), hy + r * math.sin(ang)))
    return Polygon(pts)


def item_polygon(it: Item) -> Polygon:
    w, d = (it.w, it.d) if it.rotation in (0, 180) else (it.d, it.w)
    return Polygon(
        [
            (it.x - w / 2, it.y - d / 2),
            (it.x + w / 2, it.y - d / 2),
            (it.x + w / 2, it.y + d / 2),
            (it.x - w / 2, it.y + d / 2),
        ]
    )
