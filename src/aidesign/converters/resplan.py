"""ResPlan → Plan.

ResPlan (https://github.com/m-agour/ResPlan, данные CC BY 4.0) — 17 000 планов,
каждый — словарь Shapely-геометрий: wall, door, window, front_door, inner,
living, kitchen, bedroom, bathroom, balcony, storage, stair, ... и числа area,
net_area, wall_depth.

Что проверено на самом датасете:
- координаты лежат на холсте ~256 единиц, а не в метрах (вопреки README);
- поле area — площадь в м², соответствующая геометрии inner, поэтому масштаб
  считаем как sqrt(area / inner.area); на 3000 планах это даёт медианную ширину
  двери ~1,0 м и толщину стены ~0,23 м;
- стены — ортогональные полигоны одинаковой толщины wall_depth, двери
  вырезаны из стен.

Чего в ResPlan нет: стояков, типа стен, направления открывания дверей.
Стояк ставим эвристикой (inferred=True), стены — structural=unknown.
"""

from __future__ import annotations

import math
import pickle
import statistics
from pathlib import Path
from typing import Any

import shapely
from shapely import affinity
from shapely.geometry import Point as SPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..geometry.core import rect_along
from ..geometry.walls import _parts, _rect_segment, polygons_to_segments
from ..schemas import (
    Fixture, FixtureKind, Opening, OpeningKind, Plan, PlanSource, Room, RoomType,
    Structural, Wall, WallPosition,
)

# порядок важен: living идёт последним и «уступает» пересечения остальным помещениям
ROOM_KEYS: dict[str, RoomType] = {
    "kitchen": RoomType.kitchen,
    "bedroom": RoomType.bedroom,
    "bathroom": RoomType.bathroom,
    "balcony": RoomType.balcony,
    "storage": RoomType.storage,
    "stair": RoomType.other,
    "living": RoomType.living,
}
DOOR_MM = 900.0  # запасной масштаб: типичная ширина двери


def load_resplan(path: str | Path) -> list[dict[str, Any]]:
    """Загрузить ResPlan.pkl (распакуйте ResPlan.zip из репозитория датасета)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    for p in data:
        if "balacony" in p:  # опечатка в части исходных данных
            p["balcony"] = p.pop("balacony")
    return data


def _geom(p: dict, key: str) -> BaseGeometry | None:
    g = p.get(key)
    if g is None or not isinstance(g, BaseGeometry) or g.is_empty:
        return None
    return g


def is_apartment_like(p: dict, max_area_m2: float = 120.0, max_bedrooms: int = 3) -> bool:
    """Отсеять частные дома: лестницы, участки, бассейны, парковки, большие площади."""
    if _geom(p, "stair") is not None:
        return False
    for key in ("garden", "pool", "parking"):
        if _geom(p, key) is not None:
            return False
    area = float(p.get("area") or 0)
    if not 15 <= area <= max_area_m2:
        return False
    bedrooms = _geom(p, "bedroom")
    n_bed = len(_parts(bedrooms)) if bedrooms is not None else 0
    return n_bed <= max_bedrooms and _geom(p, "bathroom") is not None and _geom(p, "front_door") is not None


def _mm_per_unit(p: dict) -> float:
    inner = _geom(p, "inner")
    area = float(p.get("area") or 0)
    if inner is not None and area > 0 and inner.area > 0:
        k = math.sqrt(area / inner.area) * 1000
        if 10 <= k <= 200:
            return k
    doors = _geom(p, "door")
    if doors is not None:
        lengths = [math.dist(*_rect_segment(d)[:2]) for d in _parts(doors)]
        if lengths:
            return DOOR_MM / statistics.median(lengths)
    raise ValueError("не удалось определить масштаб плана")


def resplan_to_plan(p: dict, ref: str | None = None) -> Plan:
    k = _mm_per_unit(p)
    every = [g for g in (p.get(key) for key in p) if isinstance(g, BaseGeometry) and not g.is_empty]
    minx = min(g.bounds[0] for g in every)
    miny = min(g.bounds[1] for g in every)

    def tf(g: BaseGeometry) -> BaseGeometry:
        return affinity.scale(affinity.translate(g, -minx, -miny), k, k, origin=(0, 0))

    def pt(xy) -> tuple[float, float]:
        return (round((xy[0] - minx) * k, 1), round((xy[1] - miny) * k, 1))

    # стены
    # в ResPlan проёмы вырезаны из стен; в нашей схеме стена цельная, а проёмы
    # вырезаются при отрисовке — поэтому сначала «заделываем» проёмы
    wall_geom = _geom(p, "wall")
    fill = [g for g in (_geom(p, k) for k in ("door", "window", "front_door")) if g is not None]
    if wall_geom is not None:
        wall_geom = unary_union([wall_geom, *fill])
    t_units = float(p.get("wall_depth") or 0) or None
    segs = polygons_to_segments(wall_geom, t_units) if wall_geom is not None else []
    inner = _geom(p, "inner")
    footprint = unary_union([g for g in (inner, wall_geom) if g is not None])
    footprint = unary_union([Polygon(x.exterior) for x in _parts(footprint)])
    walls: list[Wall] = []
    for i, (a, b, t) in enumerate(segs, 1):
        rect = rect_along(a, b, t)
        exterior = not footprint.buffer(t * 0.05).contains(rect.buffer(t * 0.6))
        walls.append(Wall(
            id=f"w{i}", a=pt(a), b=pt(b), thickness=round(t * k, 1),
            position=WallPosition.exterior if exterior else WallPosition.interior,
            structural=Structural.unknown,
        ))

    # проёмы
    openings: list[Opening] = []
    for key, kind, prefix in (("front_door", OpeningKind.entrance, "e"),
                              ("door", OpeningKind.door, "d"),
                              ("window", OpeningKind.window, "win")):
        g = _geom(p, key)
        if g is None:
            continue
        for j, part in enumerate(_parts(g), 1):
            a, b, _ = _rect_segment(part)
            openings.append(Opening(id=f"{prefix}{j}", kind=kind, a=pt(a), b=pt(b)))

    # помещения: без заходов в стены и без взаимных перекрытий
    walls_mm = unary_union([rect_along(w.a, w.b, w.thickness) for w in walls])
    taken = Polygon()
    rooms: list[Room] = []
    counters: dict[str, int] = {}
    for key, rtype in ROOM_KEYS.items():
        g = _geom(p, key)
        if g is None:
            continue
        g = tf(g).buffer(0).difference(walls_mm).difference(taken)
        taken = taken.union(g)
        for part in _parts(g):
            if part.area < 0.5e6:  # меньше 0,5 м² — артефакт
                continue
            counters[key] = counters.get(key, 0) + 1
            poly = _clean_polygon(part)
            if poly is None:
                continue
            coords = [(round(x, 1), round(y, 1)) for x, y in list(poly.exterior.coords)[:-1]]
            rooms.append(Room(id=f"{key}{counters[key]}", type=rtype, polygon=coords))

    # в ResPlan нет прихожих и коридоров: это часть inner, не попавшая ни в одно помещение
    if inner is not None:
        residual = tf(inner).buffer(0).difference(walls_mm).difference(taken)
        entrance = next((o for o in openings if o.kind == OpeningKind.entrance), None)
        n_hall = 0
        for part in sorted(_parts(residual), key=lambda g: -g.area):
            poly = _clean_polygon(part) if part.area > 1.0e6 else None
            if poly is None:
                continue
            n_hall += 1
            is_hall = entrance is not None and poly.distance(rect_along(entrance.a, entrance.b, 600)) < 1
            rooms.append(Room(
                id=f"hall{n_hall}", type=RoomType.hall if is_hall else RoomType.corridor,
                polygon=[(round(x, 1), round(y, 1)) for x, y in list(poly.exterior.coords)[:-1]],
            ))

    plan = Plan(
        walls=walls, openings=openings, rooms=rooms,
        source=PlanSource(kind="resplan", ref=ref or str(p.get("id")), mm_per_px=round(k, 3)),
    )
    plan.fixtures = infer_risers(plan)
    return plan


def _clean_polygon(part: Polygon) -> Polygon | None:
    """Валидный полигон без дыр и шипов, на сетке 1 мм (схема хранит только контур)."""
    g = part.buffer(-15).buffer(15)  # убрать щели и шипы тоньше 3 см
    parts = _parts(g)
    if not parts:
        return None
    poly = Polygon(max(parts, key=lambda x: x.area).exterior).simplify(5)
    poly = shapely.set_precision(poly, 1.0)
    parts = _parts(poly)
    if not parts:
        return None
    poly = Polygon(max(parts, key=lambda x: x.area).exterior)
    return poly if poly.is_valid and poly.area > 0.5e6 else None


def infer_risers(plan: Plan) -> list[Fixture]:
    """Эвристика: стояк — в углу санузла, дальнем от его двери, в 150 мм от стен."""
    out: list[Fixture] = []
    doors = [o for o in plan.openings if o.kind != OpeningKind.window]
    for r in plan.rooms:
        if r.type not in (RoomType.bathroom, RoomType.toilet):
            continue
        poly = Polygon(r.polygon)
        near = [o for o in doors if poly.buffer(300).intersects(SPoint(_mid(o)))]
        ref = SPoint(_mid(near[0])) if near else poly.centroid
        corner = max(poly.exterior.coords[:-1], key=lambda c: SPoint(c).distance(ref))
        c = poly.centroid
        d = math.dist(corner, (c.x, c.y)) or 1
        at = (corner[0] + (c.x - corner[0]) / d * 150, corner[1] + (c.y - corner[1]) / d * 150)
        out.append(Fixture(id=f"riser_{r.id}", kind=FixtureKind.water_riser,
                           at=(round(at[0], 1), round(at[1], 1)), inferred=True))
    return out


def _mid(o: Opening):
    return ((o.a[0] + o.b[0]) / 2, (o.a[1] + o.b[1]) / 2)
