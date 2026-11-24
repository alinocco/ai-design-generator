"""Разметка LabelMe → Plan. Правила разметки — в docs/annotation.md.

Метки (label):
  wall, wall:ext                    — полигон/прямоугольник стены (можно Г- и Т-образный)
  room:<тип>, room:<тип>:<м²>       — полигон помещения по чистовым граням;
                                      тип из RoomType, площадь — как подписана на плане
  door, door:<a|b>:<left|right>     — линия от косяка до косяка (или прямоугольник);
                                      a/b — у какой точки петли, left/right — куда открывается
  entrance[:...], window, passage   — то же для входной двери, окна, проёма без двери
  riser, vent, gas                  — точка
  scale:<мм>                        — линия известной длины (например, размер на плане)

Если линии масштаба нет, он вычисляется по подписанным площадям помещений.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..geometry.walls import _rect_segment, polygons_to_segments
from ..schemas import (
    Fixture, FixtureKind, Opening, OpeningKind, Plan, PlanSource, Room, RoomType,
    Swing, Wall, WallPosition,
)


class AnnotationError(ValueError):
    pass


def _poly(shape) -> Polygon:
    pts = shape["points"]
    if shape.get("shape_type") == "rectangle":
        (x1, y1), (x2, y2) = pts
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return Polygon(pts)


def _segment(shape):
    pts = shape["points"]
    if shape.get("shape_type") in ("line", "linestrip") and len(pts) >= 2:
        return tuple(pts[0]), tuple(pts[-1])
    a, b, _ = _rect_segment(_poly(shape))
    return a, b


def labelme_to_plan(path: str | Path) -> Plan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    shapes = data.get("shapes", [])

    scale = None
    for s in shapes:
        if s["label"].startswith("scale:"):
            a, b = _segment(s)
            scale = float(s["label"].split(":")[1]) / math.dist(a, b)

    rooms_px = []
    for s in shapes:
        parts = s["label"].split(":")
        if parts[0] != "room":
            continue
        try:
            rtype = RoomType(parts[1])
        except (IndexError, ValueError) as e:
            raise AnnotationError(f"неизвестный тип помещения в метке «{s['label']}»") from e
        area = float(parts[2].replace(",", ".")) if len(parts) > 2 and parts[2] else None
        rooms_px.append((rtype, _poly(s), area))

    if scale is None:
        labelled = [(p, a) for _, p, a in rooms_px if a]
        if not labelled:
            raise AnnotationError("нет ни линии scale:<мм>, ни подписанных площадей помещений")
        scale = math.sqrt(sum(a for _, a in labelled) * 1e6 / sum(p.area for p, _ in labelled))

    def pt(xy):
        return (round(xy[0] * scale, 1), round(xy[1] * scale, 1))

    # стены
    wall_shapes = [s for s in shapes if s["label"].split(":")[0] == "wall"]
    ext_union = unary_union([_poly(s) for s in wall_shapes if s["label"] == "wall:ext"])
    walls: list[Wall] = []
    # каждую размеченную фигуру раскладываем отдельно: прямоугольник — одна стена,
    # Г- или Т-образный полигон — несколько
    segs = [seg for s in wall_shapes for seg in polygons_to_segments(_poly(s))]
    for i, (a, b, t) in enumerate(segs, 1):
        mid = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        is_ext = not ext_union.is_empty and ext_union.buffer(t * 0.25).contains(mid)
        walls.append(Wall(id=f"w{i}", a=pt(a), b=pt(b), thickness=round(t * scale, 1),
                          position=WallPosition.exterior if is_ext else WallPosition.interior))

    # проёмы
    kinds = {"door": OpeningKind.door, "entrance": OpeningKind.entrance,
             "window": OpeningKind.window, "passage": OpeningKind.passage}
    openings: list[Opening] = []
    counters: dict[str, int] = {}
    for s in shapes:
        parts = s["label"].split(":")
        if parts[0] not in kinds:
            continue
        kind = kinds[parts[0]]
        a, b = _segment(s)
        swing = None
        if len(parts) == 3 and kind in (OpeningKind.door, OpeningKind.entrance):
            swing = Swing(hinge=parts[1], side=parts[2])
        counters[parts[0]] = counters.get(parts[0], 0) + 1
        openings.append(Opening(id=f"{parts[0]}{counters[parts[0]]}", kind=kind,
                                a=pt(a), b=pt(b), swing=swing))

    rooms = []
    rcount: dict[str, int] = {}
    for rtype, poly, area in rooms_px:
        rcount[rtype.value] = rcount.get(rtype.value, 0) + 1
        coords = [pt(c) for c in list(poly.exterior.coords)[:-1]]
        rooms.append(Room(id=f"{rtype.value}{rcount[rtype.value]}", type=rtype,
                          polygon=coords, label_area_m2=area))

    fixtures = []
    fk = {"riser": FixtureKind.water_riser, "vent": FixtureKind.vent, "gas": FixtureKind.gas}
    for s in shapes:
        if s["label"] in fk:
            fixtures.append(Fixture(id=f"{s['label']}{len(fixtures) + 1}", kind=fk[s["label"]],
                                    at=pt(s["points"][0])))

    return Plan(walls=walls, openings=openings, rooms=rooms, fixtures=fixtures,
                source=PlanSource(kind="labelme", ref=data.get("imagePath") or str(path),
                                  mm_per_px=round(scale, 4)))
