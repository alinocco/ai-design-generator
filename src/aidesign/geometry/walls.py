"""Разложение полигона стен на отрезки с толщиной.

Нужно конвертерам: в ResPlan и в разметке LabelMe стены заданы полигонами
(часто Г- и Т-образными), а в нашей схеме стена — отрезок a→b с толщиной.

Алгоритм для ортогональных полигонов: режем полигон сеткой по координатам его
вершин, в каждой горизонтальной полосе высотой ≈ толщине стены ищем непрерывные
отрезки внутри полигона — это горизонтальные стены; то же для вертикальных полос.
Узлы (углы, Т-примыкания) попадают и в горизонтальную, и в вертикальную стену,
поэтому стены в углах перекрываются — как и требует схема.
Неортогональные части аппроксимируются минимальным повёрнутым прямоугольником.
"""

from __future__ import annotations

import math

from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep


def _parts(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _cluster(values: list[float], tol: float) -> list[float]:
    vals = sorted(values)
    out: list[float] = []
    for v in vals:
        if out and v - out[-1] <= tol:
            continue
        out.append(v)
    return out



def _rect_segment(poly: Polygon) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Отрезок по длинной оси минимального повёрнутого прямоугольника и его толщина."""
    r = poly.minimum_rotated_rectangle
    c = list(r.exterior.coords)[:4]
    e1 = math.dist(c[0], c[1])
    e2 = math.dist(c[1], c[2])
    if e1 >= e2:  # длинная сторона c0-c1, толщина e2
        a = ((c[0][0] + c[3][0]) / 2, (c[0][1] + c[3][1]) / 2)
        b = ((c[1][0] + c[2][0]) / 2, (c[1][1] + c[2][1]) / 2)
        return a, b, e2
    a = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)
    b = ((c[3][0] + c[2][0]) / 2, (c[3][1] + c[2][1]) / 2)
    return a, b, e1


def _runs(cells_inside: list[bool], edges: list[float]) -> list[tuple[float, float]]:
    runs, start = [], None
    for i, inside in enumerate(cells_inside):
        if inside and start is None:
            start = edges[i]
        if not inside and start is not None:
            runs.append((start, edges[i]))
            start = None
    if start is not None:
        runs.append((start, edges[-1]))
    return runs


def _orthogonal_segments(poly: Polygon, t: float, snap: float, lo_k: float = 0.55, hi_k: float = 1.6):
    xs = _cluster([x for x, _ in poly.exterior.coords], snap)
    ys = _cluster([y for _, y in poly.exterior.coords], snap)
    for ring in poly.interiors:
        xs = _cluster(xs + [x for x, _ in ring.coords], snap)
        ys = _cluster(ys + [y for _, y in ring.coords], snap)
    prepared = prep(poly)

    def inside(x, y):
        return prepared.contains(Point(x, y))

    lo, hi = lo_k * t, hi_k * t
    horiz: list[tuple[float, float, float, float]] = []  # x0, x1, y0, y1
    vert: list[tuple[float, float, float, float]] = []  # y0, y1, x0, x1
    for y0, y1 in zip(ys, ys[1:]):
        h = y1 - y0
        if not lo <= h <= hi:
            continue
        yc = (y0 + y1) / 2
        flags = [inside((x0 + x1) / 2, yc) for x0, x1 in zip(xs, xs[1:])]
        horiz += [(x0, x1, y0, y1) for x0, x1 in _runs(flags, xs) if x1 - x0 >= 1.5 * h]
    for x0, x1 in zip(xs, xs[1:]):
        w = x1 - x0
        if not lo <= w <= hi:
            continue
        xc = (x0 + x1) / 2
        flags = [inside(xc, (y0 + y1) / 2) for y0, y1 in zip(ys, ys[1:])]
        vert += [(y0, y1, x0, x1) for y0, y1 in _runs(flags, ys) if y1 - y0 >= 1.5 * w]

    horiz = _merge_bands(horiz, snap, hi)
    vert = _merge_bands(vert, snap, hi)
    h_rects = [box(x0, y0, x1, y1) for x0, x1, y0, y1 in horiz]
    v_rects = [box(x0, y0, x1, y1) for y0, y1, x0, x1 in vert]
    h_union = unary_union(h_rects) if h_rects else Polygon()
    v_union = unary_union(v_rects) if v_rects else Polygon()

    segs = []
    # ложная «стена» — полоса, почти целиком лежащая внутри перпендикулярной стены
    # (её рождают вершины соседних стен, режущие толстую стену сеткой)
    for (x0, x1, y0, y1), r in zip(horiz, h_rects):
        if r.intersection(v_union).area <= 0.7 * r.area:
            segs.append(((x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2), y1 - y0))
    for (y0, y1, x0, x1), r in zip(vert, v_rects):
        if r.intersection(h_union).area <= 0.7 * r.area:
            segs.append((((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1), x1 - x0))
    return segs


def _merge_bands(bars, snap: float, max_t: float):
    """Склеить соседние параллельные полосы одной стены, разрезанные лишними вершинами.

    bars — (s0, s1, c0, c1): s — вдоль стены, c — поперёк.
    """
    bars = sorted(bars, key=lambda b: (b[2], b[0]))
    changed = True
    while changed:
        changed = False
        out: list[tuple[float, float, float, float]] = []
        for b in bars:
            for i, o in enumerate(out):
                touch = abs(o[3] - b[2]) <= snap or abs(b[3] - o[2]) <= snap
                overlap = min(o[1], b[1]) - max(o[0], b[0])
                shorter = min(o[1] - o[0], b[1] - b[0])
                if touch and overlap >= 0.8 * shorter and max(o[3], b[3]) - min(o[2], b[2]) <= max_t:
                    out[i] = (min(o[0], b[0]), max(o[1], b[1]), min(o[2], b[2]), max(o[3], b[3]))
                    changed = True
                    break
            else:
                out.append(b)
        bars = out
    return bars


def polygons_to_segments(
    geom: BaseGeometry, thickness: float | None = None
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """(a, b, thickness) для каждой стены в полигоне(ах).

    thickness — типичная толщина стены; если не задана, оценивается как 2·площадь/периметр.
    """
    parts = _parts(geom)
    if not parts:
        return []
    # толщина известна (ResPlan) — узкий допуск; оценена — широкий, стены бывают разные
    lo_k, hi_k = (0.55, 1.6) if thickness is not None else (0.3, 4.0)
    if thickness is None:
        # для тонких вытянутых фигур площадь ≈ длина × t, периметр ≈ 2 × длина
        perim = sum(p.exterior.length + sum(r.length for r in p.interiors) for p in parts)
        thickness = 2 * sum(p.area for p in parts) / perim
    snap = 0.12 * thickness
    out = []
    for p in parts:
        if p.area < (thickness**2) * 0.5:
            continue  # мусор и пилоны
        rect = p.minimum_rotated_rectangle
        if p.area / rect.area > 0.95:
            out.append(_rect_segment(p))
            continue
        simplified = p.simplify(0.1 * thickness)
        if not isinstance(simplified, Polygon) or simplified.is_empty:
            simplified = p
        segs = _orthogonal_segments(simplified, thickness, snap, lo_k, hi_k)
        out.extend(segs)
        # что не покрыли ортогональные стены (косые участки) — прямоугольниками
        covered = unary_union([_rect(a, b, t) for a, b, t in segs]) if segs else Polygon()
        rest = p.difference(covered.buffer(snap))
        for r in _parts(rest):
            if r.area < thickness**2:
                continue
            rr = r.minimum_rotated_rectangle
            if r.area / rr.area > 0.6:
                out.append(_rect_segment(r))
    return out


def _rect(a, b, width):
    from .core import rect_along

    return rect_along(a, b, width)
