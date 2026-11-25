"""Обмерщик v0: распознавание маркетингового плана застройщика без обучения.

Рассчитан на «чистые» планы: стены залиты сплошным тёмным цветом (чёрным или
серым), двери нарисованы тонкими дугами, площади подписаны цифрами.

Конвейер:
 1. Маска стен — тёмные малонасыщенные пиксели основного «стенового» тона;
    морфологическое открытие убирает тонкие линии: дуги дверей, окна, мебель, текст.
 2. Векторизация — контуры маски → полигоны → отрезки стен (geometry.walls).
 3. Проёмы — от каждого свободного торца стены «смотрим» вдоль её оси: если на
    расстоянии 1,5–12 толщин стены есть другая стена, промежуток — проём.
    Тип проёма — по тонким линиям в пикселях вокруг: дуга → дверь (с открыванием),
    линии внутри наружной стены → окно, пусто → проём без двери.
 4. Помещения — связные области внутри контура квартиры после «заделки» проёмов.
 5. Масштаб — по площадям, прочитанным OCR: медиана sqrt(подпись / площадь в px²).
    Подписи — эталон, по которому мы проверяем, что геометрия распознана верно.

Всё, в чём конвейер не уверен, помечается confidence < 1, чтобы Валидатор и
пользователь видели спорные места.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..geometry.core import rect_along
from ..geometry.walls import _parts, polygons_to_segments
from ..schemas import (
    Opening, OpeningKind, Plan, PlanSource, Room, RoomType, Swing, Wall, WallPosition,
)

NUM_RE = re.compile(r"^\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:м2|м²|m2)?\s*$")


@dataclass
class Recognition:
    plan: Plan
    image_size: tuple[int, int]
    mm_per_px: float | None
    labels: list[tuple[float, float, float]] = field(default_factory=list)  # x, y (px), м²
    wall_px: float = 0.0
    notes: list[str] = field(default_factory=list)


# --- 1. маска стен -------------------------------------------------------------------

def wall_mask(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Маска стен и оценка толщины стены в пикселях."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    dark = (sat < 60) & (val < 140)
    if dark.sum() == 0:
        raise ValueError("на картинке нет тёмных малонасыщенных пикселей — это не план?")
    # основной тон стен — самый частый уровень яркости среди тёмных пикселей
    hist = np.bincount(val[dark].ravel(), minlength=256)
    peak = int(np.argmax(hist))
    mask = (dark & (np.abs(val.astype(int) - peak) <= 28)).astype(np.uint8)

    # открытие маленьким ядром: тонкие линии (дуги дверей, окна, мебель, текст) уходят,
    # самые тонкие перегородки (~0,1 м) остаются
    k = max(3, int(round(0.004 * max(img.shape[:2]))) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    # оставить только крупные куски (стены квартиры), выкинуть компас, значки, текст
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise ValueError("стены не найдены")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(areas.max() * 0.02, k * k * 16)
    mask = keep[lab].astype(np.uint8)
    # типичная толщина стены: для тонких вытянутых фигур площадь ≈ длина × t, периметр ≈ 2 × длина
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    t = 2 * float(mask.sum()) / max(perim, 1.0)
    return mask, t


def mask_to_polygons(mask: np.ndarray, simplify_px: float) -> list[Polygon]:
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[Polygon] = []
    if hier is None:
        return polys
    hier = hier[0]
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:
            continue  # внутренние контуры обработаем как дыры
        holes = []
        j = hier[i][2]
        while j != -1:
            if len(contours[j]) >= 3:
                holes.append(contours[j][:, 0, :].astype(float))
            j = hier[j][0]
        p = Polygon(c[:, 0, :].astype(float), holes).buffer(0)
        for part in _parts(p.simplify(simplify_px)):
            if part.area > 0:
                polys.append(part)
    return polys


# --- 3. проёмы -----------------------------------------------------------------------

@dataclass
class _Gap:
    a: tuple[float, float]
    b: tuple[float, float]
    t: float


def find_gaps(segs, walls_geom, t: float) -> list[_Gap]:
    """Промежутки между торцом стены и следующей стеной на её оси."""
    gaps: list[_Gap] = []
    rects = [rect_along(a, b, th) for a, b, th in segs]
    for idx, (a, b, th) in enumerate(segs):
        L = math.dist(a, b)
        if L < 1e-6:
            continue
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        others = unary_union([r for j, r in enumerate(rects) if j != idx])
        for end, sign in ((b, 1), (a, -1)):
            ax_dx, ax_dy = ux * sign, uy * sign
            # торец свободен, только если концевой кусок стены не лежит в другой стене
            # (на Т-стыке конец перегородки — на дальней грани перпендикулярной стены)
            cap_len = min(L, max(th, 0.5 * t))
            cap = rect_along((end[0] - ax_dx * cap_len, end[1] - ax_dy * cap_len), end, th * 0.8)
            if not others.is_empty and cap.intersection(others).area > 0.4 * cap.area:
                continue
            best = None
            # вдоль оси стены — окна и проёмы между двумя торцами;
            # поперёк — дверь, когда стена обрывается у самой двери, а косяк — соседняя стена
            for dx, dy, along in ((ax_dx, ax_dy, True), (-ax_dy, ax_dx, False), (ax_dy, -ax_dx, False)):
                if along:
                    origin = end
                else:  # поперечный луч выходит из боковой грани у самого торца
                    back = (end[0] - ax_dx * th / 2, end[1] - ax_dy * th / 2)
                    origin = (back[0] + dx * th / 2, back[1] + dy * th / 2)
                start = (origin[0] + dx * 1.0, origin[1] + dy * 1.0)
                far = (origin[0] + dx * 20 * t, origin[1] + dy * 20 * t)
                hit = LineString([start, far]).intersection(walls_geom)
                if hit.is_empty:
                    continue
                d = min(Point(start).distance(g) for g in getattr(hit, "geoms", [hit]))
                if d < (1.5 if along else 2.5) * t:
                    continue
                stop = (start[0] + dx * d, start[1] + dy * d)
                # напротив — торец стены на той же оси (окно, широкий проём)
                # или грань перпендикулярной стены (дверь у угла: не шире ~6 толщин)
                facing_end = along and any(
                    abs((sb[0] - sa[0]) * dy - (sb[1] - sa[1]) * dx) < 0.09 * math.dist(sa, sb)
                    and min(math.dist(stop, sa), math.dist(stop, sb)) <= max(t, sth)
                    for j, (sa, sb, sth) in enumerate(segs) if j != idx and math.dist(sa, sb) > 0
                )
                if not facing_end and d > 6 * t:
                    continue
                if best is None or d < best[0]:
                    best = (d, origin, stop)
            if best is not None:
                gaps.append(_Gap(a=best[1], b=best[2], t=th))
    # ложный проём: идёт вдоль стены в её толще или вплотную к ней
    near_walls = walls_geom.buffer(t * 0.5)
    gaps = [g for g in gaps
            if rect_along(g.a, g.b, g.t).intersection(near_walls).area
            <= 0.5 * rect_along(g.a, g.b, g.t).area]
    # дубли: два торца смотрят друг на друга
    uniq: list[_Gap] = []
    for g in gaps:
        mid = Point((g.a[0] + g.b[0]) / 2, (g.a[1] + g.b[1]) / 2)
        if any(Point((u.a[0] + u.b[0]) / 2, (u.a[1] + u.b[1]) / 2).distance(mid) < t for u in uniq):
            continue
        uniq.append(g)
    return uniq


def classify_gap(gray_thin: np.ndarray, g: _Gap, exterior: bool):
    """Тип проёма по тонким линиям вокруг: дуга двери, рама окна или пусто."""
    h, w = gray_thin.shape
    L = math.dist(g.a, g.b)
    ux, uy = (g.b[0] - g.a[0]) / L, (g.b[1] - g.a[1]) / L
    nx, ny = -uy, ux

    def ink(poly: Polygon) -> tuple[float, Point | None]:
        x0, y0, x1, y1 = (int(max(0, v)) for v in poly.bounds)
        x1, y1 = min(w, x1 + 1), min(h, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0, None
        sub = gray_thin[y0:y1, x0:x1]
        poly_mask = np.zeros_like(sub)
        pts = np.array([[(x - x0, y - y0) for x, y in poly.exterior.coords]], np.int32)
        cv2.fillPoly(poly_mask, pts, 1)
        ys, xs = np.nonzero(sub & poly_mask)
        if len(xs) == 0:
            return 0.0, None
        return len(xs) / max(1.0, poly.area), Point(xs.mean() + x0, ys.mean() + y0)

    half = g.t / 2
    side = {}
    for s in (1, -1):
        ox, oy = nx * s, ny * s
        quad = Polygon([
            (g.a[0] + ox * half, g.a[1] + oy * half), (g.b[0] + ox * half, g.b[1] + oy * half),
            (g.b[0] + ox * (half + L), g.b[1] + oy * (half + L)),
            (g.a[0] + ox * (half + L), g.a[1] + oy * (half + L)),
        ])
        side[s] = ink(quad)
    inside_gap, _ = ink(rect_along(g.a, g.b, g.t * 0.9))

    if exterior and inside_gap > 0.04:
        return OpeningKind.window, None  # рама окна нарисована прямо в проёме
    best = max(side, key=lambda s: side[s][0])
    density, centroid = side[best]
    if density > 0.02 and centroid is not None:
        # центроид четверти круга смещён к петле
        hinge = "a" if centroid.distance(Point(g.a)) < centroid.distance(Point(g.b)) else "b"
        # «left» — нормаль (by-ay, ax-bx), см. geometry.core.door_swing
        left = (g.b[1] - g.a[1], g.a[0] - g.b[0])
        is_left = (left[0] * nx * best + left[1] * ny * best) > 0
        kind = OpeningKind.entrance if exterior else OpeningKind.door
        return kind, Swing(hinge=hinge, side="left" if is_left else "right")
    if exterior:
        return OpeningKind.window, None  # окно без рамы встречается чаще, чем вход без двери
    return OpeningKind.passage, None


# --- OCR -----------------------------------------------------------------------------

_OCR = None


def read_area_labels(img: np.ndarray) -> list[tuple[float, float, float]]:
    """(x, y, м²) для всех числовых подписей. Нужен пакет rapidocr-onnxruntime."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR = RapidOCR()
    res, _ = _OCR(img)
    out = []
    for box_pts, text, score in res or []:
        m = NUM_RE.match(text.replace(" ", ""))
        if not m or float(score) < 0.5:
            continue
        val = float(m.group(1).replace(",", "."))
        if 0.8 <= val <= 120:
            xs = [p[0] for p in box_pts]
            ys = [p[1] for p in box_pts]
            out.append((sum(xs) / 4, sum(ys) / 4, val))
    return out


# --- сборка ---------------------------------------------------------------------------

def recognize(path: str | Path, use_ocr: bool = True) -> Recognition:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    mask, t = wall_mask(img)
    polys = mask_to_polygons(mask, simplify_px=max(1.0, t * 0.15))
    walls_px = unary_union(polys)
    segs = polygons_to_segments(walls_px)  # толщина у наружных и внутренних стен разная
    notes: list[str] = []

    # тонкие линии = всё тёмное, что не стена
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thin = ((hsv[..., 1] < 60) & (hsv[..., 2] < 200)).astype(np.uint8)
    thin[cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0] = 0

    gaps = find_gaps(segs, walls_px, t)
    closed = unary_union([walls_px] + [rect_along(g.a, g.b, g.t) for g in gaps])
    # контур квартиры: «заливаем» стены вместе с длинными окнами и балконными дверями,
    # которые не закрылись как проёмы, — морфологическим закрытием радиусом R
    R = 10 * t
    hull = closed.buffer(R, join_style=2).buffer(-R, join_style=2)
    footprint = unary_union([Polygon(p.exterior) for p in _parts(hull)])
    outer = max(_parts(footprint), key=lambda p: p.area)
    outer = outer.intersection(Polygon(max(_parts(unary_union(
        [Polygon(p.exterior) for p in _parts(closed)] + [outer.buffer(-t)])), key=lambda p: p.area).exterior))
    boundary = outer.exterior

    labels = read_area_labels(img) if use_ocr else []
    labels_in = [(x, y, a) for x, y, a in labels if outer.contains(Point(x, y))]

    # помещения: watershed по свободной площади, семена — подписи площадей в центрах комнат;
    # граница между соседними семенами ложится в самое узкое место — в дверной проём
    room_polys, seed_area = _rooms_watershed(img.shape[:2], outer, closed, mask, labels_in, t)
    if not labels_in:
        notes.append("подписи площадей не найдены — помещения выделены только по проёмам")

    pairs = [(i, a) for i, a in seed_area.items()]
    ratios = [math.sqrt(a * 1e6 / room_polys[i].area) for i, a in pairs]
    if ratios:
        # масштаб, при котором с подписями совпадает больше всего помещений:
        # одна неверно найденная комната не должна сбивать масштаб всего плана
        def agree(kk):
            return sum(abs(room_polys[i].area * kk * kk / 1e6 - a) <= 0.05 * a for i, a in pairs)

        best = max(ratios, key=lambda kk: (agree(kk), -abs(kk - float(np.median(ratios)))))
        inliers = [r for r in ratios if abs(r - best) / best <= 0.03]
        k = float(np.median(inliers))
    else:
        k = 200.0 / t  # запасной вариант: стена ≈ 200 мм
        notes.append("нет подписей площадей — масштаб оценён по толщине стен, точность низкая")

    def pt(xy):
        return (round(xy[0] * k, 1), round(xy[1] * k, 1))

    walls: list[Wall] = []
    for i, (a, b, th) in enumerate(segs, 1):
        mid = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ext = boundary.distance(mid) < th
        walls.append(Wall(id=f"w{i}", a=pt(a), b=pt(b), thickness=round(th * k, 1),
                          position=WallPosition.exterior if ext else WallPosition.interior,
                          confidence=0.8))

    openings: list[Opening] = []
    counters: dict[str, int] = {}
    for g in gaps:
        mid = Point((g.a[0] + g.b[0]) / 2, (g.a[1] + g.b[1]) / 2)
        ext = boundary.distance(mid) < g.t
        kind, swing = classify_gap(thin, g, ext)
        counters[kind.value] = counters.get(kind.value, 0) + 1
        openings.append(Opening(id=f"{kind.value}{counters[kind.value]}", kind=kind,
                                a=pt(g.a), b=pt(g.b), swing=swing, confidence=0.7))
    if not any(o.kind == OpeningKind.entrance for o in openings):
        notes.append("входная дверь не найдена")

    label_of = seed_area
    rooms: list[Room] = []
    for i, p in enumerate(room_polys):
        poly = Polygon(p.exterior).simplify(t * 0.2)
        coords = [pt(c) for c in list(poly.exterior.coords)[:-1]]
        rooms.append(Room(id=f"r{i + 1}", type=RoomType.other, polygon=coords,
                          label_area_m2=label_of.get(i), confidence=0.7))

    plan = Plan(walls=walls, openings=openings, rooms=rooms,
                source=PlanSource(kind="recognized", ref=Path(path).name, mm_per_px=round(k, 3)))
    from ..geometry.room_types import infer_room_types

    infer_room_types(plan)  # грубая догадка по геометрии; позже — VLM
    return Recognition(plan=plan, image_size=(img.shape[1], img.shape[0]), mm_per_px=k,
                       labels=labels, wall_px=t, notes=notes)


def _rooms_watershed(shape, outer: Polygon, closed, wall_mask_px: np.ndarray, labels, t: float):
    """Помещения как области watershed. Возвращает полигоны и {индекс: подписанная площадь}."""
    h, w = shape
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [np.array(outer.exterior.coords, np.int32)], 1)
    for hole in outer.interiors:
        cv2.fillPoly(inside, [np.array(hole.coords, np.int32)], 0)
    walls = wall_mask_px.copy()
    # закрытые проёмы тоже считаем стеной: они помогают разделить комнаты
    for g in _parts(closed):
        layer = np.zeros((h, w), np.uint8)
        cv2.fillPoly(layer, [np.array(g.exterior.coords, np.int32)], 1)
        for hole in g.interiors:
            cv2.fillPoly(layer, [np.array(hole.coords, np.int32)], 0)
        walls |= layer
    free = (inside & (1 - walls)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    markers = np.zeros((h, w), np.int32)
    seeds = []
    for x, y, a in labels:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h and free[yi, xi]:
            seeds.append(a)
            cv2.circle(markers, (xi, yi), max(2, int(t / 3)), len(seeds) + 1, -1)
    markers[free == 0] = 0
    # области без подписи (кладовые, коридоры без площади) — свои семена в центрах
    n_cc, cc = cv2.connectedComponents(free)
    for cid in range(1, n_cc):
        region = cc == cid
        if region.sum() < (4 * t) ** 2 or (markers[region] > 0).any():
            continue
        dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        yi, xi = np.unravel_index(np.argmax(dist), dist.shape)
        seeds.append(None)
        cv2.circle(markers, (int(xi), int(yi)), max(2, int(t / 3)), len(seeds) + 1, -1)
    from skimage.segmentation import watershed

    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    markers = watershed(-dist, markers, mask=free.astype(bool))

    polys, seed_area = [], {}
    for sid, area in enumerate(seeds, start=2):
        region = ((markers == sid) & (free == 1)).astype(np.uint8)
        if region.sum() < (2 * t) ** 2:
            continue
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 3:
            continue
        poly = Polygon(cnt[:, 0, :].astype(float)).buffer(0)
        poly = max(_parts(poly), key=lambda p: p.area) if not isinstance(poly, Polygon) else poly
        if area is not None:
            seed_area[len(polys)] = area
        polys.append(poly)
    return polys, seed_area


def area_agreement(plan: Plan, tol: float = 0.05) -> tuple[int, int]:
    """Сколько помещений с подписью совпали по площади (в пределах tol) — из скольких."""
    from ..geometry.core import room_area_m2

    labelled = [r for r in plan.rooms if r.label_area_m2]
    ok = sum(abs(room_area_m2(r) - r.label_area_m2) / r.label_area_m2 <= tol for r in labelled)
    return ok, len(labelled)
