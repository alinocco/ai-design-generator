"""SVG-рендер ProjectState: план, проёмы, подписи с площадями, размеры и расстановка.

Площади в подписях всегда вычисляются из полигонов — тем же кодом, что и везде.
"""

from __future__ import annotations

import math
from html import escape

from shapely.geometry import Point as SPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polylabel, unary_union

from ..geometry.core import (
    door_swing,
    item_polygon,
    opening_polygon,
    opening_thickness,
    room_area_m2,
    room_polygon,
    wall_polygon,
    walls_geometry,
)
from ..geometry.validate import Issue
from ..schemas import Layout, OpeningKind, Plan, RoomType

ROOM_FILL = {
    RoomType.living: "#F1E8DA", RoomType.kitchen: "#F4EAD5", RoomType.kitchen_living: "#F1E8DA",
    RoomType.bedroom: "#EAE6F2", RoomType.bathroom: "#E4E9EF", RoomType.toilet: "#E4E9EF",
    RoomType.hall: "#EEEEEA", RoomType.corridor: "#EEEEEA", RoomType.storage: "#ECE7E0",
    RoomType.balcony: "#E9EFE6", RoomType.other: "#F2F2F2",
}
INK, MUTE, ACCENT, WARN = "#1B1F24", "#5B6470", "#2F5BEA", "#B45309"


def _path(geom: BaseGeometry) -> str:
    polys = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
    out = []
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            pts = list(ring.coords)
            out.append("M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in pts) + " Z")
    return " ".join(out)


def _fmt_area(a: float) -> str:
    return f"{a:.1f}".replace(".", ",")


def render_plan(
    plan: Plan,
    layout: Layout | None = None,
    issues: list[Issue] | None = None,
    dims: bool = True,
    width_px: int = 1000,
) -> str:
    rooms = [room_polygon(r) for r in plan.rooms]
    geoms = rooms + [wall_polygon(w) for w in plan.walls]
    if not geoms:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"
    minx = min(g.bounds[0] for g in geoms)
    miny = min(g.bounds[1] for g in geoms)
    maxx = max(g.bounds[2] for g in geoms)
    maxy = max(g.bounds[3] for g in geoms)
    span = max(maxx - minx, maxy - miny)
    pad = span * (0.12 if dims else 0.04)
    font = span / 55
    sw = span / 900  # тонкая линия
    vb = (minx - pad, miny - pad, maxx - minx + 2 * pad, maxy - miny + 2 * pad)
    height_px = int(width_px * vb[3] / vb[2])
    o: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{vb[0]:.0f} {vb[1]:.0f} {vb[2]:.0f} {vb[3]:.0f}' "
        f"width='{width_px}' height='{height_px}' font-family='Helvetica, Arial, sans-serif'>",
        f"<rect x='{vb[0]:.0f}' y='{vb[1]:.0f}' width='{vb[2]:.0f}' height='{vb[3]:.0f}' fill='#FFFFFF'/>",
    ]
    flagged = {i.object_id for i in issues or [] if i.object_id}

    for r, poly in zip(plan.rooms, rooms):
        o.append(f"<path d='{_path(poly)}' fill='{ROOM_FILL[r.type]}' stroke='none'/>")

    o.append(f"<path d='{_path(walls_geometry(plan))}' fill='{INK}' fill-rule='evenodd'/>")
    holes = unary_union([opening_polygon(plan, op) for op in plan.openings])
    for w in plan.walls:
        # подсвечиваем стены с проблемами и стены, в типе которых система не уверена
        if w.id in flagged or (w.structural.value == "unknown" and w.confidence < 0.9):
            o.append(f"<path d='{_path(wall_polygon(w).difference(holes))}' fill='{WARN}' "
                     "fill-opacity='0.85'/>")

    for op in plan.openings:
        t = opening_thickness(plan, op)
        if op.kind == OpeningKind.window:
            for k in (-0.5, 0, 0.5):
                (ax, ay), (bx, by) = op.a, op.b
                L = math.dist(op.a, op.b) or 1
                nx, ny = -(by - ay) / L * t * k, (bx - ax) / L * t * k
                o.append(f"<line x1='{ax + nx:.0f}' y1='{ay + ny:.0f}' x2='{bx + nx:.0f}' "
                         f"y2='{by + ny:.0f}' stroke='{INK}' stroke-width='{sw * 2:.1f}'/>")
        else:
            swing = door_swing(plan, op)
            if swing is not None:
                pts = list(swing.exterior.coords)
                hinge, arc = pts[0], pts[1:-1]
                o.append(f"<line x1='{hinge[0]:.0f}' y1='{hinge[1]:.0f}' x2='{arc[-1][0]:.0f}' "
                         f"y2='{arc[-1][1]:.0f}' stroke='{INK}' stroke-width='{sw * 2.5:.1f}'/>")
                d = "M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in arc)
                o.append(f"<path d='{d}' fill='none' stroke='{INK}' stroke-width='{sw:.1f}' "
                         f"stroke-dasharray='{sw * 6:.0f} {sw * 4:.0f}'/>")
            if op.kind == OpeningKind.entrance:
                mx, my = (op.a[0] + op.b[0]) / 2, (op.a[1] + op.b[1]) / 2
                o.append(f"<circle cx='{mx:.0f}' cy='{my:.0f}' r='{font * 0.35:.0f}' fill='{ACCENT}'/>")

    if layout:
        for it in layout.items:
            p = item_polygon(it)
            stroke = "#C2410C" if it.id in flagged else INK
            o.append(f"<path d='{_path(p)}' fill='#FFFFFF' fill-opacity='0.92' stroke='{stroke}' "
                     f"stroke-width='{sw * 2:.1f}'/>")
            c = p.centroid
            o.append(f"<text x='{c.x:.0f}' y='{c.y + font * 0.3:.0f}' font-size='{font * 0.75:.0f}' "
                     f"fill='{MUTE}' text-anchor='middle'>{escape(it.type)}</text>")

    for f in plan.fixtures:
        color = ACCENT if f.kind.value.endswith("riser") else MUTE
        dash = f" stroke='{color}' stroke-dasharray='{sw * 4:.0f}'" if f.inferred else ""
        o.append(f"<circle cx='{f.at[0]:.0f}' cy='{f.at[1]:.0f}' r='{font * 0.45:.0f}' "
                 f"fill='{color}' fill-opacity='{0.4 if f.inferred else 1}'{dash}/>")

    obstacles = [item_polygon(it) for it in layout.items] if layout else []
    obstacles += [s for op in plan.openings if (s := door_swing(plan, op)) is not None]
    obstacles += [SPoint(f.at).buffer(font * 1.5) for f in plan.fixtures]
    taken = unary_union(obstacles) if obstacles else None
    for r, poly in zip(plan.rooms, rooms):
        c = _label_point(poly, taken, font)
        color = WARN if r.id in flagged else INK
        o.append(f"<text x='{c.x:.0f}' y='{c.y:.0f}' font-size='{font:.0f}' font-weight='600' "
                 f"fill='{color}' text-anchor='middle'>{escape(r.display_name)}</text>")
        o.append(f"<text x='{c.x:.0f}' y='{c.y + font * 1.25:.0f}' font-size='{font * 0.85:.0f}' "
                 f"fill='{MUTE}' text-anchor='middle'>{_fmt_area(room_area_m2(r))} м²</text>")

    if dims:
        o.extend(_overall_dims(minx, miny, maxx, maxy, font, sw))
    o.append("</svg>")
    return "".join(o)


def _label_point(poly: Polygon, taken, font: float):
    """Точка для подписи: самое «просторное» место комнаты, свободное от мебели."""
    free = poly.difference(taken) if taken is not None else poly
    parts = [g for g in getattr(free, "geoms", [free]) if isinstance(g, Polygon) and not g.is_empty]
    if not parts:
        return poly.representative_point()
    target = max(parts, key=lambda g: g.area)
    try:
        return polylabel(target, tolerance=font / 4)
    except Exception:  # noqa: BLE001
        return target.representative_point()


def _overall_dims(minx, miny, maxx, maxy, font, sw) -> list[str]:
    off = font * 2.2
    t = font * 0.4
    out = []
    y = miny - off
    out.append(f"<line x1='{minx:.0f}' y1='{y:.0f}' x2='{maxx:.0f}' y2='{y:.0f}' stroke='{MUTE}' stroke-width='{sw:.1f}'/>")
    for x in (minx, maxx):
        out.append(f"<line x1='{x - t:.0f}' y1='{y + t:.0f}' x2='{x + t:.0f}' y2='{y - t:.0f}' stroke='{MUTE}' stroke-width='{sw * 2:.1f}'/>")
    out.append(f"<text x='{(minx + maxx) / 2:.0f}' y='{y - font * 0.4:.0f}' font-size='{font * 0.85:.0f}' fill='{MUTE}' text-anchor='middle'>{maxx - minx:.0f}</text>")
    x = minx - off
    out.append(f"<line x1='{x:.0f}' y1='{miny:.0f}' x2='{x:.0f}' y2='{maxy:.0f}' stroke='{MUTE}' stroke-width='{sw:.1f}'/>")
    for yy in (miny, maxy):
        out.append(f"<line x1='{x - t:.0f}' y1='{yy + t:.0f}' x2='{x + t:.0f}' y2='{yy - t:.0f}' stroke='{MUTE}' stroke-width='{sw * 2:.1f}'/>")
    cy = (miny + maxy) / 2
    out.append(f"<text x='{x - font * 0.4:.0f}' y='{cy:.0f}' font-size='{font * 0.85:.0f}' fill='{MUTE}' text-anchor='middle' transform='rotate(-90 {x - font * 0.4:.0f} {cy:.0f})'>{maxy - miny:.0f}</text>")
    return out

