"""Расстановка мебели в одной комнате: модель CP-SAT (OR-Tools).

Модель:
- комната и всё, куда ставить нельзя (зоны открывания дверей, подходы к проёмам,
  стояки), растрируются на сетку с шагом `grid` мм; запрещённые клетки — неподвижные
  прямоугольники в ограничении NoOverlap2D;
- каждый вариант предмета (тип × размер × поворот) — опциональный прямоугольник
  «предмет + зона обслуживания»; из вариантов одного запроса выбирается не больше одного;
- «спинкой к стене» — дизъюнкция по подходящим отрезкам границы комнаты;
- сантехника — жёстко в радиусе от стояка (если стояк в комнате есть);
- «у окна» — мягкое пожелание: штраф за расстояние до ближайшего окна.

Цель: поставить все обязательные предметы, затем как можно больше необязательных
(с весами), затем по возможности крупные варианты и мягкие пожелания.
LLM в модели не участвует: она формирует запросы и ограничения, координаты считает солвер.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
from ortools.sat.python import cp_model
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..catalog import CATALOG, Request, Spec
from ..geometry.core import door_swing, item_polygon, opening_polygon, room_polygon
from ..schemas import Constraint, FixtureKind, Item, OpeningKind, Plan, Rule

REQUIRED_WEIGHT = 10_000
DEFAULT_RISER_RADIUS = 1500
APPROACH_DEPTH = 600  # свободная зона перед любым проёмом внутри комнаты


@dataclass
class RoomResult:
    room_id: str
    items: list[Item] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)  # ключи обязательных запросов
    status: str = ""
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class _Option:
    req: Request
    spec: Spec
    size: tuple[int, int]
    size_rank: int
    rot: int
    p: cp_model.IntVar
    x: cp_model.IntVar
    y: cp_model.IntVar
    ext: tuple[int, int]  # размеры прямоугольника с зоной обслуживания, в клетках
    foot_off: tuple[int, int]
    foot: tuple[int, int]
    attach: tuple = ()  # пристроенные предметы этого варианта
    alt_rank: int = 0


def _footprint_mm(size: tuple[int, int], rot: int) -> tuple[int, int]:
    w, d = size
    return (w, d) if rot in (0, 180) else (d, w)


def _blocked_geometry(plan: Plan, room_id: str, room: Polygon):
    """Что занято в комнате: (жёстко — створки дверей и стояки; подходы к проёмам).

    В подходы нельзя ставить предметы, но зона обслуживания может на них заходить:
    это тот же свободный пол."""
    parts, approach = [], []
    for o in plan.openings:
        zone = opening_polygon(plan, o).buffer(60)
        if not room.intersects(zone):
            continue
        if o.kind == OpeningKind.window:
            continue
        swing = door_swing(plan, o)
        if swing is not None and swing.intersects(room):
            parts.append(swing)
        # подход: прямоугольник внутрь комнаты на ширину проёма
        ax, ay = o.a
        bx, by = o.b
        L = math.dist(o.a, o.b) or 1
        nx, ny = -(by - ay) / L, (bx - ax) / L
        for s in (1, -1):
            depth = APPROACH_DEPTH + 150
            quad = Polygon([(ax, ay), (bx, by), (bx + nx * s * depth, by + ny * s * depth),
                            (ax + nx * s * depth, ay + ny * s * depth)])
            approach.append(quad.intersection(room))
    for f in plan.fixtures:
        # угаданный стояк (inferred) — лишь ориентир для радиуса сантехники, место не занимает
        if not getattr(f, "inferred", False) and room.buffer(200).contains(Point(f.at)):
            parts.append(Point(f.at).buffer(180, cap_style=3))
    hard = unary_union(parts) if parts else Polygon()
    appr = unary_union(approach) if approach else Polygon()
    return hard, appr


def _blocked_rects(allowed: Polygon, x0: int, y0: int, nx: int, ny: int, g: int):
    """Клетки сетки вне разрешённой области → прямоугольники (в клетках)."""
    xs = (np.arange(nx) + x0) * g
    ys = (np.arange(ny) + y0) * g
    X, Y = np.meshgrid(xs, ys)
    inset = 2
    ok = np.ones_like(X, dtype=bool)
    for dx, dy in ((inset, inset), (g - inset, inset), (inset, g - inset), (g - inset, g - inset),
                   (g / 2, g / 2)):
        ok &= shapely.contains_xy(allowed, X + dx, Y + dy)
    rects = []
    # горизонтальные пробеги запрещённых клеток, затем склейка по вертикали
    runs_prev: dict[tuple[int, int], list[int]] = {}
    for j in range(ny):
        row = ~ok[j]
        runs = []
        i = 0
        while i < nx:
            if row[i]:
                s = i
                while i < nx and row[i]:
                    i += 1
                runs.append((s, i))
            else:
                i += 1
        cur: dict[tuple[int, int], list[int]] = {}
        for r in runs:
            if r in runs_prev:
                rec = runs_prev[r]
                rec[3] = j + 1
                cur[r] = rec
            else:
                rec = [r[0], j, r[1], j + 1]
                rects.append(rec)
                cur[r] = rec
        runs_prev = cur
    return [(a + x0, b + y0, c - a, d - b) for a, b, c, d in rects]  # x, y, w, h


def _wall_edges(room: Polygon, min_len: float = 300.0):
    """Почти осевые отрезки границы (допуск 2%) и сторона, где внутренность комнаты."""
    out = {0: [], 90: [], 180: [], 270: []}  # поворот предмета, спинка которого ляжет на отрезок
    coords = list(room.exterior.coords)
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx >= min_len and dy <= max(20.0, 0.02 * dx):  # горизонтальный
            y = (y1 + y2) / 2
            xa, xb = sorted((x1, x2))
            mx = (xa + xb) / 2
            if room.contains(Point(mx, y + 30)):
                out[0].append(("h", max(y1, y2), xa, xb))  # внутри ниже → спинка вверху
            elif room.contains(Point(mx, y - 30)):
                out[180].append(("h", min(y1, y2), xa, xb))
        elif dy >= min_len and dx <= max(20.0, 0.02 * dy):  # вертикальный
            x = (x1 + x2) / 2
            ya, yb = sorted((y1, y2))
            my = (ya + yb) / 2
            if room.contains(Point(x - 30, my)):
                out[90].append(("v", min(x1, x2), ya, yb))  # внутри левее → спинка справа
            elif room.contains(Point(x + 30, my)):
                out[270].append(("v", max(x1, x2), ya, yb))
    return out


def solve_room(
    plan: Plan,
    room_id: str,
    requests: list[Request],
    constraints: list[Constraint] | None = None,
    fixed: list[Item] | None = None,
    grid: int | None = None,
    time_limit: float = 5.0,
    seed: int = 0,
    avoid: list[Item] | None = None,
) -> RoomResult:
    """avoid — прежние положения предметов, которые пользователь просил переставить:
    предмет того же типа не может снова накрыть центр старого места."""
    t0 = time.perf_counter()
    res = RoomResult(room_id=room_id)
    constraints = constraints or []
    fixed = fixed or []
    room = room_polygon(plan.room(room_id))
    if not room.is_valid or room.area < 1e6:
        res.status = "skipped"
        return res
    if grid is None:  # в санузлах счёт идёт на сантиметры: шаг 50 мм съедает до 10 см ширины
        grid = 25 if room.area < 8e6 else 50

    # ограничения агентов → параметры модели
    excluded = {c.subject for c in constraints if c.rule == Rule.exclude}
    riser_r: dict[str, int] = {}
    clear_override: dict[str, int] = {}
    hard_wall: set[str] = set()
    near: list[tuple[str, str, str]] = []  # (тип, цель, hard|soft)
    for c in constraints:
        if c.rule == Rule.within_riser_radius:
            riser_r[c.subject] = int(c.params.get("mm", DEFAULT_RISER_RADIUS))
        elif c.rule == Rule.min_clearance:
            clear_override[c.subject] = int(c.params.get("mm", 0))
        elif c.rule == Rule.against_wall and c.kind == "hard":
            hard_wall.add(c.subject)
        elif c.rule == Rule.near and c.target:
            near.append((c.subject, c.target, c.kind))
        elif c.rule not in (Rule.exclude, Rule.not_in_swing, Rule.keep, Rule.in_room, Rule.require):
            res.notes.append(f"ограничение {c.id} ({c.rule.value}) солвер пока не учитывает")

    hard, appr = _blocked_geometry(plan, room_id, room)
    for it in fixed:
        hard = hard.union(item_polygon(it))
    free_hard = room.difference(hard)
    free_all = free_hard.difference(appr)
    minx, miny, maxx, maxy = room.bounds
    gx0, gy0 = math.floor(minx / grid), math.floor(miny / grid)
    gx1, gy1 = math.ceil(maxx / grid), math.ceil(maxy / grid)
    nx, ny = gx1 - gx0, gy1 - gy0
    shapely.prepare(free_hard)
    shapely.prepare(free_all)
    rects = _blocked_rects(free_all, gx0, gy0, nx, ny, grid)
    rects_hard = _blocked_rects(free_hard, gx0, gy0, nx, ny, grid)
    edges = _wall_edges(room)

    windows = [o for o in plan.openings if o.kind == OpeningKind.window
               and room.buffer(300).intersects(opening_polygon(plan, o))]
    risers = [f for f in plan.fixtures
              if f.kind in (FixtureKind.water_riser, FixtureKind.sewer_riser)
              and room.buffer(300).contains(Point(f.at))]
    fixtures_by_id = {f.id: f for f in plan.fixtures}

    m = cp_model.CpModel()
    # Два уровня непересечения:
    #  1) предмет + половина его зоны обслуживания не пересекаются друг с другом и со всем занятым —
    #     так между двумя предметами, смотрящими друг на друга, остаётся сумма половин, а зоны
    #     соседних предметов (унитаз и раковина) могут делить общий свободный пол;
    #  2) полная зона обслуживания каждого предмета целиком в комнате и не на створке двери.
    xs_iv, ys_iv = [], []
    for (x, y, w, h) in rects:
        xs_iv.append(m.NewFixedSizeIntervalVar(x, w, f"bx{x}_{y}"))
        ys_iv.append(m.NewFixedSizeIntervalVar(y, h, f"by{x}_{y}"))
    hard_iv = [(m.NewFixedSizeIntervalVar(x, w, ""), m.NewFixedSizeIntervalVar(y, h, ""))
               for (x, y, w, h) in rects_hard]

    options: list[_Option] = []
    objective = []
    by_req: dict[str, list[_Option]] = {}
    for req in requests:
        alts = [a for a in req.alternatives if a not in excluded]
        for alt_rank, alt in enumerate(alts):
            spec = CATALOG[alt]
            cf = clear_override.get(alt, spec.clear_front)
            ca = spec.clear_around
            wall = spec.against_wall or alt in hard_wall
            attach = tuple(a for a in spec.attach if a[0] not in excluded)
            variants = []  # (размер с пристройками, ранг, пристройки, сам размер)
            for rank, size in enumerate(spec.sizes):
                if attach:
                    extra = sum(aw * (2 if side == "both" else 1) for _, aw, _, side in attach)
                    variants.append(((size[0] + extra, size[1]), rank, attach, size))
                variants.append((size, rank + (len(spec.sizes) if attach else 0), (), size))
            for size, rank, att, own in variants:
                for rot in (0, 90, 180, 270):
                    fw_mm, fh_mm = _footprint_mm(size, rot)
                    fw, fh = math.ceil(fw_mm / grid), math.ceil(fh_mm / grid)
                    if fw > nx or fh > ny:
                        continue
                    cand_edges = edges[rot] if wall else []
                    if wall and not cand_edges:
                        continue

                    def margins(c_f, c_a):
                        l = r = t = b = c_a
                        if rot == 0:
                            b += c_f
                        elif rot == 180:
                            t += c_f
                        elif rot == 90:
                            l += c_f
                        else:
                            r += c_f
                        return l, r, t, b

                    full_m = margins(math.ceil(cf / grid), math.ceil(ca / grid))
                    half_m = margins(math.ceil(cf / 2 / grid), math.ceil(ca / 2 / grid))
                    if (fw + full_m[0] + full_m[1] > nx) or (fh + full_m[2] + full_m[3] > ny):
                        continue
                    tag = f"{req.key}_{alt}_{rank}_{rot}"
                    p = m.NewBoolVar(f"p_{tag}")
                    fx = m.NewIntVar(gx0 + full_m[0], gx1 - fw - full_m[1], f"x_{tag}")
                    fy = m.NewIntVar(gy0 + full_m[2], gy1 - fh - full_m[3], f"y_{tag}")
                    x, y = fx, fy
                    opt = _Option(req, spec, own, rank, rot, p, x, y, (fw, fh), (0, 0), (fw, fh),
                                  attach=att, alt_rank=alt_rank)
                    l, r_, t, b = half_m
                    xs_iv.append(m.NewOptionalFixedSizeIntervalVar(fx - l, fw + l + r_, p, ""))
                    ys_iv.append(m.NewOptionalFixedSizeIntervalVar(fy - t, fh + t + b, p, ""))
                    l, r_, t, b = full_m
                    if hard_iv and (l or r_ or t or b):
                        zx = m.NewOptionalFixedSizeIntervalVar(fx - l, fw + l + r_, p, "")
                        zy = m.NewOptionalFixedSizeIntervalVar(fy - t, fh + t + b, p, "")
                        m.AddNoOverlap2D([a for a, _ in hard_iv] + [zx], [c for _, c in hard_iv] + [zy])

                    if wall:
                        bs = []
                        for kind, c, a, b in cand_edges:
                            if kind == "h":
                                lo, hi = math.ceil(a / grid), math.floor(b / grid) - fw
                                if hi < lo:
                                    continue
                                bvar = m.NewBoolVar("")
                                if rot == 0:
                                    m.Add(fy == math.ceil(c / grid)).OnlyEnforceIf(bvar)
                                else:
                                    m.Add(fy + fh == math.floor(c / grid)).OnlyEnforceIf(bvar)
                                m.Add(fx >= lo).OnlyEnforceIf(bvar)
                                m.Add(fx <= hi).OnlyEnforceIf(bvar)
                            else:
                                lo, hi = math.ceil(a / grid), math.floor(b / grid) - fh
                                if hi < lo:
                                    continue
                                bvar = m.NewBoolVar("")
                                if rot == 270:
                                    m.Add(fx == math.ceil(c / grid)).OnlyEnforceIf(bvar)
                                else:
                                    m.Add(fx + fw == math.floor(c / grid)).OnlyEnforceIf(bvar)
                                m.Add(fy >= lo).OnlyEnforceIf(bvar)
                                m.Add(fy <= hi).OnlyEnforceIf(bvar)
                            bs.append(bvar)
                        if not bs:
                            m.Add(p == 0)
                        else:
                            m.Add(sum(bs) == p)

                    for old in avoid or []:
                        if old.type != alt:
                            continue
                        ox, oy = old.x / grid, old.y / grid  # центр старого места, в клетках
                        outs = [m.NewBoolVar("") for _ in range(4)]
                        m.Add(fx > math.floor(ox)).OnlyEnforceIf(outs[0])
                        m.Add(fx + fw < math.ceil(ox)).OnlyEnforceIf(outs[1])
                        m.Add(fy > math.floor(oy)).OnlyEnforceIf(outs[2])
                        m.Add(fy + fh < math.ceil(oy)).OnlyEnforceIf(outs[3])
                        m.AddBoolOr(outs).OnlyEnforceIf(p)

                    # центр предмета в полуклетках: 2·cx = 2·fx + fw
                    cx2, cy2 = 2 * fx + fw, 2 * fy + fh
                    if spec.plumbing and risers:
                        r = risers[0]
                        R2 = 2 * math.ceil(riser_r.get(alt, DEFAULT_RISER_RADIUS) / grid)
                        rx2, ry2 = round(2 * r.at[0] / grid), round(2 * r.at[1] / grid)
                        m.Add(cx2 - rx2 <= R2).OnlyEnforceIf(p)
                        m.Add(rx2 - cx2 <= R2).OnlyEnforceIf(p)
                        m.Add(cy2 - ry2 <= R2).OnlyEnforceIf(p)
                        m.Add(ry2 - cy2 <= R2).OnlyEnforceIf(p)

                    targets = []
                    if spec.near_window and windows:
                        targets.append(("window", "soft", 3))
                    for subj, tgt, kind in near:
                        if subj == alt:
                            targets.append((tgt, kind, 6))
                    for tgt, kind, wgt in targets:
                        pts = []
                        if tgt == "window":
                            pts = [((o.a[0] + o.b[0]) / 2, (o.a[1] + o.b[1]) / 2) for o in windows]
                        elif tgt in fixtures_by_id:
                            pts = [fixtures_by_id[tgt].at]
                        if not pts:
                            continue
                        dists = []
                        for px, py in pts:
                            dx = m.NewIntVar(0, 4 * (nx + ny), "")
                            dy = m.NewIntVar(0, 4 * (nx + ny), "")
                            tx, ty = round(2 * px / grid), round(2 * py / grid)
                            m.AddAbsEquality(dx, cx2 - tx)
                            m.AddAbsEquality(dy, cy2 - ty)
                            dists.append(dx + dy)
                        dmin = m.NewIntVar(0, 8 * (nx + ny), "")
                        m.AddMinEquality(dmin, dists)
                        if kind == "hard":
                            m.Add(dmin <= 2 * math.ceil(1200 / grid)).OnlyEnforceIf(p)
                        else:
                            pen = m.NewIntVar(0, 8 * (nx + ny), "")
                            m.Add(pen == dmin).OnlyEnforceIf(p)
                            m.Add(pen == 0).OnlyEnforceIf(p.Not())
                            objective.append(-wgt * pen)

                    value = REQUIRED_WEIGHT if req.required else req.weight
                    objective.append(value * p)
                    # предпочтения: первая альтернатива, комплект с пристройками, крупный размер
                    objective.append((200 * (len(alts) - 1 - alt_rank) + 80 * bool(att)
                                      + 5 * (2 * len(spec.sizes) - rank)) * p)
                    options.append(opt)
                    by_req.setdefault(req.key, []).append(opt)

    for key, opts in by_req.items():
        m.Add(sum(o.p for o in opts) <= 1)
    if options:
        m.AddNoOverlap2D(xs_iv, ys_iv)
    m.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = seed
    status = solver.Solve(m)
    res.status = solver.StatusName(status)
    res.seconds = time.perf_counter() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res.unplaced = [r.key for r in requests if r.required]
        return res

    placed_keys = set()
    for o in options:
        if not solver.Value(o.p):
            continue
        fx = (solver.Value(o.x) + o.foot_off[0]) * grid
        fy = (solver.Value(o.y) + o.foot_off[1]) * grid
        extra = sum(aw * (2 if side == "both" else 1) for _, aw, _, side in o.attach)
        full = (o.size[0] + extra, o.size[1])
        fw_mm, fh_mm = _footprint_mm(full, o.rot)
        cellw, cellh = o.foot[0] * grid, o.foot[1] * grid
        # прижимаем комплект спинкой к стене внутри своих клеток
        if o.rot == 0:
            cx, cy = fx + cellw / 2, fy + fh_mm / 2
        elif o.rot == 180:
            cx, cy = fx + cellw / 2, fy + cellh - fh_mm / 2
        elif o.rot == 270:
            cx, cy = fx + fw_mm / 2, fy + cellh / 2
        else:
            cx, cy = fx + cellw - fw_mm / 2, fy + cellh / 2
        res.items.extend(_split(room_id, o, cx, cy, full))
        placed_keys.add(o.req.key)
    res.unplaced = [r.key for r in requests if r.required and r.key not in placed_keys]
    return res



def _split(room_id: str, o: _Option, cx: float, cy: float, full: tuple[int, int]) -> list[Item]:
    """Комплект → основной предмет и пристройки, спинками к одной стене."""
    # локальные оси: u — вдоль стены (вправо при rotation=0), v — от стены
    ux, uy = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[o.rot]
    vx, vy = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}[o.rot]
    W, D = full
    back = -D / 2  # спинка комплекта в координате v

    def place(u_center, w, d, typ, key):
        v_center = back + d / 2
        return Item(id=f"{room_id}_{key}", type=typ, room=room_id,
                    x=round(cx + ux * u_center + vx * v_center, 1),
                    y=round(cy + uy * u_center + vy * v_center, 1), w=w, d=d, rotation=o.rot)

    left_w = sum(aw for _, aw, _, side in o.attach if side in ("left", "both"))
    main_u = -W / 2 + left_w + o.size[0] / 2
    items = [place(main_u, o.size[0], o.size[1], o.spec.type, o.req.key)]
    u_left, u_right = -W / 2, -W / 2 + left_w + o.size[0]
    for n, (typ, aw, ad, side) in enumerate(o.attach):
        if side in ("left", "both"):
            items.append(place(u_left + aw / 2, aw, ad, typ, f"{o.req.key}_{typ}_l{n}"))
            u_left += aw
        if side in ("right", "both"):
            items.append(place(u_right + aw / 2, aw, ad, typ, f"{o.req.key}_{typ}_r{n}"))
            u_right += aw
    return items
