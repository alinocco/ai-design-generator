"""Тип помещения по геометрии — пока нет VLM.

Правила грубые, но дают расстановке осмысленную программу комнаты:
- без окна и у входной двери → прихожая;
- без окна, до 2,5 м² → кладовая/туалет (кладовая, если есть санузел крупнее);
- без окна, до 9 м² → санузел; без окна и больше → коридор;
- с окном: самое большое → гостиная (кухня-гостиная, если отдельной кухни не нашлось),
  самое маленькое из 5–13 м² при трёх и более комнатах с окном → кухня, остальные → спальни.

Уверенность ставим 0.4: пользователь видит тип на экране подтверждения и исправляет.
"""

from __future__ import annotations

from ..schemas import OpeningKind, Plan, RoomType
from .core import opening_polygon, room_area_m2, room_polygon


def infer_room_types(plan: Plan, only_unknown: bool = True) -> Plan:
    polys = {r.id: room_polygon(r) for r in plan.rooms}
    windows = [opening_polygon(plan, o).buffer(80) for o in plan.openings if o.kind == OpeningKind.window]
    entrance = [opening_polygon(plan, o).buffer(80) for o in plan.openings if o.kind == OpeningKind.entrance]
    targets = [r for r in plan.rooms if not only_unknown or r.type == RoomType.other]
    if not targets:
        return plan

    def has(rid, zones):
        return any(polys[rid].intersects(z) for z in zones)

    lit = sorted([r for r in targets if has(r.id, windows)], key=room_area_m2)
    dark = [r for r in targets if not has(r.id, windows)]

    for r in dark:
        a = room_area_m2(r)
        if has(r.id, entrance):
            r.type = RoomType.hall
        elif a <= 2.5:
            r.type = RoomType.storage if any(room_area_m2(x) > 2.5 and x in dark for x in dark) else RoomType.toilet
        elif a <= 9:
            r.type = RoomType.bathroom
        else:
            r.type = RoomType.corridor
        r.confidence = min(r.confidence, 0.4)

    kitchen = None
    if len(lit) >= 3:
        kitchen = next((r for r in lit if 5 <= room_area_m2(r) <= 13), None)
    for r in lit:
        if r is lit[-1]:
            r.type = RoomType.living if kitchen else RoomType.kitchen_living
        elif r is kitchen:
            r.type = RoomType.kitchen
        else:
            r.type = RoomType.bedroom
        r.confidence = min(r.confidence, 0.4)
    return plan
