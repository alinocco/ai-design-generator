"""Каталог мебели и сантехники и «программа» помещения: что ставить в комнату каждого типа.

Соглашение об ориентации предмета (совпадает с Item.rotation, ось y вниз):
при rotation=0 спинка предмета вверху (у стены с меньшим y), лицевая сторона смотрит вниз (+y).
Поворот — по часовой стрелке: 90 — спинка справа, 180 — снизу, 270 — слева.

Зона обслуживания (clear_front) — свободное место перед лицевой стороной: чтобы открыть
шкаф, сесть за стол, подойти к кровати. Солвер не даёт зонам перекрываться с мебелью,
дверями и друг с другом.

Габариты — типичные для российского рынка; уточняйте под свой проект.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import RoomType


@dataclass(frozen=True)
class Spec:
    type: str
    name_ru: str
    sizes: tuple[tuple[int, int], ...]  # варианты (ширина, глубина), мм — от большего к меньшему
    clear_front: int = 0  # мм перед лицевой стороной
    clear_around: int = 0  # мм со всех сторон (стол, к которому подходят отовсюду)
    against_wall: bool = True  # спинкой к стене
    plumbing: bool = False  # нужен стояк рядом
    near_window: bool = False  # лучше у окна (мягкое пожелание)
    # пристраиваемые предметы: (тип, ширина, глубина, сторона "left" | "right" | "both")
    # солвер ставит их вместе с основным — тумбочки у кровати, холодильник у гарнитура;
    # если вместе не помещается, пробует основной предмет без них
    attach: tuple[tuple[str, int, int, str], ...] = ()


CATALOG: dict[str, Spec] = {s.type: s for s in [
    # спальня
    Spec("bed_double", "Кровать двуспальная", ((1600, 2000), (1400, 2000)), clear_front=600,
         attach=(("nightstand", 450, 400, "both"),)),
    Spec("bed_single", "Кровать односпальная", ((900, 2000),), clear_front=600,
         attach=(("nightstand", 450, 400, "left"),)),
    Spec("nightstand", "Тумба прикроватная", ((450, 400),)),
    Spec("wardrobe", "Шкаф", ((1800, 600), (1200, 600), (900, 600)), clear_front=700),
    Spec("desk", "Рабочий стол", ((1200, 600), (1000, 550)), clear_front=800, near_window=True),
    # гостиная
    Spec("sofa", "Диван", ((2200, 950), (1800, 900)), clear_front=700),
    Spec("armchair", "Кресло", ((850, 850),), clear_front=500),
    Spec("tv_stand", "Тумба под ТВ", ((1600, 400), (1200, 400)), clear_front=300),
    Spec("bookcase", "Стеллаж", ((900, 350),), clear_front=600),
    # кухня
    Spec("kitchen_run", "Кухонный гарнитур", ((3000, 600), (2400, 600), (1800, 600)),
         clear_front=900, plumbing=True, attach=(("fridge", 600, 650, "right"),)),
    Spec("fridge", "Холодильник", ((600, 650),), clear_front=700),
    Spec("dining_table", "Обеденный стол", ((1200, 800), (800, 800)), clear_around=600,
         against_wall=False),
    # санузел
    Spec("bathtub", "Ванна", ((1700, 750), (1500, 700), (1400, 700)), clear_front=600, plumbing=True),
    Spec("shower", "Душевая", ((900, 900), (800, 800)), clear_front=600, plumbing=True),
    Spec("toilet", "Унитаз", ((400, 650),), clear_front=600, plumbing=True),
    Spec("sink", "Раковина", ((600, 450), (500, 400)), clear_front=600, plumbing=True),
    Spec("washer", "Стиральная машина", ((600, 600),), clear_front=600, plumbing=True),
    # прихожая
    Spec("hall_wardrobe", "Шкаф в прихожей", ((1200, 450), (800, 450)), clear_front=700),
]}


@dataclass
class Request:
    """Что нужно поставить. alternatives — взаимозаменяемые типы в порядке предпочтения
    (ванна, иначе душ)."""

    alternatives: tuple[str, ...]
    required: bool = True
    weight: int = 100  # ценность для необязательного предмета
    key: str = ""

    def __post_init__(self):
        self.key = self.key or self.alternatives[0]


@dataclass
class RoomProgram:
    room_type: RoomType
    requests: list[Request] = field(default_factory=list)


def program_for(room_type: RoomType, area_m2: float, remote_work: bool = False,
                adults: int = 2) -> list[Request]:
    """Набор мебели для помещения по его типу и площади."""
    R = Request
    if room_type == RoomType.bedroom:
        bed = ("bed_double", "bed_single") if adults >= 2 and area_m2 >= 8 else ("bed_single",)
        reqs = [R(bed, key="bed"), R(("wardrobe",))]
        reqs.append(R(("desk",), required=False, weight=400 if remote_work else 60))
        return reqs
    if room_type == RoomType.living:
        return [R(("sofa",)), R(("tv_stand",)),
                R(("armchair",), required=False, weight=50),
                R(("desk",), required=False, weight=300 if remote_work else 40),
                R(("bookcase",), required=False, weight=30)]
    if room_type == RoomType.kitchen:
        return [R(("kitchen_run",)),
                R(("dining_table",), required=area_m2 >= 9, weight=200)]
    if room_type == RoomType.kitchen_living:
        return [R(("kitchen_run",)), R(("sofa",)),
                R(("dining_table",), required=False, weight=200),
                R(("tv_stand",), required=False, weight=100)]
    if room_type == RoomType.bathroom and area_m2 < 2.2:  # по площади это туалет
        room_type = RoomType.toilet
    if room_type == RoomType.bathroom:
        return [R(("bathtub", "shower"), key="bath"), R(("toilet",)), R(("sink",)),
                R(("washer",), required=False, weight=150)]
    if room_type == RoomType.toilet:
        return [R(("toilet",)), R(("sink",), required=False, weight=50)]
    if room_type == RoomType.hall:
        return [R(("hall_wardrobe",), required=False, weight=150)]
    return []
