"""Схема плана квартиры.

Соглашения:
- единицы — миллиметры;
- система координат как у картинки: x вправо, y вниз;
- стена — прямоугольник, натянутый на отрезок a→b шириной thickness (торцы плоские,
  за точки a и b стена не выходит). В углах стены должны перекрываться: продлевайте
  отрезок на половину толщины примыкающей стены;
- проём задаётся своими точками a→b на оси стены; стена в месте проёма может быть
  как цельной (проём «вырезается» при отрисовке), так и разорванной — оба варианта валидны;
- полигон помещения — его внутренний контур (по чистовым граням стен). Площади
  НЕ хранятся, а вычисляются из полигонов (aidesign.geometry).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Point = tuple[float, float]


class WallPosition(str, Enum):
    exterior = "exterior"
    interior = "interior"


class Structural(str, Enum):
    load_bearing = "load_bearing"
    partition = "partition"
    unknown = "unknown"


class Wall(BaseModel):
    id: str
    a: Point
    b: Point
    thickness: float = Field(gt=0, description="мм")
    position: WallPosition = WallPosition.interior
    structural: Structural = Structural.unknown
    confidence: float = Field(
        1.0, ge=0, le=1,
        description="уверенность Обмерщика в стене и её типе; < 0.9 — показать пользователю",
    )


class OpeningKind(str, Enum):
    door = "door"
    entrance = "entrance"  # входная дверь в квартиру
    window = "window"
    passage = "passage"  # проём без двери


class Swing(BaseModel):
    """Открывание двери относительно направления a→b проёма."""

    hinge: Literal["a", "b"] = "a"
    side: Literal["left", "right"] = "left"


class Opening(BaseModel):
    id: str
    kind: OpeningKind
    a: Point
    b: Point
    wall: str | None = Field(None, description="id стены, если известен")
    swing: Swing | None = None
    confidence: float = Field(1.0, ge=0, le=1)


class RoomType(str, Enum):
    living = "living"
    kitchen = "kitchen"
    kitchen_living = "kitchen_living"
    bedroom = "bedroom"
    bathroom = "bathroom"
    toilet = "toilet"
    hall = "hall"
    corridor = "corridor"
    storage = "storage"
    balcony = "balcony"
    other = "other"


ROOM_NAMES_RU: dict[RoomType, str] = {
    RoomType.living: "Гостиная",
    RoomType.kitchen: "Кухня",
    RoomType.kitchen_living: "Кухня-гостиная",
    RoomType.bedroom: "Спальня",
    RoomType.bathroom: "Санузел",
    RoomType.toilet: "Туалет",
    RoomType.hall: "Прихожая",
    RoomType.corridor: "Коридор",
    RoomType.storage: "Гардеробная",
    RoomType.balcony: "Балкон",
    RoomType.other: "Помещение",
}


class Room(BaseModel):
    id: str
    type: RoomType
    polygon: list[Point] = Field(min_length=3)
    name: str | None = None
    label_area_m2: float | None = Field(
        None, description="площадь, подписанная на плане застройщика (для валидации)"
    )
    confidence: float = Field(1.0, ge=0, le=1)

    @property
    def display_name(self) -> str:
        return self.name or ROOM_NAMES_RU[self.type]


class FixtureKind(str, Enum):
    water_riser = "water_riser"
    sewer_riser = "sewer_riser"
    vent = "vent"
    gas = "gas"


class Fixture(BaseModel):
    """Неподвижный инженерный элемент: стояк, вентканал, газ."""

    id: str
    kind: FixtureKind
    at: Point
    inferred: bool = Field(False, description="поставлен эвристикой, а не взят с плана")


class PlanSource(BaseModel):
    kind: Literal["manual", "labelme", "resplan", "recognized"] = "manual"
    ref: str | None = None
    mm_per_px: float | None = None


class Plan(BaseModel):
    units: Literal["mm"] = "mm"
    walls: list[Wall] = []
    openings: list[Opening] = []
    rooms: list[Room] = []
    fixtures: list[Fixture] = []
    source: PlanSource = PlanSource()

    @field_validator("walls", "openings", "rooms", "fixtures")
    @classmethod
    def _unique_ids(cls, items):
        ids = [i.id for i in items]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"повторяющиеся id: {sorted(dup)}")
        return items

    def room(self, room_id: str) -> Room:
        return next(r for r in self.rooms if r.id == room_id)

    def wall(self, wall_id: str) -> Wall:
        return next(w for w in self.walls if w.id == wall_id)
