import pytest
from shapely.geometry import Polygon

from aidesign.geometry import (
    check_layout, door_swing, polygons_to_segments, rect_along, room_area_m2, rooms_graph,
    total_area_m2, validate_plan,
)
from aidesign.schemas import Item, Layout


def test_demo_areas(demo_plan):
    areas = {r.id: round(room_area_m2(r), 2) for r in demo_plan.rooms}
    assert areas == {"r1": 23.77, "r2": 11.39, "r3": 4.84, "r4": 2.42}
    assert total_area_m2(demo_plan) == pytest.approx(42.41, abs=0.01)


def test_demo_plan_is_valid(demo_plan):
    assert validate_plan(demo_plan) == []


def test_rooms_graph(demo_plan):
    g = rooms_graph(demo_plan)
    assert g["r4"] == {"r1", "r2", "r3"}  # из прихожей — во все помещения


def test_unreachable_room_detected(demo_plan):
    demo_plan.openings = [o for o in demo_plan.openings if o.id != "d2"]  # убрали дверь в санузел
    codes = {(i.code, i.object_id) for i in validate_plan(demo_plan)}
    assert ("room_unreachable", "r3") in codes


def test_area_mismatch_detected(demo_plan):
    demo_plan.rooms[1].label_area_m2 = 13.0
    assert any(i.code == "area_mismatch" for i in validate_plan(demo_plan))


def test_door_swings_into_room(demo_plan):
    d1 = next(o for o in demo_plan.openings if o.id == "d1")  # дверь в спальню
    swing = door_swing(demo_plan, d1)
    bedroom = Polygon(demo_plan.room("r2").polygon)
    assert swing.intersection(bedroom).area / swing.area > 0.95


def test_l_shaped_wall_decomposition():
    t = 200
    l_shape = rect_along((0, 0), (3000, 0), t).union(rect_along((-100, 0), (-100, 2000), t))
    segs = polygons_to_segments(l_shape, t)
    assert len(segs) == 2
    assert all(th == pytest.approx(t) for _, _, th in segs)


def test_layout_checks(demo_state):
    issues = check_layout(demo_state.plan, demo_state.layout)
    assert not [i for i in issues if i.level == "error"]


def test_blocking_wardrobe_detected(demo_plan):
    # шкаф v3 прямо в зоне открывания двери спальни
    wardrobe = Item(id="i1", type="wardrobe", room="r2", x=4800, y=2500, w=1800, d=600, rotation=90)
    codes = {i.code for i in check_layout(demo_plan, Layout(items=[wardrobe]))}
    assert "blocks_door" in codes
