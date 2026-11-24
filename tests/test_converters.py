import json

import pytest
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

from aidesign.converters import labelme_to_plan, resplan_to_plan
from aidesign.geometry import room_area_m2, validate_plan


def _shape(label, points, shape_type="polygon"):
    return {"label": label, "points": points, "shape_type": shape_type}


def test_labelme(tmp_path):
    # квартира 6×4 м, 1 px = 10 мм; стены 20 px = 200 мм
    shapes = [
        _shape("wall:ext", [[-10, -10], [610, 10]], "rectangle"),
        _shape("wall:ext", [[-10, 390], [610, 410]], "rectangle"),
        _shape("wall:ext", [[-10, -10], [10, 410]], "rectangle"),
        _shape("wall:ext", [[590, -10], [610, 410]], "rectangle"),
        _shape("wall", [[290, 0], [310, 400]], "rectangle"),
        _shape("room:living:16,8", [[10, 10], [290, 10], [290, 390], [10, 390]]),
        _shape("room:bedroom", [[310, 10], [590, 10], [590, 390], [310, 390]]),
        _shape("entrance:a:right", [[100, 400], [190, 400]], "line"),
        _shape("door:a:left", [[300, 150], [300, 230]], "line"),
        _shape("window", [[400, 0], [520, 0]], "line"),
        _shape("scale:6000", [[0, 0], [600, 0]], "line"),
    ]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"shapes": shapes, "imagePath": "plan.png"}), encoding="utf-8")
    plan = labelme_to_plan(path)
    assert plan.source.mm_per_px == pytest.approx(10)
    assert len(plan.walls) == 5
    assert sum(w.position.value == "exterior" for w in plan.walls) == 4
    living = next(r for r in plan.rooms if r.type.value == "living")
    assert room_area_m2(living) == pytest.approx(10.64)
    assert living.label_area_m2 == 16.8
    codes = {i.code for i in validate_plan(plan)}
    assert "area_mismatch" in codes  # подпись 16,8 не совпадает с геометрией — валидатор ловит
    assert "room_unreachable" not in codes


def test_resplan_synthetic():
    # мини-план в «единицах ResPlan»: 100×50, стены 2, area задаёт масштаб
    t = 2
    walls = unary_union([
        box(-1, -1, 101, 1).difference(box(70, -1, 80, 1)),
        box(-1, 49, 101, 51).difference(box(20, 49, 30, 51)),
        box(-1, -1, 1, 51),
        box(99, -1, 101, 51),
        box(49, 1, 51, 49).difference(box(49, 20, 51, 28)),
    ])
    p = {
        "id": 1, "wall": walls, "wall_depth": t,
        "inner": box(1, 1, 99, 49),
        "living": box(1, 1, 49, 49), "bedroom": MultiPolygon([box(51, 1, 99, 49)]),
        "bathroom": None, "kitchen": None,
        "door": MultiPolygon([box(49, 20, 51, 28)]),
        "window": MultiPolygon([box(70, -1, 80, 1)]),
        "front_door": box(20, 49, 30, 51),
        "area": 98 * 48 * 0.01,  # 1 единица = 100 мм
    }
    plan = resplan_to_plan(p)
    assert plan.source.mm_per_px == pytest.approx(100, rel=1e-3)
    assert {o.kind.value for o in plan.openings} == {"door", "window", "entrance"}
    assert all(w.thickness == pytest.approx(200, rel=0.05) for w in plan.walls)
    errors = [i for i in validate_plan(plan) if i.level == "error"]
    assert errors == []
