import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("skimage")

from aidesign.geometry import room_area_m2  # noqa: E402
from aidesign.recognition import recognize  # noqa: E402


def _draw(plan, path, px_per_mm=0.1):
    """Нарисовать план «как у застройщика»: чёрные стены, проёмы — разрывами."""
    from aidesign.geometry.core import opening_polygon, wall_polygon

    img = np.full((800, 1000, 3), 245, np.uint8)
    off = 80

    def fill(poly, color):
        pts = np.array([[(x * px_per_mm + off), (y * px_per_mm + off)] for x, y in poly.exterior.coords],
                       np.int32)
        cv2.fillPoly(img, [pts], color)

    for w in plan.walls:
        fill(wall_polygon(w), (20, 20, 20))
    for o in plan.openings:
        fill(opening_polygon(plan, o), (245, 245, 245))
    cv2.imwrite(str(path), img)


def test_recognize_demo_without_ocr(tmp_path, demo_plan):
    path = tmp_path / "demo.png"
    _draw(demo_plan, path)
    rec = recognize(path, use_ocr=False)
    assert len(rec.plan.rooms) == 4  # кухня-гостиная, спальня, санузел, прихожая
    assert len(rec.plan.walls) >= 8
    # без подписей масштаб берётся по толщине стен — проверяем только пропорции:
    # спальня (11,4 м²) больше санузла (4,8 м²) больше прихожей (2,4 м²)
    areas = sorted(room_area_m2(r) for r in rec.plan.rooms)
    assert areas[0] < areas[1] < areas[2] < areas[3]
