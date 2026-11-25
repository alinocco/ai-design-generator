"""Отладочная картинка: распознанный план поверх исходного изображения."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from ..geometry.core import room_area_m2
from .cv_baseline import Recognition

OPENING_BGR = {
    "door": (255, 0, 0),  # синий
    "entrance": (255, 0, 255),  # пурпурный
    "window": (0, 200, 255),  # жёлтый
    "passage": (0, 180, 0),  # зелёный
}


def overlay(image_path: str | Path, rec: Recognition) -> np.ndarray:
    """Комнаты — цветом с подписью «распознано / на плане, м²», стены — красным, проёмы — по типу."""
    img = cv2.imread(str(image_path))
    k = rec.mm_per_px
    layer = img.copy()
    rnd = random.Random(0)
    for r in rec.plan.rooms:
        pts = np.array([[x / k, y / k] for x, y in r.polygon], np.int32)
        cv2.fillPoly(layer, [pts], [rnd.randint(80, 230) for _ in range(3)])
    img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
    for r in rec.plan.rooms:
        pts = np.array([[x / k, y / k] for x, y in r.polygon], np.int32)
        cx, cy = pts.mean(0).astype(int)
        text = f"{room_area_m2(r):.1f}" + (f"/{r.label_area_m2}" if r.label_area_m2 else "")
        ok = r.label_area_m2 and abs(room_area_m2(r) - r.label_area_m2) <= 0.05 * r.label_area_m2
        color = (0, 140, 0) if ok else (0, 0, 200) if r.label_area_m2 else (90, 90, 90)
        cv2.putText(img, text, (int(cx) - 30, int(cy) + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for w in rec.plan.walls:
        cv2.line(img, tuple(int(v / k) for v in w.a), tuple(int(v / k) for v in w.b), (0, 0, 255), 2)
    for o in rec.plan.openings:
        cv2.line(img, tuple(int(v / k) for v in o.a), tuple(int(v / k) for v in o.b),
                 OPENING_BGR[o.kind.value], 5)
    return img
