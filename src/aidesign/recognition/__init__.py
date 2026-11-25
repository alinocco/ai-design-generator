"""Шаг 1: распознавание плана. Нужны дополнительные зависимости: pip install -e '.[recognition]'."""

from .cv_baseline import Recognition, area_agreement, recognize

__all__ = ["Recognition", "area_agreement", "recognize"]
