from pathlib import Path

import pytest

from aidesign.schemas import Plan, ProjectState

DATA = Path(__file__).resolve().parents[1] / "data" / "demo"


@pytest.fixture
def demo_plan() -> Plan:
    return Plan.model_validate_json((DATA / "demo_48.json").read_text(encoding="utf-8"))


@pytest.fixture
def demo_state() -> ProjectState:
    return ProjectState.model_validate_json((DATA / "demo_48_state.json").read_text(encoding="utf-8"))
