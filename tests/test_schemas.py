import pytest
from pydantic import ValidationError

from aidesign.schemas import Constraint, Plan, ProjectState, Wall
from aidesign.store import VersionStore


def test_roundtrip(demo_state):
    again = ProjectState.model_validate_json(demo_state.model_dump_json())
    assert again == demo_state


def test_duplicate_ids_rejected():
    w = Wall(id="w1", a=(0, 0), b=(1000, 0), thickness=200)
    with pytest.raises(ValidationError):
        Plan(walls=[w, w])


def test_user_constraint_needs_comment():
    with pytest.raises(ValidationError):
        Constraint(id="c1", kind="hard", rule="keep", subject="desk", source="user")


def test_hard_constraint_has_no_weight():
    with pytest.raises(ValidationError):
        Constraint(id="c1", kind="hard", rule="keep", subject="desk", source="brief", weight=2)


def test_versions_are_immutable(tmp_path, demo_state):
    store = VersionStore(tmp_path)
    store.save(demo_state)
    nxt = demo_state.next_version("тест")
    store.save(nxt)
    assert store.versions(demo_state.project_id) == [demo_state.version, demo_state.version + 1]
    assert store.load(demo_state.project_id).parent_version == demo_state.version
    with pytest.raises(FileExistsError):
        store.save(nxt)
