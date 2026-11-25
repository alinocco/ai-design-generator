import math

import pytest

pytest.importorskip("ortools")

from aidesign.catalog import CATALOG, program_for  # noqa: E402
from aidesign.geometry import infer_room_types  # noqa: E402
from aidesign.schemas import Constraint, ConstraintSource, FixtureKind, RoomType, Rule  # noqa: E402
from aidesign.solver import generate_layout  # noqa: E402


def _errors(res):
    return [i for i in res.issues if i.level == "error"]


def test_demo_layout_is_clean(demo_plan):
    res = generate_layout(demo_plan, time_limit=5)
    assert not _errors(res)
    assert not res.unplaced
    assert not [i for i in res.issues if i.code == "no_passage"]
    types = {i.type for i in res.layout.items}
    assert {"kitchen_run", "sofa", "toilet", "sink"} <= types
    assert types & {"bed_double", "bed_single"}
    assert types & {"bathtub", "shower"}


def test_plumbing_near_riser(demo_plan):
    res = generate_layout(demo_plan, time_limit=5)
    risers = [f for f in demo_plan.fixtures
              if f.kind in (FixtureKind.water_riser, FixtureKind.sewer_riser)]
    assert risers
    wet = {r.id for r in demo_plan.rooms if r.type in (RoomType.bathroom, RoomType.toilet)}
    checked = 0
    for it in res.layout.items:
        if it.room in wet and CATALOG.get(it.type) and CATALOG[it.type].plumbing:
            checked += 1
            d = min(max(abs(it.x - r.at[0]), abs(it.y - r.at[1])) for r in risers)
            assert d <= 1500 + 50, it
    assert checked >= 3


def test_partial_regeneration_keeps_other_rooms(demo_plan):
    first = generate_layout(demo_plan, time_limit=5)
    bedroom = next(r.id for r in demo_plan.rooms if r.type == RoomType.bedroom)
    bed = next(i for i in first.layout.items if i.room == bedroom and i.type.startswith("bed_"))
    keep = Constraint(id="c1", kind="hard", rule=Rule.keep, subject=bed.id,
                      source=ConstraintSource.system)
    second = generate_layout(demo_plan, constraints=[keep], previous=first.layout,
                             rooms=[bedroom], seed=5, time_limit=5)
    before = {i.id: i for i in first.layout.items}
    after = {i.id: i for i in second.layout.items}
    for iid, it in before.items():
        if it.room != bedroom or iid == bed.id or iid.startswith(bed.id + "_"):
            assert after[iid] == it
    assert not _errors(second)


def test_exclude_constraint(demo_plan):
    ex = Constraint(id="c2", kind="hard", rule=Rule.exclude, subject="desk",
                    source=ConstraintSource.system)
    res = generate_layout(demo_plan, constraints=[ex], time_limit=5)
    assert not [i for i in res.layout.items if i.type == "desk"]


def test_program_small_bathroom_is_wc():
    keys = {r.key for r in program_for(RoomType.bathroom, 1.6)}
    assert "bath" not in keys and "toilet" in keys


def test_infer_room_types(demo_plan):
    plan = demo_plan.model_copy(deep=True)
    for r in plan.rooms:
        r.type = RoomType.other
    infer_room_types(plan)
    got = {r.type for r in plan.rooms}
    assert RoomType.bathroom in got or RoomType.toilet in got
    assert got & {RoomType.living, RoomType.kitchen_living, RoomType.bedroom}
    assert all(not math.isnan(r.confidence or 0) for r in plan.rooms)
