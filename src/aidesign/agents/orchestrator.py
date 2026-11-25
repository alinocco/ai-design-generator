"""Оркестратор: ведёт ProjectState по версиям и решает, кого из агентов звать.

    бриф ──► Аналитик брифа ──► ограничения ─┐
                                              ├─► Планировщик (солвер) ─► Критик ─► версия N
    правка ─► Агент правок ──► ограничения ──┘        ▲  только затронутые комнаты

LLM участвует только в переводе текста в ограничения. Расстановку, проверку и решение,
какие комнаты пересчитать, делает код — поэтому ошибку модели видно в списке ограничений,
а не в «кривой» мебели.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..catalog import CATALOG
from ..geometry.validate import Issue
from ..schemas import Comment, Constraint, ConstraintSource, Plan, ProjectState, Rule, Stage
from ..solver import generate_layout
from ..store import VersionStore
from .brief_analyst import analyze_brief
from .drafts import merge_constraints
from .edit_agent import parse_edit
from .llm import LLMClient, Usage


@dataclass
class Turn:
    state: ProjectState
    reply: str
    changed: bool = True
    rooms: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class Orchestrator:
    def __init__(self, llm: LLMClient, store: VersionStore | None = None, *,
                 time_limit: float = 5.0):
        self.llm = llm
        self.store = store
        self.time_limit = time_limit

    # --- новый проект ---------------------------------------------------------------
    def start(self, plan: Plan, brief_text: str, project_id: str) -> Turn:
        n0 = len(self.llm.log)
        state = ProjectState(project_id=project_id, plan=plan, stage=Stage.plan_confirmed,
                             note="план и бриф")
        br = analyze_brief(self.llm, plan, brief_text)
        state.brief, state.constraints = br.brief, br.constraints
        res = generate_layout(plan, state.brief, state.constraints, time_limit=self.time_limit)
        state.layout, state.stage = res.layout, Stage.layout
        reply = _summary(res, [r.id for r in plan.rooms], prefix="Расставил мебель по брифу.")
        self._save(state, n0)
        return Turn(state, reply, True, [r.id for r in plan.rooms], res.issues, br.warnings,
                    Usage.of(self.llm.log[n0:]))

    # --- правка к текущей версии ------------------------------------------------------
    def comment(self, state: ProjectState, text: str, target: str | None = None) -> Turn:
        n0 = len(self.llm.log)
        cid = f"cm{len(state.comments) + 1}"
        ed = parse_edit(self.llm, state, text, comment_id=cid, target=target)
        usage = Usage.of(self.llm.log[n0:])
        if ed.intent != "change" or not ed.rooms:
            reply = ed.reply or "Изменений не потребовалось."
            return Turn(state, reply, False, [], [], ed.warnings, usage)

        new = state.next_version(f"правка {cid}: {text.strip()[:80]}")
        new.constraints, dropped = merge_constraints(list(state.constraints), ed.constraints)
        new.comments.append(Comment(id=cid, stage=Stage.layout, text=text.strip(), target=target,
                                    parsed_to=[c.id for c in ed.constraints]))
        warnings = list(ed.warnings)
        if dropped:
            warnings.append("отменены прежние правила: " + ", ".join(dropped))

        res = self._relayout(new, ed.rooms, ed.move, cid, warnings)
        new.layout, new.stage = res.layout, Stage.layout
        reply = _summary(res, ed.rooms, prefix=ed.reply)
        self._save(new, n0)
        return Turn(new, reply, True, ed.rooms, res.issues, warnings, Usage.of(self.llm.log[n0:]))

    def _relayout(self, state: ProjectState, rooms: list[str], move: list[str], cid: str,
                  warnings: list[str]):
        prev = state.layout
        items = {i.id: i for i in (prev.items if prev else [])}
        excluded = {(c.subject, c.room) for c in state.constraints if c.rule == Rule.exclude}
        # «перенеси шкаф»: остальное в его комнате фиксируем на время пересчёта
        transient: list[Constraint] = []
        moved_rooms = {items[m].room for m in move}
        for it in items.values():
            if it.room not in moved_rooms or it.id in move or any(it.id.startswith(m + "_") for m in move):
                continue
            if (it.type, None) in excluded or (it.type, it.room) in excluded:
                continue
            if any(it.id.startswith(j + "_") for j in items if j != it.id):
                continue  # пристройку (тумбочку) держит основной предмет
            transient.append(Constraint(id=f"{cid}_tmp_{it.id}", kind="hard", rule=Rule.keep,
                                        subject=it.id, source=ConstraintSource.system))
        res = generate_layout(state.plan, state.brief, state.constraints + transient, previous=prev,
                              rooms=rooms, time_limit=self.time_limit, move=move)
        after = {i.id: i for i in res.layout.items}
        stuck = [m for m in move if not _moved(items[m], after.get(m))]
        if stuck:
            warnings.append("не нашлось другого места для: " + ", ".join(stuck))
        unplaced_moved = [m for m in move if m not in after and items[m].type not in
                          {c.subject for c in state.constraints if c.rule == Rule.exclude}]
        if unplaced_moved:
            warnings.append("на новом месте не поместилось: " + ", ".join(unplaced_moved))
        return res

    def _save(self, state: ProjectState, n0: int):
        if not self.store:
            return
        self.store.save(state)
        log = Path(self.store.root) / state.project_id / "llm_log.jsonl"
        with log.open("a", encoding="utf-8") as f:
            for call in self.llm.log[n0:]:
                rec = asdict(call) | {"version": state.version}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _moved(a, b) -> bool:
    return b is None or abs(a.x - b.x) > 100 or abs(a.y - b.y) > 100 or a.rotation != b.rotation


def _summary(res, rooms: list[str], prefix: str = "") -> str:
    parts = [prefix.strip()] if prefix.strip() else []
    for room_id, keys in res.unplaced.items():
        names = [CATALOG[k].name_ru.lower() if k in CATALOG else k for k in keys]
        parts.append(f"В {room_id} не поместилось: {', '.join(names)}.")
    errors = [i for i in res.issues if i.level == "error"]
    warns = [i for i in res.issues if i.level == "warning" and i.code != "not_placed"]
    if errors:
        parts.append(f"Критик нашёл ошибки: {len(errors)}.")
    if warns:
        parts.append("Замечания Критика: " + " ".join(i.message for i in warns[:3]))
    return " ".join(parts)
