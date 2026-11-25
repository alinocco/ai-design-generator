"""Оценка агентов на размеченных примерах (data/agents_eval/*.jsonl).

Ограничение считается найденным, если совпали правило и предмет, а также комната и цель —
когда они указаны в ожидаемом ответе. Лишние правила снижают точность (precision),
пропущенные — полноту (recall). Для брифа отдельно проверяются поля о жильцах,
для правок — намерение (change / question / unsupported).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import Constraint, Layout, Plan, ProjectState
from .brief_analyst import analyze_brief
from .edit_agent import parse_edit
from .llm import LLMClient, LLMError, Usage


def _key_match(exp: dict, c: Constraint) -> bool:
    if exp["rule"] != c.rule.value or exp["subject"] != c.subject:
        return False
    if "room" in exp and exp["room"] != c.room:
        return False
    if "target" in exp and exp["target"] != c.target:
        return False
    return True


def match(expected: list[dict], got: list[Constraint],
          optional: list[dict] = ()) -> tuple[int, int, int]:
    """(tp, fp, fn) — жадное сопоставление один к одному.

    optional — допустимые, но не обязательные правила: их наличие не штрафуется."""
    left = list(got)
    tp = 0
    for e in expected:
        hit = next((c for c in left if _key_match(e, c)), None)
        if hit is not None:
            left.remove(hit)
            tp += 1
    left = [c for c in left if not any(_key_match(o, c) for o in optional)]
    return tp, len(left), len(expected) - tp


@dataclass
class CaseResult:
    id: str
    ok: bool
    tp: int = 0
    fp: int = 0
    fn: int = 0
    attempts: int = 0
    seconds: float = 0.0
    error: str | None = None
    detail: str = ""


@dataclass
class Report:
    kind: str
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    extra: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    @property
    def precision(self) -> float:
        tp = sum(c.tp for c in self.cases)
        return tp / max(tp + sum(c.fp for c in self.cases), 1)

    @property
    def recall(self) -> float:
        tp = sum(c.tp for c in self.cases)
        return tp / max(tp + sum(c.fn for c in self.cases), 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def summary(self) -> dict:
        n = len(self.cases)
        return {
            "kind": self.kind, "model": self.model, "cases": n,
            "exact": round(sum(c.ok for c in self.cases) / max(n, 1), 3),
            "precision": round(self.precision, 3), "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "failed_json": sum(c.error is not None for c in self.cases),
            "first_try_json": round(sum(c.attempts == 1 for c in self.cases) / max(n, 1), 3),
            "median_s": round(sorted(c.seconds for c in self.cases)[n // 2], 2) if n else 0,
            **{k: round(v, 3) for k, v in self.extra.items()},
            "prompt_tokens": self.usage.prompt_tokens, "completion_tokens": self.usage.completion_tokens,
        }


def load_cases(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("//")]


def eval_briefs(llm: LLMClient, plan: Plan, cases: list[dict]) -> Report:
    rep = Report("brief", llm.config.model)
    n0 = len(llm.log)
    occ_total = occ_ok = 0
    for case in cases:
        t0 = time.perf_counter()
        try:
            res = analyze_brief(llm, plan, case["text"])
        except LLMError as e:
            rep.cases.append(CaseResult(case["id"], False, fn=len(case["expect"].get("constraints", [])),
                                        seconds=time.perf_counter() - t0, error=str(e)))
            continue
        exp = case["expect"]
        tp, fp, fn = match(exp.get("constraints", []), res.constraints, exp.get("optional", []))
        occ = exp.get("occupants", {})
        got_occ = res.brief.occupants.model_dump()
        occ_good = sum(got_occ[k] == v for k, v in occ.items())
        occ_total += len(occ)
        occ_ok += occ_good
        ok = fp == 0 and fn == 0 and occ_good == len(occ)
        detail = "; ".join(f"{c.rule.value} {c.subject}" + (f"@{c.room}" if c.room else "")
                           + (f"→{c.target}" if c.target else "") for c in res.constraints)
        rep.cases.append(CaseResult(case["id"], ok, tp, fp, fn, res.attempts,
                                    time.perf_counter() - t0, detail=detail))
    rep.extra["occupants_acc"] = occ_ok / max(occ_total, 1)
    rep.usage = Usage.of(llm.log[n0:])
    return rep


def eval_edits(llm: LLMClient, state: ProjectState, cases: list[dict]) -> Report:
    rep = Report("edit", llm.config.model)
    n0 = len(llm.log)
    intent_ok = 0
    move_tp = move_fp = move_fn = 0
    for k, case in enumerate(cases):
        t0 = time.perf_counter()
        try:
            res = parse_edit(llm, state, case["text"], comment_id=f"e{k}", target=case.get("target"))
        except LLMError as e:
            rep.cases.append(CaseResult(case["id"], False, fn=len(case["expect"].get("constraints", [])),
                                        seconds=time.perf_counter() - t0, error=str(e)))
            continue
        exp = case["expect"]
        tp, fp, fn = match(exp.get("constraints", []), res.constraints, exp.get("optional", []))
        em = set(exp.get("move", []))
        gm = set(res.move)
        move_tp += len(em & gm)
        move_fp += len(gm - em)
        move_fn += len(em - gm)
        good_intent = res.intent == exp.get("intent", "change")
        intent_ok += good_intent
        ok = good_intent and fp == 0 and fn == 0 and em == gm
        detail = (f"[{res.intent}] " + "; ".join(
            f"{c.rule.value} {c.subject}" + (f"@{c.room}" if c.room else "")
            + (f"→{c.target}" if c.target else "") for c in res.constraints)
            + (f" move={sorted(gm)}" if gm else "") + f" rooms={res.rooms}")
        rep.cases.append(CaseResult(case["id"], ok, tp, fp, fn, res.attempts,
                                    time.perf_counter() - t0, detail=detail))
    rep.extra["intent_acc"] = intent_ok / max(len(cases), 1)
    rep.extra["move_recall"] = move_tp / max(move_tp + move_fn, 1)
    rep.extra["move_precision"] = move_tp / max(move_tp + move_fp, 1)
    rep.usage = Usage.of(llm.log[n0:])
    return rep


def report_markdown(reports: list[Report]) -> str:
    lines = []
    for r in reports:
        s = r.summary()
        lines.append(f"### {r.kind} — {r.model}\n")
        lines.append("| " + " | ".join(s) + " |")
        lines.append("|" + "---|" * len(s))
        lines.append("| " + " | ".join(str(v) for v in s.values()) + " |\n")
        bad = [c for c in r.cases if not c.ok]
        if bad:
            lines.append("Ошибки:\n")
            for c in bad:
                lines.append(f"- `{c.id}` tp={c.tp} fp={c.fp} fn={c.fn}"
                             + (f" — {c.error}" if c.error else f" — получено: {c.detail}"))
            lines.append("")
    return "\n".join(lines)


def demo_state(plan: Plan, layout: Layout) -> ProjectState:
    return ProjectState(project_id="eval", plan=plan, layout=layout)
