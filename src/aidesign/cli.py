"""Командная строка шага 0.

  aidesign validate data/demo/demo_48.json
  aidesign render data/demo/demo_48.json -o out/demo.svg
  aidesign convert-labelme разметка.json -o data/plans/plan_01.json
  aidesign convert-resplan ResPlan.pkl -o data/resplan --limit 200
  aidesign recognize data/raw_plans/plan.png --debug
  aidesign eval-recognition data/raw_plans
  aidesign layout data/demo/demo_48.json -o out/demo_layout.json   (+ рядом .svg)
  aidesign layout out/demo_layout.json --rooms r2 --keep r2_bed      (перегенерировать спальню)
  aidesign llm-check                                                  (проверить модель из .env)
  aidesign brief data/demo/demo_48.json "Живём вдвоём, работаю из дома" --project demo
  aidesign comment demo "Убери стол из спальни"
  aidesign eval-agents -o out/agents_eval.md            (+ --split holdout для честной оценки)

Файл на входе validate/render — Plan или ProjectState в JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .geometry import total_area_m2, validate_plan
from .render import render_plan
from .schemas import Plan, ProjectState


def _load(path: str) -> tuple[Plan, ProjectState | None]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "plan" in data:
        state = ProjectState.model_validate(data)
        return state.plan, state
    return Plan.model_validate(data), None


def cmd_validate(args) -> int:
    plan, _ = _load(args.path)
    issues = validate_plan(plan)
    print(f"Помещений: {len(plan.rooms)}, стен: {len(plan.walls)}, проёмов: {len(plan.openings)}, "
          f"площадь без балконов: {total_area_m2(plan):.1f} м²")
    for i in issues:
        print(f"[{i.level}] {i.code} {i.object_id or ''}: {i.message}")
    if not issues:
        print("Проблем не найдено.")
    return 1 if any(i.level == "error" for i in issues) else 0


def cmd_render(args) -> int:
    plan, state = _load(args.path)
    issues = validate_plan(plan) if args.issues else None
    svg = render_plan(plan, layout=state.layout if state else None, issues=issues)
    out = Path(args.output or Path(args.path).with_suffix(".svg"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(out)
    return 0


def cmd_convert_labelme(args) -> int:
    from .converters import labelme_to_plan

    plan = labelme_to_plan(args.path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(out)
    return 0


def cmd_convert_resplan(args) -> int:
    from .converters import is_apartment_like, load_resplan, resplan_to_plan

    data = load_resplan(args.path)
    ids = None
    if args.split:
        splits = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        ids = set(splits[args.split])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    for p in data:
        if done >= args.limit:
            break
        if ids is not None and p.get("id") not in ids:
            continue
        if not is_apartment_like(p, max_area_m2=args.max_area, max_bedrooms=args.max_bedrooms):
            skipped += 1
            continue
        try:
            plan = resplan_to_plan(p)
        except Exception as e:  # noqa: BLE001 — грязные планы пропускаем, но считаем
            failed += 1
            print(f"plan {p.get('id')}: {e}", file=sys.stderr)
            continue
        if not args.keep_invalid and any(i.level == "error" for i in validate_plan(plan)):
            failed += 1
            continue
        (out / f"resplan_{p['id']}.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        done += 1
    print(f"сохранено {done}, отфильтровано как дома {skipped}, отброшено с ошибками {failed} → {out}")
    return 0


def cmd_recognize(args) -> int:
    from .recognition import area_agreement, recognize

    rec = recognize(args.path)
    out = Path(args.output or Path(args.path).with_suffix(".plan.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rec.plan.model_dump_json(indent=2), encoding="utf-8")
    ok, n = area_agreement(rec.plan)
    print(f"{out}: помещений {len(rec.plan.rooms)}, проёмов {len(rec.plan.openings)}, "
          f"масштаб {rec.mm_per_px:.1f} мм/px, площадь совпала у {ok} из {n} подписанных")
    for note in rec.notes:
        print(f"  ! {note}")
    if args.debug:
        import cv2

        from .recognition.debug import overlay

        dbg = out.with_suffix(".debug.png")
        cv2.imwrite(str(dbg), overlay(args.path, rec))
        print(f"  отладочная картинка: {dbg}")
    return 0


def cmd_eval_recognition(args) -> int:
    """Метрика без ручной разметки: совпадение распознанных площадей с подписанными на плане."""
    from .recognition import area_agreement, recognize

    files = sorted(p for p in Path(args.folder).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    tot5 = tot10 = total = failed = 0
    for f in files:
        try:
            rec = recognize(f)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"{f.name}: ошибка — {e}")
            continue
        ok5, n = area_agreement(rec.plan, 0.05)
        ok10, _ = area_agreement(rec.plan, 0.10)
        tot5, tot10, total = tot5 + ok5, tot10 + ok10, total + n
        print(f"{f.name:40s} помещений {len(rec.plan.rooms):2d}  ±5%: {ok5}/{n}  ±10%: {ok10}/{n}")
    if total:
        print(f"\nИТОГО: планов {len(files)}, с ошибкой {failed}; помещений с подписью {total}; "
              f"площадь совпала ±5% — {tot5 / total:.0%}, ±10% — {tot10 / total:.0%}")
    return 0


def cmd_layout(args) -> int:
    from .schemas import Constraint, ConstraintSource, Rule, Stage
    from .solver import generate_layout

    plan, state = _load(args.path)
    if state is None:
        state = ProjectState(project_id=Path(args.path).stem, plan=plan)
    constraints = list(state.constraints)
    for item_id in args.keep or []:
        constraints.append(Constraint(id=f"cli_keep_{item_id}", kind="hard", rule=Rule.keep,
                                      subject=item_id, source=ConstraintSource.system))
    rooms = args.rooms.split(",") if args.rooms else None
    res = generate_layout(plan, state.brief, constraints, previous=state.layout, rooms=rooms,
                          seed=args.seed, time_limit=args.time_limit)
    for rr in res.rooms:
        extra = f" не поместилось: {', '.join(rr.unplaced)}" if rr.unplaced else ""
        print(f"{rr.room_id:>12} {plan.room(rr.room_id).type.value:<14} {rr.status:<10} "
              f"{rr.seconds:5.2f} с, предметов {len(rr.items)}{extra}")
        for n in rr.notes:
            print(f"{'':>14}{n}")
    for i in res.issues:
        print(f"[{i.level}] {i.code} {i.object_id or ''}: {i.message}")
    if state.layout is not None:
        new = state.next_version("расстановка" + (f": {args.rooms}" if args.rooms else ""))
    else:
        new = state.model_copy(deep=True)
    new.layout, new.stage = res.layout, Stage.layout
    out = Path(args.output or Path(args.path).with_name(Path(args.path).stem + "_layout.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(new.model_dump_json(indent=1), encoding="utf-8")
    out.with_suffix(".svg").write_text(render_plan(plan, layout=res.layout, issues=res.issues),
                                       encoding="utf-8")
    print(out, out.with_suffix(".svg"))
    return 0


def _agents():
    from . import agents
    from .store import VersionStore
    return agents, VersionStore


def _print_turn(turn, store_root: str) -> None:
    print(f"Версия {turn.state.version}: {turn.reply}")
    for w in turn.warnings:
        print(f"  ! {w}")
    if turn.changed:
        cons = turn.state.constraints
        print("Ограничения:")
        for c in cons:
            extra = (f" @{c.room}" if c.room else "") + (f" → {c.target}" if c.target else "") \
                + (f" {c.params}" if c.params else "")
            print(f"  {c.id:<10} {c.kind:<4} {c.rule.value:<14} {c.subject}{extra}  [{c.source.value}]")
        svg = Path(store_root) / turn.state.project_id / f"v{turn.state.version:04d}.svg"
        svg.write_text(render_plan(turn.state.plan, layout=turn.state.layout, issues=turn.issues),
                       encoding="utf-8")
        print(svg)
    u = turn.usage
    print(f"LLM: вызовов {u.calls}, {u.seconds:.1f} с, токенов {u.prompt_tokens}+{u.completion_tokens}")


def cmd_llm_check(args) -> int:
    agents, _ = _agents()
    from pydantic import BaseModel

    from .agents.structured import ask_structured

    class Ping(BaseModel):
        ok: bool
        room: str

    cfg = agents.LLMConfig.from_env()
    print("Модель:", cfg.describe(), "| JSON:", cfg.json_mode if cfg.provider == "openai" else "format=schema")
    llm = agents.make_client(cfg)
    res = ask_structured(llm, Ping, "Ты проверяешь связь.", "Ответь ok=true и room — как по-русски "
                         "называется комната, где спят.", agent="ping")
    print("Ответ:", res.value.model_dump(), f"| попыток {res.attempts}, {llm.log[-1].seconds:.1f} с")
    return 0


def cmd_brief(args) -> int:
    agents, VersionStore = _agents()
    plan, _ = _load(args.plan)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    store = VersionStore(args.store)
    orch = agents.Orchestrator(agents.make_client(), store, time_limit=args.time_limit)
    turn = orch.start(plan, text or "", args.project)
    _print_turn(turn, args.store)
    return 0


def cmd_comment(args) -> int:
    agents, VersionStore = _agents()
    store = VersionStore(args.store)
    state = store.load(args.project, args.version)
    orch = agents.Orchestrator(agents.make_client(), store, time_limit=args.time_limit)
    turn = orch.comment(state, args.text, target=args.target)
    _print_turn(turn, args.store)
    return 0


def cmd_eval_agents(args) -> int:
    agents, _ = _agents()
    from .agents.eval import demo_state, eval_briefs, eval_edits, load_cases, report_markdown
    from .schemas import Layout

    root = Path(args.data)
    plan = Plan.model_validate_json((root.parent / "demo" / "demo_48.json").read_text(encoding="utf-8"))
    layout = Layout.model_validate_json((root / "demo_layout.json").read_text(encoding="utf-8"))
    llm = agents.make_client()
    print("Модель:", llm.config.describe())
    reports = []
    suffix = "" if args.split == "dev" else "_holdout"
    if args.set in ("briefs", "all"):
        reports.append(eval_briefs(llm, plan, load_cases(root / f"briefs{suffix}.jsonl")[: args.limit]))
        reports[-1].kind += f" ({args.split})"
        print(reports[-1].summary())
    if args.set in ("edits", "all"):
        reports.append(eval_edits(llm, demo_state(plan, layout),
                                  load_cases(root / f"edits{suffix}.jsonl")[: args.limit]))
        reports[-1].kind += f" ({args.split})"
        print(reports[-1].summary())
    md = report_markdown(reports)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(out)
    else:
        print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aidesign")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="проверить план")
    p.add_argument("path")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("render", help="нарисовать план в SVG")
    p.add_argument("path")
    p.add_argument("-o", "--output")
    p.add_argument("--issues", action="store_true", help="подсветить проблемы валидатора")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("convert-labelme", help="разметка LabelMe → plan.json")
    p.add_argument("path")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_convert_labelme)

    p = sub.add_parser("convert-resplan", help="ResPlan.pkl → набор plan.json")
    p.add_argument("path")
    p.add_argument("-o", "--output", default="data/resplan")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-area", type=float, default=120.0)
    p.add_argument("--max-bedrooms", type=int, default=3)
    p.add_argument("--split", choices=["train", "val", "test"], help="взять только этот сплит")
    p.add_argument("--split-file", default="split.json")
    p.add_argument("--keep-invalid", action="store_true",
                   help="сохранять и планы, на которых валидатор нашёл ошибки")
    p.set_defaults(func=cmd_convert_resplan)

    p = sub.add_parser("recognize", help="картинка плана → plan.json (Обмерщик v0)")
    p.add_argument("path")
    p.add_argument("-o", "--output")
    p.add_argument("--debug", action="store_true", help="сохранить картинку с наложением")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("eval-recognition", help="метрика распознавания по подписанным площадям")
    p.add_argument("folder")
    p.set_defaults(func=cmd_eval_recognition)

    p = sub.add_parser("layout", help="расставить мебель (каталог + солвер CP-SAT + Критик)")
    p.add_argument("path", help="plan.json или state.json")
    p.add_argument("-o", "--output")
    p.add_argument("--rooms", help="пересчитать только эти комнаты, через запятую")
    p.add_argument("--keep", nargs="*", help="id предметов, которые оставить на месте")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=5.0)
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("llm-check", help="проверить связь с моделью из .env")
    p.set_defaults(func=cmd_llm_check)

    p = sub.add_parser("brief", help="новый проект: план + бриф текстом → первая версия")
    p.add_argument("plan", help="plan.json")
    p.add_argument("text", nargs="?", default="", help="пожелания клиента")
    p.add_argument("--file", help="или файл с пожеланиями")
    p.add_argument("--project", required=True, help="id проекта (папка в --store)")
    p.add_argument("--store", default="projects")
    p.add_argument("--time-limit", type=float, default=5.0)
    p.set_defaults(func=cmd_brief)

    p = sub.add_parser("comment", help="правка текстом к последней версии проекта")
    p.add_argument("project")
    p.add_argument("text")
    p.add_argument("--target", help="id объекта, к которому правка")
    p.add_argument("--version", type=int, help="к какой версии (по умолчанию последняя)")
    p.add_argument("--store", default="projects")
    p.add_argument("--time-limit", type=float, default=5.0)
    p.set_defaults(func=cmd_comment)

    p = sub.add_parser("eval-agents", help="метрика агентов на data/agents_eval")
    p.add_argument("--set", choices=["briefs", "edits", "all"], default="all")
    p.add_argument("--split", choices=["dev", "holdout"], default="dev",
                   help="dev — набор для настройки промптов, holdout — отложенный, для отчёта")
    p.add_argument("--data", default="data/agents_eval")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_eval_agents)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
