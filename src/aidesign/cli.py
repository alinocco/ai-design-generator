"""Командная строка шага 0.

  aidesign validate data/demo/demo_48.json
  aidesign render data/demo/demo_48.json -o out/demo.svg
  aidesign convert-labelme разметка.json -o data/plans/plan_01.json
  aidesign convert-resplan ResPlan.pkl -o data/resplan --limit 200

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

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
