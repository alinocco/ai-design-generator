"""Командная строка шага 0.

  aidesign validate data/demo/demo_48.json
  aidesign render data/demo/demo_48.json -o out/demo.svg
  aidesign convert-labelme разметка.json -o data/plans/plan_01.json
  aidesign convert-resplan ResPlan.pkl -o data/resplan --limit 200
  aidesign recognize data/raw_plans/plan.png --debug
  aidesign eval-recognition data/raw_plans

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

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
