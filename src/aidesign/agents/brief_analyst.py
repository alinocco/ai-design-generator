"""Аналитик брифа: текст пожеланий клиента → Brief и ограничения для солвера."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Brief, Constraint, ConstraintSource, Occupants, Plan, Rule
from .drafts import RULES_TEXT, BriefDraft, catalog_text, plan_text, to_constraints
from .llm import LLMClient
from .structured import ask_structured

SYSTEM = """Ты — Аналитик брифа в системе, которая проектирует интерьер квартиры.
Тебе дают план квартиры (список комнат) и пожелания клиента обычным текстом.
Извлеки из текста:
1) кто живёт: adults, children, remote_work (работает ли кто-то из дома), pets;
2) style — стиль словами клиента, must_have / nice_to_have / avoid — короткие пункты;
3) constraints — только то, что прямо влияет на расстановку мебели, по словарю ниже.

Правила:
- subject — только тип из каталога (английский идентификатор), room — только id из плана.
- Не выдумывай ограничения, которых нет в тексте. Если клиент ничего не сказал про предмет — не пиши правило.
- «нужна ванна, а не душ» → exclude shower; «только душ» → exclude bathtub.
- «работаю из дома» → remote_work=true и require desk.
- «без телевизора» → exclude tv_stand.
- «по возможности», «хорошо бы» → hard=false.
- «X у окна» → near X target "window" и require X (сам предмет тоже нужен).
- «X не там, а в комнате K» → exclude X в первой комнате и require X в комнате K.
- Координаты и размеры комнат не нужны: их считает солвер.

Каталог предметов:
{catalog}

Правила (rule):
{rules}

Примеры (пустые поля опущены; room — id комнаты из плана, только если комната названа в тексте):
«Живу один, есть кот. Хочу душ вместо ванны. Телевизор не нужен, а кресло для чтения — хорошо бы.» →
{{"adults": 1, "pets": true, "must_have": ["душ"], "nice_to_have": ["кресло для чтения"], "avoid": ["телевизор"],
"constraints": [{{"rule": "exclude", "subject": "bathtub"}}, {{"rule": "exclude", "subject": "tv_stand"}},
{{"rule": "require", "subject": "armchair", "hard": false}}]}}
«Нас трое: мы с женой и сын. Кресло не в спальне, а в гостиной.» →
{{"adults": 2, "children": 1, "constraints": [{{"rule": "exclude", "subject": "armchair", "room": "<id спальни>"}},
{{"rule": "require", "subject": "armchair", "room": "<id гостиной>"}}]}}
«Мы с мужем хотим светлый сканди, бюджет 2 млн.» →
{{"adults": 2, "style": ["скандинавский", "светлые тона"], "budget_rub": 2000000, "constraints": []}}
Последний пример: если в тексте нет пожеланий к мебели, constraints пустой. Не добавляй «стол у окна»
или другие правила, если о них не сказано."""


@dataclass
class BriefResult:
    brief: Brief
    constraints: list[Constraint]
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1


def analyze_brief(llm: LLMClient, plan: Plan, text: str, *, id_prefix: str = "b") -> BriefResult:
    if not text.strip():
        return BriefResult(Brief(), [])
    system = SYSTEM.format(catalog=catalog_text(), rules=RULES_TEXT)
    user = f"План квартиры:\n{plan_text(plan)}\n\nПожелания клиента:\n{text.strip()}"
    res = ask_structured(llm, BriefDraft, system, user, agent="brief_analyst")
    d: BriefDraft = res.value  # type: ignore[assignment]
    brief = Brief(raw_text=text.strip(),
                  occupants=Occupants(adults=d.adults, children=d.children,
                                      remote_work=d.remote_work, pets=d.pets),
                  style=d.style, must_have=d.must_have, nice_to_have=d.nice_to_have,
                  avoid=d.avoid, budget_rub=d.budget_rub)
    conv = to_constraints(d.constraints, plan, None, source=ConstraintSource.brief, id_prefix=id_prefix)
    cons = implied_requirements(conv.constraints, id_prefix)
    return BriefResult(brief, cons, conv.warnings, res.attempts)


def implied_requirements(cons: list[Constraint], id_prefix: str) -> list[Constraint]:
    """«Кресло у окна» значит, что кресло нужно: near без require дополняется require.

    Модели часто пишут одно правило из двух; дописываем второе кодом, если предмет не исключён."""
    have = {(c.subject, c.room) for c in cons if c.rule == Rule.require}
    have_any = {c.subject for c in cons if c.rule == Rule.require}
    excluded = {c.subject for c in cons if c.rule == Rule.exclude and c.room is None}
    out = list(cons)
    n = len(cons)
    for c in cons:
        if c.rule != Rule.near or c.subject in excluded or c.subject in have_any:
            continue
        if (c.subject, c.room) in have:
            continue
        n += 1
        out.append(Constraint(id=f"{id_prefix}{n}", kind=c.kind, rule=Rule.require, subject=c.subject,
                              room=c.room, weight=c.weight, source=c.source))
        have_any.add(c.subject)
    return out
