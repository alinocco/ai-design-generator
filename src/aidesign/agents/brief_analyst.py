"""Аналитик брифа: текст пожеланий клиента → Brief и ограничения для солвера.

Модель только разбирает текст: кто живёт и какие действия с мебелью названы — обычными
словами. В типы каталога и правила их переводит код (agents/resolve.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Brief, Constraint, ConstraintSource, Occupants, Plan, Rule
from .drafts import BriefActionsDraft, plan_text
from .llm import LLMClient
from .resolve import actions_to_constraints
from .structured import ask_structured

SYSTEM = """Ты разбираешь пожелания клиента к интерьеру квартиры.

Заполни:
- adults, children — сколько взрослых и детей живёт; remote_work — работает ли кто-то из дома;
  pets — есть ли животные;
- style, must_have, nice_to_have, avoid — короткими фразами из текста;
- budget_rub — бюджет в рублях, если назван;
- actions — КАЖДОЕ пожелание про мебель и сантехнику отдельным действием.

Действия (action):
- «убрать» — предмет не нужен: «телевизор не нужен», «без кресел», «вместо ванны душ» (убрать ванну);
- «добавить» — предмет нужен: «нужен рабочий стол», «обеденный стол обязателен», «хочу кресло»,
  «вместо ванны душ» (добавить душ);
- «к окну» — предмет у окна;
- «больше места» — свободное место перед предметом;
- «к стене», «не мешать двери».
object — предмет в именительном падеже, как в тексте: «рабочий стол», «стиральная машина».
room — комната словами, только если названа: «спальня», «кухня», «санузел».
hard=false — если «хорошо бы», «по возможности», «желательно».

Подсказки:
- «работаем из дома», «удалёнка» → remote_work=true и действие «добавить — рабочий стол».
- Если предмет нужен в одной комнате, а не в другой — два действия: «убрать» с первой комнатой
  и «добавить» со второй.
- Если про мебель ничего не сказано, actions пустой. Не придумывай пожеланий.

Пример: «Живу один, есть кот. Душ вместо ванны. Телевизор не нужен, кресло для чтения — хорошо бы.»
{"adults": 1, "pets": true, "must_have": ["душ"], "avoid": ["телевизор"], "actions": [
{"action": "убрать", "object": "ванна"}, {"action": "добавить", "object": "душ"},
{"action": "убрать", "object": "телевизор"}, {"action": "добавить", "object": "кресло", "hard": false}]}

Пример: «Мы с мужем и двое детей. Кресло не в спальне, а в гостиной. Стиль — светлый сканди.»
{"adults": 2, "children": 2, "style": ["скандинавский", "светлый"], "actions": [
{"action": "убрать", "object": "кресло", "room": "спальня"},
{"action": "добавить", "object": "кресло", "room": "гостиная"}]}"""


@dataclass
class BriefResult:
    brief: Brief
    constraints: list[Constraint]
    warnings: list[str] = field(default_factory=list)
    attempts: int = 1


def analyze_brief(llm: LLMClient, plan: Plan, text: str, *, id_prefix: str = "b") -> BriefResult:
    if not text.strip():
        return BriefResult(Brief(), [])
    user = f"Комнаты квартиры:\n{plan_text(plan)}\n\nПожелания клиента:\n{text.strip()}"
    res = ask_structured(llm, BriefActionsDraft, SYSTEM, user, agent="brief_analyst")
    d: BriefActionsDraft = res.value  # type: ignore[assignment]
    brief = Brief(raw_text=text.strip(),
                  occupants=Occupants(adults=d.adults, children=d.children,
                                      remote_work=d.remote_work, pets=d.pets),
                  style=d.style, must_have=d.must_have, nice_to_have=d.nice_to_have,
                  avoid=d.avoid, budget_rub=d.budget_rub)
    conv = actions_to_constraints(d.actions, plan, None, source=ConstraintSource.brief,
                                  id_prefix=id_prefix, brief=True)
    cons = implied_requirements(conv.constraints, id_prefix)
    if d.remote_work and not any(c.subject == "desk" for c in cons):
        cons.append(Constraint(id=f"{id_prefix}{len(cons) + 1}", kind="hard", rule=Rule.require,
                               subject="desk", source=ConstraintSource.brief))
    return BriefResult(brief, cons, conv.warnings, res.attempts)


def implied_requirements(cons: list[Constraint], id_prefix: str) -> list[Constraint]:
    """«Кресло у окна» значит, что кресло нужно: near без require дополняется require."""
    have_any = {c.subject for c in cons if c.rule == Rule.require}
    excluded = {c.subject for c in cons if c.rule == Rule.exclude and c.room is None}
    out = list(cons)
    n = len(cons)
    for c in cons:
        if c.rule != Rule.near or c.subject in excluded or c.subject in have_any:
            continue
        n += 1
        out.append(Constraint(id=f"{id_prefix}{n}", kind=c.kind, rule=Rule.require, subject=c.subject,
                              room=c.room, weight=c.weight, source=c.source))
        have_any.add(c.subject)
    return out
