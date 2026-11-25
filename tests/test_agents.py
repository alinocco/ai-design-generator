import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

pytest.importorskip("ortools")

from aidesign.agents import (FakeLLM, LLMConfig, LLMError, OllamaClient,  # noqa: E402
                             OpenAICompatClient, Orchestrator)
from aidesign.agents.drafts import ConstraintDraft, merge_constraints, to_constraints  # noqa: E402
from aidesign.agents.eval import eval_briefs, eval_edits, load_cases  # noqa: E402
from aidesign.agents.structured import ask_structured, extract_json, inline_schema  # noqa: E402
from aidesign.agents.drafts import EditDraft  # noqa: E402
from aidesign.schemas import ConstraintSource, Layout, ProjectState, Rule  # noqa: E402
from aidesign.store import VersionStore  # noqa: E402

EVAL = Path(__file__).resolve().parents[1] / "data" / "agents_eval"


@pytest.fixture
def demo_layout() -> Layout:
    return Layout.model_validate_json((EVAL / "demo_layout.json").read_text(encoding="utf-8"))


def _brief(**kw):
    d = {"adults": 2, "children": 0, "remote_work": False, "pets": False, "style": [],
         "must_have": [], "nice_to_have": [], "avoid": [], "budget_rub": None, "actions": []}
    d.update(kw)
    return json.dumps(d, ensure_ascii=False)


def _edit(intent="change", actions=(), reply="ок"):
    return json.dumps({"intent": intent, "actions": list(actions), "reply": reply}, ensure_ascii=False)


def _a(action, obj, room=None, **kw):
    return {"action": action, "object": obj, "room": room, **kw}


# --- разбор ответа -------------------------------------------------------------------

def test_extract_json_from_fences():
    assert json.loads(extract_json('Вот:\n```json\n{"a": 1}\n```\nготово')) == {"a": 1}
    assert json.loads(extract_json('ответ {"a": {"b": 2}} конец')) == {"a": {"b": 2}}


def test_inline_schema_has_no_refs():
    s = json.dumps(inline_schema(EditDraft))
    assert "$ref" not in s and "$defs" not in s and "not_in_swing" in s


def test_retry_after_invalid_answer():
    llm = FakeLLM(["я не знаю", '{"intent": "maybe"}',
                   json.dumps({"intent": "question", "constraints": [], "move": [], "rooms": [], "reply": ""})])
    res = ask_structured(llm, EditDraft, "sys", "user", agent="t")
    assert res.attempts == 3 and res.value.intent == "question"
    # модель видит свою ошибку
    assert "не прошёл проверку" in llm.calls[-1][-1]["content"]


def test_gives_up_after_retries():
    with pytest.raises(LLMError):
        ask_structured(FakeLLM(["x", "y", "z"]), EditDraft, "s", "u", agent="t")


# --- черновик → ограничения ---------------------------------------------------------

def test_to_constraints_resolves_names_and_ids(demo_plan, demo_layout):
    drafts = [
        ConstraintDraft(rule="exclude", subject="Рабочий стол", room="спальня"),
        ConstraintDraft(rule="near", subject="r1_sofa", target="window"),
        ConstraintDraft(rule="keep", subject="bed_double"),
        ConstraintDraft(rule="require", subject="jacuzzi"),
        ConstraintDraft(rule="near", subject="desk", target="moon"),
        ConstraintDraft(rule="min_clearance", subject="wardrobe", mm=50, hard=False),
    ]
    conv = to_constraints(drafts, demo_plan, demo_layout, source=ConstraintSource.user,
                          id_prefix="c", comment_id="cm1")
    got = [(c.rule.value, c.subject, c.room, c.target) for c in conv.constraints]
    assert ("exclude", "desk", "r2", None) in got
    assert ("near", "sofa", "r1", "window") in got
    assert ("keep", "r2_bed", None, None) in got
    mc = next(c for c in conv.constraints if c.rule == Rule.min_clearance)
    assert mc.params["mm"] == 900 and mc.kind == "soft"
    assert len(conv.warnings) == 2  # jacuzzi и moon отброшены
    assert all(c.comment_id == "cm1" for c in conv.constraints)


def test_merge_new_rule_replaces_old(demo_plan):
    old = to_constraints([ConstraintDraft(rule="require", subject="desk")], demo_plan, None,
                         source=ConstraintSource.brief, id_prefix="b").constraints
    new = to_constraints([ConstraintDraft(rule="exclude", subject="desk")], demo_plan, None,
                         source=ConstraintSource.user, id_prefix="u", comment_id="cm1").constraints
    merged, dropped = merge_constraints(old, new)
    assert dropped == ["b1"] and [c.id for c in merged] == ["u1"]


# --- оркестратор --------------------------------------------------------------------

def test_orchestrator_brief_then_edit(demo_plan, tmp_path):
    llm = FakeLLM([
        _brief(remote_work=True, actions=[_a("добавить", "рабочий стол"), _a("убрать", "ванна")]),
        _edit(actions=[_a("оставить", "кровать"), _a("убрать", "стол", "спальня"),
                       _a("переставить", "шкаф")], reply="Убираю стол, шкаф переставлю."),
        _edit("question", reply="Площадь спальни 11,4 м²."),
    ])
    store = VersionStore(tmp_path)
    orch = Orchestrator(llm, store)
    t1 = orch.start(demo_plan, "Двое, работаем дома, только душ", "p1")
    types1 = {i.type for i in t1.state.layout.items}
    assert "desk" in types1 and "shower" in types1 and "bathtub" not in types1
    assert t1.state.version == 1 and t1.state.brief.occupants.remote_work

    t2 = orch.comment(t1.state, "Убери стол, шкаф переставь, кровать не трогай")
    assert t2.state.version == 2 and t2.state.parent_version == 1 and t2.rooms == ["r2"]
    a = {i.id: i for i in t1.state.layout.items}
    b = {i.id: i for i in t2.state.layout.items}
    assert "r2_desk" not in b
    assert b["r2_bed"] == a["r2_bed"]
    w1, w2 = a["r2_wardrobe"], b["r2_wardrobe"]
    assert (w1.x, w1.y, w1.rotation) != (w2.x, w2.y, w2.rotation)
    for iid, it in a.items():  # другие комнаты — предмет в предмет
        if it.room != "r2":
            assert b[iid] == it
    assert t2.state.comments[-1].parsed_to and not any("tmp" in c.id for c in t2.state.constraints)
    assert "b1" not in {c.id for c in t2.state.constraints}  # «нужен стол» вытеснен правкой

    t3 = orch.comment(t2.state, "Какая площадь спальни?")
    assert not t3.changed and t3.state.version == 2 and "11,4" in t3.reply
    assert store.versions("p1") == [1, 2]
    log = (tmp_path / "p1" / "llm_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2  # вопрос новую версию не создаёт и в журнал версии не пишется


# --- HTTP-клиенты: локальный сервер вместо Ollama и API ------------------------------

class _Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append((self.path, body, self.headers.get("Authorization")))
        if self.path == "/api/chat":
            out = {"message": {"content": '{"ok": true}'}, "prompt_eval_count": 10, "eval_count": 3}
        else:
            out = {"choices": [{"message": {"content": '{"ok": true}'}}],
                   "usage": {"prompt_tokens": 11, "completion_tokens": 4}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Handler.seen = []
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_client_sends_schema(server):
    llm = OllamaClient(LLMConfig(provider="ollama", model="qwen", base_url=server))
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    assert llm.chat([{"role": "user", "content": "hi"}], schema, agent="t") == '{"ok": true}'
    path, body, _ = _Handler.seen[-1]
    assert path == "/api/chat" and body["format"] == schema and body["stream"] is False
    assert body["options"]["num_ctx"] >= 8192
    assert llm.log[-1].prompt_tokens == 10


@pytest.mark.parametrize("mode,expected", [("schema", "json_schema"), ("object", "json_object"),
                                           ("none", None)])
def test_openai_client_response_format(server, mode, expected):
    llm = OpenAICompatClient(LLMConfig(provider="openai", model="m", base_url=server + "/v1",
                                       api_key="sk-test", json_mode=mode))
    llm.chat([{"role": "user", "content": "hi"}], {"type": "object"}, agent="t")
    path, body, auth = _Handler.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert (body.get("response_format") or {}).get("type") == expected
    assert llm.log[-1].completion_tokens == 4


def test_config_from_env_file(tmp_path, monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_JSON_MODE"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    env.write_text("# комментарий\nLLM_PROVIDER=openai\nLLM_MODEL=deepseek-chat\n"
                   "LLM_BASE_URL=https://api.deepseek.com/\nLLM_JSON_MODE=object\n", encoding="utf-8")
    cfg = LLMConfig.from_env(env)
    assert (cfg.provider, cfg.model, cfg.base_url, cfg.json_mode) == \
        ("openai", "deepseek-chat", "https://api.deepseek.com", "object")
    monkeypatch.setenv("LLM_MODEL", "other")  # переменная окружения важнее файла
    assert LLMConfig.from_env(env).model == "other"


# --- набор для оценки согласован с планом: «идеальная» модель получает 100% -----------

_ACT = {"exclude": "убрать", "require": "добавить", "keep": "оставить", "near": "к окну",
        "min_clearance": "больше места", "not_in_swing": "не мешать двери", "against_wall": "к стене"}


def _oracle(cases, kind):
    """«Идеальная» модель: переводит ожидаемые правила обратно в действия словами."""
    from aidesign.catalog import CATALOG
    by_text = {c["text"]: c for c in cases}

    def reply(messages):
        user = messages[1]["content"]
        case = next(c for t, c in by_text.items() if user.rstrip().endswith(t))
        exp = case["expect"]
        acts = []
        for c in exp.get("constraints", []):
            obj = c["subject"] if c["rule"] == "keep" else CATALOG[c["subject"]].name_ru
            acts.append(_a(_ACT[c["rule"]], obj, c.get("room")))
        acts += [_a("переставить", m) for m in exp.get("move", [])]
        if kind == "brief":
            occ = {"adults": 1, "children": 0, "remote_work": False, "pets": False}
            occ.update(exp.get("occupants", {}))
            return _brief(**occ, actions=acts)
        return _edit(exp.get("intent", "change"), acts)
    return reply


@pytest.mark.parametrize("suffix,nb,ne", [("", 10, 15), ("_holdout", 5, 8)])
def test_eval_sets_are_consistent(demo_plan, demo_layout, suffix, nb, ne):
    briefs = load_cases(EVAL / f"briefs{suffix}.jsonl")
    edits = load_cases(EVAL / f"edits{suffix}.jsonl")
    assert len(briefs) >= nb and len(edits) >= ne
    rb = eval_briefs(FakeLLM(_oracle(briefs, "brief")), demo_plan, briefs)
    re_ = eval_edits(FakeLLM(_oracle(edits, "edit")),
                     ProjectState(project_id="e", plan=demo_plan, layout=demo_layout), edits)
    for rep in (rb, re_):
        bad = [(c.id, c.detail, c.error) for c in rep.cases if not c.ok]
        assert not bad, bad
        assert rep.f1 == 1.0


# --- проверки согласованности поверх ответа модели ------------------------------------

def test_reconcile_drops_contradictions(demo_plan, demo_layout):
    from aidesign.agents.edit_agent import reconcile
    items = {i.id: i for i in demo_layout.items}
    conv = to_constraints([ConstraintDraft(rule="exclude", subject="bathtub", room="r3"),
                           ConstraintDraft(rule="near", subject="sofa", room="r1", target="window"),
                           ConstraintDraft(rule="require", subject="washer", room="r3"),
                           ConstraintDraft(rule="keep", subject="r2_wardrobe")],
                          demo_plan, demo_layout, source=ConstraintSource.user, id_prefix="c",
                          comment_id="cm1")
    cons, move, notes = reconcile(conv.constraints, ["r3_bath", "r1_sofa", "r3_washer", "r2_wardrobe"],
                                  items)
    assert move == ["r3_washer", "r2_wardrobe"]  # ванну убирают, диван ставят у окна
    rules = {(c.rule.value, c.subject) for c in cons}
    assert ("require", "washer") not in rules and ("keep", "r2_wardrobe") not in rules
    assert len(notes) == 4


@pytest.mark.parametrize("text,shift", [
    ("Сдвинь кровать на 20 сантиметров левее.", True),
    ("подвинь диван на 30 см вправо", True),
    ("Передвинь шкаф на полметра", True),
    ("Перед шкафом нужно хотя бы метр.", False),
    ("Переставь диван куда-нибудь", False),
])
def test_exact_shift_is_unsupported(text, shift):
    from aidesign.agents.edit_agent import _EXACT_SHIFT
    assert bool(_EXACT_SHIFT.search(text)) == shift


def test_edit_question_and_shift_never_change(demo_plan, demo_layout):
    from aidesign.agents import parse_edit
    state = ProjectState(project_id="x", plan=demo_plan, layout=demo_layout)
    # модель ошиблась с намерением — код всё равно не даёт точный сдвиг
    llm = FakeLLM([_edit(actions=[_a("больше места", "кровать"), _a("переставить", "кровать")])])
    res = parse_edit(llm, state, "Сдвинь кровать на 20 см левее", comment_id="c1")
    assert res.intent == "unsupported" and not res.constraints and not res.move
    # question с правилами — правила отбрасываются
    llm = FakeLLM([_edit("question", actions=[_a("убрать", "тумбочка")])])
    res = parse_edit(llm, state, "Зачем две тумбочки?", comment_id="c2")
    assert res.intent == "question" and not res.constraints and not res.rooms


def test_brief_near_implies_require(demo_plan):
    from aidesign.agents import analyze_brief
    llm = FakeLLM([_brief(actions=[_a("к окну", "кресло"), _a("к окну", "рабочий стол"),
                                   _a("добавить", "стол")])])
    res = analyze_brief(llm, demo_plan, "Кресло и стол у окна")
    got = [(c.rule.value, c.subject) for c in res.constraints]
    assert got.count(("require", "armchair")) == 1 and got.count(("require", "desk")) == 1


def test_layout_text_groups_by_room(demo_plan, demo_layout):
    from aidesign.agents.drafts import layout_text
    txt = layout_text(demo_layout, demo_plan)
    bedroom = txt.split("r2 — Спальня:")[1].split("\nr3")[0]
    assert "r2_desk — Рабочий стол" in bedroom and "dining_table" not in bedroom


@pytest.mark.parametrize("phrase,room,expected", [
    ("стол", "r2", ["r2_desk"]), ("стол", "r1", ["r1_dining_table"]),
    ("обеденный стол", None, ["r1_dining_table"]), ("рабочее место", None, ["r2_desk"]),
    ("тумба под телевизор", None, ["r1_tv_stand"]), ("стиралку", None, ["r3_washer"]),
    ("тумбочки", "r2", ["r2_bed_nightstand_l0", "r2_bed_nightstand_r0"]), ("r2_bed", None, ["r2_bed"]),
    ("джакузи", None, ["r3_bath"]), ("пианино", None, []),
])
def test_resolve_items(demo_layout, phrase, room, expected):
    from aidesign.agents.resolve import resolve_items
    assert [i.id for i in resolve_items(phrase, room, demo_layout)] == expected


@pytest.mark.parametrize("phrase,room_id", [("спальня", "r2"), ("в спальне", "r2"), ("гостиная", "r1"),
                                            ("кухня", "r1"), ("ванная", "r3"), ("санузел", "r3"),
                                            ("прихожая", "r4"), ("r3", "r3"), ("балкон", None)])
def test_resolve_room(demo_plan, phrase, room_id):
    from aidesign.agents.resolve import resolve_room
    assert resolve_room(demo_plan, phrase)[0] == room_id


def test_window_phrase_fixes_move(demo_plan, demo_layout):
    from aidesign.agents import parse_edit
    state = ProjectState(project_id="x", plan=demo_plan, layout=demo_layout)
    res = parse_edit(FakeLLM([_edit(actions=[_a("переставить", "диван")])]), state,
                     "Хочу, чтобы диван стоял у окна.", comment_id="c1")
    assert not res.move and [(c.rule.value, c.subject, c.target) for c in res.constraints] == \
        [("near", "sofa", "window")]


def test_shower_replaces_bath_and_dedupe(demo_plan):
    from aidesign.agents.resolve import actions_to_constraints
    from aidesign.agents.drafts import ActionDraft as A
    res = actions_to_constraints([A(action="добавить", object="душевая"),
                                  A(action="добавить", object="рабочий стол", room="спальня"),
                                  A(action="добавить", object="стол")],
                                 demo_plan, None, source=ConstraintSource.brief, id_prefix="b")
    got = [(c.rule.value, c.subject, c.room) for c in res.constraints]
    assert ("exclude", "bathtub", None) in got
    assert got.count(("require", "desk", "r2")) == 1 and ("require", "desk", None) not in got


def test_move_to_other_room(demo_plan, demo_layout):
    from aidesign.agents import parse_edit
    state = ProjectState(project_id="x", plan=demo_plan, layout=demo_layout)
    res = parse_edit(FakeLLM([_edit(actions=[_a("переставить", "стиральная машина", "кухня")])]), state,
                     "Стиральная машина пусть будет на кухне", comment_id="c1")
    got = {(c.rule.value, c.subject, c.room) for c in res.constraints}
    assert got == {("exclude", "washer", "r3"), ("require", "washer", "r1")} and not res.move
    assert set(res.rooms) == {"r1", "r3"}
