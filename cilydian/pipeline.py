"""CiLydian: self-hosted agentic CI pipeline.

Flow: spec -> coder -> LLM judge (advisory by default) -> deterministic gates
(ruff + pytest), with a bounded repair loop and human escalation when HITL is on.

Models are pluggable per role via config.toml: local Ollama or any
OpenAI-compatible endpoint (OpenRouter, Groq, Gemini compat, DeepSeek, vLLM).
API keys are never stored in config: roles reference environment variable names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from datetime import datetime
from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

RUN_ROOT = "runs"
LOG_DIR = os.path.join(RUN_ROOT, "latest")  # rebound per run in begin_run()
LOG_CAP = 5_000_000
MEMORY_DB = "memory.db"
MEMORY_CAP = 500
DEFAULT_IDEA = "Rate limiter on asyncio.Semaphore for CVE scanner"
_FENCE_BLOCK = re.compile(r"```[\w-]*[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL)

# Populated by init_models/init_from_config; nodes read these at call time.
SPEC_LLM = None
CODER = None
JUDGE = None
JUDGE_MODEL = ""
JUDGE_PROVIDER = ""
JUDGE_TEMPERATURE = 0.0
MAX_ITER = 3
JUDGE_BLOCKING = False
HITL = False
PREFLIGHT_LLM = True

JUDGE_PROMPT_VERSION = "v3-nofences-specbound"  # bump on every judge prompt change


# ---------- env + config ----------


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader; avoids a python-dotenv dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[config] {path} not found; see config.toml in repo root")


def build_llm(cfg: dict, default_temperature: float = 0.0):
    """Instantiate a chat model for one pipeline role."""
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return ChatOllama(
            model=cfg["model"],
            base_url=cfg.get("base_url", "http://localhost:11434"),
            num_ctx=cfg.get("num_ctx", 4096),
            temperature=cfg.get("temperature", default_temperature),
        )
    if provider == "openai_compat":
        key_env = cfg.get("api_key_env", "")
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise SystemExit(
                f"[config] role needs env var {key_env!r} but it is not set. "
                f"Copy .env.example to .env and fill it in."
            )
        return ChatOpenAI(
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=api_key,
            temperature=cfg.get("temperature", default_temperature),
            max_retries=3,  # exponential backoff on 429 (free-tier rate limits)
        )
    raise SystemExit(f"[config] unknown provider: {provider!r}")


def init_models(cfg: dict) -> None:
    global SPEC_LLM, CODER, JUDGE, JUDGE_MODEL, JUDGE_PROVIDER, JUDGE_TEMPERATURE
    roles = cfg.get("roles", {})
    for role in ("spec", "coder", "judge"):
        if role not in roles:
            raise SystemExit(f"[config] missing [roles.{role}] section")
    SPEC_LLM = build_llm(roles["spec"], default_temperature=0.0)
    CODER = build_llm(roles["coder"], default_temperature=0.2)
    JUDGE = build_llm(roles["judge"], default_temperature=0.0)
    JUDGE_MODEL = roles["judge"].get("model", "unknown")
    JUDGE_PROVIDER = roles["judge"].get("provider", "ollama")
    JUDGE_TEMPERATURE = float(roles["judge"].get("temperature", 0.0))


def init_from_config(path: str, hitl: bool | None = None) -> None:
    """Single entry point for CLI and studio. Idempotent."""
    global MAX_ITER, JUDGE_BLOCKING, HITL, PREFLIGHT_LLM
    load_dotenv()
    cfg = load_config(path)
    init_models(cfg)
    pipe = cfg.get("pipeline", {})
    MAX_ITER = int(pipe.get("max_iter", 3))
    JUDGE_BLOCKING = bool(pipe.get("judge_blocking", False))
    HITL = bool(pipe.get("hitl", False)) if hitl is None else hitl
    PREFLIGHT_LLM = bool(pipe.get("preflight_llm", True))
    memory_init()


# ---------- runs/ + logs ----------


def begin_run(run_id: str | None = None) -> str:
    """Create an isolated artifact directory for one pipeline run."""
    global LOG_DIR
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(RUN_ROOT, run_id)
    ensure_isolation()
    reset_logs()
    return LOG_DIR


def log(agent: str, msg: str) -> None:
    path = os.path.join(LOG_DIR, f"{agent}.log")
    if os.path.exists(path) and os.path.getsize(path) > LOG_CAP:
        os.replace(path, path + ".old")
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def reset_logs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    for agent in ("spec", "coder", "judge", "tester"):
        open(os.path.join(LOG_DIR, f"{agent}.log"), "w", encoding="utf-8").close()


def ensure_isolation() -> None:
    """Stop ruff/pytest from inheriting the parent project's configs."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(LOG_DIR, "ruff.toml")):
        open(os.path.join(LOG_DIR, "ruff.toml"), "w", encoding="utf-8").close()
    if not os.path.exists(os.path.join(LOG_DIR, "pytest.ini")):
        with open(os.path.join(LOG_DIR, "pytest.ini"), "w", encoding="utf-8") as f:
            f.write("[pytest]\nasyncio_mode = auto\n")


def run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )


def preflight() -> None:
    """Cheap local checks always; LLM ping only if PREFLIGHT_LLM is true."""
    for tool in ("ruff", "pytest"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"[preflight] {tool} not found on PATH (see requirements.txt)"
            )
    if not PREFLIGHT_LLM:
        return
    try:
        CODER.invoke("ping")
    except Exception as exc:
        raise SystemExit(f"[preflight] coder endpoint unreachable: {exc}") from exc


# ---------- verdict memory + instincts (SQLite, capped) ----------


def memory_init() -> None:
    """Schema with migrations: verdicts (spec/code added for export-eval), instincts."""
    with sqlite3.connect(MEMORY_DB) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""CREATE TABLE IF NOT EXISTS verdicts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            model TEXT, score INTEGER, verdict TEXT, issues TEXT,
            ts TEXT DEFAULT CURRENT_TIMESTAMP)""")
        cols = {r[1] for r in con.execute("PRAGMA table_info(verdicts)")}
        for col in ("spec", "code"):
            if col not in cols:
                con.execute(f"ALTER TABLE verdicts ADD COLUMN {col} TEXT")
        con.execute("""CREATE TABLE IF NOT EXISTS instincts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger TEXT UNIQUE NOT NULL,
            lesson TEXT NOT NULL,
            hits INTEGER DEFAULT 1,
            created TEXT DEFAULT CURRENT_TIMESTAMP,
            last_seen TEXT DEFAULT CURRENT_TIMESTAMP)""")


def memory_key(spec: str, code: str) -> str:
    return hashlib.sha256(
        f"{JUDGE_PROVIDER}|{JUDGE_MODEL}|{JUDGE_TEMPERATURE}|"
        f"{JUDGE_PROMPT_VERSION}|{spec}|{code}".encode()
    ).hexdigest()


def memory_get(key: str):
    with sqlite3.connect(MEMORY_DB) as con:
        return con.execute(
            "SELECT score, verdict, issues FROM verdicts WHERE key=?", (key,)
        ).fetchone()


def memory_put(key: str, verdict: dict, spec: str = "", code: str = "") -> None:
    with sqlite3.connect(MEMORY_DB) as con:
        con.execute(
            "INSERT OR REPLACE INTO verdicts(key, model, score, verdict, issues, spec, code)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                key,
                JUDGE_MODEL,
                verdict.get("score"),
                verdict["verdict"],
                json.dumps(verdict["issues"], ensure_ascii=False),
                spec,
                code,
            ),
        )
        # eviction by insertion order (id is monotonic), not by ts (1s resolution)
        con.execute(
            "DELETE FROM verdicts WHERE id NOT IN"
            " (SELECT id FROM verdicts ORDER BY id DESC LIMIT ?)",
            (MEMORY_CAP,),
        )


def instinct_bump(trigger: str, lesson: str) -> None:
    """Record or refresh a lesson learned from a runtime event."""
    with sqlite3.connect(MEMORY_DB) as con:
        con.execute(
            "INSERT INTO instincts(trigger, lesson) VALUES(?,?)"
            " ON CONFLICT(trigger) DO UPDATE SET"
            " hits = hits + 1, last_seen = CURRENT_TIMESTAMP",
            (trigger, lesson),
        )


def instinct_top(n: int = 10) -> list[tuple[str, str, int]]:
    """Top-N lessons by hits, for injection into the coder prompt."""
    with sqlite3.connect(MEMORY_DB) as con:
        return con.execute(
            "SELECT trigger, lesson, hits FROM instincts ORDER BY hits DESC LIMIT ?",
            (n,),
        ).fetchall()


def instinct_clear() -> int:
    with sqlite3.connect(MEMORY_DB) as con:
        n = con.execute("SELECT COUNT(*) FROM instincts").fetchone()[0]
        con.execute("DELETE FROM instincts")
        return n


# ---------- utilities ----------


def strip_fences(text: str) -> str:
    """Extract the first fenced code block; models append prose after it."""
    m = _FENCE_BLOCK.search(text)
    if m:
        return m.group(1)
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if "```" in t:
            t = t.split("```", 1)[0]
        return t.rstrip()
    return t


def _f821_names(ruff_stdout: str) -> list[str]:
    """Extract undefined names from ruff F821 lines for instinct seeding."""
    names: list[str] = []
    for line in ruff_stdout.splitlines():
        m = re.search(r"Undefined name [`'\"]([A-Za-z_][A-Za-z0-9_]*)[`'\"]", line)
        if m and m.group(1) not in names:
            names.append(m.group(1))
    return names


# ---------- state + nodes ----------


class State(TypedDict, total=False):
    idea: str
    spec: str
    spec_path: str
    code: str
    code_hash: str
    prev_code_hash: str
    verdict: dict
    judge_verdict: dict
    gates_verdict: dict
    iteration: int
    human_action: str


def _instincts_block() -> str:
    """Render top lessons as a prompt preface for the coder."""
    rows = instinct_top(10)
    if not rows:
        return ""
    lines = [
        "PATTERNS FROM PAST RUNS (apply only if directly relevant):",
    ]
    for trigger, lesson, hits in rows:
        lines.append(f"- [{trigger}, seen {hits}x] {lesson}")
    return "\n".join(lines) + "\n"


def spec_writer(s: State):
    """Stage 0: human-authored spec wins; otherwise generate and persist."""
    if s.get("spec_path"):
        with open(s["spec_path"], encoding="utf-8") as f:
            spec = f.read()
    else:
        spec = SPEC_LLM.invoke(
            "Turn this idea into a MINIMAL SPEC. Include ONLY what the idea states.\n"
            "Sections: Role; Interface (exact PYTHON class/method signatures the "
            "implementation MUST expose - never Java/Go/other languages); Constraints; "
            "Measurable Acceptance Criteria (max 5, unit-testable); Test Plan (max 5 "
            "tests, 1:1 mapping to criteria).\n"
            "If the task involves time, windows, or delays: the Interface MUST include "
            "an injectable clock parameter (e.g. clock: Callable[[], float] = "
            "time.monotonic), and the Test Plan MUST drive time through a fake clock. "
            "Real sleeps and wall-clock measurements are FORBIDDEN in tests.\n"
            "FORBIDDEN unless the idea explicitly requests them: security compliance, "
            "abuse prevention, authentication, monitoring, logging, UAT, post-release "
            "monitoring, performance measurement, config files, documentation.\n"
            f"Idea: {s['idea']}"
        ).content

    with open(os.path.join(LOG_DIR, "SPEC.md"), "w", encoding="utf-8") as f:
        f.write(spec)

    if HITL:
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "stage": "spec",
                "spec": spec,
                "actions": ["continue", "edit <new spec text>", "reject"],
            }
        )
        d = decision if isinstance(decision, dict) else {"action": str(decision)}
        if d.get("action") == "reject":
            return {"spec": "", "iteration": s["iteration"] + 1}
        if d.get("action") == "edit":
            spec = d.get("spec", spec)
            with open(os.path.join(LOG_DIR, "SPEC.md"), "w", encoding="utf-8") as f:
                f.write(spec)

    tests = CODER.invoke(
        f"Generate a pytest file for this spec:\n{spec}\n"
        "Write only test functions, no implementation. "
        "Import the module under test as `from candidate import ...`, "
        "NOT as `from logs.candidate import ...`. "
        "Output raw python code only, no markdown fences."
        "Tests must call ONLY the class and methods listed in the SPEC Interface "
        "section, with exact names and signatures. No invented API.\n"
        "Tests must NEVER call time.sleep or measure wall-clock time. If the SPEC "
        "defines an injectable clock, construct a fake clock and advance it "
        "explicitly to simulate window expiry.\n"
    ).content
    with open(os.path.join(LOG_DIR, "test_candidate.py"), "w", encoding="utf-8") as f:
        f.write(strip_fences(tests))
    log("spec", spec)
    return {"spec": spec}


def coder(s: State):
    """Generate/repair the implementation; tests are the naming contract."""
    feedback = s["verdict"].get("issues", [])
    prompt = f"SPEC:\n{s['spec']}\n"
    try:
        with open(os.path.join(LOG_DIR, "test_candidate.py"), encoding="utf-8") as f:
            prompt += (
                f"TESTS that must pass (match names/signatures exactly):\n{f.read()}\n"
            )
    except OSError:
        pass
    prompt += (
        f"Issues to fix: {feedback}\n"
        "PRIORITY: if pytest failed on a specific test, making that test pass "
        "is your #1 priority. Style/judge suggestions are secondary.\n"
        "If you cannot fix the failing test, do NOT rewrite the same code - "
        "instead add a `# TODO:` comment explaining what is blocked.\n"
        + _instincts_block()
        + "Write only the Python implementation for module `candidate`. "
        "Raw code only, no markdown fences, no tests. "
        "NO explanations, NO headers, NO prose after the code."
    )
    code = strip_fences(CODER.invoke(prompt).content)
    log("coder", code)
    new_hash = hashlib.sha256(code.encode()).hexdigest()[:12]
    return {
        "code": code,
        "iteration": s["iteration"] + 1,
        "code_hash": new_hash,
        "prev_code_hash": s.get("code_hash", ""),
    }


def judge(s: State):
    """LLM review with defensive parsing; verdict memory cache saves tokens."""
    key = memory_key(s["spec"], s["code"])
    hit = memory_get(key)
    if hit:
        log("judge", f"cache hit key={key[:8]} score={hit[0]} verdict={hit[1]}")
        verdict = {
            "score": hit[0],
            "verdict": hit[1],
            "issues": json.loads(hit[2]),
            "gate": "judge",
        }
        return {"verdict": verdict, "judge_verdict": verdict}

    raw = JUDGE.invoke(
        "Rate 1-5: correctness, security, simplicity. Judge ONLY against "
        "the SPEC's measurable acceptance criteria; do not request features "
        f"beyond the SPEC. Check against SPEC:\n{s['spec']}\nCODE:\n{s['code']}\n"
        'Return JSON only: {"score": int, "verdict": "pass"|"fail", "issues": [str]}\n\n'
        "CRITICAL: Output ONLY raw JSON. NO markdown fences.\n"
        'CORRECT: {"score": 4, "verdict": "pass", "issues": []}\n'
        'WRONG: ```json{"score": 4, "verdict": "pass", "issues": []}```'
    ).content
    a, b = raw.find("{"), raw.rfind("}")
    try:
        verdict = json.loads(raw[a : b + 1]) if a != -1 else {}
    except json.JSONDecodeError:
        verdict = {}
    if not isinstance(verdict, dict):
        verdict = {}
    verdict.setdefault("issues", [])
    raw_score = verdict.get("score", 5)
    try:
        score = int(float(raw_score))
    except (TypeError, ValueError):
        score = 0
        verdict["issues"].append(f"Unparseable score: {raw_score!r}")
    verdict["verdict"] = str(verdict.get("verdict", "fail")).lower()
    if verdict["verdict"] not in ("pass", "fail"):
        verdict["verdict"] = "fail"
    if score < 4 and verdict["verdict"] == "pass":
        verdict["verdict"] = "fail"
        verdict["issues"].append(f"Auto-downgraded: score={score} contradicts 'pass'.")
    verdict["gate"] = "judge"
    log("judge", json.dumps(verdict, ensure_ascii=False))
    memory_put(key, verdict, s["spec"], s["code"])
    return {"verdict": verdict, "judge_verdict": verdict}


def tester(s: State):
    """Deterministic gates: ruff (auto-fix then check) and pytest."""
    candidate = os.path.join(LOG_DIR, "candidate.py")
    with open(candidate, "w", encoding="utf-8") as f:
        f.write(s["code"])
    with open(os.path.join(LOG_DIR, "conftest.py"), "w", encoding="utf-8") as f:
        f.write("import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n")
    run(["ruff", "check", "--fix", "candidate.py"], LOG_DIR)
    with open(candidate, encoding="utf-8") as f:
        fixed = f.read()
    ruff = run(["ruff", "check", "candidate.py"], LOG_DIR)
    if ruff.returncode != 0:
        log("tester", f"ruff rc={ruff.returncode}\n{ruff.stdout[:500]}")
        # seed instincts from F821 undefined names
        for name in _f821_names(ruff.stdout):
            if name in ("time", "asyncio"):
                instinct_bump(f"f821:{name}", f"add `import {name}` at top of file")
            elif name == "Callable":
                instinct_bump("f821:Callable", "add `from typing import Callable`")
            elif name == "deque":
                instinct_bump("f821:deque", "add `from collections import deque`")
            elif name == "Decimal":
                instinct_bump("f821:Decimal", "add `from decimal import Decimal`")
            else:
                instinct_bump(f"f821:{name}", f"add missing import for `{name}`")
        gates = {
            "ruff": "fail",
            "pytest": "skipped",
            "issues": [f"ruff:{ruff.stdout[:300]}"],
        }
        return {
            "code": fixed,
            "verdict": {
                **s["verdict"],
                "gate": "ruff+pytest",
                "verdict": "fail",
                "issues": s["verdict"].get("issues", []) + gates["issues"],
            },
            "gates_verdict": gates,
        }
    pytest_run = run(["pytest", "test_candidate.py", "-q", "--tb=short"], LOG_DIR)
    ok = pytest_run.returncode == 0
    log(
        "tester",
        f"ruff rc=0\npytest rc={pytest_run.returncode}\n{pytest_run.stdout[-500:]}",
    )
    extra_issues = [] if ok else [f"pytest:{pytest_run.stdout[:300]}"]
    gates = {
        "ruff": "pass",
        "pytest": "pass" if ok else "fail",
        "issues": extra_issues,
    }
    return {
        "code": fixed,
        "verdict": {
            **s["verdict"],
            "gate": "ruff+pytest",
            "verdict": "pass" if ok else "fail",
            "issues": s["verdict"].get("issues", []) + extra_issues,
        },
        "gates_verdict": gates,
    }


def escalate(s: State):
    """Human decision point when the bounded loop is exhausted."""
    from langgraph.types import interrupt

    decision = interrupt(
        {
            "stage": "escalation",
            "verdict": s["verdict"],
            "actions": ["accept", "retry <feedback>", "abort"],
        }
    )
    d = decision if isinstance(decision, dict) else {"action": str(decision)}
    action = d.get("action", "abort")
    if action == "accept":
        return {
            "human_action": "accept",
            "verdict": {
                **s["verdict"],
                "verdict": "pass",
                "issues": s["verdict"].get("issues", [])
                + ["human: accepted failing gates"],
            },
        }
    if action == "retry":
        return {
            "human_action": "retry",
            "verdict": {
                **s["verdict"],
                "issues": s["verdict"].get("issues", [])
                + [f"human:{d.get('feedback', '')}"],
            },
        }
    return {"human_action": "abort"}


# ---------- routing ----------


def route_spec(s: State) -> Literal["coder", "end"]:
    """Empty spec (HITL reject or empty --spec file) ends the run."""
    return "end" if not s.get("spec", "").strip() else "coder"


def route(s: State) -> Literal["coder", "tester"]:
    if (
        JUDGE_BLOCKING
        and s["verdict"]["verdict"] == "fail"
        and s["iteration"] < MAX_ITER
    ):
        return "coder"
    return "tester"


def route_tester(s: State) -> Literal["coder", "escalate", "end"]:
    if s["verdict"]["verdict"] == "pass":
        return "end"
    # Early abort: coder produced byte-identical code across iterations.
    # Seed an instinct so future runs see the lesson in the coder prompt.
    prev = s.get("prev_code_hash", "")
    curr = s.get("code_hash", "")
    if prev and curr and prev == curr:
        log("tester", f"early abort: coder stuck at hash={curr}")
        instinct_bump(
            "early-abort",
            "when the previous iteration produced identical code, add a `# TODO:`"
            " explaining what is blocked instead of rewriting the same code",
        )
        return "escalate" if HITL else "end"
    if s["iteration"] < MAX_ITER:
        return "coder"
    return "escalate" if HITL else "end"


def route_escalate(s: State) -> Literal["coder", "end"]:
    return "coder" if s.get("human_action") == "retry" else "end"


# ---------- graph build (no module-level compile!) ----------


def build_graph() -> StateGraph:
    g = StateGraph(State)
    g.add_node("spec", spec_writer)
    g.add_node("coder", coder)
    g.add_node("judge", judge)
    g.add_node("tester", tester)
    g.add_node("escalate", escalate)
    g.set_entry_point("spec")
    g.add_conditional_edges("spec", route_spec, {"coder": "coder", "end": END})
    g.add_edge("coder", "judge")
    g.add_conditional_edges("judge", route, {"coder": "coder", "tester": "tester"})
    g.add_conditional_edges(
        "tester", route_tester, {"coder": "coder", "escalate": "escalate", "end": END}
    )
    g.add_conditional_edges("escalate", route_escalate, {"coder": "coder", "end": END})
    return g


def build_app(checkpointer=None):
    """Plain app for CLI; checkpointed app for HITL clients (studio)."""
    if checkpointer is None and HITL:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as e:
            raise SystemExit("[hitl] pip install langgraph-checkpoint-sqlite") from e
        checkpointer = SqliteSaver(
            sqlite3.connect("checkpoints.db", check_same_thread=False)
        )
    return build_graph().compile(checkpointer=checkpointer)


# ---------- export-eval ----------


_EVAL_ASSERT_SRC = '''"""promptfoo python assertion: compare judge verdict against cached expectation."""
import json

def get_assert(output: str, context: dict) -> dict:
    vars_ = context.get("vars") or {}
    expected = str(vars_.get("expected_verdict", "")).lower()

    # Find the start of JSON
    start = output.find("{")
    if start == -1:
        return {"pass": False, "reason": "no json found in output"}

    try:
        # raw_decode stops at the end of the valid JSON object (ignores trailing text)
        parsed, _ = json.JSONDecoder().raw_decode(output[start:])
    except json.JSONDecodeError as e:
        return {"pass": False, "reason": f"invalid json: {e}"}

    got = str(parsed.get("verdict", "")).lower()

    return {
        "pass": got == expected,
        "reason": f"expected verdict={expected}, got={got}",
    }
'''


def _yaml_block(text: str, indent: int) -> list[str]:
    pad = " " * indent
    return [
        pad + ln if ln.strip() else pad.rstrip()
        for ln in (text or "").splitlines() or [""]
    ]


def export_eval(path: str = "promptfoo_eval.yaml", limit: int = 50) -> int:
    """Dump cached verdicts as promptfoo tests; corpus grows with every run."""
    with sqlite3.connect(MEMORY_DB) as con:
        rows = con.execute(
            "SELECT spec, code, score, verdict FROM verdicts"
            " WHERE spec IS NOT NULL AND code IS NOT NULL AND spec <> ''"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        raise SystemExit("[export-eval] no rows with spec/code in memory.db yet")

    if JUDGE_PROVIDER == "ollama":
        pf_provider = f"ollama:chat:{JUDGE_MODEL}"
    elif JUDGE_PROVIDER == "openai_compat":
        if (
            "openrouter" in str(os.environ.get("OPENROUTER_API_KEY", ""))
            or "openrouter" in str(JUDGE_MODEL).lower()
        ):
            pf_provider = f"openrouter:{JUDGE_MODEL}"
        else:
            pf_provider = f"openai:{JUDGE_MODEL}"
    else:
        pf_provider = f"ollama:chat:{JUDGE_MODEL}"

    lines = [
        "# generated by cilydian --export-eval; do not edit by hand",
        f"# judge_prompt_version: {JUDGE_PROMPT_VERSION}",
        "# run: promptfoo eval -c promptfoo_eval.yaml",
        "providers:",
        f"  - {pf_provider}",
        "prompts:",
        "  - |",
        *_yaml_block(
            "Rate 1-5: correctness, security, simplicity. Judge ONLY against "
            "the SPEC's measurable acceptance criteria; do not request features "
            "beyond the SPEC. Check against SPEC:\n{{spec}}\nCODE:\n{{code}}\n"
            'Return JSON only: {"score": int, "verdict": "pass"|"fail", "issues": [str]}\n\n'
            "CRITICAL: Output ONLY raw JSON. NO markdown fences.\n"
            'CORRECT: {"score": 4, "verdict": "pass", "issues": []}\n'
            'WRONG: ```json{"score": 4, "verdict": "pass", "issues": []}```',
            4,
        ),
        "tests:",
    ]
    for spec, code, score, verdict in rows:
        lines += [
            "  - vars:",
            "      spec: |",
            *_yaml_block(spec, 8),
            "      code: |",
            *_yaml_block(code, 8),
            f"      expected_verdict: {verdict}",
            "    assert:",
            "      - type: javascript",
            "        value: |",
            "          const expected = context.vars.expected_verdict;",
            '          const match = output.match(/"verdict"\\s*:\\s*"(pass|fail)"/i);',
            "          if (!match) return false;",
            "          return match[1].toLowerCase() === expected.toLowerCase();",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Always overwrite eval_assert.py to ensure it's the latest robust version
    with open("eval_assert.py", "w", encoding="utf-8") as f:
        f.write(_EVAL_ASSERT_SRC)

    return len(rows)


# ---------- instincts CLI ----------


def instincts_show(limit: int = 25) -> int:
    rows = instinct_top(limit)
    if not rows:
        print("(instincts table is empty)")
        return 0
    for trigger, lesson, hits in rows:
        print(f"[{hits:3d}] {trigger:30s}  {lesson}")
    return len(rows)


def instincts_add(arg: str) -> None:
    if "|" not in arg:
        raise SystemExit(
            "[instincts] use format: trigger|lesson  "
            "(e.g. 'f821:asyncio|add `import asyncio` at top')"
        )
    trigger, lesson = arg.split("|", 1)
    instinct_bump(trigger.strip(), lesson.strip())
    print(f"[instincts] recorded: {trigger.strip()}")


# ---------- CLI ----------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agentic CI pipeline: spec -> coder -> judge -> deterministic gates"
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="run the pipeline (default)")
    run_p.add_argument(
        "idea", nargs="?", default=DEFAULT_IDEA, help="One-line task description"
    )
    run_p.add_argument("--config", default="config.toml")
    run_p.add_argument("--spec", default="")
    run_p.add_argument("--run-id", default=None)

    ev = sub.add_parser("export-eval", help="dump verdict memory as promptfoo tests")
    ev.add_argument("--config", default="config.toml")
    ev.add_argument("--out", default="promptfoo_eval.yaml")
    ev.add_argument("--limit", type=int, default=50)

    ins = sub.add_parser("instincts", help="manage the learned lessons table")
    ins.add_argument("action", choices=["show", "add", "clear"])
    ins.add_argument("arg", nargs="?", default="")
    ins.add_argument("--config", default="config.toml")
    ins.add_argument("--limit", type=int, default=25)

    if len(sys.argv) > 1 and sys.argv[1] not in (
        "run",
        "export-eval",
        "instincts",
        "-h",
        "--help",
    ):
        args = parser.parse_args(["run"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if args.cmd is None:
        args.cmd = "run"
        args.idea = DEFAULT_IDEA
        args.config = "config.toml"
        args.spec = ""
        args.run_id = None

    init_from_config(args.config)

    if args.cmd == "export-eval":
        print(
            f"[export-eval] wrote {export_eval(args.out, args.limit)} tests to {args.out}"
        )
        return 0

    if args.cmd == "instincts":
        if args.action == "show":
            instincts_show(args.limit)
        elif args.action == "add":
            instincts_add(args.arg)
        elif args.action == "clear":
            n = instinct_clear()
            print(f"[instincts] cleared {n} lessons")
        return 0

    # run
    if args.spec and not os.path.exists(args.spec):
        raise SystemExit(
            f"[config] --spec file not found: {args.spec!r} (cwd: {os.getcwd()})"
        )
    preflight()
    begin_run(args.run_id)

    app = build_app()
    result = app.invoke(
        {
            "idea": args.idea,
            "spec": "",
            "spec_path": args.spec,
            "code": "",
            "code_hash": "",
            "prev_code_hash": "",
            "verdict": {},
            "judge_verdict": {},
            "gates_verdict": {},
            "iteration": 0,
            "human_action": "",
        }
    )
    final = result.get("verdict", {})
    output = {
        "final_verdict": final.get("verdict", "unknown"),
        "iterations": result.get("iteration", 0),
        "judge": result.get("judge_verdict", {}),
        "gates": result.get("gates_verdict", {}),
        "all_issues": final.get("issues", []),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if final.get("verdict") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
