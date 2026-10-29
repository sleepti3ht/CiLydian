"""CiLydian: self-hosted agentic CI pipeline.

Flow: spec -> coder -> LLM judge (advisory by default) -> deterministic gates
(ruff + pytest), with a bounded repair loop and human escalation.

Models are pluggable per role via config.toml: local Ollama or any
OpenAI-compatible endpoint (OpenRouter, Groq, Gemini compat, DeepSeek, vLLM).
API keys are never stored in config: roles reference environment variable names.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tomllib
from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

LOG_DIR = "logs"
LOG_CAP = 5_000_000  # rotate a log once it exceeds this many bytes
DEFAULT_IDEA = "Rate limiter on asyncio.Semaphore for CVE scanner"
_FENCE_BLOCK = re.compile(r"```[\w-]*[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL)

# Populated by init_models(); graph nodes read these globals at call time.
SPEC_LLM = None
CODER = None
JUDGE = None
MAX_ITER = 3
JUDGE_BLOCKING = False


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


def build_llm(cfg: dict):
    """Instantiate a chat model for one pipeline role.

    provider = "ollama"        -> local Ollama server
    provider = "openai_compat" -> any OpenAI-compatible API (OpenRouter, Groq,
                                  Gemini compat, DeepSeek, vLLM, ...)
    """
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        return ChatOllama(
            model=cfg["model"],
            base_url=cfg.get("base_url", "http://localhost:11434"),
            num_ctx=cfg.get("num_ctx", 4096),
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
            temperature=cfg.get("temperature", 0.2),
            max_retries=3,  # exponential backoff on 429 (free-tier rate limits)
        )
    raise SystemExit(f"[config] unknown provider: {provider!r}")


def load_config(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[config] {path} not found; see config.toml in repo root")


def init_models(cfg: dict) -> None:
    global SPEC_LLM, CODER, JUDGE
    roles = cfg.get("roles", {})
    for role in ("spec", "coder", "judge"):
        if role not in roles:
            raise SystemExit(f"[config] missing [roles.{role}] section")
    SPEC_LLM = build_llm(roles["spec"])
    CODER = build_llm(roles["coder"])
    JUDGE = build_llm(roles["judge"])


def strip_fences(text: str) -> str:
    """Extract the first fenced code block; models append prose after it."""
    m = _FENCE_BLOCK.search(text)
    if m:
        return m.group(1)
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if "```" in t:               # closing fence with prose after it
            t = t.split("```", 1)[0]
        return t.rstrip()
    return t


def log(agent: str, msg: str) -> None:
    path = os.path.join(LOG_DIR, f"{agent}.log")
    if os.path.exists(path) and os.path.getsize(path) > LOG_CAP:
        os.replace(path, path + ".old")  # atomic on Windows; single writer, no race
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def reset_logs() -> None:
    """Truncate per run: live-tail panels stay alive, history = one cycle."""
    os.makedirs(LOG_DIR, exist_ok=True)
    for agent in ("spec", "coder", "judge", "tester"):
        open(os.path.join(LOG_DIR, f"{agent}.log"), "w", encoding="utf-8").close()


def ensure_isolation() -> None:
    """Stop ruff/pytest from inheriting the parent project's configs."""
    os.makedirs(LOG_DIR, exist_ok=True)  # ← Добавленная строка
    if not os.path.exists(os.path.join(LOG_DIR, "ruff.toml")):
        open(os.path.join(LOG_DIR, "ruff.toml"), "w", encoding="utf-8").close()
    if not os.path.exists(os.path.join(LOG_DIR, "pytest.ini")):
        with open(os.path.join(LOG_DIR, "pytest.ini"), "w", encoding="utf-8") as f:
            f.write("[pytest]\nasyncio_mode = auto\n")


def run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run a gate tool; force UTF-8 decoding (Windows locales default to cp1251)."""
    return subprocess.run(
        cmd, check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", cwd=cwd,
    )


def preflight() -> None:
    for tool in ("ruff", "pytest"):
        if shutil.which(tool) is None:
            raise SystemExit(f"[preflight] {tool} not found on PATH (see requirements.txt)")
    try:
        CODER.invoke("ping")
    except Exception as exc:
        raise SystemExit(f"[preflight] coder endpoint unreachable: {exc}") from exc


class State(TypedDict):
    idea: str
    spec: str
    code: str
    verdict: dict
    iteration: int


def spec_writer(s: State):
    """Stage 0: turn a one-line idea into a SPEC plus a pytest skeleton."""
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
            prompt += f"TESTS that must pass (match names/signatures exactly):\n{f.read()}\n"
    except OSError:
        pass
    prompt += (
        f"Issues to fix: {feedback}\n"
        "Write only the Python implementation for module `candidate`. "
        "Raw code only, no markdown fences, no tests. "
        "NO explanations, NO headers, NO prose after the code."
    )
    code = strip_fences(CODER.invoke(prompt).content)
    log("coder", code)
    return {"code": code, "iteration": s["iteration"] + 1}


def judge(s: State):
    """LLM review with defensive parsing: models ignore JSON schemas regularly."""
    raw = JUDGE.invoke(
        "Rate 1-5: correctness, security, simplicity. Judge ONLY against "
        "the SPEC's measurable acceptance criteria; do not request features "
        f"beyond the SPEC. Check against SPEC:\n{s['spec']}\nCODE:\n{s['code']}\n"
        'Return JSON only: {"score": int, "verdict": "pass"|"fail", "issues": [str]}\n\n'
        'CRITICAL: Output ONLY raw JSON. NO markdown fences.\n'
        'CORRECT: {"score": 4, "verdict": "pass", "issues": []}\n'
        'WRONG: ```json{"score": 4, "verdict": "pass", "issues": []}```'
    ).content
    a, b = raw.find("{"), raw.rfind("}")
    try:
        verdict = json.loads(raw[a:b + 1]) if a != -1 else {}
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
    return {"verdict": verdict}


def tester(s: State):
    """Deterministic gates: ruff (auto-fix then check) and pytest. These block."""
    candidate = os.path.join(LOG_DIR, "candidate.py")
    with open(candidate, "w", encoding="utf-8") as f:
        f.write(s["code"])
    with open(os.path.join(LOG_DIR, "conftest.py"), "w", encoding="utf-8") as f:
        f.write("import sys, os\nsys.path.insert(0, os.path.dirname(__file__))\n")
    run(["ruff", "check", "--fix", "candidate.py"], LOG_DIR)
    with open(candidate, encoding="utf-8") as f:
        fixed = f.read()  # State must match the artifact on disk after --fix
    ruff = run(["ruff", "check", "candidate.py"], LOG_DIR)
    if ruff.returncode != 0:
        # Short-circuit: no point running pytest on code that fails lint/syntax.
        log("tester", f"ruff rc={ruff.returncode}\n{ruff.stdout[:500]}")
        return {"code": fixed, "verdict": {
            **s["verdict"], "gate": "ruff+pytest", "verdict": "fail",
            "issues": s["verdict"].get("issues", []) + [f"ruff:{ruff.stdout[:300]}"],
        }}
    pytest_run = run(["pytest", "test_candidate.py", "-q", "--tb=short"], LOG_DIR)
    ok = pytest_run.returncode == 0
    log("tester", f"ruff rc=0\npytest rc={pytest_run.returncode}\n{pytest_run.stdout[-500:]}")
    extra_issues = [] if ok else [f"pytest:{pytest_run.stdout[:300]}"]
    return {"code": fixed, "verdict": {
        **s["verdict"], "gate": "ruff+pytest",
        "verdict": "pass" if ok else "fail",
        "issues": s["verdict"].get("issues", []) + extra_issues,
    }}


def route(s: State) -> Literal["coder", "tester"]:
    """Judge is advisory by default; flip JUDGE_BLOCKING to make it a gate."""
    if JUDGE_BLOCKING and s["verdict"]["verdict"] == "fail" and s["iteration"] < MAX_ITER:
        return "coder"
    return "tester"


def route_tester(s: State) -> Literal["coder", "end"]:
    """Deterministic failures loop back to coder, bounded by MAX_ITER."""
    if s["verdict"]["verdict"] == "fail" and s["iteration"] < MAX_ITER:
        return "coder"
    return "end"


g = StateGraph(State)
g.add_node("spec", spec_writer)
g.add_node("coder", coder)
g.add_node("judge", judge)
g.add_node("tester", tester)
g.set_entry_point("spec")
g.add_edge("spec", "coder")
g.add_edge("coder", "judge")
g.add_conditional_edges("judge", route, {"coder": "coder", "tester": "tester"})
g.add_conditional_edges("tester", route_tester, {"coder": "coder", "end": END})
app = g.compile()


def main() -> int:
    global MAX_ITER, JUDGE_BLOCKING
    parser = argparse.ArgumentParser(
        description="Agentic CI pipeline: spec -> coder -> judge -> deterministic gates"
    )
    parser.add_argument(
        "idea", 
        nargs="?", 
        default=DEFAULT_IDEA,
        help="One-line task description (e.g., 'Async rate limiter with token bucket')"
    )
    parser.add_argument("--config", default="config.toml",
                        help="path to role/model config (default: config.toml)")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    init_models(cfg)
    pipe = cfg.get("pipeline", {})
    MAX_ITER = int(pipe.get("max_iter", 3))
    JUDGE_BLOCKING = bool(pipe.get("judge_blocking", False))

    preflight()
    ensure_isolation()
    reset_logs()
    result = app.invoke(
        {"idea": args.idea, "spec": "", "code": "", "verdict": {}, "iteration": 0}
    )
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2))
    return 0 if result["verdict"].get("verdict") == "pass" else 1  # CI-friendly exit code


if __name__ == "__main__":
    raise SystemExit(main())