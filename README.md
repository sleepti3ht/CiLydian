<div align="center">

# CiLydian

[![python](https://img.shields.io/badge/Python-3.12%2B-fdf6e3?style=flat&logo=python&logoColor=657b83)](https://python.org)
[![langgraph](https://img.shields.io/badge/LangGraph-1.0%2B-b58900?style=flat)](https://langchain-ai.github.io/langgraph/)
[![status](https://img.shields.io/badge/status-active-859900?style=flat)](https://github.com)
[![license](https://img.shields.io/badge/license-MIT-fdf6e3?style=flat)](LICENSE)

</div>

> Self-hosted agentic CI pipeline on LangGraph: **spec → coder → LLM judge (advisory) → deterministic gates (ruff + pytest)**, with a bounded repair loop and human escalation. Models are pluggable per role — local Ollama or any OpenAI-compatible endpoint.

---

## Why CiLydian?

Most multi-agent coding tools stop at "generate and hope." CiLydian treats LLM output as untrusted input: mechanical gates (ruff, pytest) are the source of truth, the LLM judge is advisory feedback, and the loop is bounded. If the pipeline can't converge in `MAX_ITER`, it exits with a status code and the full log — never hangs, never burns tokens indefinitely.

Compared to AutoGen Studio or CrewAI: those are chat-orchestration frameworks. CiLydian is a **CI primitive** — it runs, exits with a status code, and can be wired into git hooks, GitHub Actions, or cron.

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sleepti3ht/cilydian.git
cd cilydian

# Unix / macOS
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Run with local Ollama (default)
python pipeline.py "Your task description"

# Or use cloud profile (free tiers)
cp config.cloud.toml.example config.toml
cp .env.example .env  # fill with OpenRouter/Groq keys
python pipeline.py "Your task description"
```

Stack: `langgraph`, `langchain-ollama`, `langchain-openai`, `ruff`, `pytest`. See `requirements.txt`.

---

## Features

### Core
- **Spec-driven generation**: turn one-line ideas into a `SPEC.md` and a `test_candidate.py` skeleton before any code is written
- **Bounded repair loop**: coder iterates on judge feedback up to `MAX_ITER` times, then escalates to a human
- **Advisory LLM judge**: semantic review that never blocks on its own verdict (see [Design notes](#design-notes))
- **Deterministic gates**: `ruff` (auto-fix + check) and `pytest`; these are what actually block the pipeline
- **Short-circuit**: pytest is skipped if ruff fails, saves 10–20s per failed cycle
- **Config-driven models**: mix Ollama local and OpenAI-compatible cloud per role
- **Rate-limit resilience**: exponential backoff for 429s on free tiers
- **UTF-8 on Windows**: subprocess encoding prevents `cp1251` mojibake in logs

### Reliability
- **Defensive JSON parsing**: judge verdicts normalized via `try: int(float(raw))`
- **Programmatic guardrail**: `score < 4` with `verdict="pass"` is auto-downgraded to `fail` (see [Design notes](#design-notes))
- **Markdown fence stripping**: handles LLM code wrapped in \`\`\` fences
- **Log rotation**: 5 MB cap with atomic `os.replace()`; single writer, no race
- **Isolated gates**: `logs/ruff.toml` and `logs/pytest.ini` are auto-created on first run to prevent parent-config bleed
- **Preflight checks**: verifies `ruff` / `pytest` are on PATH and the coder endpoint is reachable before spending a single token

---

## Design notes

**The judge is advisory, the guardrails are not.** Local 7B–14B judges drift beyond SPEC — they request absent features, over- or under-rate code, and sometimes contradict themselves. So the LLM's verdict alone never blocks the pipeline. A separate programmatic rule handles one specific pathological case: if the LLM outputs `verdict="pass"` while scoring below 4, the guardrail downgrades it to `fail`. That's not the judge blocking — it's a deterministic check that doesn't trust a self-contradicting model. Mechanical gates (ruff, pytest) remain the ultimate source of truth.

**The repair loop is bounded on purpose.** `MAX_ITER` (default 3) caps attempts. Exhaustion means human escalation, not a hang. On complex tasks this may give up earlier than a human reviewer would prefer — acceptable for CI, where the cost of a false-negative is a re-run, not a lost afternoon.

**Mechanical fixes belong to the tool.** `ruff --fix` runs before `ruff check` and auto-corrects imports and formatting. LLM tokens are too expensive for style fixes a Rust binary does in 40 milliseconds.

---

## Architecture

```mermaid
flowchart LR
    classDef solarizedBase3 fill:#fdf6e3,stroke:#93a1a1,color:#657b83,stroke-width:1px
    classDef solarizedBase02 fill:#eee8d5,stroke:#93a1a1,color:#586e75,stroke-width:1px
    classDef solarizedYellow fill:#b58900,stroke:#b58900,color:#fdf6e3,stroke-width:2px
    classDef solarizedOrange fill:#cb4b16,stroke:#cb4b16,color:#fdf6e3,stroke-width:2px
    classDef solarizedGreen fill:#859900,stroke:#859900,color:#fdf6e3,stroke-width:2px
    classDef solarizedBlue fill:#268bd2,stroke:#268bd2,color:#fdf6e3,stroke-width:2px
    classDef solarizedCyan fill:#2aa198,stroke:#2aa198,color:#fdf6e3,stroke-width:2px

    A[idea]:::solarizedYellow
    S[spec_writer<br>role: spec]:::solarizedBlue
    C[coder<br>role: coder]:::solarizedGreen
    J[judge<br>advisory]:::solarizedOrange
    T[tester<br>ruff + pytest]:::solarizedCyan
    E[final verdict<br>exit code]:::solarizedBase02

    A --> S
    S --> C
    C --> J
    J -->|advisory| T
    T -->|fail and iter < max_iter| C
    T -->|pass or budget exhausted| E
```

---

## Configuration

### Local Ollama profile (default)

```toml
# config.toml
[pipeline]
max_iter = 3
judge_blocking = false

[roles.spec]
provider = "ollama"
model = "qwen2.5-coder:7b"

[roles.coder]
provider = "ollama"
model = "qwen2.5-coder:7b"

[roles.judge]
provider = "ollama"
model = "qwen2.5:14b"
```

### Cloud hybrid profile (free tiers)

```toml
# config.cloud.toml.example
[pipeline]
max_iter = 3
judge_blocking = false

[roles.spec]
provider = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-chat"
api_key_env = "OPENROUTER_API_KEY"

[roles.coder]
provider = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-chat"
api_key_env = "OPENROUTER_API_KEY"

[roles.judge]
provider = "openai_compat"
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.3-70b-versatile"
api_key_env = "GROQ_API_KEY"
```

```bash
# .env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

> ⚠️ Never commit `.env` — keep secrets private. Config references env var names, not values.

---

## Performance

| Metric | Local (7B coder) | Cloud (frontier) |
|---|---|---|
| Coder throughput | ~15 tok/s | ~50–100 tok/s |
| Judge throughput | ~3–6 tok/s (partial CPU offload) | ~30–60 tok/s |
| Full cycle wall time | 4–15 min | 1–3 min |
| Exit code | 0 = gates passed, 1 = failed | 0 = gates passed, 1 = failed |

**Typical run:**

```log
[preflight] ruff: OK
[preflight] pytest: OK
[preflight] coder endpoint: reachable
[spec] Generated SPEC.md + test_candidate.py (1.2s)
[coder] Iteration 1/3 — wrote candidate.py (3.4s)
[judge] score=3, verdict=fail (advisory)
[tester] ruff rc=0, pytest rc=1 (fail)
[coder] Iteration 2/3 — fixed issues (2.8s)
[judge] score=4, verdict=pass (advisory)
[tester] ruff rc=0, pytest rc=0 (pass)
✓ Pipeline completed: gates passed (exit code 0)
```

---

## Honest Limitations

- **Local models (7b-14b) fail convergence** on mid-complexity tasks (e.g., sliding-window rate limiter with exact boolean return) within 3 iterations.
- **Cloud free-tier models (Groq gpt-oss-120b)** provide better judge quality but have rate limits (200K tokens/day).
- **CiLydian is an experiment and portfolio artifact**, not a daily development tool. For production code, use manual multi-vendor workflow (DeepSeek/Qwen for prompting, Claude for architecture, Gemini for review, local ruff/pytest for gates).

---

## Screenshot

![CiLydian Terminal](screenshots/terminal.png)

---

## Roadmap

- [ ] **GitHub Actions integration** — trigger CiLydian on issue labels
- [ ] **Multi-language support** — extend beyond Python (JavaScript, Go)
- [ ] **Persistent state** — resume interrupted runs from checkpoint
- [ ] **Web dashboard** — visualize pipeline runs and logs
- [ ] **Plugin system** — custom gates beyond ruff/pytest (mypy, bandit)

---

## Contributing

Contributions are welcome! Please open an issue or submit a PR.

### Development setup

```bash
git clone https://github.com/sleepti3ht/cilydian.git
cd cilydian

# Unix / macOS
python -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install pre-commit
pre-commit install  # enforces ruff + pytest on commit

# Run the pipeline on a sample task
python pipeline.py "Test task"
```

### Code style

- Commit messages: Conventional Commits (`feat(pipeline): ...`, `fix(tester): ...`)
- Run `ruff check` and `pytest` before pushing (enforced by pre-commit)

---

## License

MIT
