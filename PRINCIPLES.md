# Global Rules for Claude Code

These rules apply to every project. Each project's `CLAUDE.md`
overrides anything here when there's a conflict.

## Source of Truth

- Code is authoritative. If documentation contradicts the code,
  trust the code and propose an edit to the documentation.
- Before answering questions about defaults, architecture, or
  implementation details: read the relevant file. Do not recite
  from memory.
- Treat `CLAUDE.md` files as code. If a rule no longer matches
  reality, stop, propose a concrete edit, and confirm before
  continuing the original task.

## Plan Before You Build

- For non-trivial work (≥3 steps, architectural decisions,
  changes across multiple files), enter plan mode first. State
  the plan with explicit verification steps, confirm, then
  implement.
- Translate vague tasks into verifiable goals before starting:
  "Add validation" → "Write tests for invalid inputs, then make
  them pass". "Fix the bug" → "Write a test that reproduces it,
  then make it pass". Strong success criteria let you loop
  independently.
- If something goes sideways mid-implementation, stop and
  re-plan. Don't keep pushing through a broken approach.
- Skip planning for one-line fixes and obvious changes. Don't
  over-engineer trivial work.

## Git Discipline

- Run `git status` before making changes. Understand the
  current state before modifying it.
- Before risky or larger edits, ensure a committed baseline
  exists to return to.
- Make atomic commits: one logical change per commit. Don't
  bundle unrelated changes.
- Never mention Claude, AI, or assistant tools in commit
  messages, code comments, or documentation. Write as the
  developer.
- Never push to a remote unless explicitly asked.

## Verification Before Done

- Never mark a task complete without proving it works.
- Run the relevant tests, check the logs, demonstrate the
  behavior change concretely.
- Before declaring success, ask: "Would a senior engineer
  approve this in review?" If unsure, iterate.

## Root Causes Over Quick Fixes

- When debugging, find the actual root cause. Symptomatic
  patches mask problems and create technical debt.
- If a fix feels hacky, stop and ask: "knowing what I now know,
  what's the elegant solution?"
- Given a bug report, fix it. Don't ask for hand-holding —
  point at logs, errors, and failing tests, then resolve them.

## Scope Discipline

- Changes touch only what's necessary. Don't refactor adjacent
  code "while you're there" without asking.
- Prefer minimal, surgical edits over rewrites. Match existing
  style even if you'd do it differently.
- If you discover unrelated issues, note them. Don't fix them
  in the same change.
- Don't add what wasn't asked for: no speculative features,
  no abstractions for single-use code, no configurability that
  wasn't requested, no error handling for impossible scenarios.
- Every changed line should trace directly to the user's request.
  Remove imports and variables that your own changes orphaned;
  don't remove pre-existing dead code unless asked.

## Subagent Strategy

- Use subagents liberally to keep the main context window
  clean. Offload research, exploration, and parallel analysis.
- One focused task per subagent. Don't bundle multiple goals
  into a single subagent invocation — they get worse, not
  better, when overloaded.
- For complex problems, throw more compute at it via parallel
  subagents rather than sequencing everything in the main loop.
- Skip subagents for trivial work (single-file edits, one-shot
  greps, simple lookups). The orchestration overhead isn't
  worth it for tasks the main loop can finish in one tool call.

## Communication

- When uncertain, say so explicitly. Do not fabricate. State
  assumptions before acting on them.
- If multiple valid interpretations of a request exist, present
  them — don't pick silently.
- If a simpler approach exists than what was asked for, say so
  and push back when warranted.
- Don't pad short answers with preamble or recap what you're
  about to do — just do it.
- Surface decisions that affect the user before making them
  irreversible.

## Compaction

When compacting a conversation, preserve: list of modified
files, pending verifications, active config or schema changes,
unresolved errors, and any decisions reached with the user.

---

## Python Project Defaults

Apply only when the project is Python. Project `CLAUDE.md`
overrides this if conventions differ.

**If the project uses uv:**
- Run code with `uv run` (never bare `python` or `pytest`)
- Add dependencies with `uv add`, never edit `pyproject.toml`
  by hand
- Use `uvx` for one-off tools, never `pip install`
- Don't create virtual environments manually — uv handles
  `.venv/` automatically
- Don't create `requirements.txt` — `pyproject.toml` and
  `uv.lock` are the source of truth

**General Python conventions:**
- Use ruff for linting and formatting (not black + isort + flake8)
- Use modern type hints: `list[str]`, `str | None`, `dict[K, V]`
- Use `pathlib.Path`, not `os.path` string manipulation
- Don't add `# type: ignore` without an error code and a reason
