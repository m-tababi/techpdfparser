# Project Rules

## Core Principles

- **Simplicity First:** Make every change as simple as possible. Impact minimal code.
- **No Laziness:** Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact:** Changes should only touch what's necessary. Avoid introducing bugs.

---

## Code Style

- **Small functions:** Every function has exactly one job. More than ~20–30 lines → split it up.
- **Descriptive names:** Function and variable names explain themselves. No comment needed for `calculateTotalPrice()`.
- **No deep nesting:** Maximum 2–3 levels. If more: extract a function or use early return.
- **No magic numbers:** Named constants instead of raw values (`MAX_RETRIES = 3`, not just `3`).
- **No dead code:** Don't comment out and leave behind. Either delete it or add a TODO with a reason.

---

## Comments

- Comments explain the **why**, not the **what**. The code itself explains the what.
- Complex logic or non-obvious decisions get a short comment above them.
- Public functions/classes: brief docstring describing what it does and what it returns.

```python
# BAD
x = x + 1  # increment x by 1

# GOOD
# Offset by 1 because API is 0-indexed but UI is 1-indexed
display_index = raw_index + 1
```

---

## Modularity

- **Swappable components:** Hide external tools (APIs, databases, services) behind an interface/adapter.
  - Example: No direct `openai.chat()` calls scattered everywhere → use an `llm_client.complete()` interface instead.
- **No circular dependencies:** Modules only depend in one direction.
- **Clear boundaries:** Every module/folder has a single, well-defined responsibility. No god-modules.

---

## Project Structure

```
src/
  core/          # Business logic, no external dependencies
  adapters/      # External tools & services (swappable)
  utils/         # Small, reusable helper functions
tasks/
  todo.md        # Current plan with checkable items
  lessons.md     # Lessons learned from corrections
tests/
```

---

## Workflow Orchestration

### 1. Plan Node Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy

- Use subagents liberally to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop

- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until the mistake rate drops
- Review lessons at the start of each session for the relevant project

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First:** Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan:** Check in before starting implementation
3. **Track Progress:** Mark items complete as you go
4. **Explain Changes:** High-level summary at each step
5. **Document Results:** Add review section to `tasks/todo.md`
6. **Capture Lessons:** Update `tasks/lessons.md` after any correction

---

## What to Avoid

- **Overengineering:** Don't build abstractions that aren't needed yet (YAGNI).
- **Copy-paste code:** If something appears twice → extract a shared function.
- **Global state:** Pass data explicitly, don't share via global variables.
- **Bloated files:** IMPORTANT: No function does more than one thing. IMPORTANT: No file grows beyond ~200–300 lines — split it up.
