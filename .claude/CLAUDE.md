# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

The project is structured in modular, decoupled blocks. The first block focuses on processing technical PDFs and transforming text, tables, diagrams, formulas, images, and technical drawings into a unified structured output. This output will serve as the foundation for later stages. In the long term, the system should verify technical statements, trace them back to their supporting evidence, and support the approval of safety-critical containers. See @README.md for stack, install, and usage.

## Core Behavior

@docs/principles.md

## Source-of-Truth Rules

- Code is authoritative. If this file and the code disagree, trust the code and propose an edit to CLAUDE.md.
- Never duplicate code state here (adapter lists, default values, pipeline contents, file layouts). Always point to the authoritative location.
- Before answering questions about defaults, architecture shape, or pipeline contents: read the relevant file. Do not recite from memory.

## Where to Look

Pointers, not duplicated content:

- **Extraction config + defaults:** `extraction/config.py`
- **Extraction pipeline:** `extraction/pipeline.py`
- **Registry mechanics:** `extraction/registry.py`
- **Interfaces (Protocols):** `extraction/interfaces.py`
- **Output models:** `extraction/models.py`
- **Quality gate scope (mypy, ruff):** @pyproject.toml

## Commands

```bash
# Extract a PDF
python -m extraction extract path/to/document.pdf --config config.yaml --output outputs/

# Tests / lint / types
pytest -q
pytest extraction/tests/test_models.py::test_name   # single test
ruff check extraction
mypy
```

## Workflow Rules

- **Plan before non-trivial implementation.** For anything beyond a one-file change, state a brief plan with verify-steps first (see Goal-Driven Execution in @docs/principles.md). Confirm, then implement.
Always check `git status` before making changes and make sure the current state is understood.
Before risky or larger edits, ensure there is a committed baseline to return to if needed.
Commit and push every stable, meaningful progress point to GitHub.
Do not mention Claude, AI, or assistant tools in commit messages or code comments.## Architecture Invariants

## Compaction

When compacting this conversation, preserve: list of modified files, pending verifications, active config or adapter changes, and any open schema mismatches.

## Self-Update Rule

If you notice this file describes something that no longer matches the code — an adapter that doesn't exist, a pipeline that was renamed, an invariant that was deliberately relaxed — stop, propose a concrete edit to CLAUDE.md, and ask me before continuing with the original task. Treat this file as code: keep it correct or fix it.
