# techpdfparser

A technical PDF parsing library.

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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run Tests

```bash
pytest
```
