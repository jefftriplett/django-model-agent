# Contributing

## Setup

Clone the repository and install dependencies:

```console
git clone https://github.com/jefftriplett/django-model-agent
cd django-model-agent
uv sync --group dev
```

## Running tests

```console
just test
```

Or directly:

```console
uv run pytest -v
```

Run a specific test file:

```console
just test tests/test_model_agent.py
```

Tests use an in-memory SQLite database and pydantic-ai's `TestModel` for
agent integration tests — no external API calls or database setup needed.

## Linting

```console
just lint
```

## Building the docs

```console
just docs         # live preview
just docs-build   # full build with llms.txt generation
```

## Project structure

```
src/django_model_agent/
├── __init__.py      # Public API exports
├── base.py          # ModelAgent, ModelAgentContext, decorators
├── tools.py         # ModelTool, ReadOnlyTool, UpdateTool, DiffAwareUpdateTool
├── memory.py        # AgentMemory model, AgentMemoryMixin
└── examples.py      # PlaceAgent examples

tests/
├── conftest.py      # Fixtures
├── models.py        # Test Django models
├── settings.py      # Test Django settings
├── test_model_agent.py
├── test_original.py
└── test_pydantic_ai.py

docs/                # Zensical documentation source
scripts/
└── gen_llms.py      # Post-build llms.txt generation
```

## Conventions

- All code uses `from __future__ import annotations`.
- Tests use `pytest` with `pytest-django`.
- pydantic-ai integration tests use `TestModel` to avoid real API calls.
- `ModelTool` subclasses use `ClassVar` for class-level attributes.
