# Don't load any .env files by default
set dotenv-load := false

# Default command - shows list of available commands
@_default:
    just --list

# Format the justfile itself using built-in formatter
@fmt:
    just --fmt --unstable

# Run pre-commit hooks on all files
@lint *ARGS:
    uv --quiet tool run prek {{ ARGS }} --all-files

# Update pre-commit hooks to latest versions
@lint-autoupdate:
    uv --quiet tool run prek autoupdate

# Update prek hooks to latest versions
@update:
    uv --quiet tool run prek autoupdate

# Update Python dependencies - regenerates uv.lock from pyproject.toml
@lock *ARGS:
    uv lock {{ ARGS }}

# Upgrade dependencies and lock
@upgrade:
    just lock --upgrade

# Run pytest test suite (accepts optional arguments like path or test name)
@test *ARGS:
    uv run --group dev pytest {{ ARGS }}

# Serve docs locally with live reload
@docs:
    uv run zensical serve

# Build docs and generate llms.txt
@docs-build:
    uv run zensical build --clean --strict
    uv run python scripts/gen_llms.py site
