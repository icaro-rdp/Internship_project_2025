# AGENTS.md

start each response with "using custom instructions:"

## Exploring the repository

When using an AI agent to explore the codebase, start with high-level questions about the project structure, main modules, and functionality.For instance, if a utility function is needed, first check if a similar function already exists in the codebase in `main/Utils` to avoid duplication. If you need to understand how a specific module works, look for its docstrings and any related documentation in the repository.

## Setup & commands

This project uses **uv** for both dependency management and running code. Run
_everything_ through `uv` so it executes inside the project environment; never
call `pip`, `python`, or `pytest` directly.

| Task                 | Command                     |
| -------------------- | --------------------------- |
| Install / sync deps  | `uv sync`                   |
| Add a dependency     | `uv add <package>`          |
| Add a dev dependency | `uv add --dev <package>`    |
| Remove a dependency  | `uv remove <package>`       |
| Run a script         | `uv run <script.py>`        |
| Run a module         | `uv run python -m <module>` |
| Run tests            | `uv run pytest`             |
| Lint                 | `uv run ruff check`         |
| Format               | `uv run ruff format`        |

- Do not edit dependency entries in `pyproject.toml` by hand — use `uv add` /
  `uv remove` so the lockfile stays consistent.
- If you must invoke the interpreter outside `uv` for a one-off, use `python3`,
  never `python`.

## Language & version

- Python 3.12. Use modern syntax freely (`match`, `X | Y` unions, `tomllib`).

## Formatting (delegated to tooling — do not hand-format)

- `ruff format` is the source of truth for layout: 88-char lines, blank-line
  spacing, quote style. Don't manually reflow to other limits.
- `ruff check` enforces import order (stdlib → third-party → local) and lint
  rules. Use **absolute imports**, never relative.
- Write code that already passes `uv run ruff check`; don't lean on a later pass.

## Simplicity & file hygiene

- After creating a new file, review it once and remove redundant, unused, or
  decorative code, comments, imports, helpers, and configuration.
- Keep every file as easy to read as possible for both humans and machines:
  prefer direct structure, clear names, and only the abstractions needed by the
  current behavior.
- Do not keep placeholder code, speculative extension points, or unused
  examples unless they are required by the task or an established project
  pattern.

## Naming

Standard casing is assumed (`snake_case` functions/vars, `PascalCase`
classes/exceptions, `UPPER_SNAKE_CASE` constants, `_leading_underscore` for
non-public). Beyond that:

- Choose intention-revealing names: `circle_radius`, not `r`; `retry_count`,
  not `n`.
- Single letters only for well-known math/coordinate contexts and loop
  throwaways. Never use `l`, `O`, or `I`.

## Type hints

- Fully annotate every signature, including `-> None` returns.
- Always parametrize generics: `dict[str, int]`, not bare `dict`.
- Prefer `X | None` over `Optional[X]`.
- Treat `Any` as a code smell; if unavoidable, add a comment explaining why.

```python
def process_items(items: list[str], limit: int | None = None) -> dict[str, int]:
    ...
```

## Docstrings (Google style)

Every public module, class, function, and method needs a docstring. Signatures
are typed, so **do not repeat types** in the docstring — describe meaning.

```python
def fetch_user(user_id: int) -> dict[str, str]:
    """Fetch user details from the database.

    Args:
        user_id: Unique identifier of the user. Must be non-negative.

    Returns:
        Mapping of attribute names to values (e.g. name, email).

    Raises:
        ValueError: If user_id is negative.
        ConnectionError: If the database is unreachable.
    """
```

## Pythonic patterns

- **Truthiness**: `if items:`, not `if len(items) > 0:`.
- **Resources**: manage files, locks, and sockets with `with`.
- **Iteration**: `enumerate()` for indices, `zip()` for parallel iteration.
- **Comprehensions**: prefer them for simple transforms; use explicit loops
  when logic gets complex (readability wins).
- **Strings**: f-strings everywhere **except** logging calls (see below).
- **Mutable defaults**: never use mutable default arguments — use `None` as the
  sentinel and assign inside the body.

## Error handling

- Catch specific exceptions; never `except Exception:` or bare `except:`.
- Prefer EAFP (`try`/`except`) over LBYL guard checks.
- Define domain errors as subclasses of one project base exception
  (e.g. `class MyProjectError(Exception)`) so callers can catch the family.

## Logging

- Use the `logging` module, never `print()`, for diagnostics.
- Use lazy `%`-style args so messages format only when emitted:

```python
logger.info("Fetched %d users for tenant %s", count, tenant_id)  # correct
logger.info(f"Fetched {count} users for tenant {tenant_id}")      # avoid
```
