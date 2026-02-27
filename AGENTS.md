# Repository Guidelines

## Project Structure & Module Organization
- `useapi_nodes.py` contains all ComfyUI node classes, request utilities, polling logic, and cache helpers.
- `__init__.py` exports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` for ComfyUI discovery.
- `tests/test_structure.py` runs offline contract checks for node registration, categories, and input/output signatures.
- `tests/test_integration.py` runs live Useapi.net integration tests (skipped automatically when `USEAPI_TOKEN` is not set).
- `docs/ai/` stores internal project/agent documentation; treat it as reference material, not runtime code.
- `.env.example` documents supported environment variables and safe local setup.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install project dependencies (core runtime deps are typically provided by ComfyUI).
- `pip install opencv-python`: optional dependency required by `UseapiLoadVideoFrame`.
- `python -m pytest`: run the full test suite; integration tests auto-skip when no token is available.
- `python -m pytest tests/test_structure.py -v`: run fast offline validation before opening a PR.
- `USEAPI_TOKEN=... python -m pytest tests/test_integration.py -v -s`: run live API checks.
- No packaging/build step is required; place this repo in `ComfyUI/custom_nodes/` and restart ComfyUI.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and concise docstrings for public helpers.
- Use `snake_case` for functions; prefix private helpers with `_` (for example, `_runway_poll`).
- Use `PascalCase` for node classes with the `Useapi` prefix (for example, `UseapiRunwayGenerate`).
- Keep `NODE_CLASS_MAPPINGS` keys aligned with class names and display mappings.
- Use uppercase for module-level constants (`BASE_URL`, `LOG`).

## Testing Guidelines
- Tests use `unittest` style and run through `pytest`.
- Add structural tests for every new node: registration, category, `INPUT_TYPES`, `RETURN_TYPES`, and bound `FUNCTION`.
- For network behavior, cover at least one success path and one explicit error path.
- No numeric coverage gate is enforced today; do not merge behavior changes without tests.

## Commit & Pull Request Guidelines
- Prefer Conventional Commit prefixes used in history: `feat:`, `fix:`, `docs:`.
- Keep commits focused and imperative (one logical change per commit).
- PRs should include: problem summary, key changes, test commands run, and API/token handling notes.
- For UI-visible node changes, include a workflow snippet or screenshot showing expected output.

## Security & Configuration Tips
- Never commit real credentials or `.env`; use `.env.example` as the template.
- Prefer environment variables (`USEAPI_TOKEN`, optional email vars) instead of hardcoded tokens.
