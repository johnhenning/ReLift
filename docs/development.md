# Python development

The project runs on Python 3.13 through `uv`. Install development tools with
`uv sync --extra dev`. Ruff and Pyrefly are pinned in the lockfile.

```bash
uv run ruff format .
uv run ruff check .
uv run pyrefly check --min-severity warn
uv run pytest tests --ignore=tests/gpu -q
```

The GitHub workflow runs a dedicated Python-quality job for every push and pull
request, with separate Ruff-formatting, Ruff-lint and Pyrefly-type steps. Unit tests
run independently in the CPU job. Both jobs use the shared setup action to install
the same locked development dependencies with CPU PyTorch wheels.

Ruff requires function annotations, checks imports and common correctness issues,
and formats the full first-party source tree. Pyrefly covers `relift/`, `scripts/`,
`tests/`, the frontend packaging adapter and `stubs/`. It reports warnings as
failures too.

Use concrete types and protocols for stable interfaces. Dynamic checkpoint,
provider and JSON boundaries may use `Any`. Optional state needs an explicit
check before use. The small external stubs in [stubs/README.md](../stubs/README.md)
describe pinned dynamic provider imports; they do not replace checking our code.

Downloaded dependencies, generated artifacts, weights and frozen benchmark sources
are excluded. Historical experiment snapshots must retain their original bytes.
Run the CUDA tests that cover any changed inference or native path on a CUDA host.
For an explicit native build, use `scripts/run_source_tests.py --source .
--native-build PATH --target tests/gpu/FILE.py` so imports target the chosen source.
