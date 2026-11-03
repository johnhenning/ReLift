# Development checks

Use Python 3.13 and `uv`. Annotate every first-party Python function, including
scripts, tests, fixtures and packaging hooks. Prefer concrete types and protocols;
reserve `Any` for dynamic model/checkpoint, JSON and external library boundaries.
Narrow optional values explicitly and keep dependency stubs in `stubs/` specific
to the pinned upstream API. Do not silence an entire module to pass type checking.

Represent fixed-field dictionaries with named `TypedDict` schemas in
`relift/records/`; use them in producers and consumers. Use concrete value types
for mappings keyed by filenames, scenes or injected backends. Keep arbitrary
metadata and raw historical JSON extensible at their actual boundaries. See
`docs/typed-records.md` for optional fields, narrowing and frozen-source imports.

Before publishing Python changes, run:

```sh
uv run ruff format .
uv run ruff check .
uv run pyrefly check --min-severity warn
uv run pytest tests --ignore=tests/gpu -q
```

Run relevant CUDA tests when changing inference, memory ownership or native
backends. Preserve benchmark snapshots and raw results in `outputs/`; do not
format or rewrite historical sources. Keep report generation based on recorded
technical evidence and place interpretation in separate analysis documents.

The dedicated CI quality job enforces Ruff formatting/lint and Pyrefly types;
the CPU job runs unit and viewer tests separately. Check all first-party code;
vendored dependencies, generated artifacts, weights and historical outputs are
outside this scope.
