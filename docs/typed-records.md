# Typed dictionary interfaces

Fixed-field dictionaries have named `TypedDict` schemas in `relift/records/`.
They describe the dictionaries already passed between components and written to
JSON. They do not add wrapper objects or change the stored representation.

| Module | Contracts |
|---|---|
| `comparison.py` | Numerical errors, array comparisons, full trajectory reports, shape failures and campaign parity |
| `configuration.py` | Serialized component settings and optimization overrides |
| `runtime.py` | Stage metrics, cache and commit counters, mapped archives and trajectory manifests |
| `model.py` | Checkpoints, training batches, provider geometry and evaluation results |
| `benchmark.py` | Experiment definitions, worker results, telemetry, trials and summaries |
| `artifacts.py` | Viewer data, architecture nodes, native builds and profiling outputs |
| `reporting.py` | Gallery state and normalized report/chart records |

For example, `ErrorStats.report()` returns `ErrorReport`. Pyrefly now checks the
presence and spelling of its fields and distinguishes integer counts, boolean
flags and nullable error measurements. `compare()` returns either a full
`TrajectoryComparison` or `ShapeMismatch`. Consumers use
`is_trajectory_comparison()` before reading the full result's error groups.
A shape mismatch remains a failed comparison rather than producing a missing-key
exception in campaign validation.

Annotate both ends of an interface. Returning a typed record helps little if the
consumer immediately widens it to `Any`. Use `record.copy()` when retaining the
same schema; `dict(record)` erases its field-level types. Prefer `Mapping` for
read-only collections where mutation is unnecessary.

Use `NotRequired[T]` when a key may be absent and `T | None` when the key is
present but its value may be null. Legacy partial configuration and gallery
records deliberately allow missing fields. Use discriminated unions for records
whose available fields depend on a status, such as prepared-array comparisons.

An open collection is still a dictionary. Scene names, array filenames and
backend identifiers are dynamic keys, so those collections use concrete value
types such as `dict[str, FileReference]`. User-supplied manifest metadata,
checkpoint training arguments and raw historical report inputs remain extensible.
They are not forced into an inaccurate fixed schema.

`TypedDict` provides static checking, not runtime validation. JSON and checkpoint
loads remain explicit dynamic boundaries. Keep casts local to those boundaries;
do not cast arbitrary dictionaries throughout the program to suppress errors.
Existing configuration validation and storage checksum checks still enforce their
runtime invariants.

Experiment workers choose a frozen source tree before importing `relift`.
Their record imports therefore live under `TYPE_CHECKING`, with postponed
annotations. Runtime dictionary construction stays ordinary Python so the current
worker can also execute older snapshots that predate the record package.

The record-contract tests include invalid snippets that must fail Pyrefly for
misspelled keys, missing required fields and incorrect value types. A fixture
captured from the preceding stack revision checks that numerical report values
serialize identically, including nonfinite inputs. To capture that fixture from
a supplied immutable source snapshot into a fresh file:

```sh
uv run python scripts/capture_record_wire_fixture.py \
  --source outputs/experiments/typed-records-before-source \
  --out outputs/maintenance/error-report-wire.json
```

Run the standard development checks in [development.md](development.md).
Ruff now includes the first-party `relift/data/` package while excluding downloaded
root-level `data/`. Working-tree source snapshots make the same distinction.
