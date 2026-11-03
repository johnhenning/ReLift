# External API boundaries

These stubs describe the small pinned upstream interfaces imported after ReLift
selects a provider's source tree at runtime. Pyrefly checks the full first-party
library, scripts, tests and frontend build hook. It does not type-check downloaded
provider source, generated viewers or frozen benchmark snapshots.

`nets.alltracker`, `tapvid3d_utils`, `track_eval_util` and `track_eval_dense` describe
the imported tracker/evaluation interfaces. The `xformers` namespace represents
the optional import probed by the providers; declaring it does not install or
enable an xformers backend. Update these interfaces when their pinned upstream
versions change.

PyYAML uses the `types-PyYAML` development dependency. Runtime dependencies with
their own typing information use that information directly.
