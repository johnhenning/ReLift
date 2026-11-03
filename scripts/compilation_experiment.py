"""Separate first-use compilation cost from repeated full-clip correction passes."""
import torch


def run(args, options, phases, model, info):
    from relift.storage import file_sha256
    from relift.streaming import stream_correct
    from torch._dynamo.utils import counters

    enabled = options.pop("compile")
    passes = options.pop("passes", 4)
    if options or not isinstance(enabled, bool) or passes < 2:
        raise ValueError("Expected compile flag and at least one cold and one warm pass")
    counters.clear()
    if enabled:
        with phases.stage("compile_wrapper"):
            model = torch.compile(model, dynamic=False)
    metadata = dict(checkpoint_sha256=file_sha256(args.root / "checkpoints/released_checkpoint.pt"),
                    checkpoint_step=info["step"], compiled=enabled)
    runs = []
    for index in range(passes):
        name = "correction_cold" if index == 0 else f"correction_warm_{index}"
        destination = args.output / f"pass-{index:02}"
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with phases.stage(name):
            store = stream_correct(args.output / "window", destination, model, metadata, group=1024, batch=8, seed=0)
        runs.append(dict(phase=name, output=str(destination.resolve()), shape=store.shape, finite=bool(store.finite())))
    return dict(task="compilation", phases=phases.results, runs=runs, compiled=enabled,
                compiler_counters={str(key): {str(name): value for name, value in counter.items()} for key, counter in counters.items()},
                source=str(args.source.resolve()),
                scope="Fresh process and compiler cache; first pass includes lazy compilation and mapped preparation. Warm passes reuse model and prepared mappings but commit fresh trajectory stores.")
