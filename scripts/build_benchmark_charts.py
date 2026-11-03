#!/usr/bin/env python
"""Plot saved, comparable benchmark trials without rerunning inference.

    uv run python scripts/build_benchmark_charts.py --benchmark-roots ROOT... \
      --analysis docs/performance-opportunities.md --out-dir outputs/benchmark-charts

Writes PNG/SVG figures, an offline Plotly viewer, CSV and provenance JSON.
Measurements come from trial.json. Dense-output disk allocation is inventoried
now, separately from recorded timed I/O; temporary scratch peak is not inferred.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_offload_results import canonical, load_trials, lookup, number, source_version, workload

# Names/units describe the measurement schema, not a particular campaign.
METRICS = (
    ("runtime_s", "Runtime", "seconds", "wall_seconds", 1),
    ("host_rss_gib", "Host RAM · sampled process-tree RSS", "GiB", "telemetry.sampled_peak_process_tree_rss_bytes", 2**30),
    ("gpu_gib", "GPU memory · sampled whole device", "GiB", "telemetry.sampled_peak_gpu_device_used_bytes", 2**30),
    ("disk_writes_gib", "Physical disk writes · timed run", "GiB", "telemetry.process_tree_write_bytes", 2**30),
    ("dense_mib", "Saved dense outputs · allocated on disk", "MiB", "inventory.dense_allocated_bytes", 2**20),
    ("disk_reads_gib", "Physical disk reads · timed run", "GiB", "telemetry.process_tree_read_bytes", 2**30),
)
ORDER = {name: index for index, name in enumerate(("existing", "none", "streamed", "ram", "disk", "auto"))}
NAMES = {"existing": "Original", "none": "Dense", "streamed": "Streamed", "ram": "RAM offload", "disk": "Disk offload", "auto": "Automatic"}


def inventory_output(path):
    """Stat canonical dense output only; never extract archives or count reader caches."""
    root = Path(path)
    manifest = root / "tracks.store/manifest.json"
    shape = None
    if manifest.is_file():
        record = json.loads(manifest.read_text())
        if record.get("state") != "complete":
            return dict(dense_bytes=None, dense_allocated_bytes=None, shape=None)
        files = [p for p in manifest.parent.rglob("*") if p.is_file()]
        shape = record.get("shape")
    elif (root / "tracks.npz").is_file():
        files = [root / "tracks.npz"]
        import numpy as np
        import zipfile
        with zipfile.ZipFile(files[0]) as archive, archive.open("xyz.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, _ = reader(stream)
    else:
        return dict(dense_bytes=None, dense_allocated_bytes=None, shape=None)
    stats = [p.stat() for p in files]
    return dict(dense_bytes=sum(s.st_size for s in stats),
                dense_allocated_bytes=sum(s.st_blocks * 512 for s in stats) if all(hasattr(s, "st_blocks") for s in stats) else None,
                shape=list(shape) if shape else None, files=len(files))


def collect(roots):
    """Split workloads and implementation versions; retain failures as outcomes."""
    records = load_trials(roots)
    if not records:
        raise ValueError("No trial.json records found in the selected roots")
    buckets = defaultdict(lambda: defaultdict(list))
    for record in records:
        data = dict(record["data"])
        if data.get("status") == "complete" and number(data.get("wall_seconds")) and data["wall_seconds"] > 0:
            data["inventory"] = inventory_output(data["output"]) if data.get("output") else {}
        else:
            data["inventory"] = {}
        record = dict(record, data=data)
        # Unknown workload settings must not accidentally match across trials.
        settings = workload(data)
        runtime = data.get("environment")
        environment_lock = lookup(data, "environment.environment_lockfile_sha256") or data.get("lockfile_sha256")
        signature = dict(workload=settings, lockfile=environment_lock, runtime=runtime, wrapper=data.get("wrapper_sha256"))
        if any(value is None for key, value in settings.items() if key != "command_settings"):
            signature["unmatched_record"] = str(record["path"])
        key = canonical(signature)
        variant = canonical(dict(root=str(record["root"]), mode=data.get("mode"), version=source_version(data), settings=lookup(data, "metrics.settings")))
        buckets[key][variant].append(record)
    result = []
    for signature, variants in buckets.items():
        spec = json.loads(signature)
        rows = []
        for members in variants.values():
            first = members[0]["data"]
            valid = [r for r in members if r["data"].get("status") == "complete" and number(r["data"].get("wall_seconds")) and r["data"]["wall_seconds"] > 0]
            stats = {}
            for key, _, _, field, divisor in METRICS:
                values = [lookup(r["data"], field) / divisor for r in valid if number(lookup(r["data"], field))]
                stats[key] = dict(values=values, median=median(values) if values else None,
                                  min=min(values) if values else None, max=max(values) if values else None, n=len(values))
            resolved = sorted({lookup(r["data"], "metrics.settings.offload_resolved") for r in members} - {None})
            rows.append(dict(mode=first.get("mode", "unknown"), root=members[0]["root"].name,
                             source_version=source_version(first), n=len(valid), attempts=len(members),
                             statuses=dict(Counter(r["data"].get("status", "unknown") for r in members)),
                             resolved_modes=resolved, metrics=stats,
                             records=[dict(path=str(r["path"]), sha256=hashlib.sha256(r["path"].read_bytes()).hexdigest(),
                                           inventory=r["data"].get("inventory"),
                                           stages=lookup(r["data"], "metrics.stages"),
                                           telemetry_interval_seconds=r["data"].get("telemetry_interval_seconds")) for r in members]))
        rows.sort(key=lambda r: (ORDER.get(r["mode"], len(ORDER)), r["mode"], r["root"]))
        counts = Counter(row["mode"] for row in rows)
        for row in rows:
            name = NAMES.get(row["mode"], row["mode"])
            version = hashlib.sha256(canonical(row["source_version"]).encode()).hexdigest()[:6]
            row["label"] = name + (f" · {row['root']} · {version}" if counts[row["mode"]] > 1 else "") + f" ({row['n']}/{row['attempts']})"
        scene = spec["workload"].get("scene", "unknown")
        slug = re.sub(r"[^a-zA-Z0-9-]+", "-", str(scene)).strip("-") or "workload"
        result.append(dict(id=slug + "-" + hashlib.sha256(signature.encode()).hexdigest()[:8], settings=spec, rows=rows))
    return sorted(result, key=lambda w: (-len({r["mode"] for r in w["rows"]}), str(w["settings"]["workload"].get("scene"))))


def display(value):
    if value is None:
        return "not recorded"
    return f"{value:.3g}" if abs(value) < .01 and value != 0 else f"{value:.2f}".rstrip("0").rstrip(".")


def title_for(item):
    settings = item["settings"]["workload"]
    shapes = {tuple(record["inventory"]["shape"]) for row in item["rows"] for record in row["records"]
              if record.get("inventory", {}).get("shape")}
    if len(shapes) == 1:
        total, height, width, _ = next(iter(shapes))
        return f"{settings.get('scene', 'unknown')} · {total} frames · {height}×{width} grid"
    return f"{settings.get('scene', 'unknown')} · processing resolution {settings.get('process_res', 'unknown')}"


def save_static(item, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows = item["rows"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")
    for ax, (key, title, unit, _, _) in zip(axes.flat, METRICS):
        for index, row in enumerate(rows):
            stat = row["metrics"][key]
            if stat["median"] is None:
                label = "not recorded" if row["n"] else " / ".join(row["statuses"])
                ax.text(.02, index, label, transform=ax.get_yaxis_transform(), va="center", fontsize=12)
                continue
            value = stat["median"]
            ax.barh(index, value, height=.55, color="#356b9b", alpha=.85)
            ax.errorbar(value, index, xerr=[[value - stat["min"]], [stat["max"] - value]], fmt="none", ecolor="#172b42", capsize=4)
            offsets = np.linspace(-.13, .13, len(stat["values"])) if len(stat["values"]) > 1 else [0]
            ax.scatter(stat["values"], index + np.asarray(offsets), s=14, color="#172b42", zorder=3)
        values = [r["metrics"][key]["max"] for r in rows if r["metrics"][key]["max"] is not None]
        ceiling = max(values, default=1) or 1
        ax.set_xlim(0, ceiling * 1.32)
        for index, row in enumerate(rows):
            value = row["metrics"][key]["median"]
            if value is not None:
                ax.text(row["metrics"][key]["max"] + ceiling * .025, index - .02, display(value), va="center", fontsize=13)
        ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel(unit, fontsize=13)
        ax.set_title(title, fontsize=14, loc="left", pad=14)
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="x", color="#dbe1e8", linewidth=.6)
        ax.set_axisbelow(True)
        for edge in ("top", "right", "left"):
            ax.spines[edge].set_visible(False)
    fig.suptitle(title_for(item), fontsize=17)
    fig.supxlabel("Labels: completed/attempted trials. Bars: medians; dots: trials; whiskers: min–max, not confidence intervals.\nMemory peaks are sampled. Disk traffic is timed I/O; saved dense allocation is a current inventory, not peak scratch.", fontsize=11)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{item['id']}.{extension}", dpi=170)
    plt.close(fig)


def interactive_chart(item, metric, include_library):
    import plotly.graph_objects as go
    import plotly.io as pio
    key, title, unit, _, _ = metric
    rows = item["rows"]
    names = [r["label"] for r in rows]
    stats = [r["metrics"][key] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[s["median"] for s in stats], y=names, orientation="h", marker_color="#356b9b",
        cliponaxis=False,
        error_x=dict(type="data", symmetric=False,
                     array=[s["max"] - s["median"] if s["median"] is not None else None for s in stats],
                     arrayminus=[s["median"] - s["min"] if s["median"] is not None else None for s in stats]),
        hovertemplate="%{y}<br>Median: %{x:.5g} " + unit + "<extra></extra>"))
    for row in rows:
        stat = row["metrics"][key]
        fig.add_trace(go.Scatter(x=stat["values"], y=[row["label"]] * len(stat["values"]), mode="markers",
                                marker=dict(size=6, color="#172b42"), hovertemplate="Trial: %{x:.6g} " + unit + "<extra></extra>"))
        if stat["median"] is None:
            label = "not recorded" if row["n"] else " / ".join(row["statuses"])
            fig.add_annotation(x=0, y=row["label"], text=label, showarrow=False, xanchor="left")
    high = max((s["max"] for s in stats if s["max"] is not None), default=1) or 1
    for row in rows:
        stat = row["metrics"][key]
        if stat["median"] is not None:
            fig.add_annotation(x=stat["max"] + high * .025, y=row["label"], text=display(stat["median"]), showarrow=False, xanchor="left")
    fig.update_layout(template="plotly_white", title=dict(text=title, font_size=16), showlegend=False,
                      height=max(320, 125 + len(rows) * 44), margin=dict(l=140, r=55, t=60, b=55),
                      xaxis=dict(title=unit, range=[0, high * 1.3]),
                      yaxis=dict(categoryorder="array", categoryarray=names, autorange="reversed"), font=dict(size=12))
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_library, config=dict(responsive=True, displaylogo=False))


def render_analysis(path, destination):
    from markdown_it import MarkdownIt
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = renderer.parse(path.read_text())
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href")
                if not urlsplit(href).scheme and not href.startswith("#"):
                    target = (path.parent / href).resolve()
                    child.attrSet("href", quote(os.path.relpath(target, destination.parent), safe="/._-#"))
    return renderer.renderer.render(tokens, renderer.options, {})


def write_outputs(items, roots, folder, analysis=None):
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(inventoried_at=datetime.now(timezone.utc).isoformat(), benchmark_roots=[str(Path(p).resolve()) for p in roots],
                   definitions=dict(host_ram="Sampled process-tree RSS; not CPU utilization or total host memory",
                                    gpu="Sampled whole-device use, including desktop and other processes",
                                    saved_disk="Current canonical dense outputs only; excludes checkpoints, decoded/provider/mapped caches and cleaned scratch",
                                    io="Recorded process-tree physical I/O; OS page cache retained",
                                    scratch_peak="Not recorded; no peak temporary disk-space claim"), workloads=items)
    (folder / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (folder / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["workload", "mode", "root", "completed", "attempted", "statuses", "resolved_modes", *[m[0] for m in METRICS]])
        for item in items:
            for row in item["rows"]:
                writer.writerow([item["id"], row["mode"], row["root"], row["n"], row["attempts"], canonical(row["statuses"]), canonical(row["resolved_modes"]), *[row["metrics"][m[0]]["median"] for m in METRICS]])
    sections, library = [], True
    for item in items:
        save_static(item, folder)
        charts = []
        for metric in METRICS:
            charts.append(interactive_chart(item, metric, library))
            library = False
        sources = "".join(f'<li><a href="{quote(os.path.relpath(r["path"], folder), safe="/._-")}">{html.escape(row["label"])} · {html.escape(Path(r["path"]).parent.name)}</a></li>' for row in item["rows"] for r in row["records"])
        sections.append(f'<section id="{item["id"]}"><h2>{html.escape(title_for(item))}</h2><div class="chart-grid">' + "".join(charts) +
                        f'</div><p><a href="{item["id"]}.png">PNG</a> · <a href="{item["id"]}.svg">SVG</a></p><details><summary>Trial sources and recorded settings</summary><pre>{html.escape(json.dumps(item["settings"], indent=2))}</pre><ul>{sources}</ul></details></section>')
    template = (ROOT / "docs/benchmark-charts.template.html").read_text()
    destination = folder / "index.html"
    replacements = dict(__CHARTS__="\n".join(sections),
                        __NAV__=" · ".join(f'<a href="#{item["id"]}">{html.escape(str(item["settings"]["workload"]["scene"]))}</a>' for item in items),
                        __ANALYSIS__=render_analysis(analysis, destination) if analysis else "")
    destination.write_text(re.sub(r"__[A-Z_]+__", lambda match: replacements[match.group(0)], template))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--scene", help="optional exact scene filter; otherwise render each workload separately")
    parser.add_argument("--analysis", type=Path, help="separate reviewed Markdown to include below charts")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    items = collect(args.benchmark_roots)
    if args.scene:
        items = [item for item in items if item["settings"]["workload"].get("scene") == args.scene]
    if not items:
        parser.error("No selected workloads")
    destination = write_outputs(items, args.benchmark_roots, args.out_dir.resolve(), args.analysis)
    print(destination)
    for item in items:
        print(args.out_dir / f"{item['id']}.png")


if __name__ == "__main__":
    main()
