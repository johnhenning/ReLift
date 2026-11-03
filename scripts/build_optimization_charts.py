#!/usr/bin/env python
"""Render recorded experiment summaries; interpretation comes from separate Markdown.

    uv run python scripts/build_optimization_charts.py --summaries SUMMARY.json ... \
        --analysis docs/optimization-experiment-log.md --out-dir outputs/optimization-charts
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

METRICS = (
    ("wall_median", "Fresh process runtime", "seconds", 1),
    ("sampled_peak_process_tree_rss_bytes", "Host RAM · sampled tree RSS", "GiB", 2**30),
    ("sampled_peak_gpu_device_used_bytes", "GPU memory · sampled whole device", "GiB", 2**30),
    ("process_tree_write_bytes", "Physical disk writes", "GiB", 2**30),
    ("sampled_output_peak_allocated_bytes", "Output allocation · sampled peak", "GiB", 2**30),
    ("process_rss_highwater_bytes", "Main process RAM · OS RSS high water", "GiB", 2**30),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sections, graphs, records = [], [], []
    for source in args.summaries:
        rows = json.loads(source.read_text())
        records.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=rows))
        groups = defaultdict(list)
        for row in rows:
            if not row.get("diagnostic") and row.get("measurement_role") != "validation":
                groups[row["experiment"]].append(row)
        for name, group in groups.items():
            title = f"{name} · {source.parent.name}"
            if any(row.get("measurement_role") == "screening" for row in group):
                title += " · exploratory screening"
            identity = hashlib.sha256(title.encode()).hexdigest()[:10]
            counts = Counter(row["variant"] for row in group)
            labels = [f"{row['variant']}" + (f" · {row['source_fingerprint'][:6]}" if counts[row["variant"]] > 1 else "") + (f" · gate {row['quality']}" if row.get("quality") else "") + f" ({row['n']}/{row['attempts']})" for row in group]
            colors = ["#b94949" if row.get("quality") == "fail" else "#356b9b" for row in group]
            graph = make_subplots(rows=3, cols=2, subplot_titles=[item[1] for item in METRICS], horizontal_spacing=.2)
            figure, axes = plt.subplots(3, 2, figsize=(13, 12), layout="constrained")
            figure.suptitle(title, fontsize=16)
            for index, (key, heading, unit, scale) in enumerate(METRICS):
                values = [row[key] / scale if row.get(key) is not None else None for row in group]
                plus = [(row["wall_max"] - row["wall_median"]) if row.get("wall_median") is not None else 0 for row in group] if key == "wall_median" else None
                minus = [(row["wall_median"] - row["wall_min"]) if row.get("wall_median") is not None else 0 for row in group] if key == "wall_median" else None
                graph.add_trace(go.Bar(x=values, y=labels, orientation="h", showlegend=False, marker_color=colors,
                    error_x=dict(type="data", symmetric=False, array=plus, arrayminus=minus) if plus is not None else None,
                    hovertemplate=f"%{{y}}<br>%{{x:.3f}} {unit}<extra></extra>"), row=index // 2 + 1, col=index % 2 + 1)
                graph.update_xaxes(title_text=unit, rangemode="tozero", row=index // 2 + 1, col=index % 2 + 1)
                axis = axes.flat[index]
                axis.set_title(heading)
                axis.set_xlabel(unit)
                axis.set_yticks(range(len(labels)), labels)
                for j, value in enumerate(values):
                    if value is None:
                        axis.text(.02, j, "not recorded", transform=axis.get_yaxis_transform())
                    else:
                        axis.barh(j, value, color=colors[j])
                        axis.text(value + (plus[j] if plus is not None else 0), j, f" {value:.3g}", va="center", fontsize=10)
                        if plus is not None:
                            axis.errorbar(value, j, xerr=[[minus[j]], [plus[j]]], color="#172b42", capsize=4)
                axis.invert_yaxis()
                axis.set_xlim(0, max((value for value in values if value is not None), default=1) * 1.3 or 1)
                axis.grid(axis="x", alpha=.2)
            figure.supxlabel("Medians; runtime whiskers show observed min–max. Labels show completed/attempted trials.\nTree/device/output peaks are sampled; main-process RSS high water is OS-reported. Output allocation includes fixtures.", fontsize=10)
            figure.savefig(args.out_dir / f"{identity}.png", dpi=150)
            plt.close(figure)
            graph.update_layout(height=950, template="plotly_white", title=title, margin=dict(l=180, r=40, b=60))
            graph.update_yaxes(autorange="reversed")
            graphs.append(dict(id=f"chart-{identity}", figure=graph.to_plotly_json()))
            stages = "\n".join(f"<tr><td>{html.escape(label)}</td><td>{html.escape(json.dumps(row['phase_medians'], sort_keys=True))}</td></tr>" for row, label in zip(group, labels))
            sections.append(f'<section><h2>{html.escape(title)}</h2><div id="chart-{identity}"></div><details><summary>Stage median seconds</summary><table>{stages}</table></details></section>')
    analysis = ""
    if args.analysis:
        from markdown_it import MarkdownIt
        analysis = MarkdownIt("commonmark", {"html": False}).render(args.analysis.read_text())
    from plotly.utils import PlotlyJSONEncoder
    payload = json.dumps(graphs, cls=PlotlyJSONEncoder).replace("<", "\\u003c")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Optimization experiments</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem;color:#182c40}section{margin:3rem 0}td{padding:.5rem;overflow-wrap:anywhere}table{table-layout:fixed;width:100%}details{margin:1rem 0}</style><h1>Recorded optimization experiments</h1><p>Each panel compares one experiment in its recorded environment. Process runtime and stage runtime measure different scopes. Runtime whiskers are observed ranges, not confidence intervals. Resource peaks are sampled; device totals include desktop use. Output allocation includes prepared fixtures and compiler caches; it is separate from cumulative physical disk I/O. Diagnostics are excluded.</p>'
    document += '<p>Main-process RSS high water is OS-reported, separately from sampled tree RSS. Red bars identify failed numerical gates.</p>'
    document += "".join(sections) + '<article>' + analysis + '</article><script>' + get_plotlyjs() + '</script><script type="application/json" id="charts">' + payload + '</script><script>for(const item of JSON.parse(document.getElementById("charts").textContent)){Plotly.newPlot(item.id,item.figure.data,item.figure.layout,{responsive:true,displaylogo:false});}</script></html>'
    (args.out_dir / "index.html").write_text(document)
    (args.out_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()
