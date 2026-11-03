#!/usr/bin/env python
"""Build the inspectable architecture diagram from current source code.

    uv run python scripts/build_architecture_view.py
    uv run python scripts/build_architecture_view.py --out PATH/architecture.html

Outputs one offline HTML document with the diagram, Markdown walkthrough and
embedded source files. Rebuild after editing the implementation or documentation.
"""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]


def excerpt(filename, symbol, focus=None, limit=28):
    path = ROOT / filename
    text = path.read_text()
    parts = symbol.split(".")
    owner = ast.parse(text)
    for part in parts:
        owner = next(node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part)
    start, stop = owner.lineno, owner.end_lineno
    lines = text.splitlines()
    if focus:
        match = next(i for i in range(start - 1, stop) if focus in lines[i])
        start = max(start, match + 1 - 3)
    stop = min(stop, start + limit - 1)
    return dict(file=filename, symbol=symbol, start=start, stop=stop,
                code="\n".join(f"{i + 1:>3}  {lines[i]}" for i in range(start - 1, stop)),
                sha256=hashlib.sha256(text.encode()).hexdigest())


def node(key, title, purpose, inputs, outputs, filename, symbol, focus=None):
    return dict(id=key, title=title, purpose=purpose, inputs=inputs, outputs=outputs,
                source=excerpt(filename, symbol, focus))


def graph_data():
    nodes = [
        node("decode", "Read video", "Decode the source clip while preserving its temporal frame order.", "Video or image directory", "RGB frames · T×H×W×3", "scripts/demo_video.py", "read_frames"),
        node("geometry", "DA3 geometry", "Estimate visible depth and camera motion; define the grid used by the tracker.", "RGB clip", "Depth, intrinsics, poses, processed RGB", "relift/providers/da3.py", "DepthAnything3Provider.run"),
        node("tracking", "AllTracker", "Follow the identity of every first-frame grid pixel through the clip.", "Processed RGB clip", "2D tracks · T×H×W×2; visibility", "relift/providers/alltracker.py", "AllTrackerProvider.run"),
        node("lift", "Lift into 3D", "Combine tracked pixels, depth and cameras in the first camera's coordinate system.", "2D tracks, depth, intrinsics, poses", "Camera centres, unit rays, ranges", "relift/rays.py", "build_rays"),
        node("visibility", "Observed mask", "Use visibility and valid range to identify supporting observations.", "Visibility and range", "Observed mask; hidden observations withheld", "relift/infer.py", "predict_dense", "obs ="),
        node("interpolation", "Fill missing motion", "Interpolate between observed positions and hold the nearest position at open ends.", "Lifted observations and observed mask", "Initial trajectory and gap lengths", "relift/features.py", "interp_init"),
        node("features", "14 geometry features", "Represent the initial path, observed lift, reliability, rays and camera motion relative to an anchor.", "Initial path, observations, mask, rays, cameras", "Features · B×T×P×14; anchor · B×P×3", "relift/features.py", "features_clean"),
        node("grouping", "Group reference points", "Use the CUDA seed to form scene-wide groups of 1,024 tracks; process eight groups per baseline batch.", "Valid reference pixel IDs", "Seeded track groups with full timelines", "relift/infer.py", "predict_dense", "perm ="),
        node("corrector", "ReLift corrector", "Alternate temporal and cross-track attention to predict a learned 3D residual.", "14 geometric channels plus anchor", "Residual for each track and frame", "relift/model.py", "Corrector.forward"),
        node("arrays", "Save dense trajectories", "Restore clip scale and save every reference point's trajectory and provenance.", "Corrected paths, raw lift, masks and cameras", "Dense arrays · T×H×W×3", "scripts/demo_video.py", "main", "np.savez"),
        node("plotly", "Interactive 3D views", "Display selected reference tracks while retaining every temporal frame.", "Saved dense trajectories", "Offline Plotly HTML and export manifest", "scripts/plotly_demo.py", "export_viewer"),
        node("pack", "Pack 4 frames", "Embed four frames' 14-channel features into one 384-wide token.", "B×T×P×14 features", "B×P×(T/4)×384 tokens", "relift/model.py", "Corrector.forward", "x = feats"),
        node("time", "Time attention", "Each point attends across the clip's temporal chunks.", "Tokens for one track over time", "Motion context along that trajectory", "relift/model.py", "Block.forward", "self.time"),
        node("space", "Space attention", "Tracks in one group exchange information within the same temporal chunk.", "Tokens across tracks at one temporal chunk", "Cross-track motion and geometry context", "relift/model.py", "Block.forward", "y = self.n2"),
        node("residual", "Add the repair", "Add the predicted residual to the interpolated path in normalized coordinates.", "Initial path and learned residual", "Repaired trajectory before clip scale is restored", "relift/features.py", "correct"),
    ]
    offload = [
        ("mapped-store", "SSD trajectory store", "Store full-timeline trajectory shards and read only requested reference pixels.", "Completed point groups", "Mapped shards and verified manifest", "relift/storage.py", "TrajectoryStore.write_shard"),
        ("streamed-correction", "Streamed correction", "Construct CUDA rays and correct one full-timeline point batch at a time.", "Mapped geometry and 2D tracking", "Corrected shards without a complete CUDA output tensor", "relift/streaming.py", "stream_correct"),
        ("da3-pager", "DA3 RAM or disk paging", "Move inactive DA3 blocks and retained features off CUDA while preserving temporal context.", "Pinned DA3 model and full clip", "Active GPU blocks with cold state in RAM or disk", "relift/da3_offload.py", "DA3Pager.page_block"),
    ]
    for spec in offload:
        if (ROOT / spec[-2]).exists():
            nodes.append(node(*spec))
    return dict(nodes=nodes, edges=[
        ["decode", "geometry"], ["geometry", "tracking"], ["geometry", "lift"], ["tracking", "lift"],
        ["tracking", "visibility"], ["lift", "visibility"], ["lift", "interpolation"], ["visibility", "interpolation"],
        ["interpolation", "features"], ["visibility", "features"], ["features", "grouping"],
        ["grouping", "corrector"], ["corrector", "arrays"], ["arrays", "plotly"],
    ])


def render_walkthrough(markdown, source=ROOT / "docs/architecture.md"):
    """Render the authoritative document and route repository links to embedded files."""
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = renderer.parse(markdown)
    files, headings, used_ids = {}, [], set()

    def add_file(path):
        relative = str(path.relative_to(ROOT))
        files[relative] = path.read_text()
        return relative

    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            title = tokens[index + 1].content
            slug = "walkthrough-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            original, serial = slug, 2
            while slug in used_ids:
                slug, serial = f"{original}-{serial}", serial + 1
            used_ids.add(slug)
            token.attrSet("id", slug)
            if token.tag == "h2":
                headings.append(dict(title=title, id=slug))
        for child in token.children or []:
            if child.type != "link_open":
                continue
            href = child.attrGet("href")
            url = urlsplit(href)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (source.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT):
                continue
            if target == ROOT / "docs/architecture-view.html":
                child.attrSet("href", "#pipeline")
                continue
            if target.is_dir():
                choices = sorted(target.glob("*.py"))
                for choice in choices:
                    add_file(choice)
                target = choices[0] if choices else target
            if target.is_file():
                filename = add_file(target)
                child.attrSet("href", "#source-files")
                child.attrSet("data-source-file", filename)
    ordinary_fence = renderer.renderer.rules["fence"]

    def fence(items, index, options, env):
        token = items[index]
        if token.info.strip() == "mermaid":
            return '<pre class="mermaid">' + html.escape(token.content) + '</pre>\n'
        return ordinary_fence(items, index, options, env)

    renderer.renderer.rules["fence"] = fence
    return renderer.renderer.render(tokens, renderer.options, {}), headings, files


def safe_json(value):
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_page():
    data = graph_data()
    template = (ROOT / "docs/architecture-view.template.html").read_text()
    fragment = template.replace("__ARCHITECTURE_CSS__", (ROOT / "docs/architecture-view.css").read_text())
    fragment = fragment.replace("__ARCHITECTURE_DATA__", safe_json(data))
    fragment = fragment.replace("__ARCHITECTURE_JS__", (ROOT / "docs/architecture-view.js").read_text())
    walkthrough, headings, files = render_walkthrough((ROOT / "docs/architecture.md").read_text())
    for node_data in data["nodes"]:
        filename = node_data["source"]["file"]
        files[filename] = (ROOT / filename).read_text()
    vendor = ROOT / "docs/vendor/mermaid"
    manifest = json.loads((vendor / "source.json").read_text())
    for filename, record in manifest["files"].items():
        if hashlib.sha256((vendor / filename).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Vendored architecture asset checksum mismatch: {filename}")
    bundle = re.sub(r"</script", lambda match: "<\\/script", (vendor / "mermaid.min.js").read_text(), flags=re.I)
    # Disable source-map lookups; all executable assets are embedded in the viewer.
    bundle = re.sub(r"//# sourceMappingURL=.*", "", bundle)
    page = (ROOT / "docs/architecture-standalone.template.html").read_text()
    replacements = {
        "__STANDALONE_CSS__": (ROOT / "docs/architecture-standalone.css").read_text(),
        "__COMPONENT_EXPLORER__": fragment,
        "__WALKTHROUGH__": walkthrough,
        "__CONTENTS__": "\n".join(f'<a href="#{h["id"]}">{html.escape(h["title"])}</a>' for h in headings),
        "__SOURCE_FILES__": safe_json(files),
        "__SOURCE_OPTIONS__": "\n".join(f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>' for name in sorted(files)),
        "__MERMAID_BUNDLE__": bundle,
        "__MERMAID_LICENSE__": html.escape((vendor / "LICENSE").read_text()),
        "__STANDALONE_JS__": (ROOT / "docs/architecture-standalone.js").read_text(),
    }
    # Replace only tokens in the original template. Embedded source files can
    # themselves contain these marker strings and must remain byte-for-byte intact.
    return re.sub(r"__[A-Z_]+__", lambda match: replacements[match.group(0)], page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/architecture-view.html")
    args = ap.parse_args()
    page = build_page()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Built {args.out} ({len(page.encode()):,} bytes)")


if __name__ == "__main__":
    main()
