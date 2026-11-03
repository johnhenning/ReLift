"""The standalone viewer carries the authoritative document and source snapshot."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from scripts.build_architecture_view import ROOT, build_page, render_walkthrough


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts, self.resources = {}, []
        self.active = None

    def handle_starttag(self, tag: Any, attrs: Any) -> None:
        attrs = dict(attrs)
        if tag in {"script", "link", "img", "iframe"} and (attrs.get("src") or attrs.get("href")):
            self.resources.append((tag, attrs))
        if tag == "script":
            self.active = attrs.get("id")
            if self.active:
                self.scripts[self.active] = ""

    def handle_data(self, data: Any) -> None:
        if self.active:
            self.scripts[self.active] += data

    def handle_endtag(self, tag: Any) -> None:
        if tag == "script":
            self.active = None


def test_standalone_page_embeds_document_sources_and_all_diagram_assets() -> None:
    page = build_page()
    parsed = Page()
    parsed.feed(page)
    assert page.lower().startswith("<!doctype html>")
    assert '<meta name="viewport"' in page
    assert not parsed.resources, "The standalone viewer must not need remote or adjacent assets"
    assert "connect-src 'none'" in page
    assert "window.openai" not in (ROOT / "docs/architecture-view.js").read_text()
    data = json.loads(parsed.scripts["relift-architecture-data"])
    files = json.loads(parsed.scripts["relift-source-files"])
    for node in data["nodes"]:
        name = node["source"]["file"]
        assert files[name] == (ROOT / name).read_text()
    # Source files contain template markers themselves: interpolation must not
    # rewrite embedded code or recursively embed the generated viewer.
    assert (
        files["scripts/build_architecture_view.py"]
        == Path("scripts/build_architecture_view.py").read_text()
    )
    assert "docs/architecture-view.html" not in files
    markdown = (ROOT / "docs/architecture.md").read_text()
    assert page.count('<pre class="mermaid">') == markdown.count("```mermaid")
    _, headings, _ = render_walkthrough(markdown)
    for heading in headings:
        assert f'id="{heading["id"]}"' in page
        assert f'href="#{heading["id"]}"' in page


def test_local_document_links_open_embedded_source_instead_of_requesting_files() -> None:
    text = "## Cameras\n\n[ray code](../relift/rays.py) and [viewer](architecture-view.html).\n"
    rendered, headings, files = render_walkthrough(text)
    assert 'href="#source-files" data-source-file="relift/rays.py"' in rendered
    assert 'href="#pipeline"' in rendered
    assert files["relift/rays.py"] == (ROOT / "relift/rays.py").read_text()
    assert headings == [dict(title="Cameras", id="walkthrough-cameras")]
