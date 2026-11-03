// Tests the component-to-source interaction without a browser or visual renderer.
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function harness(initialState) {
  const page = fs.readFileSync(path.join(__dirname, "architecture-view.html"), "utf8");
  const data = JSON.parse(page.match(/<script id="relift-architecture-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const saved = [], parts = {}, events = {};
  for (const name of ["title", "purpose", "inputs", "outputs", "location", "code", "file"]) parts[name] = {textContent: "", dataset: {}};
  const buttons = data.nodes.map(node => ({dataset: {node: node.id}, attributes: {}, events: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    addEventListener(key, value) { this.events[key] = value; }}));
  const box = {left: 0, top: 0, right: 736, bottom: 430, width: 736, height: 430};
  const diagram = {getBoundingClientRect: () => box};
  const svg = {setAttribute() {}, innerHTML: ""};
  const root = {querySelectorAll: () => buttons, querySelector(selector) {
    if (selector === "#relift-architecture-data") return {textContent: JSON.stringify(data)};
    if (selector === ".arch-flow") return diagram;
    if (selector === ".arch-connectors") return svg;
    if (selector.startsWith("[data-position=")) return {getBoundingClientRect: () => ({...box, width: 300, height: 40, right: 300, bottom: 40})};
    if (selector.startsWith("#relift-selected-")) return parts[selector.slice("#relift-selected-".length)];
    throw new Error(`Unknown selector ${selector}`);
  }};
  const context = {document: {getElementById: () => root}, ResizeObserver: class {observe() {}},
    localStorage: {getItem() { return initialState; }, setItem(key, value) { saved.push({key, value}); }},
    window: {addEventListener(name, callback) { events[name] = callback; }}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "architecture-view.js"), "utf8"), context);
  return {data, saved, parts, buttons, svg, events};
}

test("every component selects its current code excerpt and remembers the selection", () => {
  const {data, saved, parts, buttons, svg} = harness();
  assert.equal(parts.title.textContent, "Lift into 3D");
  for (const node of data.nodes) {
    const button = buttons.find(button => button.dataset.node === node.id);
    button.events.click();
    assert.equal(parts.code.textContent, node.source.code);
    assert.match(parts.location.textContent, new RegExp(node.source.file.replaceAll(".", "\\.")));
    assert.equal(button.attributes["aria-controls"], "relift-code-panel");
    assert.equal(buttons.filter(button => button.attributes["aria-pressed"] === "true").length, 1);
    assert.equal(saved.at(-1).value, node.id);
    assert.equal(parts.file.dataset.sourceFile, node.source.file);
    assert.equal(parts.file.dataset.sourceLine, String(node.source.start));
  }
  assert.ok(svg.innerHTML.includes("data-active="));
});

test("restored selections work and unknown saved components fall back safely", () => {
  const {parts, saved, events} = harness("space");
  assert.equal(parts.title.textContent, "Space attention");
  events.storage({key: "relift-architecture:selected", newValue: "time"});
  assert.equal(parts.title.textContent, "Time attention");
  events.storage({key: "relift-architecture:selected", newValue: "retired-component"});
  assert.equal(parts.title.textContent, "Lift into 3D");
  assert.equal(saved.length, 0, "restoring saved state must not write it back");
});

test("all displayed excerpts and line numbers match the current source files", () => {
  const {data} = harness();
  for (const {source} of data.nodes) {
    const text = fs.readFileSync(path.join(__dirname, "..", source.file), "utf8");
    assert.equal(crypto.createHash("sha256").update(text).digest("hex"), source.sha256,
      `${source.file} changed; rebuild the architecture view`);
    const lines = text.split(/\r?\n/);
    const expected = lines.slice(source.start - 1, source.stop)
      .map((line, index) => `${String(source.start + index).padStart(3)}  ${line}`).join("\n");
    assert.equal(source.code, expected, `${source.file}:${source.start}–${source.stop}`);
  }
});

test("standalone source links open complete files and highlight the referenced line", () => {
  const files = {"model.py": "one\ntwo\nthree", "rays.py": "<script>literal source</script>\nray"};
  const events = {}, picker = {value: "model.py", addEventListener(name, callback) {this[name] = callback;}};
  const location = {textContent: ""};
  const output = {children: [], offsetTop: 0, parentElement: {scrollTop: 0},
    replaceChildren() {this.children = [];}, append(span) {this.children.push(span);},
    querySelector() {return this.children.find(span => span.className.includes("is-reference"));}};
  const document = {documentElement: {},
    getElementById(id) {return ({"relift-source-files": {textContent: JSON.stringify(files)},
      "relift-file-select": picker, "relift-file-code": output, "relift-file-location": location})[id];},
    createElement() {return {offsetTop: 100};},
    addEventListener(name, callback) {events[name] = callback;}, querySelectorAll() {return [];}};
  const context = {document, console, getComputedStyle() {return {getPropertyValue() {return "";}};},
    mermaid: {initialize() {}, async run() {}}, matchMedia() {return {addEventListener() {}};}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "architecture-standalone.js"), "utf8"), context);
  assert.equal(output.children.length, 3);
  events.click({target: {closest() {return {dataset: {sourceFile: "rays.py", sourceLine: "2"}};}}});
  assert.equal(picker.value, "rays.py");
  assert.equal(location.textContent, "rays.py · line 2");
  assert.ok(output.children[0].textContent.includes("<script>literal source</script>"));
  assert.equal(output.children[1].className, "source-line is-reference");
  picker.value = "model.py";
  picker.change();
  assert.equal(output.children.length, 3);
  assert.ok(output.children.every(span => span.className === "source-line"));
});
