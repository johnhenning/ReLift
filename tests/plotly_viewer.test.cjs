// Control tests without a browser: the real Plotly API is replaced by a recorder.
// This checks event wiring/data updates, not WebGL rendering or visual appearance.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

async function viewer() {
  const packed = new Float32Array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25]);
  const payload = {frames: 3, points: 2, dense_points: 2, fps: 2, dynamic: [0], pixels: [[0, 0], [1, 0]],
    raw: Buffer.from(packed.buffer).toString("base64"), corrected: Buffer.from(packed.buffer).toString("base64"),
    observed: Buffer.from([1, 1, 0, 1, 1, 1]).toString("base64")};
  const elements = {};
  for (const id of ["trajectory-data", "trajectory-plot", "frame-slider", "frame-label", "play", "extent", "selected", "error", "paths"]) {
    elements[id] = {textContent: "", value: 0, events: {}, data: [],
      addEventListener(name, callback) { this.events[name] = callback; },
      on(name, callback) { this.events[name] = callback; }};
  }
  elements["trajectory-data"].textContent = JSON.stringify(payload);
  const calls = [], errors = [];
  let animation;
  const context = {document: {getElementById: id => elements[id], documentElement: {dataset: {}}},
    window: {reliftPlotReady: true, innerWidth: 1200, addEventListener() {}},
    Plotly: {async restyle(gd, change, indices) { calls.push({type: "restyle", change, indices}); },
      async relayout(gd, change) { calls.push({type: "relayout", change}); }},
    console: {error: error => errors.push(error)}, performance: {now: () => 0},
    requestAnimationFrame: callback => { animation = callback; return 1; }, cancelAnimationFrame() {},
    atob, Uint8Array, Float32Array};
  const script = fs.readFileSync(path.join(__dirname, "../scripts/plotly_viewer.js"), "utf8");
  await vm.runInNewContext(script, context);
  return {elements, calls, errors, tick: time => animation(time)};
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test("scrubbing reaches the last frame and selecting a point uses its complete path", async () => {
  const v = await viewer();
  v.elements["frame-slider"].value = 2;
  v.elements["frame-slider"].events.input();
  await flush();
  const update = v.calls.filter(c => c.type === "restyle" && c.indices[0] === 0).at(-1);
  assert.deepEqual(Array.from(update.change.x[0]), [20, 23]);
  assert.match(v.elements["frame-label"].textContent, /2 \/ 2/);
  v.elements["trajectory-plot"].events.plotly_click({points: [{curveNumber: 1, pointNumber: 1}]});
  await flush();
  const selection = v.calls.filter(c => c.type === "restyle" && c.indices[0] === 4).at(-1);
  assert.deepEqual(Array.from(selection.change.x[1]), [3, 13, 23, null]);
  assert.match(v.elements.selected.textContent, /observed in 3\/3 frames/);
  assert.equal(v.errors.length, 0);
});

test("playback stops at the final source frame and camera changes synchronize", async () => {
  const v = await viewer();
  v.elements.play.events.click();
  assert.equal(v.elements.play.textContent, "Pause");
  v.tick(1000);
  await flush();
  assert.equal(v.elements.play.textContent, "Play");
  assert.match(v.elements["frame-label"].textContent, /2 \/ 2/);
  const camera = {eye: {x: 1, y: 2, z: 3}};
  v.elements["trajectory-plot"].events.plotly_relayout({"scene.camera": camera});
  await flush();
  assert.deepEqual(v.calls.at(-1).change["scene2.camera"], camera);
  assert.equal(v.errors.length, 0);
});
