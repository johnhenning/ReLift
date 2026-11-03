/* No server or network requests: all sampled trajectories and Plotly are embedded. */
(async function () {
  "use strict";
  const data = JSON.parse(document.getElementById("trajectory-data").textContent);
  const gd = document.getElementById("trajectory-plot");
  const slider = document.getElementById("frame-slider");
  const label = document.getElementById("frame-label");
  const play = document.getElementById("play");
  function decode(text, Type) {
    const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0));
    return new Type(bytes.buffer);
  }
  const raw = decode(data.raw, Float32Array);
  const fixed = decode(data.corrected, Float32Array);
  const observed = decode(data.observed, Uint8Array);
  let selected = data.dynamic[0], current = 0, target = 0, revision = 0, rendered = -1;
  let busy = false, playing = false, animation = 0, startTime = 0, startFrame = 0;
  slider.max = data.frames - 1;
  // A uniform subset of pixels is displayed, with every frame of every sampled path.
  const grid = data.grid ? ` · ${data.grid[1]}×${data.grid[0]} processing grid` : "";
  document.getElementById("extent").textContent = `${data.frames} source frames · ${(data.frames / data.fps).toFixed(2)} s at ${data.fps} fps · ${data.points.toLocaleString()} displayed / ${data.dense_points.toLocaleString()} dense trajectories${grid}`;
  function coords(array, frame, indices = null) {
    const x = [], y = [], z = [];
    for (const p of indices || Array.from({length: data.points}, (_, i) => i)) {
      const base = (frame * data.points + p) * 3;
      x.push(array[base]); y.push(array[base + 1]); z.push(array[base + 2]);
    }
    return {x, y, z};
  }
  function paths(array, indices) {
    const x = [], y = [], z = [];
    for (const p of indices) {
      for (let t = 0; t < data.frames; t++) {
        const base = (t * data.points + p) * 3;
        x.push(array[base]); y.push(array[base + 1]); z.push(array[base + 2]);
      }
      x.push(null); y.push(null); z.push(null);
    }
    return {x, y, z};
  }
  function report(error) {
    console.error(error);
    document.getElementById("error").textContent = `Could not update the viewer: ${error.message}`;
    stop();
  }
  async function showFrame(frame) {
    target = Math.min(data.frames - 1, Math.max(0, frame));
    revision++;
    if (busy) return;
    busy = true;
    try {
      while (rendered !== revision) {
        const version = revision, t = target;
        const series = [coords(raw, t), coords(fixed, t), coords(raw, t, [selected]), coords(fixed, t, [selected])];
        await Plotly.restyle(gd, {x: series.map(s => s.x), y: series.map(s => s.y), z: series.map(s => s.z)}, [0, 1, 6, 7]);
        current = t; slider.value = t;
        label.textContent = `${t} / ${data.frames - 1} · ${(t / data.fps).toFixed(2)} s`;
        rendered = version;
      }
    } catch (error) { report(error); }
    finally { busy = false; }
  }
  async function selectPoint(point) {
    selected = point;
    const series = [paths(raw, [selected]), paths(fixed, [selected])];
    await Plotly.restyle(gd, {x: series.map(s => s.x), y: series.map(s => s.y), z: series.map(s => s.z)}, [4, 5]);
    let seen = 0;
    for (let t = 0; t < data.frames; t++) seen += observed[t * data.points + selected];
    const [u, v] = data.pixels[selected];
    document.getElementById("selected").textContent = `Selected reference pixel (${u}, ${v}) · observed in ${seen}/${data.frames} frames · orange path spans the entire video`;
    await showFrame(current);
  }
  function stop() {
    playing = false; play.textContent = "Play";
    cancelAnimationFrame(animation);
  }
  function tick(now) {
    if (!playing) return;
    const frame = startFrame + Math.floor((now - startTime) * data.fps / 1000);
    if (frame !== target) showFrame(Math.min(frame, data.frames - 1));
    if (frame >= data.frames - 1) stop();
    else animation = requestAnimationFrame(tick);
  }
  try {
    if (!window.reliftPlotReady) {
      await new Promise(resolve => window.addEventListener("relift-plot-ready", resolve, {once: true}));
    }
    if (!gd.data) throw new Error("Plotly did not initialize");
    const representative = [paths(raw, data.dynamic), paths(fixed, data.dynamic)];
    await Plotly.restyle(gd, {x: representative.map(s => s.x), y: representative.map(s => s.y), z: representative.map(s => s.z)}, [2, 3]);
    await selectPoint(selected);
    gd.on("plotly_click", event => {
      const point = event.points[0];
      if (point && (point.curveNumber === 0 || point.curveNumber === 1)) selectPoint(point.pointNumber).catch(report);
    });
    let syncing = false;
    gd.on("plotly_relayout", event => {
      if (syncing) return;
      const changes = {};
      for (const [key, value] of Object.entries(event)) {
        if (key.startsWith("scene.camera")) changes[key.replace("scene.camera", "scene2.camera")] = value;
        if (key.startsWith("scene2.camera")) changes[key.replace("scene2.camera", "scene.camera")] = value;
      }
      if (!Object.keys(changes).length) return;
      syncing = true;
      Plotly.relayout(gd, changes).catch(report).finally(() => { syncing = false; });
    });
    slider.addEventListener("input", () => { stop(); showFrame(Number(slider.value)); });
    play.addEventListener("click", () => {
      if (playing) { stop(); return; }
      startFrame = current >= data.frames - 1 ? 0 : current;
      startTime = performance.now(); playing = true; play.textContent = "Pause";
      animation = requestAnimationFrame(tick);
    });
    document.getElementById("paths").addEventListener("change", event => {
      Plotly.restyle(gd, {visible: event.target.checked}, [2, 3]).catch(report);
    });
    function resize() {
      const small = window.innerWidth < 720;
      Plotly.relayout(gd, {
        height: small ? 900 : 620,
        "scene.domain": small ? {x: [0, 1], y: [.55, 1]} : {x: [0, .49], y: [0, 1]},
        "scene2.domain": small ? {x: [0, 1], y: [0, .45]} : {x: [.51, 1], y: [0, 1]},
        "annotations[0].x": small ? .5 : .245, "annotations[0].y": 1.04,
        "annotations[1].x": small ? .5 : .755, "annotations[1].y": small ? .49 : 1.04,
      }).catch(report);
    }
    window.addEventListener("resize", resize);
    resize();
    document.documentElement.dataset.viewerReady = "true";
  } catch (error) { report(error); }
})();
