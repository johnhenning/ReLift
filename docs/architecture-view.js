(function () {
  "use strict";
  const root = document.getElementById("relift-architecture");
  const data = JSON.parse(root.querySelector("#relift-architecture-data").textContent);
  const nodes = new Map(data.nodes.map(node => [node.id, node]));
  const flow = root.querySelector(".arch-flow");
  const svg = root.querySelector(".arch-connectors");
  let selected = "lift";
  const internals = new Set(["pack", "time", "space", "residual"]);
  function draw() {
    const bounds = flow.getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
    const active = internals.has(selected) ? "corrector" : selected;
    const pieces = ['<defs><marker id="relift-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><polygon points="0,0 8,4 0,8"></polygon></marker></defs>'];
    for (const [source, target] of data.edges) {
      const a = root.querySelector(`[data-position="${source}"]`).getBoundingClientRect();
      const b = root.querySelector(`[data-position="${target}"]`).getBoundingClientRect();
      let x1 = a.left + a.width / 2 - bounds.left;
      let y1 = a.bottom - bounds.top;
      let x2 = b.left + b.width / 2 - bounds.left;
      let y2 = b.top - bounds.top;
      let route;
      if (Math.abs(a.top - b.top) < 2) {
        x1 = a.right - bounds.left; y1 = a.top + a.height / 2 - bounds.top;
        x2 = b.left - bounds.left; y2 = b.top + b.height / 2 - bounds.top;
        route = `M ${x1} ${y1} L ${x2 - 3} ${y2}`;
      } else {
        const middle = (y1 + y2) / 2;
        route = `M ${x1} ${y1} C ${x1} ${middle}, ${x2} ${middle}, ${x2} ${y2 - 3}`;
      }
      pieces.push(`<path d="${route}" marker-end="url(#relift-arrow)" data-active="${source === active || target === active}"></path>`);
    }
    svg.innerHTML = pieces.join("");
  }
  function select(id, save = false) {
    if (!nodes.has(id)) id = "lift";
    selected = id;
    const node = nodes.get(id);
    root.querySelectorAll("[data-node]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.node === id));
    });
    root.querySelector("#relift-selected-title").textContent = node.title;
    root.querySelector("#relift-selected-purpose").textContent = node.purpose;
    root.querySelector("#relift-selected-inputs").textContent = node.inputs;
    root.querySelector("#relift-selected-outputs").textContent = node.outputs;
    root.querySelector("#relift-selected-location").textContent = `${node.source.file}:${node.source.start}–${node.source.stop} · ${node.source.symbol}`;
    root.querySelector("#relift-selected-code").textContent = node.source.code;
    const sourceLink = root.querySelector("#relift-selected-file");
    sourceLink.dataset.sourceFile = node.source.file;
    sourceLink.dataset.sourceLine = String(node.source.start);
    draw();
    if (save) { try { localStorage.setItem("relift-architecture:selected", id); } catch (_) {} }
  }
  root.querySelectorAll("[data-node]").forEach(button => {
    if (!nodes.has(button.dataset.node)) { button.hidden = true; return; }
    button.setAttribute("aria-controls", "relift-code-panel");
    button.addEventListener("click", () => select(button.dataset.node, true));
  });
  let saved = "lift";
  try { saved = localStorage.getItem("relift-architecture:selected") || saved; } catch (_) {}
  select(saved);
  window.addEventListener("storage", event => {
    if (event.key === "relift-architecture:selected") select(event.newValue);
  });
  new ResizeObserver(draw).observe(flow);
})();
