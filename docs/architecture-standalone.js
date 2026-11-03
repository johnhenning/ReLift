(function () {
  "use strict";
  const files = JSON.parse(document.getElementById("relift-source-files").textContent);
  const picker = document.getElementById("relift-file-select");
  const output = document.getElementById("relift-file-code");
  const location = document.getElementById("relift-file-location");
  function showFile(filename, line = 0) {
    if (!Object.hasOwn(files, filename)) return;
    picker.value = filename;
    location.textContent = filename + (line ? ` · line ${line}` : "");
    output.replaceChildren();
    files[filename].split("\n").forEach((text, index) => {
      const span = document.createElement("span");
      span.className = "source-line" + (index + 1 === line ? " is-reference" : "");
      span.textContent = `${String(index + 1).padStart(4)}  ${text}`;
      output.append(span);
    });
    const highlighted = output.querySelector(".is-reference");
    output.parentElement.scrollTop = highlighted ? highlighted.offsetTop - output.offsetTop - 60 : 0;
  }
  picker.addEventListener("change", () => showFile(picker.value));
  document.addEventListener("click", event => {
    const source = event.target.closest("[data-source-file]");
    if (source) showFile(source.dataset.sourceFile, Number(source.dataset.sourceLine || 0));
  });
  showFile(picker.value);
  const diagrams = [...document.querySelectorAll("pre.mermaid")].map(element => ({element, source: element.textContent}));
  async function drawDiagrams() {
    const style = getComputedStyle(document.documentElement);
    mermaid.initialize({startOnLoad: false, securityLevel: "strict", theme: "base", themeVariables: {
      primaryColor: style.getPropertyValue("--code").trim(),
      primaryTextColor: style.getPropertyValue("--foreground").trim(),
      primaryBorderColor: style.getPropertyValue("--border").trim(),
      lineColor: style.getPropertyValue("--muted-foreground").trim(),
      secondaryColor: style.getPropertyValue("--panel").trim(),
      tertiaryColor: style.getPropertyValue("--background").trim(),
      fontFamily: "system-ui, sans-serif"
    }});
    for (const {element, source} of diagrams) {
      element.removeAttribute("data-processed");
      element.textContent = source;
    }
    try { await mermaid.run({nodes: diagrams.map(d => d.element)}); }
    catch (error) { console.error("Architecture diagram rendering failed", error); }
  }
  drawDiagrams();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawDiagrams);
})();
