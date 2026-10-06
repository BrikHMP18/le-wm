"use strict";

const COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f", "#b07aa1", "#edc948", "#76b7b2", "#ff9da7"];
const HIDDEN_SERIES = new Set(["timestamp", "episode_id", "episode_index", "ep_idx"]);
const PREFETCH = 40;

const $ = (id) => document.getElementById(id);
const state = {
  file: null, info: null, episodes: null, visible: [], ep: null, data: null,
  times: [], t: 0, playing: false, clock: 0, last: 0, hidden: {}, frameCache: new Map(),
};

async function api(path, params) {
  const query = new URLSearchParams({ file: state.file, ...params });
  const response = await fetch(`${path}?${query}`);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

const frameUrl = (col, ep, t) =>
  `/api/frame?${new URLSearchParams({ file: state.file, col, ep, t })}`;
const imageColumns = () => state.info.columns.filter((c) => c.kind === "image");
const seriesColumns = () =>
  state.info.columns.filter((c) => c.kind === "series" && !HIDDEN_SERIES.has(c.name));

function readHash() {
  return Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
}

function writeHash() {
  const hash = new URLSearchParams({ file: state.file, ep: state.ep, t: state.t });
  history.replaceState(null, "", `#${hash}`);
}

async function init() {
  const files = await (await fetch("/api/files")).json();
  const select = $("file");
  for (const name of files) select.add(new Option(name, name));
  const wanted = readHash();
  select.value = files.includes(wanted.file) ? wanted.file : files[0];
  select.onchange = () => loadFile(select.value, 0, 0);
  bindControls();
  await loadFile(select.value, Number(wanted.ep) || 0, Number(wanted.t) || 0);
}

async function loadFile(file, ep, t) {
  pause();
  state.file = file;
  [state.info, state.episodes] = await Promise.all([api("/api/info"), api("/api/episodes")]);
  const { info } = state;
  $("summary").textContent =
    `${info.episodes.toLocaleString()} episodes · ${info.rows.toLocaleString()} frames · ` +
    `${info.columns.length} columns${info.fps ? ` · ${+info.fps.toFixed(2)} fps` : ""}`;
  $("metadata").textContent = JSON.stringify(info.metadata, null, 2);
  buildCameras();
  buildFacets();
  applyFilters();
  await loadEpisode(Math.min(ep, info.episodes - 1), t);
}

function buildCameras() {
  const root = $("cameras");
  root.replaceChildren();
  for (const col of imageColumns()) {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.dataset.col = col.name;
    img.alt = col.name;
    const caption = document.createElement("figcaption");
    caption.textContent = `${col.name} · ${col.shape.slice(1).join("×")} ${col.dtype}`;
    figure.append(img, caption);
    root.append(figure);
  }
  if (!imageColumns().length) root.innerHTML = '<p class="muted">No image columns.</p>';
}

function buildFacets() {
  const root = $("facets");
  root.replaceChildren();
  const attributes = state.episodes.attributes;
  if (!attributes) return;
  const values = {};
  for (const attrs of attributes) {
    for (const [key, value] of Object.entries(attrs)) {
      if (typeof value === "string" || typeof value === "boolean") {
        (values[key] ??= new Map()).set(String(value), ((values[key].get(String(value))) || 0) + 1);
      }
    }
  }
  for (const [key, counts] of Object.entries(values)) {
    if (counts.size < 2 || counts.size > 20) continue;
    const select = document.createElement("select");
    select.dataset.key = key;
    select.add(new Option(`${key}: all`, ""));
    for (const [value, n] of [...counts].sort()) select.add(new Option(`${key}: ${value} (${n})`, value));
    select.onchange = applyFilters;
    root.append(select);
  }
}

function applyFilters() {
  const filters = [...$("facets").querySelectorAll("select")]
    .filter((s) => s.value !== "")
    .map((s) => [s.dataset.key, s.value]);
  const { lengths, attributes } = state.episodes;
  state.visible = lengths
    .map((_, i) => i)
    .filter((i) => filters.every(([k, v]) => String(attributes?.[i]?.[k]) === v));
  $("count").textContent = `${state.visible.length.toLocaleString()} of ${lengths.length.toLocaleString()} episodes`;

  const facetKeys = new Set([...$("facets").querySelectorAll("select")].map((s) => s.dataset.key));
  const fragment = document.createDocumentFragment();
  for (const i of state.visible) {
    const li = document.createElement("li");
    li.dataset.ep = i;
    const attrs = attributes?.[i] || {};
    const tags = [...facetKeys].map((k) => attrs[k]).filter((v) => v !== undefined).join(" · ");
    li.innerHTML = `<span>ep ${i}</span><span class="meta">${lengths[i]} f${tags ? ` · ${tags}` : ""}</span>`;
    li.onclick = () => loadEpisode(i, 0);
    fragment.append(li);
  }
  $("episodes").replaceChildren(fragment);
  markActive();
}

function markActive() {
  for (const li of $("episodes").querySelectorAll("li.active")) li.classList.remove("active");
  const li = $("episodes").querySelector(`li[data-ep="${state.ep}"]`);
  if (li) {
    li.classList.add("active");
    li.scrollIntoView({ block: "nearest" });
  }
}

async function loadEpisode(ep, t = 0) {
  if (ep < 0 || ep >= state.info.episodes) return;
  const data = await api("/api/episode", { ep });
  state.ep = ep;
  state.data = data;
  state.frameCache = new Map();
  const fps = state.info.fps || 10;
  const ts = data.series.timestamp?.map((row) => row[0]);
  state.times = ts && ts.every((v) => v !== null)
    ? ts.map((v) => v - ts[0])
    : Array.from({ length: data.length }, (_, i) => i / fps);
  $("scrub").max = data.length - 1;
  const attrs = data.attributes || {};
  $("attributes").innerHTML = Object.entries(attrs)
    .map(([k, v]) => `<span>${k}: ${typeof v === "number" ? +v.toFixed(4) : v}</span>`)
    .join("");
  buildPlots();
  markActive();
  seek(Math.min(t, data.length - 1));
}

function prefetch(from) {
  const { ep, data } = state;
  for (let t = from; t < Math.min(from + PREFETCH, data.length); t++) {
    for (const col of imageColumns()) {
      const key = `${col.name}:${t}`;
      if (state.frameCache.has(key)) continue;
      const img = new Image();
      img.src = frameUrl(col.name, ep, t);
      state.frameCache.set(key, img);
    }
  }
}

function seek(t) {
  state.t = Math.max(0, Math.min(t, state.data.length - 1));
  state.clock = state.times[state.t];
  render();
}

function render() {
  const { t, ep, data } = state;
  for (const img of $("cameras").querySelectorAll("img")) {
    img.src = state.frameCache.get(`${img.dataset.col}:${t}`)?.src || frameUrl(img.dataset.col, ep, t);
  }
  prefetch(t + 1);
  $("scrub").value = t;
  $("status").textContent =
    `episode ${ep} · frame ${t + 1}/${data.length} · t = ${state.times[t].toFixed(2)} s · row ${data.offset + t}`;
  for (const plot of state.plots) drawPlot(plot);
  writeHash();
}

// ---------------------------------------------------------------- plots

function buildPlots() {
  const root = $("plots");
  root.replaceChildren();
  state.plots = [];
  for (const col of seriesColumns()) {
    const rows = state.data.series[col.name];
    if (!rows) continue;
    const width = rows[0]?.length || 0;
    const names = col.names || (width === 1 ? [col.name] : Array.from({ length: width }, (_, i) => `${col.name}[${i}]`));
    const hidden = (state.hidden[col.name] ??= new Set());

    const box = document.createElement("div");
    box.innerHTML = `<div class="plot-head"><h2>${col.name}</h2><span class="muted small">${col.shape.slice(1).join("×") || "scalar"} ${col.dtype}</span></div>`;
    const legend = document.createElement("div");
    legend.className = "legend";
    const canvas = document.createElement("canvas");
    box.append(legend, canvas);
    root.append(box);

    const plot = { col: col.name, rows, names, hidden, canvas, legend, values: [], bool: col.dtype === "bool" };
    names.forEach((name, i) => {
      const span = document.createElement("span");
      span.innerHTML = `<i style="background:${COLORS[i % COLORS.length]}"></i>${name} <b></b>`;
      span.classList.toggle("off", hidden.has(i));
      span.onclick = () => {
        hidden.has(i) ? hidden.delete(i) : hidden.add(i);
        span.classList.toggle("off", hidden.has(i));
        drawPlot(plot);
      };
      legend.append(span);
      plot.values.push(span.querySelector("b"));
    });

    const seekFromPointer = (event) => {
      const rect = canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left - plot.left) / (rect.width - plot.left - 8);
      const target = x * state.times.at(-1);
      let best = 0;
      for (let i = 0; i < state.times.length; i++) {
        if (Math.abs(state.times[i] - target) < Math.abs(state.times[best] - target)) best = i;
      }
      pause();
      seek(best);
    };
    canvas.onpointerdown = (event) => {
      canvas.setPointerCapture(event.pointerId);
      seekFromPointer(event);
      canvas.onpointermove = seekFromPointer;
    };
    canvas.onpointerup = () => (canvas.onpointermove = null);
    state.plots.push(plot);
  }
}

function formatTick(value) {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e4 || abs < 1e-2)) return value.toExponential(1);
  return +value.toFixed(2) + "";
}

function drawPlot(plot) {
  const { canvas, rows, hidden } = plot;
  const styles = getComputedStyle(document.documentElement);
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * dpr)) canvas.width = Math.round(width * dpr);
  if (canvas.height !== Math.round(height * dpr)) canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  let lo = Infinity, hi = -Infinity;
  for (const row of rows) {
    row.forEach((v, i) => {
      if (v !== null && !hidden.has(i)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    });
  }
  if (plot.bool || !isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06;
  lo -= pad; hi += pad;

  const left = (plot.left = 52), right = 8, top = 6, bottom = 18;
  const span = state.times.at(-1) || 1;
  const X = (t) => left + (t / span) * (width - left - right);
  const Y = (v) => top + (1 - (v - lo) / (hi - lo)) * (height - top - bottom);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = styles.getPropertyValue("--muted");
  ctx.strokeStyle = styles.getPropertyValue("--grid");
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(width - right, Y(v)); ctx.stroke();
    ctx.fillText(formatTick(v), left - 6, Y(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let k = 0; k <= 4; k++) {
    ctx.textAlign = k === 0 ? "left" : k === 4 ? "right" : "center";
    ctx.fillText(`${+((span * k) / 4).toFixed(2)}s`, X((span * k) / 4), height - bottom + 4);
  }

  plot.names.forEach((_, i) => {
    if (hidden.has(i)) return;
    ctx.strokeStyle = COLORS[i % COLORS.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let drawing = false;
    rows.forEach((row, t) => {
      const v = row[i];
      if (v === null) { drawing = false; return; }
      drawing ? ctx.lineTo(X(state.times[t]), Y(v)) : ctx.moveTo(X(state.times[t]), Y(v));
      drawing = true;
    });
    ctx.stroke();
  });

  const cx = X(state.times[state.t]);
  ctx.strokeStyle = styles.getPropertyValue("--cursor");
  ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, height - bottom); ctx.stroke();
  const current = rows[state.t] || [];
  plot.names.forEach((_, i) => {
    const v = current[i];
    plot.values[i].textContent = v === null || v === undefined ? "NaN" : formatTick(v);
    if (v === null || v === undefined || hidden.has(i)) return;
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.beginPath(); ctx.arc(cx, Y(v), 3, 0, 2 * Math.PI); ctx.fill();
  });
}

// ---------------------------------------------------------------- playback

function play() {
  if (!state.data) return;
  if (state.t >= state.data.length - 1) seek(0);
  state.playing = true;
  state.last = performance.now();
  $("play").textContent = "❚❚";
  requestAnimationFrame(tick);
}

function pause() {
  state.playing = false;
  $("play").textContent = "▶︎";
}

async function tick(now) {
  if (!state.playing) return;
  state.clock += ((now - state.last) / 1000) * Number($("speed").value);
  state.last = now;
  let t = state.t;
  while (t < state.data.length - 1 && state.times[t + 1] <= state.clock) t++;
  if (t >= state.data.length - 1 && state.clock > state.times.at(-1)) {
    if ($("autonext").checked) {
      const next = state.visible[state.visible.indexOf(state.ep) + 1];
      if (next !== undefined) {
        await loadEpisode(next, 0);
        state.last = performance.now();
        return requestAnimationFrame(tick);
      }
    }
    if ($("loop").checked) {
      seek(0);
      return requestAnimationFrame(tick);
    }
    state.t = t;
    render();
    return pause();
  }
  if (t !== state.t) { state.t = t; render(); }
  requestAnimationFrame(tick);
}

function stepEpisode(delta) {
  const position = state.visible.indexOf(state.ep);
  const next = state.visible[position === -1 ? 0 : position + delta];
  if (next !== undefined) loadEpisode(next, 0);
}

function bindControls() {
  $("play").onclick = () => (state.playing ? pause() : play());
  $("prev").onclick = () => { pause(); seek(state.t - 1); };
  $("next").onclick = () => { pause(); seek(state.t + 1); };
  $("scrub").oninput = (event) => { pause(); seek(Number(event.target.value)); };
  $("jump").onchange = (event) => {
    const ep = Number(event.target.value);
    if (Number.isInteger(ep)) loadEpisode(ep, 0);
  };
  window.addEventListener("resize", () => state.plots?.forEach(drawPlot));
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input:not([type=range]), select")) return;
    const actions = {
      " ": () => (state.playing ? pause() : play()),
      ArrowLeft: () => { pause(); seek(state.t - 1); },
      ArrowRight: () => { pause(); seek(state.t + 1); },
      ArrowUp: () => stepEpisode(-1),
      ArrowDown: () => stepEpisode(1),
    };
    if (actions[event.key]) { event.preventDefault(); actions[event.key](); }
  });
}

init().catch((error) => {
  document.querySelector("main").innerHTML = `<p>Failed to load: ${error.message}</p>`;
});
