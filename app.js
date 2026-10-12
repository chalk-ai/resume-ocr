"use strict";

// Everything the model writes comes from an uploaded document, so it is untrusted: this file
// only ever puts it on the page through textContent, never innerHTML.

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const state = { job: 0, pages: [], words: [], raw: "", data: null, done: false, stats: {} };

// ---------- partial JSON ----------

// Closes whatever the model has written so far into parseable JSON: finishes an open string,
// drops a dangling key or comma, and closes open objects and arrays.
function parsePartial(src) {
  try { return JSON.parse(src); } catch (_) { /* fall through */ }
  const stack = [];
  let inStr = false, esc = false;
  for (const ch of src) {
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === "\"") inStr = false;
    } else if (ch === "\"") inStr = true;
    else if (ch === "{" || ch === "[") stack.push(ch);
    else if (ch === "}" || ch === "]") stack.pop();
  }
  let s = src;
  if (inStr) s += "\"";
  for (let i = 0; i < 4; i++) {
    const t = s
      .replace(/\\$/, "")
      .replace(/,\s*$/, "")
      .replace(/([{,]\s*)"(?:[^"\\]|\\.)*"\s*:?\s*$/, "$1")
      .replace(/:\s*$/, "")
      .replace(/,\s*$/, "");
    if (t === s) break;
    s = t;
  }
  s += stack.reverse().map((c) => (c === "{" ? "}" : "]")).join("");
  try { return JSON.parse(s); } catch (_) { return null; }
}

// ---------- matching fields back to OCR words ----------

const norm = (s) => s.toLowerCase().replace(/[^a-z0-9@.+]/g, "");
const tokens = (s) => s.split(/\s+/).map(norm).filter(Boolean);

function editDistance(a, b) {
  if (Math.abs(a.length - b.length) > 2) return 3;
  const row = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return row[b.length];
}

const same = (a, b) => a === b || (a.length > 3 && editDistance(a, b) <= 1);

// The run of OCR words that best spells out `value`, or [] when nothing matches well enough.
function locate(value) {
  const want = tokens(String(value)).slice(0, 14);
  if (!want.length) return [];
  let best = [], bestScore = 0;
  const words = state.words;
  for (let i = 0; i < words.length; i++) {
    if (!same(words[i].n, want[0]) && !(want.length > 1 && same(words[i].n, want[1]))) continue;
    const run = [];
    let hit = 0, j = i;
    for (const t of want) {
      if (j >= words.length) break;
      if (same(words[j].n, t)) { hit++; run.push(words[j]); j++; }
      else if (j + 1 < words.length && same(words[j + 1].n, t)) { hit++; run.push(words[j], words[j + 1]); j += 2; }
    }
    const score = hit / want.length;
    if (score > bestScore) { bestScore = score; best = run; }
  }
  return bestScore >= 0.6 ? best : [];
}

function highlight(found, on) {
  for (const w of found) w.node.classList.toggle("hl", on);
  if (on && found.length) found[0].node.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// ---------- rendering ----------

function setStage(name, status, note) {
  const li = document.querySelector(`.stages li[data-stage="${name}"]`);
  li.classList.toggle("active", status === "active");
  li.classList.toggle("done", status === "done");
  li.querySelector("em").textContent = note || "";
}

function resetStages() {
  for (const s of ["render", "ocr", "extract"]) setStage(s, "idle");
}

function showError(msg) {
  const e = $("error");
  e.textContent = msg;
  e.hidden = !msg;
}

function renderPages(pages) {
  const root = $("pages");
  root.replaceChildren();
  state.pages = pages.map((p) => {
    const wrap = el("div", "page");
    const img = el("img");
    img.src = p.image;
    img.alt = "Resume page";
    const boxes = el("div", "boxes");
    const scan = el("div", "scan");
    wrap.append(img, boxes, scan);
    root.append(wrap);
    return { wrap, boxes, scan };
  });
}

function renderOcr(pageIndex, lines) {
  const page = state.pages[pageIndex];
  if (!page) return;
  page.scan.remove();
  let k = 0;
  for (const line of lines) {
    for (const w of line.words) {
      const [x0, y0, x1, y1] = w.box;
      const node = el("div", "w");
      node.style.left = `${x0 * 100}%`;
      node.style.top = `${y0 * 100}%`;
      node.style.width = `${(x1 - x0) * 100}%`;
      node.style.height = `${(y1 - y0) * 100}%`;
      node.title = `${w.text} (${Math.round(w.confidence * 100)}%)`;
      page.boxes.append(node);
      // A short top-to-bottom reveal; the OCR itself has already finished.
      setTimeout(() => node.classList.add("in"), Math.min(900, k++ * 4));
      state.words.push({ n: norm(w.text), node });
    }
  }
}

// A field element that lights up its source words on hover, when they can be found.
function field(tag, cls, value) {
  const n = el(tag, `f ${cls}`, value);
  if (!state.done || !value) return n;
  const found = locate(value);
  if (found.length) {
    n.classList.add("match");
    n.addEventListener("mouseenter", () => highlight(found, true));
    n.addEventListener("mouseleave", () => highlight(found, false));
  }
  return n;
}

function section(title) {
  const s = el("div", "sec");
  s.append(el("h3", null, title));
  return s;
}

function chips(values) {
  const c = el("div", "chips");
  for (const v of values || []) if (v) c.append(field("span", "chip", v));
  return c;
}

const filled = (a) => Array.isArray(a) && a.some((x) => x && (typeof x !== "object" || Object.values(x).some(Boolean)));

function renderFields() {
  const d = state.data;
  const root = $("fields");
  if (!d) return;
  root.replaceChildren();

  const who = el("div", "who");
  if (d.name) who.append(field("div", "name", d.name));
  if (d.headline) who.append(field("div", "headline", d.headline));
  const contact = chips([d.email, d.phone, d.location, ...(d.links || [])]);
  if (contact.childElementCount) who.append(contact);
  root.append(who);

  if (d.summary) {
    const s = section("Summary");
    s.append(field("div", "summary", d.summary));
    root.append(s);
  }
  if (filled(d.experience)) {
    const s = section("Experience");
    for (const job of d.experience) {
      const item = el("div", "item");
      const row = el("div", "row");
      row.append(field("span", "t", job.title || ""), el("span", "d", [job.start, job.end].filter(Boolean).join(" – ")));
      item.append(row);
      const org = [job.organization, job.location].filter(Boolean).join(", ");
      if (org) item.append(field("div", "o", job.organization ? job.organization : org));
      if (filled(job.highlights)) {
        const ul = el("ul");
        for (const h of job.highlights) if (h) ul.append(field("li", "", h));
        item.append(ul);
      }
      s.append(item);
    }
    root.append(s);
  }
  if (filled(d.education)) {
    const s = section("Education");
    for (const ed of d.education) {
      const item = el("div", "item");
      const row = el("div", "row");
      row.append(field("span", "t", ed.credential || ""), el("span", "d", [ed.start, ed.end].filter(Boolean).join(" – ")));
      item.append(row);
      if (ed.institution) item.append(field("div", "o", ed.institution));
      if (ed.details) item.append(field("div", "o", ed.details));
      s.append(item);
    }
    root.append(s);
  }
  for (const [key, title] of [["skills", "Skills"], ["certifications", "Certifications"], ["languages", "Languages"]]) {
    if (filled(d[key])) {
      const s = section(title);
      s.append(chips(d[key]));
      root.append(s);
    }
  }
  if (!state.done) {
    const last = root.lastElementChild;
    if (last) last.classList.add("caret");
  }
}

function renderJson() {
  const pre = $("json");
  pre.replaceChildren();
  const text = state.done && state.data ? JSON.stringify(state.data, null, 2) : state.raw;
  // Colour keys and string values without ever parsing the text into HTML.
  const re = /("(?:[^"\\]|\\.)*")(\s*:)?/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) pre.append(text.slice(last, m.index));
    pre.append(el("span", m[2] ? "k" : "s", m[1]));
    if (m[2]) pre.append(m[2]);
    last = re.lastIndex;
  }
  pre.append(text.slice(last));
}

let frame = 0;
function scheduleRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    renderFields();
    if (!$("json").hidden) renderJson();
  });
}

function renderStats() {
  const s = state.stats;
  const parts = [];
  if (s.ocr != null) parts.push(`OCR ${s.ocr} ms`);
  if (s.extract != null) parts.push(`extraction ${(s.extract / 1000).toFixed(1)} s`);
  if (s.tokens) parts.push(`${s.tokens} tokens · ${Math.round(s.tokens / (s.extract / 1000))} tok/s`);
  if (s.truncated) parts.push(`first ${s.pagesShown} of ${s.totalPages} pages`);
  $("stats").textContent = parts.join("  ·  ");
}

// ---------- running a job ----------

async function run(body) {
  const job = ++state.job;
  Object.assign(state, { pages: [], words: [], raw: "", data: null, done: false, stats: {} });
  showError("");
  resetStages();
  setStage("render", "active");
  $("pages").replaceChildren(el("p", "placeholder", "Rendering…"));
  $("fields").replaceChildren(el("p", "placeholder", "Waiting for OCR…"));
  $("json").replaceChildren();
  $("stats").textContent = "";

  let resp;
  try {
    resp = await fetch("extract", { method: "POST", body });
  } catch (_) {
    return fail(job, "Could not reach the demo server.");
  }
  if (!resp.ok) {
    let msg = `Request failed (${resp.status}).`;
    try { msg = (await resp.json()).error || msg; } catch (_) { /* keep default */ }
    return fail(job, msg);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (job !== state.job) { reader.cancel(); return; }
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (chunk.startsWith("data: ")) handle(JSON.parse(chunk.slice(6)));
    }
  }
}

function fail(job, msg) {
  if (job !== state.job) return;
  showError(msg);
  resetStages();
  $("pages").replaceChildren(el("p", "placeholder", "Pick a sample resume or upload one to start."));
  $("fields").replaceChildren();
}

function handle(msg) {
  switch (msg.type) {
    case "pages":
      setStage("render", "done", `${msg.render_ms} ms`);
      setStage("ocr", "active");
      state.stats.totalPages = msg.total_pages;
      state.stats.pagesShown = msg.pages.length;
      state.stats.truncated = msg.total_pages > msg.pages.length;
      renderPages(msg.pages);
      break;
    case "ocr":
      state.stats.ocr = (state.stats.ocr || 0) + msg.ms;
      renderOcr(msg.page, msg.lines);
      if (msg.page === state.pages.length - 1) {
        setStage("ocr", "done", `${state.stats.ocr} ms`);
        setStage("extract", "active");
        $("fields").replaceChildren(el("p", "placeholder", "Reading the OCR text…"));
      }
      renderStats();
      break;
    case "delta":
      state.raw += msg.text;
      state.data = parsePartial(state.raw) || state.data;
      scheduleRender();
      break;
    case "result":
      state.done = true;
      state.data = msg.data;
      state.stats.extract = msg.ms;
      state.stats.tokens = msg.completion_tokens;
      setStage("extract", "done", `${(msg.ms / 1000).toFixed(1)} s`);
      renderFields();
      renderJson();
      renderStats();
      break;
    case "error":
      fail(state.job, msg.message);
      break;
  }
}

// ---------- controls ----------

async function loadSamples() {
  const root = $("samples");
  let samples = [];
  try { samples = await (await fetch("samples")).json(); } catch (_) { return; }
  for (const s of samples) {
    const b = el("button", "sample");
    b.type = "button";
    b.setAttribute("aria-pressed", "false");
    const img = el("img");
    img.src = `samples/${s.id}.png`;
    img.alt = "";
    b.append(img, el("span", "sample-title", s.title), el("span", "sample-blurb", s.blurb));
    b.addEventListener("click", () => {
      for (const o of root.children) o.setAttribute("aria-pressed", String(o === b));
      const body = new FormData();
      body.append("sample", s.id);
      run(body);
    });
    root.append(b);
  }
}

$("file").addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  ev.target.value = "";
  if (!f) return;
  if (f.size > 5 * 1024 * 1024) return showError("Files up to 5 MB, please.");
  for (const o of $("samples").children) o.setAttribute("aria-pressed", "false");
  const body = new FormData();
  body.append("file", f);
  run(body);
});

$("show-boxes").addEventListener("change", (ev) => $("pages").classList.toggle("no-boxes", !ev.target.checked));

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll(".tabs button")) t.setAttribute("aria-selected", String(t === tab));
    const json = tab.dataset.tab === "json";
    $("json").hidden = !json;
    $("fields").hidden = json;
    if (json) renderJson();
  });
}

loadSamples();
