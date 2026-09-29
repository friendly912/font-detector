"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  doc: null,            // アップロード結果 (ページ・フォント情報)
  fontsByKey: new Map(),
  selected: new Set(),  // 一覧で選択されたフォントキー
  matches: [],
  searchSeq: 0,
  imageScan: null,      // 画像内OCRの結果 (ドキュメントごとに一度だけ取得)
};

const PAGE_SCALE = Math.min(3, Math.max(1.5, (window.devicePixelRatio || 1) * 1.5));

// ---------------------------------------------------------------- API

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `エラーが発生しました (${res.status})`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : msg;
    } catch (_) { /* JSON 以外の応答 */ }
    throw new Error(msg);
  }
  return res;
}

function criteria() {
  return {
    font_keys: [...state.selected],
    query: $("query").value,
    body_only: $("body-only").checked,
    include_images: $("include-images").checked && state.imageScan !== null,
    min_score: Number($("min-score").value),
  };
}

// ---------------------------------------------------------------- upload

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function upload(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    setStatus("PDFファイルを選択してください。", true);
    return;
  }
  $("file-name").textContent = file.name;
  setStatus("解析中…");
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api("/api/documents", { method: "POST", body: form });
    const doc = await res.json();
    loadDocument(doc);
    const fontCount = doc.fonts.length;
    let msg = `${doc.page_count}ページ・${fontCount}種類のフォントを検出しました。`;
    if (doc.invisible_chars > 0) msg += ` 透明テキスト${doc.invisible_chars}文字は対象外です。`;
    if (fontCount === 0) msg = "テキストレイヤーが見つかりません（スキャンPDFの可能性があります）。";
    setStatus(msg);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function loadDocument(doc) {
  state.doc = doc;
  state.fontsByKey = new Map(doc.fonts.map((f) => [f.key, f]));
  state.selected.clear();
  state.matches = [];
  state.imageScan = null;
  $("query").value = "";
  $("include-images").checked = false;
  $("include-images").disabled = doc.images.length === 0;
  $("image-controls").hidden = true;
  $("body-size").textContent = doc.body_size ? `(本文 ${doc.body_size}pt)` : "";
  renderFontList();
  renderPages();
  $("controls").hidden = false;
  $("results-panel").hidden = false;
  $("empty").hidden = true;
  runSearch();
}

// ---------------------------------------------------------------- font list

function renderFontList() {
  const list = $("font-list");
  const tpl = $("font-item");
  list.replaceChildren();
  for (const font of state.doc.fonts) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    const checkbox = node.querySelector("input");
    checkbox.value = font.key;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(font.key);
      else state.selected.delete(font.key);
      runSearch();
    });
    node.querySelector(".swatch").style.background = font.color;
    node.querySelector(".font-name").textContent = font.display;
    const embedded = { true: "埋込", false: "未埋込", null: "埋込不明" }[String(font.embedded)];
    const raw = font.raw_names.filter((n) => n !== font.display).join(", ");
    let meta = `${font.char_count.toLocaleString()}文字 · ${pagesLabel(font.pages)} · ${embedded}`;
    if (state.imageScan) meta += ` · ${referenceLabel(state.imageScan.references[font.key])}`;
    node.querySelector(".font-meta").textContent = meta + (raw ? ` · ${raw}` : "");
    node.dataset.key = font.key;
    list.append(node);
  }
}

function referenceLabel(ref) {
  if (!ref || ref.glyphs === 0) return "画像判定の見本なし";
  const src = ref.sources.map((x) => ({ pdf: "PDF", system: "システム" }[x] || x)).join("+");
  return `見本${ref.glyphs}字(${src})`;
}

function pagesLabel(pages) {
  if (pages.length <= 3) return `p.${pages.join(",")}`;
  return `${pages.length}ページ`;
}

function syncFontListActive(activeKeys) {
  for (const li of $("font-list").children) {
    const key = li.dataset.key;
    li.querySelector("input").checked = state.selected.has(key);
    li.classList.toggle("active", activeKeys.has(key));
  }
}

// ---------------------------------------------------------------- pages

function renderPages() {
  const container = $("pages");
  container.replaceChildren();
  state.doc.page_sizes.forEach((size, i) => {
    const pageNo = i + 1;
    const page = document.createElement("div");
    page.className = "page";
    page.id = `page-${pageNo}`;
    page.style.aspectRatio = `${size.width} / ${size.height}`;

    const label = document.createElement("span");
    label.className = "page-number";
    label.textContent = `${pageNo} / ${state.doc.page_count}`;

    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = `ページ ${pageNo}`;
    img.src = `/api/documents/${state.doc.id}/pages/${pageNo}.png?scale=${PAGE_SCALE}`;

    const overlay = document.createElement("div");
    overlay.className = "overlay";

    page.append(label, img, overlay);
    container.append(page);
  });
  drawImageRegions();
}

function placeBox(el, bbox, size) {
  const [x0, y0, x1, y1] = bbox;
  el.style.left = `${(x0 / size.width) * 100}%`;
  el.style.top = `${(y0 / size.height) * 100}%`;
  el.style.width = `${((x1 - x0) / size.width) * 100}%`;
  el.style.height = `${((y1 - y0) / size.height) * 100}%`;
}

function overlayOf(pageNo) {
  return document.querySelector(`#page-${pageNo} .overlay`);
}

function drawImageRegions() {
  for (const im of state.doc.images) {
    const box = document.createElement("div");
    box.className = "image-region";
    placeBox(box, im.bbox, state.doc.page_sizes[im.page - 1]);
    overlayOf(im.page).append(box);
  }
}

function fontName(key) {
  return state.fontsByKey.get(key)?.display ?? key;
}

function drawOcrLines() {
  document.querySelectorAll(".ocr-line").forEach((el) => el.remove());
  if (!state.imageScan) return;
  for (const ln of state.imageScan.lines) {
    const box = document.createElement("div");
    box.className = "ocr-line";
    const ranking = ln.ranking.length
      ? ln.ranking.map((r) => `${fontName(r.font)} ${r.score.toFixed(2)} (${r.compared}字)`).join("\n")
      : "比較できる見本がありません";
    box.title = `${ln.text}\n--- 推定 ---\n${ranking}`;
    placeBox(box, ln.bbox, state.doc.page_sizes[ln.page - 1]);
    overlayOf(ln.page).append(box);
  }
}

function drawMatches() {
  document.querySelectorAll(".hit").forEach((el) => el.remove());
  state.matches.forEach((m, idx) => {
    const font = state.fontsByKey.get(m.font);
    const box = document.createElement("div");
    box.className = "hit";
    box.dataset.idx = idx;
    if (m.source === "image") {
      box.classList.add("image-hit");
      box.style.setProperty("--hit-color", font.color);
      box.title = `画像内推定: ${font.display} (類似度 ${m.score.toFixed(2)})\n${m.text}`;
    } else {
      box.style.background = font.color;
      box.title = `${font.display} / ${m.size}pt\n${m.text}`;
    }
    box.addEventListener("click", () => focusMatch(idx, { scrollViewer: false }));
    placeBox(box, m.bbox, state.doc.page_sizes[m.page - 1]);
    overlayOf(m.page).append(box);
  });
}

// ---------------------------------------------------------------- search & results

let searchTimer = 0;
function scheduleSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(runSearch, 200);
}

async function runSearch() {
  if (!state.doc) return;
  const seq = ++state.searchSeq;
  try {
    const res = await api(`/api/documents/${state.doc.id}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(criteria()),
    });
    const data = await res.json();
    if (seq !== state.searchSeq) return; // 古い応答は捨てる
    state.matches = data.matches.sort((a, b) => a.page - b.page || a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]);
    syncFontListActive(new Set(data.font_keys));
    drawMatches();
    renderResults(data.font_keys.length > 0);
  } catch (err) {
    setStatus(err.message, true);
  }
}

function renderResults(hasTarget) {
  const list = $("results");
  list.replaceChildren();
  const matches = state.matches;
  $("download").disabled = matches.length === 0;

  if (!hasTarget) {
    $("summary").textContent = "フォントを選択または入力してください。";
    return;
  }
  const textMatches = matches.filter((m) => m.source === "text");
  const imageCount = matches.length - textMatches.length;
  const chars = textMatches.reduce((n, m) => n + m.text.trim().length, 0);
  const pages = new Set(matches.map((m) => m.page));
  let summary = `${textMatches.length}箇所 · ${chars.toLocaleString()}文字`;
  if (state.imageScan && $("include-images").checked) summary += ` · 画像内 ${imageCount}行`;
  $("summary").textContent = matches.length
    ? `${summary} · ${pages.size}ページ`
    : "該当する箇所はありません。";

  let currentPage = 0;
  matches.forEach((m, idx) => {
    if (m.page !== currentPage) {
      currentPage = m.page;
      const label = document.createElement("li");
      label.className = "page-label";
      label.textContent = `ページ ${m.page}`;
      list.append(label);
    }
    const font = state.fontsByKey.get(m.font);
    const li = document.createElement("li");
    li.className = "result-item";
    li.dataset.idx = idx;

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = font.color;
    const text = document.createElement("span");
    text.className = "result-text";
    text.textContent = m.text.trim();
    const meta = document.createElement("span");
    if (m.source === "image") {
      meta.className = "badge";
      meta.textContent = `画像 ${m.score.toFixed(2)}`;
      meta.title = "画像内の文字（OCR）から推定。数値は字形の類似度";
    } else {
      meta.className = "result-size";
      meta.textContent = `${m.size}pt`;
    }

    li.append(swatch, text, meta);
    li.addEventListener("click", () => focusMatch(idx, { scrollViewer: true }));
    list.append(li);
  });
}

function focusMatch(idx, { scrollViewer }) {
  const box = document.querySelector(`.hit[data-idx="${idx}"]`);
  const item = document.querySelector(`.result-item[data-idx="${idx}"]`);
  document.querySelectorAll(".result-item.active").forEach((el) => el.classList.remove("active"));
  if (item) {
    item.classList.add("active");
    if (!scrollViewer) item.scrollIntoView({ block: "nearest" });
  }
  if (box) {
    if (scrollViewer) box.scrollIntoView({ block: "center", behavior: "smooth" });
    box.classList.remove("flash");
    void box.offsetWidth; // アニメーションを再生し直す
    box.classList.add("flash");
  }
}

// ---------------------------------------------------------------- image scan

async function toggleImages(enabled) {
  $("image-controls").hidden = !enabled;
  if (enabled && !state.imageScan) {
    const checkbox = $("include-images");
    checkbox.disabled = true;
    const regions = state.doc.images.length;
    setStatus(`画像内の文字を解析中… (画像${regions}個・初回のみ時間がかかります)`);
    try {
      const res = await api(`/api/documents/${state.doc.id}/image-scan`, { method: "POST" });
      state.imageScan = await res.json();
      const decided = state.imageScan.lines.filter((l) => l.ranking.length > 0).length;
      setStatus(`画像${state.imageScan.regions}個から${state.imageScan.lines.length}行を読み取り、` +
                `${decided}行でフォントを比較しました。`);
      renderFontList();
      drawOcrLines();
    } catch (err) {
      checkbox.checked = false;
      $("image-controls").hidden = true;
      setStatus(err.message, true);
    } finally {
      checkbox.disabled = false;
    }
  }
  runSearch();
}

// ---------------------------------------------------------------- download

async function downloadAnnotated() {
  const button = $("download");
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "作成中…";
  try {
    const res = await api(`/api/documents/${state.doc.id}/annotated`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(criteria()),
    });
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = state.doc.filename.replace(/\.pdf$/i, "") + "_highlighted.pdf";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.textContent = label;
    button.disabled = state.matches.length === 0;
  }
}

// ---------------------------------------------------------------- wiring

function init() {
  const dropzone = $("dropzone");
  $("file-input").addEventListener("change", (e) => upload(e.target.files[0]));
  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("dragover"));
  }
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    upload(e.dataTransfer.files[0]);
  });
  // ドロップゾーン外に落としたときにブラウザがPDFを開いてしまうのを防ぐ
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("query").addEventListener("input", scheduleSearch);
  $("body-only").addEventListener("change", runSearch);
  $("include-images").addEventListener("change", (e) => toggleImages(e.target.checked));
  $("min-score").addEventListener("input", (e) => {
    $("min-score-value").textContent = Number(e.target.value).toFixed(2);
    scheduleSearch();
  });
  $("show-images").addEventListener("change", (e) => {
    $("pages").classList.toggle("hide-images", !e.target.checked);
  });
  $("clear-selection").addEventListener("click", () => {
    state.selected.clear();
    $("query").value = "";
    runSearch();
  });
  $("download").addEventListener("click", downloadAnnotated);
}

init();
