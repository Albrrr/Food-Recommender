const STORAGE_KEY = "foodrecommender_api_base";

function $(id) {
  return document.getElementById(id);
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text || "";
  el.classList.remove("ok", "err");
  if (kind === "ok") el.classList.add("ok");
  if (kind === "err") el.classList.add("err");
}

function normalizeBaseUrl(s) {
  const trimmed = (s || "").trim();
  if (!trimmed) return "";
  return trimmed.replace(/\/+$/, "");
}

function getApiBase() {
  return normalizeBaseUrl($("apiBase").value);
}

function saveApiBase() {
  const base = getApiBase();
  if (!base) {
    localStorage.removeItem(STORAGE_KEY);
    return;
  }
  localStorage.setItem(STORAGE_KEY, base);
}

function loadApiBase() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) $("apiBase").value = saved;
}

async function apiFetch(path, options) {
  const base = getApiBase();
  if (!base) throw new Error("Set Backend URL first (paste your ngrok URL).");
  const url = `${base}${path.startsWith("/") ? "" : "/"}${path}`;
  const res = await fetch(url, options);
  const contentType = res.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = payload?.detail || payload?.error || (typeof payload === "string" ? payload : JSON.stringify(payload));
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return payload;
}

function renderAnswer(text) {
  const el = $("answer");
  el.classList.remove("muted");
  el.textContent = text || "";
  if (!text) {
    el.classList.add("muted");
    el.textContent = "No answer returned.";
  }
}

function renderSources(sources) {
  const el = $("sources");
  el.innerHTML = "";
  const list = Array.isArray(sources) ? sources : [];
  if (list.length === 0) {
    el.classList.add("muted");
    const li = document.createElement("li");
    li.textContent = "No sources returned.";
    el.appendChild(li);
    return;
  }
  el.classList.remove("muted");
  for (const s of list) {
    const li = document.createElement("li");
    li.textContent = s;
    el.appendChild(li);
  }
}

function setLoading(isLoading) {
  $("submit").disabled = isLoading;
  $("checkHealth").disabled = isLoading;
  $("saveApiBase").disabled = isLoading;
}

async function onHealth() {
  setLoading(true);
  setStatus("Checking…");
  try {
    const data = await apiFetch("/health", { method: "GET" });
    if (data.status === "ok") setStatus("Backend OK", "ok");
    else setStatus(`Not ready: ${data.error || "unknown error"}`, "err");
  } catch (e) {
    setStatus(e.message || String(e), "err");
  } finally {
    setLoading(false);
  }
}

async function onRecommend() {
  const query = ($("query").value || "").trim();
  if (!query) {
    setStatus("Type a query first.", "err");
    return;
  }

  setLoading(true);
  setStatus("Generating…");
  renderAnswer("Working…");
  renderSources([]);

  try {
    const data = await apiFetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    renderAnswer(data.response);
    renderSources(data.sources);
    setStatus("Done", "ok");
  } catch (e) {
    renderAnswer("");
    renderSources([]);
    setStatus(e.message || String(e), "err");
  } finally {
    setLoading(false);
  }
}

function wire() {
  loadApiBase();

  $("saveApiBase").addEventListener("click", () => {
    saveApiBase();
    setStatus("Saved backend URL", "ok");
  });

  $("checkHealth").addEventListener("click", onHealth);
  $("submit").addEventListener("click", onRecommend);

  $("query").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onRecommend();
  });
}

wire();