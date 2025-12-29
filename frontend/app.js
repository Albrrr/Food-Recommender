const DEFAULT_BASE_URL = "https://phenotypically-multiperforate-rodolfo.ngrok-free.dev";

function normalizeBaseUrl(raw) {
  const v = String(raw || "").trim();
  if (!v) return DEFAULT_BASE_URL;
  return v.replace(/\/+$/, "");
}

function setStatus(text, kind = "info") {
  const el = document.getElementById("status");
  if (!el) return;
  el.textContent = text;
  el.dataset.kind = kind;
}

function setLatency(ms) {
  const pill = document.getElementById("latencyPill");
  if (!pill) return;
  pill.hidden = false;
  pill.textContent = `${Math.round(ms)}ms`;
}

function clearLatency() {
  const pill = document.getElementById("latencyPill");
  if (!pill) return;
  pill.hidden = true;
  pill.textContent = "";
}

function showResult(text, sources) {
  const resultText = document.getElementById("resultText");
  const sourcesWrap = document.getElementById("sources");
  const sourcesList = document.getElementById("sourcesList");

  if (resultText) resultText.textContent = text || "";

  const src = Array.isArray(sources) ? sources : [];
  if (!sourcesWrap || !sourcesList) return;

  sourcesList.innerHTML = "";
  if (src.length === 0) {
    sourcesWrap.hidden = true;
    return;
  }

  for (const s of src) {
    const li = document.createElement("li");
    li.textContent = String(s);
    sourcesList.appendChild(li);
  }
  sourcesWrap.hidden = false;
}

async function checkHealth(baseUrl) {
  const t0 = performance.now();
  const res = await fetch(`${baseUrl}/health`, { method: "GET" });
  const json = await res.json().catch(() => ({}));
  return { ok: res.ok, json, ms: performance.now() - t0 };
}

async function recommend(baseUrl, query) {
  const t0 = performance.now();
  const res = await fetch(`${baseUrl}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const json = await res.json().catch(() => ({}));
  return { ok: res.ok, json, ms: performance.now() - t0 };
}

function scrollToDemo() {
  const el = document.getElementById("demo");
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function applyExample(example) {
  const heroQueryEl = document.getElementById("heroQuery");
  const queryEl = document.getElementById("query");
  if (heroQueryEl) heroQueryEl.value = example;
  if (queryEl) queryEl.value = example;
  heroQueryEl?.focus();
}

function initExamples() {
  document.querySelectorAll("[data-example]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ex = btn.getAttribute("data-example") || "";
      applyExample(ex);
    });
  });
}

function syncQueryToHidden() {
  const heroQueryEl = document.getElementById("heroQuery");
  const queryEl = document.getElementById("query");
  if (!heroQueryEl || !queryEl) return;
  queryEl.value = String(heroQueryEl.value || "");
}

function setHeroResponse(text, enabled) {
  const el = document.getElementById("heroResponseText");
  if (el) el.textContent = String(text || "");
  document.body.classList.toggle("hasHeroResponse", Boolean(enabled));
}

function initDemo() {
  const apiBaseEl = document.getElementById("apiBaseUrl");
  const form = document.getElementById("demoForm");
  const healthBtn = document.getElementById("healthBtn");
  const heroQueryEl = document.getElementById("heroQuery");
  const heroRunBtn = document.getElementById("heroRunBtn");

  if (apiBaseEl) {
    const saved = localStorage.getItem("apiBaseUrl") || "";
    apiBaseEl.value = saved || DEFAULT_BASE_URL;
    apiBaseEl.addEventListener("change", () => {
      localStorage.setItem("apiBaseUrl", apiBaseEl.value.trim());
    });
  }

  // Keep the hidden form value in sync with the visible hero prompt.
  heroQueryEl?.addEventListener("input", () => {
    syncQueryToHidden();
    // If the user edits the prompt, show the slogan again until a new run.
    document.body.classList.remove("hasHeroResponse");
  });
  syncQueryToHidden();

  heroRunBtn?.addEventListener("click", async () => {
    syncQueryToHidden();
    const queryEl = document.getElementById("query");
    const query = String(queryEl?.value || "").trim();
    if (!query) {
      heroQueryEl?.focus();
      return;
    }

    // Submit the demo form (runs recommend()).
    form?.requestSubmit?.();
  });

  healthBtn?.addEventListener("click", async () => {
    clearLatency();
    setStatus("Checking backend health…");
    const baseUrl = normalizeBaseUrl(apiBaseEl?.value);
    try {
      const { ok, json, ms } = await checkHealth(baseUrl);
      setLatency(ms);
      if (!ok) {
        setStatus(`Health check failed (${json?.status || "error"}).`, "error");
        showResult(JSON.stringify(json, null, 2), []);
        setHeroResponse("Health check failed. See the full response below.", true);
        return;
      }
      setStatus(`Backend: ${json?.status || "ok"}`, "ok");
      showResult(
        `Backend ready.\n\nVector store: ${json?.has_vector_store ? "loaded" : "missing"}\nModel: ${
          json?.has_model ? "loaded" : "missing"
        }`,
        []
      );
      setHeroResponse("Backend is ready. Run a prompt to generate recommendations.", true);
    } catch (e) {
      setStatus("Could not reach the backend. Check the URL/CORS.", "error");
      showResult(String(e), []);
      setHeroResponse("Could not reach the backend. Check the URL/CORS and try again.", true);
    }
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearLatency();

    syncQueryToHidden();
    const queryEl = document.getElementById("query");
    const query = String(queryEl?.value || "").trim();
    if (!query) return;

    const baseUrl = normalizeBaseUrl(apiBaseEl?.value);

    heroRunBtn?.setAttribute("disabled", "true");
    setStatus("Generating recommendations…");
    showResult("Working…", []);
    setHeroResponse("Generating…", true);

    // Clear the visible prompt immediately after starting a run.
    if (heroQueryEl) heroQueryEl.value = "";
    if (queryEl) queryEl.value = "";

    try {
      const { ok, json, ms } = await recommend(baseUrl, query);
      setLatency(ms);

      if (!ok) {
        const detail = json?.detail ? `\n\n${String(json.detail)}` : "";
        setStatus("Request failed. See response for details.", "error");
        showResult(`Error (${json?.status || "HTTP error"}).${detail}`, []);
        setHeroResponse(`Error (${json?.status || "HTTP error"}).`, true);
        return;
      }

      setStatus("Done.", "ok");
      const responseText = String(json?.response || "");
      showResult(responseText, json?.sources || []);
      setHeroResponse(responseText || "Done.", true);
    } catch (err) {
      setStatus("Network error. Check the URL/CORS.", "error");
      showResult(String(err), []);
      setHeroResponse("Network error. Check the URL/CORS and try again.", true);
    } finally {
      heroRunBtn?.removeAttribute("disabled");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initExamples();
  initDemo();
});