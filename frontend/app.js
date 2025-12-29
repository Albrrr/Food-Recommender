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

function setGenerating(isGenerating, text) {
  const el = document.getElementById("heroLoadingText");
  if (el && typeof text !== "undefined") el.textContent = String(text || "");
  document.body.classList.toggle("isGenerating", Boolean(isGenerating));
}

function scrollToResult() {
  const anchor = document.getElementById("resultAnchor");
  if (!anchor) return;
  // Wait for DOM paint so layout height is correct.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      anchor.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
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
    // If the user edits the prompt, ensure we are not in generating state.
    document.body.classList.remove("isGenerating");
  });
  syncQueryToHidden();

  async function runRecommendation() {
    const query = String(heroQueryEl?.value || "").trim();
    if (!query) {
      heroQueryEl?.focus();
      return;
    }

    const baseUrl = normalizeBaseUrl(apiBaseEl?.value);

    heroRunBtn?.setAttribute("disabled", "true");
    setStatus("Generating recommendations…");
    showResult("Working…", []);
    setGenerating(true, "Generating recommendations…");

    try {
      const { ok, json, ms } = await recommend(baseUrl, query);
      setLatency(ms);

      if (!ok) {
        const detail = json?.detail ? `\n\n${String(json.detail)}` : "";
        setStatus("Request failed. See response for details.", "error");
        showResult(`Error (${json?.status || "HTTP error"}).${detail}`, []);
        // Restore slogan and scroll to error output.
        setGenerating(false);
        scrollToResult();
        return;
      }

      setStatus("Done.", "ok");
      const responseText = String(json?.response || "");
      showResult(responseText, json?.sources || []);
      // Restore slogan, clear prompt, then scroll to the full response.
      setGenerating(false);
      if (heroQueryEl) heroQueryEl.value = "";
      syncQueryToHidden();
      scrollToResult();
    } catch (err) {
      setStatus("Network error. Check the URL/CORS.", "error");
      showResult(String(err), []);
      setGenerating(false);
      scrollToResult();
    } finally {
      heroRunBtn?.removeAttribute("disabled");
    }
  }

  heroRunBtn?.addEventListener("click", runRecommendation);

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
        return;
      }
      setStatus(`Backend: ${json?.status || "ok"}`, "ok");
      showResult(
        `Backend ready.\n\nVector store: ${json?.has_vector_store ? "loaded" : "missing"}\nModel: ${
          json?.has_model ? "loaded" : "missing"
        }`,
        []
      );
    } catch (e) {
      setStatus("Could not reach the backend. Check the URL/CORS.", "error");
      showResult(String(e), []);
    }
  });

  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearLatency();
    await runRecommendation();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initExamples();
  initDemo();
});
