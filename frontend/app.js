// Foodify frontend.
//
// The API is served from the same origin as this page, so requests are relative
// and there is no backend URL to configure, no CORS handshake, and nothing to
// go stale when the deployment moves.

const ENDPOINT = "/recommend";

function el(id) {
  return document.getElementById(id);
}

function setStatus(text, kind = "info") {
  const node = el("status");
  if (!node) return;
  node.textContent = text || "";
  node.dataset.kind = kind;
}

function setGenerating(isGenerating) {
  document.body.classList.toggle("isGenerating", Boolean(isGenerating));
  const button = el("heroRunBtn");
  if (button) button.disabled = Boolean(isGenerating);
}

function setLatency(ms) {
  const pill = el("latencyPill");
  if (!pill) return;
  pill.hidden = false;
  pill.textContent = `${(ms / 1000).toFixed(1)}s`;
}

function clearResults() {
  const list = el("resultList");
  if (list) list.innerHTML = "";
  const sources = el("sources");
  if (sources) sources.hidden = true;
}

function renderMessage(text) {
  const node = el("resultMessage");
  if (node) node.textContent = text || "";
}

function renderRecommendations(recommendations) {
  const list = el("resultList");
  if (!list) return;
  list.innerHTML = "";

  if (!recommendations.length) {
    const empty = document.createElement("li");
    empty.className = "recCard recCard--empty";
    empty.textContent =
      "No dishes on the available menus match that. Try describing a flavour, ingredient or style.";
    list.appendChild(empty);
    return;
  }

  for (const rec of recommendations) {
    const card = document.createElement("li");
    card.className = "recCard";

    const dish = document.createElement("h4");
    dish.className = "recCard__dish";
    dish.textContent = rec.dish;

    const restaurant = document.createElement("div");
    restaurant.className = "recCard__restaurant";
    restaurant.textContent = rec.restaurant;

    const why = document.createElement("p");
    why.className = "recCard__why";
    why.textContent = rec.why;

    card.append(dish, restaurant, why);

    if (rec.dietary_notes) {
      const notes = document.createElement("div");
      notes.className = "recCard__diet";
      notes.textContent = rec.dietary_notes;
      card.appendChild(notes);
    }

    list.appendChild(card);
  }
}

function renderSources(sources) {
  const wrap = el("sources");
  const list = el("sourcesList");
  if (!wrap || !list) return;

  list.innerHTML = "";
  if (!sources || !sources.length) {
    wrap.hidden = true;
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = `${source.item} — ${source.restaurant}`;
    const description = document.createElement("span");
    description.textContent = ` · ${source.description}`;
    item.append(name, description);
    list.appendChild(item);
  }
  wrap.hidden = false;
}

function scrollToResults() {
  const anchor = el("resultAnchor");
  if (!anchor) return;
  requestAnimationFrame(() =>
    requestAnimationFrame(() =>
      anchor.scrollIntoView({ behavior: "smooth", block: "start" })
    )
  );
}

async function recommend(query, signal) {
  const started = performance.now();
  const response = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  const body = await response.json().catch(() => ({}));
  return { ok: response.ok, status: response.status, body, ms: performance.now() - started };
}

function errorMessageFor(status, body) {
  if (status === 422) return "That query is too long or empty. Try something shorter.";
  if (status === 429) return "Too many requests just now. Give it a moment and try again.";
  if (status === 503) return "The service is still starting up. Try again in a few seconds.";
  if (status === 504) return "That took too long. Try again.";
  if (body && typeof body.detail === "string") return body.detail;
  return "Something went wrong. Please try again.";
}

let inFlight = null;

function initDemo() {
  const input = el("heroQuery");
  const button = el("heroRunBtn");
  const form = el("demoForm");

  async function run() {
    const query = String(input?.value || "").trim();
    if (!query) {
      input?.focus();
      return;
    }

    // A new search supersedes one still running.
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();

    setGenerating(true);
    setStatus("Searching the menus…");
    clearResults();
    renderMessage("");

    try {
      const { ok, status, body, ms } = await recommend(query, inFlight.signal);
      setLatency(ms);

      if (!ok) {
        setStatus(errorMessageFor(status, body), "error");
        renderMessage("");
        renderRecommendations([]);
        scrollToResults();
        return;
      }

      setStatus("", "ok");
      renderMessage(body.message || "");
      renderRecommendations(body.recommendations || []);
      renderSources(body.sources || []);
      scrollToResults();
    } catch (error) {
      if (error.name === "AbortError") return;
      setStatus("Could not reach the service. Check your connection.", "error");
    } finally {
      setGenerating(false);
      inFlight = null;
    }
  }

  button?.addEventListener("click", run);
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    run();
  });
  input?.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") run();
  });

  document.querySelectorAll("[data-example]").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (input) {
        input.value = chip.getAttribute("data-example") || "";
        input.focus();
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", initDemo);
