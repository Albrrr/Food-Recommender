# Foodify

Retrieval-augmented restaurant recommendations over a corpus of **3,201 dishes
from 75 San Diego restaurants**. Ask for what you feel like eating; get real
dishes off real menus, with the retrieved source items shown alongside.

```
query → Qwen3 embedding → FAISS (top 30) → cross-encoder rerank (top 6) → LLM → cards
```

End to end in **~2.7s**, at roughly **$0.0005** per query.

## How it works

**Retrieval.** Every menu item is embedded with
[`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
(1024-dim) and stored in a FAISS inner-product index over L2-normalised
vectors, so inner product is cosine similarity. A query pulls 30 candidates in
40–125ms.

**Reranking.** The bi-encoder embeds queries and documents independently, so it
can only tell whether two vectors are near each other — not whether a dish
answers the request. Menu descriptions are short and share heavy vocabulary
("grilled", "fresh", "house-made"), which produces clusters of near-ties where
picking a top 3 is close to arbitrary. A cross-encoder
(`Xenova/ms-marco-MiniLM-L-6-v2`) reads query and document *together* and
rescores the 30 candidates down to 6, in 50–85ms.

It earns its place. On *"something with raw fish"* the vector search returns
Korean Steak Tartare in its top 3; the reranker drops it and promotes Peruvian
Crudo.

**Generation.** The 6 reranked items go to an LLM through OpenRouter under a
grounding constraint: recommend only dishes present in the context, and say so
plainly when nothing matches rather than offering the nearest thing. Asked for
a *"gold leaf sushi burrito with truffle caviar foam"*, it returns zero
recommendations and says none of the available menus have it. Output is a
validated JSON schema, rendered as cards.

## Running it

```bash
cp .env.example .env          # only OPENROUTER_API_KEY is required
pip install -r backend/requirements.txt
python scripts/build_index.py # ~40s, writes backend/faiss_store/
uvicorn app:app --app-dir backend --port 8000
```

Open <http://localhost:8000> — the frontend is served from the same origin as
the API, so there is no backend URL to configure and no CORS to misconfigure.

With Docker:

```bash
docker build -t foodify .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-... foodify
```

## API

| | |
|---|---|
| `GET /health` | readiness of the index, reranker and generator |
| `POST /recommend` | `{"query": "..."}` → recommendations + sources |

Every response carries an `X-Request-ID`, and every request emits one
structured JSON log line with per-stage timings and token usage:

```json
{"event":"recommend.ok","request_id":"d3a6f885837f","retrieve_ms":50.7,
 "rerank_ms":84.9,"generate_ms":2544.6,"total_ms":2680.2,
 "prompt_tokens":549,"completion_tokens":198,"cost_usd":0.00060}
```

Timing the stages separately is not decoration. The first live run took **24
seconds** — and because each stage is timed independently, the log showed at a
glance that retrieval and reranking were fine and all of it was upstream
generation: OpenRouter had routed to a provider serving ~8 tokens/second.
Setting `LLM_PROVIDER_SORT=throughput` brought it to 2.7s, for about 4× the
token price and still a twentieth of a cent per query.

Without per-stage timings that is a vague "the app feels slow" and a long
afternoon.

## Configuration

All settings are validated at startup, so a misconfigured deployment fails
immediately with a message naming the field rather than midway through a
request. Blank entries in a `.env` are treated as unset so defaults apply. See
[`.env.example`](.env.example).

Swapping the embedding model is a config change, but it invalidates the index —
the index records the model and dimension it was built with and refuses to load
against a mismatch. Rebuild with `scripts/build_index.py`.

To drop the torch dependency, set `EMBED_BACKEND=fastembed` and
`EMBED_MODEL=BAAI/bge-base-en-v1.5` (ONNX, ~0.2GB instead of ~1.2GB) and
rebuild.

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest                  # 26 tests, fully offline, ~0.1s
pytest -m integration   # opt-in: downloads the reranker model
```

The suite runs with no network and no API calls: the embedding backend is faked
with deterministic hashed vectors so the real FAISS paths are still exercised,
and the generator is injected through FastAPI's dependency overrides.

## Notes on the dataset

`DATA.json` nests unevenly. Most categories map item → description, but around
300 insert a subgroup level (`SUSHI ROLLS → Classic Rolls → California`). The
parser recurses, recovering 246 items that a flat parser silently drops — about
8% of the corpus, missing from every previous build. A further 85 entries are
the same dish listed under multiple categories and are deduplicated, so one
dish cannot occupy several recommendation slots.

## Next

An evaluation harness: ~40 queries with known-correct dishes, measuring
recall@k and MRR, so retrieval changes can be justified with numbers rather
than spot checks.
