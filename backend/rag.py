"""The retrieval-augmented generation pipeline.

Retrieval happens once, in the route, and the resulting documents are passed in
here - so the `sources` returned to the client are provably the same documents
the model was shown.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from generator import GenerationError, GenerationResult, Generator
from index_store import MenuDocument
from schemas import RECOMMENDATION_JSON_SCHEMA, Recommendation

SYSTEM_PROMPT = (
    "You are a food recommendation assistant for a restaurant menu search engine.\n"
    "\n"
    "You will be given a numbered list of menu items retrieved from the database, "
    "and a diner's request. Recommend up to three items that best match the "
    "request.\n"
    "\n"
    "Rules you must follow:\n"
    "1. Recommend ONLY dishes that appear in the provided menu items. Never invent "
    "a dish, a restaurant, a price, or a location, and never recommend something "
    "you were not shown.\n"
    "2. Copy dish and restaurant names exactly as they appear in the menu items.\n"
    "3. If none of the provided items reasonably match the request, return an empty "
    "recommendations list and explain in the message that nothing on the available "
    "menus fits. Do not offer the closest thing as if it were a match.\n"
    "4. For dietary_notes, state only what the menu description itself supports. "
    "If the description says nothing about dietary properties, use an empty string. "
    "Never infer that something is vegan, gluten-free, or allergen-free.\n"
    "5. Keep each 'why' to one or two friendly sentences explaining the fit.\n"
    "6. The 'message' is one short sentence framing the results for the diner."
)


def build_context(documents: list[MenuDocument]) -> str:
    """Render retrieved documents as a numbered list for the prompt."""
    return "\n".join(
        f"{i}. Restaurant: {doc.restaurant} | Category: {doc.category} | "
        f"Item: {doc.item} | Description: {doc.description}"
        for i, doc in enumerate(documents, start=1)
    )


def build_user_prompt(query: str, documents: list[MenuDocument]) -> str:
    return f"Menu items:\n{build_context(documents)}\n\nDiner's request: {query}"


def run_rag(
    query: str,
    documents: list[MenuDocument],
    generator: Generator,
) -> tuple[list[Recommendation], str, GenerationResult]:
    """Generate grounded recommendations from already-retrieved documents."""
    if not documents:
        raise GenerationError("No menu items were retrieved.", status_code=503)

    result = generator.generate(
        SYSTEM_PROMPT,
        build_user_prompt(query, documents),
        json_schema=RECOMMENDATION_JSON_SCHEMA,
    )

    recommendations, message = _parse(result.text)
    return recommendations, message, result


def _parse(raw: str) -> tuple[list[Recommendation], str]:
    """Parse the model's JSON. Structured output is requested but not guaranteed:
    not every model on OpenRouter enforces the schema, so validate regardless."""
    text = raw.strip()
    if text.startswith("```"):  # some models wrap JSON in a fenced block
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            "Model returned malformed JSON.", status_code=502
        ) from exc

    if not isinstance(payload, dict):
        raise GenerationError("Model returned unexpected JSON.", status_code=502)

    try:
        recommendations = [
            Recommendation.model_validate(item)
            for item in payload.get("recommendations", [])
        ]
    except ValidationError as exc:
        raise GenerationError(
            "Model returned recommendations in an unexpected shape.", status_code=502
        ) from exc

    message = str(payload.get("message") or "").strip()
    if not message:
        message = (
            "Here are some dishes that match your request."
            if recommendations
            else "Nothing on the available menus matches that request."
        )
    return recommendations, message
