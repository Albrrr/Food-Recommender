"""Request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from config import get_settings


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def _within_length_limit(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be empty.")
        limit = get_settings().max_query_length
        if len(stripped) > limit:
            raise ValueError(f"Query must be {limit} characters or fewer.")
        return stripped


class Recommendation(BaseModel):
    restaurant: str
    dish: str
    why: str
    dietary_notes: str | None = None


class Source(BaseModel):
    """A retrieved menu item that was actually shown to the model."""

    restaurant: str
    category: str
    item: str
    description: str
    score: float


class QueryResponse(BaseModel):
    recommendations: list[Recommendation]
    message: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    reranker_loaded: bool
    generator_configured: bool
    document_count: int | None = None
    embed_model: str | None = None
    llm_model: str | None = None
    error: str | None = None


# JSON schema handed to the model for structured output. Kept next to the
# Pydantic models so the two cannot drift apart unnoticed.
RECOMMENDATION_JSON_SCHEMA: dict = {
    "name": "food_recommendations",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "restaurant": {"type": "string"},
                        "dish": {"type": "string"},
                        "why": {"type": "string"},
                        "dietary_notes": {"type": "string"},
                    },
                    "required": ["restaurant", "dish", "why", "dietary_notes"],
                    "additionalProperties": False,
                },
            },
            "message": {"type": "string"},
        },
        "required": ["recommendations", "message"],
        "additionalProperties": False,
    },
}
