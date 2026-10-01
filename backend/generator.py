"""Text generation via OpenRouter.

Called through the `openai` SDK against OpenRouter's OpenAI-compatible endpoint,
so the provider is a configuration value rather than an architectural decision -
pointing LLM/base-url elsewhere needs no code change.

Everything the API can raise is translated into GenerationError with a sensible
HTTP status, so the route layer never has to know which SDK is underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import openai

from config import Settings


class GenerationError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


class Generator(Protocol):
    def generate(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> GenerationResult: ...


class OpenRouterGenerator:
    def __init__(self, settings: Settings) -> None:
        self._client = openai.OpenAI(
            api_key=settings.openrouter_api_key.get_secret_value(),
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._provider_sort = settings.llm_provider_sort

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "extra_body": {
                # Usage accounting - returns real cost alongside token counts.
                "usage": {"include": True},
                **(
                    {"provider": {"sort": self._provider_sort}}
                    if self._provider_sort != "default"
                    else {}
                ),
            },
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise GenerationError(
                "Upstream rejected our credentials.", status_code=502
            ) from exc
        except openai.BadRequestError as exc:
            raise GenerationError(
                "Upstream rejected the request.", status_code=502
            ) from exc
        except openai.RateLimitError as exc:
            raise GenerationError(
                "Upstream is rate limiting; try again shortly.", status_code=503
            ) from exc
        except openai.APITimeoutError as exc:
            raise GenerationError("Upstream timed out.", status_code=504) from exc
        except openai.APIConnectionError as exc:
            raise GenerationError("Could not reach upstream.", status_code=502) from exc
        except openai.APIStatusError as exc:
            raise GenerationError(
                f"Upstream returned {exc.status_code}.", status_code=502
            ) from exc

        if not response.choices:
            raise GenerationError("Upstream returned no choices.", status_code=502)

        text = response.choices[0].message.content or ""
        if not text.strip():
            raise GenerationError("Upstream returned empty content.", status_code=502)

        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=text.strip(),
            model=getattr(response, "model", self._model),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            cost_usd=getattr(usage, "cost", None),
        )
