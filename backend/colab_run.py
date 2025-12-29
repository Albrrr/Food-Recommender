"""
Colab-friendly launcher for the FastAPI backend.

Usage (in Colab):
  !pip install -r backend/requirements.txt
  %env HF_TOKEN=hf_...
  %env PORT=8000
  # optional:
  %env NGROK_AUTHTOKEN=...
  %env USE_NGROK=1
  !python backend/colab_run.py

It will print a public URL if ngrok is enabled.
"""

from __future__ import annotations

import os

import uvicorn


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def maybe_start_ngrok(port: int) -> str | None:
    # Only import if needed; keeps local runs simpler.
    if not _truthy(os.environ.get("USE_NGROK")) and not os.environ.get("NGROK_AUTHTOKEN"):
        return None

    from pyngrok import ngrok

    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if authtoken:
        ngrok.set_auth_token(authtoken)

    # Bind to the same port uvicorn will listen on.
    tunnel = ngrok.connect(addr=port, proto="http")
    return tunnel.public_url


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    public_url = maybe_start_ngrok(port)
    if public_url:
        print(f"Public URL: {public_url}")
        print("Set your frontend API base to this URL (no trailing slash).")

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

