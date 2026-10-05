"""Local LLM backend — OpenAI or Gemini, selected by LLM_PROVIDER."""
import time
from typing import Any, Dict, List

from ..config import (
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _gemini_models() -> List[str]:
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    return models


def gemini_generate(contents: Any, config: Dict, attempts_per_model: int = 2) -> str:
    """Call Gemini, retrying overloaded models and falling back to the next one.

    Shared by highlight ranking and Gemini transcription.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import errors as genai_errors  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    last_error: Exception = RuntimeError("no Gemini models configured")
    for model in _gemini_models():
        for attempt in range(attempts_per_model):
            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                return response.text or ""
            except genai_errors.APIError as e:
                last_error = e
                # 503 overloaded / 429 rate limited: retry, then try the next model.
                # 404: model retired for this key, go straight to the next model.
                if e.code == 404:
                    print(f"[llm/gemini] {model} unavailable (404), trying next model", flush=True)
                    break
                if e.code not in (429, 500, 503):
                    raise
                print(f"[llm/gemini] {model} returned {e.code}, retrying", flush=True)
                time.sleep(3 * (attempt + 1))
    raise last_error


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    return gemini_generate(
        prompt,
        {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai' or 'gemini'."
    )
