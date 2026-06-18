import asyncio
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("GROQ_MODEL")
    or os.getenv("OPENROUTER_MODEL")
    or os.getenv("NVIDIA_MODEL")
    or "groq/llama-3.3-70b-versatile"
)

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


async def call_llm(
    prompt: str,
    *,
    model: str = DEFAULT_LLM_MODEL,
    timeout: float = LLM_TIMEOUT,
    temperature: float = 0,
) -> str | None:
    """Single agnostic LiteLLM call. Provider determined by model string prefix.

    Returns response text or None on timeout/error. Never raises.
    """
    import litellm
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
        or None
    )
    api_base = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("NVIDIA_BASE_URL")
        or os.getenv("GROQ_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or None
    )
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **({"api_key": api_key} if api_key else {}),
                **({"api_base": api_base} if api_base else {}),
            ),
            timeout=timeout,
        )
        return response.choices[0].message.content  # type: ignore[union-attr]
    except asyncio.TimeoutError:
        logger.warning("LLM call timed out after %.0fs (model=%s)", timeout, model)
        return None
    except Exception as exc:
        logger.warning("LLM call failed: %s: %s", type(exc).__name__, exc)
        return None
