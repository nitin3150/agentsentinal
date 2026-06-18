import os
import click
from dotenv import load_dotenv

load_dotenv()

VALID_COMPLIANCE = {"hipaa", "soc2", "owasp", "pii", "All"}


def check_env() -> None:
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("GROQ_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or os.getenv("NVIDIA_MODEL")
    )
    if not model:
        raise click.ClickException(
            "LLM_MODEL is not set.\n"
            "  Example: export LLM_MODEL=groq/llama-3.3-70b-versatile\n"
            "           export LLM_API_KEY=your_key"
        )


def parse_compliance(ctx, param, value: str) -> list[str]:
    if not value:
        return []
    standards = [s.strip() for s in value.split(",")]
    invalid = set(standards) - VALID_COMPLIANCE
    if invalid:
        raise click.BadParameter(
            f"Unknown compliance standards: {', '.join(sorted(invalid))}. "
            f"Valid options: {', '.join(sorted(VALID_COMPLIANCE))}"
        )
    return standards
