"""Static per-model pricing table + cost calculation.

Hardcoded USD-per-1M-token rates for known models. An unmapped (provider, model)
pair yields cost_usd=None — unknown, never a guessed 0 — since silently looking
free when the price just isn't in the table yet would be worse than admitting
the gateway doesn't know.
"""

from dataclasses import dataclass

from app.providers.base import UsageInfo


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


PRICING_TABLE: dict[tuple[str, str], ModelPricing] = {
    ("openai", "gpt-4o"): ModelPricing(2.50, 10.00),
    ("openai", "gpt-4o-mini"): ModelPricing(0.15, 0.60),
    ("openai", "gpt-4-turbo"): ModelPricing(10.00, 30.00),
    ("openai", "gpt-3.5-turbo"): ModelPricing(0.50, 1.50),
    ("anthropic", "claude-opus-4-8"): ModelPricing(5.00, 25.00),
    ("anthropic", "claude-sonnet-5"): ModelPricing(3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): ModelPricing(1.00, 5.00),
    ("gemini", "gemini-1.5-flash"): ModelPricing(0.075, 0.30),
    ("gemini", "gemini-1.5-pro"): ModelPricing(1.25, 5.00),
    ("gemini", "gemini-2.0-flash"): ModelPricing(0.10, 0.40),
}


def calculate_cost(provider: str, model: str, usage: UsageInfo | None) -> float | None:
    if provider == "ollama":
        return 0.0  # local/free, always — never looked up in the table

    if usage is None:
        return None

    pricing = PRICING_TABLE.get((provider, model))
    if pricing is None:
        return None

    return (
        usage.prompt_tokens / 1_000_000 * pricing.input_per_million
        + usage.completion_tokens / 1_000_000 * pricing.output_per_million
    )
