import pytest

from app.providers.base import UsageInfo
from app.services.pricing import calculate_cost


class TestCalculateCost:
    def test_known_model_computes_cost(self):
        usage = UsageInfo(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        cost = calculate_cost("openai", "gpt-4o-mini", usage)
        assert cost == pytest.approx(0.75)  # $0.15 + $0.60 per 1M tokens

    def test_ollama_is_always_free_even_with_usage(self):
        usage = UsageInfo(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        assert calculate_cost("ollama", "llama3.1", usage) == 0.0

    def test_ollama_is_free_even_with_no_usage(self):
        assert calculate_cost("ollama", "llama3.1", None) == 0.0

    def test_unmapped_model_returns_none_not_zero(self):
        usage = UsageInfo(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert calculate_cost("openai", "some-future-model", usage) is None

    def test_no_usage_returns_none(self):
        assert calculate_cost("openai", "gpt-4o-mini", None) is None
