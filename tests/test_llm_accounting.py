import pytest

from app.config import Settings
from app.llm.accounting import resolve_usage_cost
from app.llm.openrouter import OpenRouterCardGenerator, OpenRouterResponseError
from app.research.facts import OpenRouterFactSheetGenerator


@pytest.mark.parametrize(
    "byok,router,upstream,expected,basis",
    [
        (True, 0, 0.0029584, 0.0029584, "openrouter_plus_byok_upstream"),
        (True, 0.0001, 0.002, 0.0021, "openrouter_plus_byok_upstream"),
        (False, 0.003, 0.002, 0.003, "openrouter_reported"),
        (False, 0, 0.002, 0, "openrouter_reported"),
    ],
)
def test_reported_receipts_respect_byok_boundary(byok, router, upstream, expected, basis):
    result = resolve_usage_cost(
        {
            "is_byok": byok,
            "cost": router,
            "prompt_tokens": 6000,
            "completion_tokens": 200,
            "cost_details": {"upstream_inference_cost": upstream},
        },
        model="unknown/model",
    )
    assert result.cost_usd == pytest.approx(expected)
    assert result.billing_basis == basis
    assert not result.estimated
    assert not result.unavailable


@pytest.mark.parametrize("reported", [None, float("nan"), -1, True])
def test_missing_or_invalid_cost_is_estimated_only_for_known_model(reported):
    usage = {"cost": reported, "prompt_tokens": 1000, "completion_tokens": 100}
    known = resolve_usage_cost(usage, model="openai/gpt-4.1-mini")
    unknown = resolve_usage_cost(usage, model="unknown/expensive")
    assert known.cost_usd == pytest.approx(0.00056)
    assert known.estimated
    assert unknown.cost_usd == 0
    assert unknown.unavailable
    assert not unknown.estimated


def test_ambiguous_zero_with_tokens_is_not_confused_with_explicit_free_nonbyok():
    usage = {"cost": 0, "prompt_tokens": 1000, "completion_tokens": 100}
    ambiguous = resolve_usage_cost(usage, model="openai/gpt-4.1-mini")
    explicit = resolve_usage_cost({**usage, "is_byok": False}, model="openai/gpt-4.1-mini")
    assert ambiguous.estimated and ambiguous.cost_usd > 0
    assert explicit.cost_usd == 0 and not explicit.estimated and not explicit.unavailable


def test_unknown_model_uses_explicit_enforced_price_ceiling_but_not_arbitrary_reservation():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    priced = resolve_usage_cost(usage, model="custom/model", price_ceiling_per_million=(0.5, 2))
    unpriced = resolve_usage_cost({}, model="custom/model", reservation_usd=0.05)
    reserved = resolve_usage_cost({}, model="openai/gpt-4.1-mini", reservation_usd=0.01)
    assert priced.cost_usd == pytest.approx(0.0007)
    assert priced.estimated
    assert unpriced.unavailable and unpriced.cost_usd == 0
    assert reserved.estimated and reserved.cost_usd == 0.01


def test_missing_byok_component_keeps_known_partial_inference_cost():
    usage = {
        "is_byok": True,
        "cost_details": {"upstream_inference_cost": 0.03},
        "prompt_tokens": 1000,
        "completion_tokens": 100,
    }
    known = resolve_usage_cost(usage, model="openai/gpt-4.1-mini")
    unknown = resolve_usage_cost(usage, model="custom/model")
    assert known.cost_usd >= 0.03
    assert unknown.cost_usd == 0.03 and unknown.unavailable


@pytest.mark.parametrize("kind", ["card", "facts"])
def test_generator_preserves_byok_cost_on_invalid_paid_response(kind):
    def transport(payload, key):
        return {
            "id": "byok-response",
            "model": "openai/gpt-4.1-mini",
            "usage": {
                "cost": 0,
                "is_byok": True,
                "prompt_tokens": 1000,
                "completion_tokens": 200,
                "cost_details": {"upstream_inference_cost": 0.003},
            },
            "choices": [{"message": {"content": "{}"}}],
        }

    settings = Settings(_env_file=None, app_env="test", openrouter_api_key="fake-key")
    with pytest.raises(OpenRouterResponseError) as caught:
        if kind == "card":
            OpenRouterCardGenerator(settings, transport=transport).generate(
                source_name="Example",
                source_role="radar",
                published_at="2026-09-07T00:00:00+00:00",
                original_text="A real saved source statement.",
                known_urls=["https://example.com"],
                model="openai/gpt-4.1-mini",
            )
        else:
            OpenRouterFactSheetGenerator(settings, transport=transport).generate(
                event={"headline": "An event"}, documents=[], model="openai/gpt-4.1-mini"
            )
    assert caught.value.cost_usd == 0.003
    assert caught.value.prompt_tokens == 1000
    assert caught.value.billing_metadata["billing_basis"] == "openrouter_plus_byok_upstream"
    assert not caught.value.billing_metadata["cost_estimated"]
