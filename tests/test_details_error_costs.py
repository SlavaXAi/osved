from types import SimpleNamespace

import pytest

from app.config import Settings
from app.details.research import (
    DetailBudgetError,
    DetailEvidenceError,
    DetailGenerationError,
    build_detail_article,
)


@pytest.fixture
def stages(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    service = SimpleNamespace(
        collect=lambda *args, **kwargs: SimpleNamespace(cost_usd=0.003, documents=[])
    )
    monkeypatch.setattr(
        "app.details.research.build_detail_fact_snapshot",
        lambda *args, **kwargs: SimpleNamespace(cost_usd=0.004, snapshot=object()),
    )
    monkeypatch.setattr(
        "app.details.research.generate_deep_article",
        lambda *args, **kwargs: SimpleNamespace(cost_usd=0.005, draft=object()),
    )
    return settings, service


@pytest.mark.parametrize("stage", ["facts", "writing"])
@pytest.mark.parametrize(
    "error_type", [DetailBudgetError, DetailGenerationError, DetailEvidenceError]
)
def test_stage_failure_retains_its_type_and_all_preceding_costs(
    stages, monkeypatch, stage, error_type
):
    settings, service = stages
    argument = "budget failure" if error_type is DetailBudgetError else ["stage failure"]
    original_error = error_type(argument, cost_usd=0.002)

    def fail(*args, **kwargs):
        raise original_error

    target = "build_detail_fact_snapshot" if stage == "facts" else "generate_deep_article"
    monkeypatch.setattr(f"app.details.research.{target}", fail)
    with pytest.raises(error_type) as raised:
        build_detail_article(settings, object(), 491, research_service=service)
    assert raised.value is original_error
    assert raised.value.cost_usd == pytest.approx(0.005 if stage == "facts" else 0.009)


@pytest.mark.parametrize("cap,total", [(0.002, 0.003), (0.006, 0.007), (0.011, 0.012)])
def test_boundary_budget_failure_reports_spent_cost(stages, cap, total):
    settings, service = stages
    settings = settings.model_copy(update={"details_max_cost_usd": cap})
    with pytest.raises(DetailBudgetError) as raised:
        build_detail_article(settings, object(), 491, research_service=service)
    assert raised.value.cost_usd == pytest.approx(total)


def test_paid_final_source_gate_failure_reports_full_request_cost(stages, monkeypatch):
    settings, service = stages
    monkeypatch.setattr("app.details.research.render_detail_article_html", lambda *_: "article")
    monkeypatch.setattr("app.details.research.source_urls_for_draft", lambda *_: ["one-source"])
    with pytest.raises(DetailEvidenceError) as raised:
        build_detail_article(settings, object(), 491, research_service=service)
    assert raised.value.cost_usd == pytest.approx(0.012)
