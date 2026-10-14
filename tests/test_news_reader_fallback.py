from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import pytest
from test_details_research import fact_documents, fact_sheet, valid_article

from app.config import Settings, SourceConfig, SourceRegistry
from app.db.repository import Repository
from app.details.research import (
    BoundedDetailResearchService,
    OpenRouterDeepArticleGenerator,
    build_detail_article,
)
from app.research.facts import OpenRouterFactSheetGenerator
from app.research.web import ScrapeResponse, SearchResponse

BAD = "https://reuters.com/atlas-unavailable"
GOOD = [
    "https://theverge.com/atlas-analysis",
    "https://arstechnica.com/atlas-methodology",
    "https://ft.com/atlas-industry",
]


@pytest.fixture
def fallback_env(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        allowed_user_id=123,
        openrouter_api_key="offline",
        openrouter_model="openai/gpt-5.6-luna",
        openrouter_card_model="openai/gpt-5.6-luna",
    )
    repo = Repository(settings.database_path)
    repo.migrate()
    repo.sync_sources(
        SourceRegistry(
            version=1,
            sources=[
                SourceConfig(
                    id="fixture",
                    name="Fixture",
                    kind="rss",
                    locator="https://openai.com/rss",
                    role="primary",
                    region="global",
                    language="en",
                )
            ],
        )
    )
    delivery = repo.create_smoke_delivery(123, "https://openai.com/atlas")
    event_id = delivery["event_id"]
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET headline=?,event_type='software_release' WHERE id=?",
            ("OpenAI выпустила Atlas 2 для корпоративной автоматизации", event_id),
        )
    original = fact_documents()[0]
    doc_id = repo.upsert_research_document(
        canonical_url=original["url"],
        url=original["url"],
        title="OpenAI Atlas 2 official release",
        publisher="OpenAI",
        source_class="primary",
        retrieved_at=datetime.now(UTC).isoformat(),
        expires_at=(datetime.now(UTC) + timedelta(days=7)).isoformat(),
        content_text=original["content_text"],
        metadata={"content_kind": "scraped_fulltext"},
    )
    repo.attach_research_document(
        event_id=event_id, research_document_id=doc_id, evidence_role="primary", relevance=1.0
    )
    ctx = {
        "baseline_card_id": delivery["card_id"],
        "baseline_text": "OpenAI выпустила Atlas 2 для корпоративной автоматизации. " * 5,
        "baseline_document_ids": [doc_id],
        "baseline_source_urls": [original["url"]],
        "context_fingerprint": "c" * 64,
    }
    return settings, repo, event_id, ctx


class FailingThenUsefulWeb:
    def __init__(self, failure):
        self.failure = failure
        self.search_queries = []
        self.scrape_urls = []

    def search(self, query, *, freshness, num_results):
        self.search_queries.append(query)
        return SearchResponse(
            query=query,
            results=[
                {
                    "url": url,
                    "title": "OpenAI Atlas 2 independent report",
                    "description": "A documentary account of Atlas capabilities and constraints.",
                    "relevance": 1.0,
                }
                for url in [BAD, *GOOD]
            ],
            cost_usd=0.00036,
        )

    def scrape(self, url):
        self.scrape_urls.append(url)
        if url == BAD:
            if self.failure == "timeout":
                raise TimeoutError("provider outcome unknown")
            if self.failure == "challenge200":
                return ScrapeResponse(
                    url=url,
                    markdown="# 403 Forbidden\nAccess to this page is denied. "
                    "Please verify your browser.",
                    metadata={"canonicalUrl": url, "title": "OpenAI Atlas 2 report"},
                    cost_usd=0.0009,
                )
            raise HTTPError(url, 403, "Forbidden", None, None)
        if url == GOOD[0]:
            text = fact_documents()[1]["content_text"]
        elif url == GOOD[1]:
            text = (
                "OpenAI Atlas 2 methodology report explains how evaluation tasks were designed. "
                * 20
            )
        else:
            text = (
                "OpenAI Atlas 2 industry report discusses corporate adoption "
                "and operational constraints. " * 20
            )
        return ScrapeResponse(
            url=url,
            markdown=text,
            metadata={"canonicalUrl": url, "title": "OpenAI Atlas 2 independent report"},
            cost_usd=0.0009,
        )


class GroundedOfflineLLM:
    def __init__(self):
        self.calls = []
        self.primary_id = None
        self.independent_id = None

    def __call__(self, payload, api_key):
        self.calls.append(payload)
        properties = payload["response_format"]["json_schema"]["schema"]["properties"]
        if "facts" in properties:
            content = payload["messages"][1]["content"]
            evidence = json.JSONDecoder().raw_decode(content[content.index("{") :])[0]
            self.primary_id = next(
                d["id"] for d in evidence["documents"] if d["source_class"] == "primary"
            )
            self.independent_id = next(
                d["id"]
                for d in evidence["documents"]
                if "72" in d["content"] and d["source_class"] == "independent"
            )
            draft = fact_sheet().model_dump()
            mapping = {1: self.primary_id, 2: self.independent_id}
            for fact in draft["facts"]:
                fact["document_ids"] = [mapping[value] for value in fact["document_ids"]]
                for quote in fact["evidence_quotes"]:
                    quote["document_id"] = mapping[quote["document_id"]]
        else:
            draft = valid_article().model_dump()
            mapping = {1: self.primary_id, 2: self.independent_id}
            draft["used_document_ids"] = [mapping[value] for value in draft["used_document_ids"]]
            for anchor in draft["source_anchors"]:
                anchor["document_id"] = mapping[anchor["document_id"]]
        return {
            "id": f"offline-{len(self.calls)}",
            "model": payload["model"],
            "choices": [{"message": {"content": json.dumps(draft)}}],
            "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 100},
        }


@pytest.mark.parametrize("failure", ["timeout", "403", "challenge200"])
def test_whole_reader_skips_unread_failure_reads_different_pages_and_keeps_reserved_cost(
    fallback_env, failure
):
    settings, repo, event_id, ctx = fallback_env
    web, llm = FailingThenUsefulWeb(failure), GroundedOfflineLLM()

    def build(depth):
        return build_detail_article(
            settings,
            repo,
            event_id,
            depth=depth,
            reader_context=ctx,
            research_service=BoundedDetailResearchService(settings, repo, client=web),
            fact_generator=OpenRouterFactSheetGenerator(settings, transport=llm),
            article_generator=OpenRouterDeepArticleGenerator(settings, transport=llm),
        )

    expanded = build("expanded")
    assert expanded.validation_status == "valid"
    assert expanded.provider_response_metadata["actual_additional_sources"] == 3
    assert expanded.provider_response_metadata["source_target_met"] is True
    assert web.scrape_urls == [BAD, *GOOD]
    assert expanded.cost_usd == pytest.approx(0.00596)
    with repo.connection() as db:
        bad = db.execute(
            "SELECT status,reserved_usd,error_json FROM details_provider_calls "
            "WHERE provider='monid/context.dev' AND stage='scrape' ORDER BY id LIMIT 1",
        ).fetchone()
        assert bad[0] == ("completed" if failure == "challenge200" else "uncertain")
        assert bad[1] == 0.0009
        if failure != "challenge200":
            assert json.loads(bad[2])["http_status"] == (None if failure == "timeout" else 403)
        unread = db.execute(
            "SELECT id,metadata_json FROM research_documents WHERE url=?", (BAD,)
        ).fetchone()
        assert json.loads(unread[1])["content_kind"] == (
            "scraped_fulltext" if failure == "challenge200" else "search_snippet"
        )
        assert unread[0] not in expanded.provider_response_metadata["additional_document_ids"]
        assert unread[0] not in expanded.fact_snapshot["document_ids"]
        first_cost = db.execute("SELECT SUM(cost_usd) FROM usage_ledger").fetchone()[0]
    article = build("article")
    assert article.validation_status == "valid"
    assert article.provider_response_metadata["actual_additional_sources"] == 3
    assert article.provider_response_metadata["target_additional_sources"] == 6
    assert article.provider_response_metadata["source_target_met"] is False
    assert web.scrape_urls == [
        BAD,
        *GOOD,
    ]  # Neither unknown A nor cached B/C/D is dispatched again.
    with repo.connection() as db:
        final_cost = db.execute("SELECT SUM(cost_usd) FROM usage_ledger").fetchone()[0]
        assert final_cost - first_cost == pytest.approx(0.00236)
    assert article.provider_response_metadata["incremental_attempt_cost_usd"] == pytest.approx(
        0.00236
    )
