from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.research.facts import FactSheetGeneration, build_event_fact_sheet
from app.research.web import (
    ResearchEventService,
    ScrapeResponse,
    SearchResponse,
    build_research_query,
    documentary_search_query,
    is_documentary_index,
    relevant_documentary_url,
)
from tests.test_research import setup_event, valid_fact_sheet


def save_document(repo, event_id, *, role="primary", text=None, url="https://x.ai/news/grok"):
    document_id = repo.upsert_research_document(
        canonical_url=url,
        url=url,
        title="Grok release",
        source_class=role,
        retrieved_at=datetime.now(UTC).isoformat(),
        expires_at=(datetime.now(UTC) + timedelta(days=1)).isoformat(),
        content_text=text
        or "Grok was released. API access costs 2 dollars per million tokens. " * 4,
        metadata={"content_kind": "scraped_fulltext"},
    )
    repo.attach_research_document(
        event_id=event_id,
        research_document_id=document_id,
        evidence_role=role,
    )
    return document_id


def test_missing_primary_stops_before_generator_and_cost_ledger(tmp_path):
    settings, repo, smoke = setup_event(tmp_path)
    save_document(repo, smoke["event_id"], role="independent")

    class NeverCalled:
        def generate(self, **kwargs):
            pytest.fail("Missing source cannot be repaired by a paid model")

    result = build_event_fact_sheet(settings, repo, smoke["event_id"], generator=NeverCalled())
    assert result["llm_skipped"] is True
    assert result["cost_usd"] == repo.current_month_cost() == 0
    assert repo.event_context(smoke["event_id"])["status"] == "research_insufficient"


def test_event_role_not_global_domain_label_controls_primary_prerequisite(tmp_path):
    settings, repo, smoke = setup_event(tmp_path)
    doc_id = save_document(repo, smoke["event_id"])
    repo.attach_research_document(
        event_id=smoke["event_id"],
        research_document_id=doc_id,
        evidence_role="corroborating",
    )
    result = build_event_fact_sheet(settings, repo, smoke["event_id"], generator=object())
    assert result["llm_skipped"] is True


def test_failed_evidence_is_not_retried_with_second_model_or_unchanged_input(tmp_path, monkeypatch):
    settings, repo, smoke = setup_event(tmp_path)
    settings.openrouter_model = "test/first"
    settings.openrouter_fallback_model = "test/second"
    doc_id = save_document(repo, smoke["event_id"])
    monkeypatch.setattr(
        "app.research.facts.validate_fact_sheet",
        lambda *_: ["fact sheet has fewer than three safe facts", "editorial hook is missing"],
    )

    class Generator:
        calls = 0

        def generate(self, *, model, **kwargs):
            self.calls += 1
            return FactSheetGeneration(
                draft=valid_fact_sheet(doc_id, doc_id),
                model=model,
                cost_usd=0.001,
                prompt_tokens=100,
                completion_tokens=100,
                response_id="test",
            )

    generator = Generator()
    first = build_event_fact_sheet(settings, repo, smoke["event_id"], generator=generator)
    second = build_event_fact_sheet(settings, repo, smoke["event_id"], generator=generator)
    assert first["status"] == "invalid"
    assert second["llm_skipped"] is True
    assert generator.calls == 1
    assert repo.current_month_cost() == pytest.approx(0.001)
    save_document(repo, smoke["event_id"], text="New evidence changed the API availability. " * 9)
    build_event_fact_sheet(settings, repo, smoke["event_id"], generator=generator)
    assert generator.calls == 2


def test_fresh_full_text_is_reused_instead_of_a_paid_scrape(tmp_path):
    settings, repo, smoke = setup_event(tmp_path)
    url = "https://x.ai/news/grok"
    doc_id = save_document(repo, smoke["event_id"], url=url)
    original = repo.event_research_documents(smoke["event_id"])[0]["content_text"]

    class Client:
        def search(self, query, **kwargs):
            return SearchResponse(
                query, [{"url": url, "title": "Grok", "description": "Tiny"}], cost_usd=0.00009
            )

        def scrape(self, url):
            pytest.fail("Fresh full text should be cached")

    result = ResearchEventService(settings, repo, client=Client()).research(
        smoke["event_id"],
        primary_url=url,
        max_pages=1,
    )
    assert result.pages_scraped == 0
    assert result.cost_usd == pytest.approx(0.00009)
    doc = next(
        doc for doc in repo.event_research_documents(smoke["event_id"]) if doc["id"] == doc_id
    )
    assert doc["content_text"] == original
    assert doc["evidence_role"] == "primary"


POLICY_EVENT = {
    "headline": "США перевели санкционное давление на Россию в законодательную основу",
    "source_url": "https://reuters.com/world/us/trump-signs-russia-sanctions-bill-into-law-2026-09-18/",
    "original_text": "Президент подписал закон о санкциях.",
}


def test_policy_query_uses_story_identity_and_responsible_institution():
    identity = build_research_query(POLICY_EVENT)
    assert "trump" in identity and "sanctions" in identity and "bill" in identity
    assert "США" not in identity
    query = documentary_search_query(POLICY_EVENT, identity)
    assert "site:whitehouse.gov" in query
    assert "site:congress.gov" in query
    assert relevant_documentary_url(
        POLICY_EVENT,
        {
            "url": "https://www.whitehouse.gov/briefings-statements/2026/09/bill-signed/",
            "title": "Trump signs Russia sanctions bill into law",
        },
    )
    for url in ("https://whitehouse.gov/", "https://whitehouse.gov.evil.test/bill-signed/"):
        assert not relevant_documentary_url(
            POLICY_EVENT,
            {
                "url": url,
                "title": "Trump signs Russia sanctions bill into law",
            },
        )
    assert not relevant_documentary_url(
        POLICY_EVENT,
        {
            "url": "https://whitehouse.gov/briefings-statements/weather-forecast/",
            "title": "Weather forecast for next week",
        },
    )


def test_institutional_catalogs_are_not_primary_documents():
    from app.research.facts import select_research_documents

    for url in (
        "https://ofac.treasury.gov/faqs/updated",
        "https://www.congress.gov/most-viewed-bills",
        "https://www.whitehouse.gov/news/",
    ):
        assert is_documentary_index(url)
        assert not relevant_documentary_url(
            POLICY_EVENT,
            {
                "url": url,
                "title": "Russia sanctions bill Trump",
            },
        )
        assert not select_research_documents(
            [
                {
                    "id": 1,
                    "url": url,
                    "source_class": "primary",
                    "content_text": "Broad FAQ. " * 1000,
                }
            ]
        )
    assert not is_documentary_index("https://ofac.treasury.gov/faqs/1225")


def test_matching_official_article_beats_alphabetical_domain_and_broad_index(tmp_path):
    settings, repo, smoke = setup_event(tmp_path)
    official = "https://www.whitehouse.gov/briefings-statements/2026/09/bill-signed/"
    with repo.connection() as db:
        db.execute(
            "UPDATE events SET headline=?,entities_json='{}' WHERE id=?",
            (POLICY_EVENT["headline"], smoke["event_id"]),
        )
        db.execute(
            "UPDATE source_items SET source_url=?,outbound_urls_json='[]'",
            (POLICY_EVENT["source_url"],),
        )
        db.execute(
            "UPDATE sources SET role='radar' WHERE id IN (SELECT source_id FROM source_items)"
        )

    class Client:
        calls = []

        def search(self, query, **kwargs):
            assert "site:whitehouse.gov" in query
            return SearchResponse(
                query,
                [
                    {"url": official, "title": "Trump signs Russia sanctions bill"},
                    {
                        "url": "https://ofac.treasury.gov/faqs/updated",
                        "title": "Updated Russia sanctions bill FAQ",
                    },
                    {
                        "url": "https://www.congress.gov/bill/119th-congress/house-bill/5334",
                        "title": "Russia sanctions bill signed by Trump",
                    },
                ],
                cost_usd=0.00027,
            )

        def scrape(self, url):
            self.calls.append(url)
            return ScrapeResponse(
                url,
                "Trump signs Russia sanctions bill into law. " * 8,
                {"canonicalUrl": url, "title": "Bill signed"},
                0.0009,
            )

    client = Client()
    ResearchEventService(settings, repo, client=client).research(smoke["event_id"], max_pages=1)
    assert client.calls == [official]
