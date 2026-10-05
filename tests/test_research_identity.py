import json
from pathlib import Path

import pytest

from app.research.web import (
    ResearchEventService,
    ScrapeResponse,
    SearchResponse,
    WebResearchError,
    build_research_query,
    cited_brand_homepage,
    event_brands,
    infer_event_primary_url,
    relevant_search_result,
)
from tests.test_research import setup_event

ARTICLE = "https://techstartups.com/2026/09/03/catch-emerges-from-stealth-with-5m-to-build-an-ai-executive-assistant/"
SITE = "https://www.catchagent.ai/"
EVENT = {
    "headline": "AI-ассистент Catch показывает спрос на делегирование административной работы",
    "source_role": "radar",
    "source_url": ARTICLE,
    "event_type": "web_signal",
    "entities_json": '{"organizations": []}',
    "original_text": "Catch вышла из stealth с seed-раундом; задачи без workflow.",
}
ARTICLE_TEXT = (
    f"[Home](https://techstartups.com/) [Catch]({SITE}) has emerged from stealth. "
    "Its AI executive assistant schedules meetings, manages email and books travel. "
    "[Nir Sabato](https://www.linkedin.com/in/nirsabato/) is the CEO. "
    "[Other](https://other.example/) is a different company."
)


def test_query_uses_brand_and_english_identity_not_russian_prose() -> None:
    query = build_research_query(EVENT, EVENT["headline"])
    assert '"Catch"' in query
    assert "stealth" in query
    assert "assistant" in query
    assert len(query) < 180
    assert not any("а" <= character.casefold() <= "я" for character in query)
    assert (
        build_research_query(EVENT, "Catch pricing site:catchagent.ai")
        == "Catch pricing site:catchagent.ai"
    )
    assert relevant_search_result(EVENT, {"title": "Catch launches its AI assistant"})
    assert not relevant_search_result(EVENT, {"title": "Best administrative AI tools for ADHD"})


def test_english_headline_boilerplate_is_not_a_company_name() -> None:
    assert event_brands({"headline": "Introducing WeatherNext Today"}) == ["WeatherNext"]
    assert event_brands({"headline": "OpenAI Safety Overview Release"}) == ["OpenAI"]
    # A real entity called Safety remains usable when explicitly extracted.
    assert event_brands(
        {
            "headline": "Safety Overview",
            "entities": {
                "organizations": ["Safety"],
            },
        }
    ) == ["Safety"]


def test_single_media_link_is_not_primary_and_unrelated_bigtech_is_not_primary() -> None:
    assert infer_event_primary_url(EVENT, [ARTICLE]) is None
    assert infer_event_primary_url(EVENT, ["https://reuters.com/news/catch"]) is None
    assert infer_event_primary_url(EVENT, ["https://openai.com/news/unrelated"]) is None
    assert infer_event_primary_url(EVENT, ["https://unrelated.example/story"]) is None
    trusted_rss = {**EVENT, "source_role": "primary", "source_url": SITE}
    assert infer_event_primary_url(trusted_rss, []) == SITE


def test_official_document_cited_as_event_source_does_not_need_company_brand():
    url = "https://www.esa.int/Applications/FLEX_and_Sentinel-3C_launched"
    event = {"headline": "Vega-C вывела два спутника", "source_url": url, "source_role": "radar"}
    assert infer_event_primary_url(event, [url]) == url
    assert infer_event_primary_url(EVENT, [url]) is None
    spoof = url.replace("esa.int", "esa.int.example.org")
    assert infer_event_primary_url({**event, "source_url": spoof}, [spoof]) is None
    homepage = "https://www.esa.int/"
    assert infer_event_primary_url({**event, "source_url": homepage}, [homepage]) is None


@pytest.mark.parametrize(
    "markdown",
    [
        "[Catch](https://reuters.com/catch)",
        "[Catch](https://unrelated.example/)",
        "[Catch](https://catch-news.example/)",
        "[Catch](https://www.catchagent.ai/?affiliate=publisher)",
        "[Nir Sabato](https://www.linkedin.com/in/nirsabato/)",
        "[Official website](https://www.catchagent.ai/)",
        "![Catch](https://www.catchagent.ai/image.jpg)",
    ],
)
def test_citation_discovery_rejects_unrelated_social_media_and_generic_links(markdown: str) -> None:
    assert cited_brand_homepage(EVENT, ARTICLE, markdown) is None


def test_citation_requires_trusted_article_and_exact_branded_anchor() -> None:
    url, basis = cited_brand_homepage(EVENT, ARTICLE, ARTICLE_TEXT)
    assert url == SITE
    assert basis["cited_by"] == ARTICLE
    assert basis["anchor"] == "Catch"
    assert cited_brand_homepage(EVENT, "https://unknown-news.example/article", ARTICLE_TEXT) is None


def test_official_redirect_is_not_rescraped_and_downgraded_by_search_result(tmp_path: Path) -> None:
    settings, repo, smoke = setup_event(tmp_path)
    requested = "https://deepmind.google/blog/introducing-weathernext-3/"
    canonical = "https://blog.google/innovation-and-ai/introducing-weathernext-3/"
    with repo.connection() as db:
        db.execute("UPDATE events SET headline='Introducing WeatherNext 3',entities_json='{}'")
        db.execute("UPDATE source_items SET source_url=?,outbound_urls_json='[]'", (requested,))
        db.execute(
            "UPDATE sources SET role='primary' WHERE id IN (SELECT source_id FROM source_items)"
        )

    class Client:
        calls = []

        def search(self, query, **kwargs):
            return SearchResponse(
                query, [{"url": canonical, "title": "Introducing WeatherNext 3"}], 0.00009
            )

        def scrape(self, url):
            self.calls.append(url)
            return ScrapeResponse(
                canonical,
                "WeatherNext 3 forecasts the weather. " * 5,
                {"canonicalUrl": canonical, "title": "WeatherNext 3"},
                0.0009,
            )

    client = Client()
    result = ResearchEventService(settings, repo, client=client).research(smoke["event_id"])
    assert client.calls == [requested]
    assert result.pages_scraped == 1
    documents = repo.event_research_documents(smoke["event_id"])
    assert len(documents) == 1
    assert documents[0]["source_class"] == documents[0]["evidence_role"] == "primary"


@pytest.mark.parametrize(
    "search_failed,site_identifies_brand", [(False, True), (True, True), (False, False)]
)
def test_unknown_startup_research_follows_one_branded_link_within_two_calls(
    tmp_path: Path,
    search_failed: bool,
    site_identifies_brand: bool,
) -> None:
    settings, repo, smoke = setup_event(tmp_path)
    event_id = smoke["event_id"]
    with repo.connection() as db:
        source_id = db.execute(
            "SELECT primary_source_item_id FROM events WHERE id=?", (event_id,)
        ).fetchone()[0]
        db.execute(
            "UPDATE events SET headline=?,entities_json=?,event_type='web_signal' WHERE id=?",
            (EVENT["headline"], EVENT["entities_json"], event_id),
        )
        db.execute(
            "UPDATE source_items SET source_url=?,outbound_urls_json='[]',original_text=? "
            "WHERE id=?",
            (ARTICLE, EVENT["original_text"], source_id),
        )
        db.execute(
            "UPDATE sources SET role='radar' "
            "WHERE id=(SELECT source_id FROM source_items WHERE id=?)",
            (source_id,),
        )

    class FakeClient:
        def __init__(self):
            self.calls = []

        def search(self, query, **kwargs):
            assert '"Catch"' in query
            self.calls.append(("search", query))
            if search_failed:
                raise WebResearchError("Unexpected token '<' in HTML provider response")
            return SearchResponse(
                query,
                [
                    {
                        "url": "https://example.com/admin-tools",
                        "title": "Best AI office tools",
                        "description": "Generic tool list",
                        "relevance": "high",
                    },
                    {
                        "url": ARTICLE,
                        "title": "Catch emerges from stealth",
                        "description": "AI executive assistant",
                        "relevance": "high",
                    },
                ],
                0.00018,
            )

        def scrape(self, url):
            self.calls.append(("scrape", url))
            return ScrapeResponse(
                url,
                ARTICLE_TEXT
                if url == ARTICLE
                else (
                    "Catch is your AI executive assistant. It schedules meetings, manages email, "
                    "books travel and connects to your Slack account without custom workflows."
                    if site_identifies_brand
                    else "Unrelated website about sports and weather."
                ),
                {
                    "title": "Catch launch" if url == ARTICLE else "Company website",
                    "canonicalUrl": url,
                },
                0.0009,
            )

    client = FakeClient()
    result = ResearchEventService(settings, repo, client=client).research(event_id, max_pages=2)
    assert [item[1] for item in client.calls if item[0] == "scrape"] == [ARTICLE, SITE]
    assert len([item for item in client.calls if item[0] == "search"]) == 1
    assert result.pages_scraped == 2
    assert result.documents_stored == 2
    documents = repo.event_research_documents(event_id)
    by_url = {document["url"]: document for document in documents}
    assert "https://example.com/admin-tools" not in by_url
    assert by_url[ARTICLE]["source_class"] == "independent"
    assert by_url[ARTICLE]["evidence_role"] == "independent"
    expected = "primary" if site_identifies_brand else "corroborating"
    assert by_url[SITE]["source_class"] == expected
    assert by_url[SITE]["evidence_role"] == expected
    if site_identifies_brand:
        assert json.loads(by_url[SITE]["metadata_json"])["primary_basis"]["cited_by"] == ARTICLE
    assert bool(result.search_error) == search_failed
    if search_failed:
        assert result.search_cost_status == "failed_request_reservation_not_invoice"
    assert repo.current_month_cost() == pytest.approx(result.cost_usd)
