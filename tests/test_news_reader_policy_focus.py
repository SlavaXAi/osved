from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from test_details_research import MemoryRepository, document

from app.config import Settings
from app.details.research import (
    BoundedDetailResearchService,
    DetailResearchPlanner,
    _reader_event,
    bounded_reader_fact_documents,
    focused_reader_fact_transport,
    reader_announcement_scope,
    reader_document_relevant,
    reader_focus_links,
    reader_policy_focus,
    select_detail_documents,
)
from app.research.web import ScrapeResponse, SearchResponse

FIXTURE = Path(__file__).parent / "fixtures/news_reader_usage_policy_20261013.json"


def synthetic_policy():
    return json.loads(FIXTURE.read_text())


def test_synthetic_policy_slug_is_an_announcement_and_unrelated_company_pages_are_excluded():
    data = synthetic_policy()
    ctx, event = data["reader"], data["event"]
    assert reader_announcement_scope(event, ctx)
    scoped = _reader_event(event, "expanded", ctx)
    assert scoped["event_type"] == "company_event"
    selected = select_detail_documents(
        [d for d in data["documents"] if reader_document_relevant(d, scoped, ctx)],
        event=scoped,
        limit=16,
    )
    assert [d["id"] for d in selected] == [2]
    genuine = document(
        99999,
        "https://reuters.com/anthropic-bans-abusive-behavior",
        "independent",
        "Anthropic updated its Usage Policy to restrict abusive or cruel behavior toward Claude. "
        "The restriction targets extreme sustained behavior rather than ordinary frustration.",
    )
    assert reader_document_relevant(genuine, scoped, ctx)


def test_policy_search_scopes_title_clause_and_seeds_core_body_links(tmp_path):
    data = synthetic_policy()
    ctx, event, docs = data["reader"], data["event"], data["documents"]
    scoped = _reader_event(event, "expanded", ctx)
    plan = DetailResearchPlanner().plan(scoped, docs)
    assert '"2026 usage policy update"' in plan.queries[0]
    assert '"Usage Policy"' in plan.queries[1] and '"abusive"' in plan.queries[1]
    links = reader_focus_links(docs, ctx, reader_policy_focus(event, ctx))
    assert links == [
        "https://www.anthropic.com/legal/aup",
        "https://www.anthropic.com/research/end-subset-conversations",
    ]
    repo = MemoryRepository(docs)
    repo.event = {**event, "id": 7}
    calls = []

    class Client:
        def search(self, query, **kwargs):
            calls.append(("search", query))
            return SearchResponse(query=query, results=[], cost_usd=0)

        def scrape(self, url):
            calls.append(("scrape", url))
            return ScrapeResponse(
                url=url,
                markdown=(
                    "Anthropic Usage Policy states substantive permitted use cases. " * 20
                    if "/legal/" in url
                    else "Claude models may end conversations with persistently abusive users. "
                    * 20
                ),
                metadata={
                    "canonicalUrl": url,
                    "title": "Usage Policy" if "/legal/" in url else "Claude conversations",
                },
                cost_usd=0.0009,
            )

    result = BoundedDetailResearchService(
        Settings(_env_file=None, app_env="test", data_dir=tmp_path), repo, client=Client()
    ).collect(7, depth="expanded", reader_context=ctx)
    assert calls[:2] == [("scrape", url) for url in links]
    assert len(result.additional_document_ids) == 2 and result.target_additional_sources == 3
    assert len([c for c in calls if c[0] == "scrape"]) == 2


def test_russian_headline_english_policy_excerpt_keeps_entire_abusive_clause_and_enforcement():
    data = synthetic_policy()
    original = next(d for d in data["documents"] if d["id"] == 2)
    scoped = _reader_event(data["event"], "expanded", data["reader"])
    excerpt = bounded_reader_fact_documents([original], scoped)[0]["content_text"]
    assert len(excerpt) < 2050
    for phrase in (
        "sustained and needless abusive or cruel behavior",
        "user frustration, pushback, dark creative themes, or model testing and research",
        "primary enforcement mechanism",
        "takes effect on November 12",
    ):
        assert phrase in excerpt
    assert (
        original["content_text"]
        == next(d for d in synthetic_policy()["documents"] if d["id"] == 2)["content_text"]
    )


def test_fact_prompt_names_only_actual_additional_documents_and_keeps_frozen_baseline():
    data = synthetic_policy()
    ctx = {**data["reader"], "additional_document_ids": [9560, 9561]}
    seen = []
    guard = SimpleNamespace(llm_transport=lambda transport, stage: transport)
    call = focused_reader_fact_transport(
        lambda payload, key: seen.append(payload), guard, ctx, True
    )
    call(
        {
            "messages": [
                {"role": "system", "content": "Facts only."},
                {"role": "user", "content": "Evidence."},
            ]
        },
        "offline",
    )
    assert (
        "Реально дополнительные прочитанные document_ids: [9560, 9561]"
        in seen[0]["messages"][1]["content"]
    )
    assert "не приписывай старому факту чужой document_id" in seen[0]["messages"][0]["content"]
    assert ctx["context_fingerprint"] == data["reader"]["context_fingerprint"]


def test_nonpolicy_reader_probes_include_meaningful_product_slug_and_ignore_generic_footer():
    event = {"headline": "Anthropic выпустила Claude Haiku", "event_type": "model_release"}
    ctx = {
        "baseline_source_urls": ["https://anthropic.com/news/introducing-claude-haiku-update"],
        "baseline_document_ids": [],
    }
    scoped = _reader_event(event, "expanded", ctx)
    plan = DetailResearchPlanner().plan(scoped, [])
    assert '"claude haiku"' in plan.queries[0]
    unrelated = document(
        8,
        "https://anthropic.com/news/other-product-update",
        "primary",
        "Anthropic introduces a different unrelated product today. " * 20
        + "\n\n[Claude Haiku update](https://anthropic.com/news/introducing-claude-haiku-update)",
    )
    assert not reader_document_relevant(unrelated, scoped, ctx)
