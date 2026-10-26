from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_details_research import (
    FactGenerator,
    MemoryRepository,
    document,
    fact_documents,
    valid_article,
)

from app.config import Settings
from app.details.models import DeepArticleParagraph, DetailFact, DetailProviderCall
from app.details.research import (
    BoundedDetailResearchService,
    DetailEvidenceError,
    DetailResearchCollection,
    DetailResearchPlanner,
    bounded_reader_fact_documents,
    build_deep_article_request,
    build_detail_article,
    build_detail_fact_snapshot,
    detail_generation_key,
    focused_reader_fact_transport,
    is_full_text_document,
    reader_announcement_scope,
    reader_document_relevant,
    reader_error_page,
    reader_new_fact_ids,
    reader_reference_text,
    reader_target_units,
    validate_deep_article,
)
from app.research.detail_policy import detail_policy_versions
from app.research.facts import (
    EvidenceQuote,
    FactCoverage,
    FactSheetDraft,
    FactSheetGeneration,
    ResearchFact,
    build_fact_sheet_request,
)
from app.research.web import ScrapeResponse, SearchResponse


def context():
    return {
        "baseline_card_id": 1,
        "baseline_text": "OpenAI выпустила Atlas 2 для корпоративной автоматизации. " * 8,
        "baseline_document_ids": [1, 2],
        "baseline_source_urls": ["https://openai.com/atlas", "https://wire.example/atlas"],
        "context_fingerprint": "frozen-baseline",
    }


class ReaderClient:
    domains = [
        "reuters.com",
        "theverge.com",
        "arstechnica.com",
        "ft.com",
        "bbc.com",
        "techcrunch.com",
    ]

    def __init__(self, count=6, *, copied=False):
        self.count, self.copied = count, copied
        self.search_calls = 0
        self.scrape_urls = []

    def search(self, query, *, freshness, num_results):
        self.search_calls += 1
        return SearchResponse(
            query=query,
            results=[
                {
                    "url": f"https://{domain}/atlas-{i}",
                    "title": f"OpenAI Atlas 2 independent context {i}",
                    "description": f"Atlas documentary source {i}",
                    "relevance": 1.0,
                }
                for i, domain in enumerate(self.domains[: self.count])
            ],
            cost_usd=self.count * 0.00009,
        )

    def scrape(self, url):
        self.scrape_urls.append(url)
        identity = "copied" if self.copied else url
        return ScrapeResponse(
            url=url,
            markdown=f"OpenAI Atlas 2 documentary investigation {identity}. " * 15,
            metadata={"canonicalUrl": url, "title": f"Atlas report {identity}"},
            cost_usd=0.0009,
        )


@pytest.mark.parametrize("depth,target", [("expanded", 3), ("article", 6)])
def test_reader_counts_additional_read_pages_beyond_already_valid_baseline(tmp_path, depth, target):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    repo = MemoryRepository(fact_documents())
    client = ReaderClient()
    result = BoundedDetailResearchService(settings, repo, client=client).collect(
        7, depth=depth, reader_context=context()
    )
    assert result.target_additional_sources == target
    assert len(result.additional_document_ids) == target
    assert {1, 2}.isdisjoint(result.additional_document_ids)
    assert len(client.scrape_urls) == target
    assert result.search_count == 1  # Cached two-domain gate cannot skip the new search.


def test_article_reuses_expanded_three_and_reads_only_the_remaining_three(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    repo = MemoryRepository(fact_documents())
    client = ReaderClient()
    service = BoundedDetailResearchService(settings, repo, client=client)
    expanded = service.collect(7, depth="expanded", reader_context=context())
    assert len(client.scrape_urls) == 3
    article = service.collect(7, depth="article", reader_context=context())
    assert len(client.scrape_urls) == 6
    assert len(set(client.scrape_urls)) == 6
    assert set(expanded.additional_document_ids).issubset(article.additional_document_ids)
    assert len(article.additional_document_ids) == 6


def test_partial_sources_are_actual_not_snippets_or_a_promised_target(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    repo = MemoryRepository(fact_documents())
    client = ReaderClient(count=2)
    result = BoundedDetailResearchService(settings, repo, client=client).collect(
        7, depth="expanded", reader_context=context()
    )
    assert result.target_additional_sources == 3 and len(result.additional_document_ids) == 2
    assert result.search_count == 2 and len(client.scrape_urls) == 2
    strict = {**context(), "allow_partial_sources": False}
    with pytest.raises(DetailEvidenceError, match="target is 3"):
        BoundedDetailResearchService(settings, repo, client=client).collect(
            7, depth="expanded", reader_context=strict
        )


def test_only_search_snippets_cannot_make_an_enriched_result(tmp_path):
    settings = Settings(
        _env_file=None, app_env="test", data_dir=tmp_path, news_reader_expanded_max_scrapes=0
    )
    repo = MemoryRepository(fact_documents())
    with pytest.raises(DetailEvidenceError, match="No additional"):
        BoundedDetailResearchService(settings, repo, client=ReaderClient()).collect(
            7, depth="expanded", reader_context=context()
        )


def test_syndicated_copy_counts_once_even_when_read_on_three_domains(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    repo = MemoryRepository(fact_documents())
    client = ReaderClient(count=3, copied=True)
    result = BoundedDetailResearchService(settings, repo, client=client).collect(
        7, depth="expanded", reader_context=context()
    )
    assert len(client.scrape_urls) == 3 and len(result.additional_document_ids) == 1


def test_unrelated_same_company_documents_do_not_enter_new_reader_corpus():
    repo = MemoryRepository()
    repo.event.update(headline="Anthropic обновила Claude Usage Policy")
    ctx = {
        "baseline_document_ids": [1],
        "baseline_source_urls": ["https://www.anthropic.com/legal/usage-policy"],
    }
    unrelated = document(
        7,
        "https://anthropic.com/news/haiku",
        "primary",
        "Anthropic introduces Claude Haiku model for quick chats. " * 20,
    )
    assert not reader_document_relevant(unrelated, repo.event, ctx)
    policy = document(
        8,
        "https://anthropic.com/legal/policy-guidance",
        "primary",
        "Claude Usage Policy documents permitted enterprise workflows. " * 20,
    )
    assert reader_document_relevant(policy, repo.event, ctx)
    unrelated["content_text"] += "\n\n[Usage Policy](https://anthropic.com/legal/usage-policy)"
    assert not reader_document_relevant(unrelated, repo.event, ctx)


def test_company_policy_reader_can_read_related_official_pages_without_faking_independence(
    tmp_path,
):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    docs = [
        document(
            i,
            f"https://anthropic.com/legal/policy-{i}",
            "primary",
            f"Anthropic Usage Policy published condition number {i}. " * 20,
        )
        for i in range(1, 5)
    ]
    repo = MemoryRepository(docs)
    repo.event.update(headline="Anthropic обновила Usage Policy", event_type="model_release")
    ctx = {
        "baseline_document_ids": [1],
        "baseline_source_urls": ["https://anthropic.com/legal/usage-policy"],
    }
    client = ReaderClient(count=0)
    assert reader_announcement_scope(repo.event, ctx)
    result = BoundedDetailResearchService(settings, repo, client=client).collect(
        7, depth="expanded", reader_context=ctx
    )
    assert result.gate.valid and len(result.additional_document_ids) == 3
    assert result.gate.independent_document_ids == []
    assert client.search_calls == 0
    with pytest.raises(DetailEvidenceError, match="independent"):
        BoundedDetailResearchService(settings, repo, client=client).collect(7)


def test_high_risk_and_nonpolicy_news_cannot_take_company_policy_shortcut():
    ctx = {"baseline_source_urls": ["https://anthropic.com/legal/usage-policy"]}
    assert not reader_announcement_scope({"headline": "Военные автономные системы"}, ctx)
    assert not reader_announcement_scope(
        {"headline": "Новая модель Claude"},
        {"baseline_source_urls": ["https://anthropic.com/news/claude-model"]},
    )


def test_policy_fact_snapshot_uses_announcement_scope_and_never_claims_independent_testing(
    tmp_path,
):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    texts = [
        "Anthropic обновила Usage Policy для корпоративных пользователей.",
        "Usage Policy разрешает обработку корпоративных документов под контролем человека.",
        "Usage Policy запрещает автономные решения в опасных сценариях.",
        "Usage Policy предусматривает ответственность компании за использование системы.",
    ]
    kinds = ["release", "availability", "limitation", "limitation"]
    docs = [
        document(i, f"https://anthropic.com/legal/policy-{i}", "primary", text * 10)
        for i, text in enumerate(texts, 1)
    ]
    repo = MemoryRepository(docs)
    repo.event.update(headline=texts[0], event_type="model_release")
    draft = FactSheetDraft(
        event_type="model_release",
        conflicts=[],
        facts=[
            ResearchFact(
                text=text,
                kind=kind,
                document_ids=[i],
                evidence_quotes=[EvidenceQuote(document_id=i, quote=text)],
            )
            for i, (text, kind) in enumerate(zip(texts, kinds, strict=True), 1)
        ],
        coverage=FactCoverage(
            event_identity=True,
            capabilities=False,
            benchmarks=False,
            pricing=False,
            availability=True,
            limitations=True,
            independent_confirmation=False,
        ),
    )
    generator = SimpleNamespace(
        generate=lambda **kwargs: FactSheetGeneration(
            draft=draft,
            model="fake",
            cost_usd=0.001,
            prompt_tokens=100,
            completion_tokens=100,
            response_id="policy-offline",
        )
    )
    ctx = {"baseline_source_urls": ["https://anthropic.com/legal/usage-policy"]}
    result = build_detail_fact_snapshot(
        settings, repo, 7, docs, generator=generator, depth="expanded", reader_context=ctx
    )
    assert result.snapshot.event_type == "company_event"
    assert result.snapshot.coverage.independent_confirmation is False
    assert len(result.snapshot.facts) == 4
    assert len(result.snapshot.source_domains) == 1


def test_reader_length_uses_visible_utf16_baseline_and_separate_tier_cache():
    ctx = {"baseline_text": "😀 " * 100}
    assert reader_target_units("expanded", ctx) == 600
    assert reader_target_units("article", ctx) == 1200
    docs = fact_documents()
    assert detail_generation_key(7, docs, depth="expanded", context_fingerprint="a") != (
        detail_generation_key(7, docs, depth="article", context_fingerprint="a")
    )
    assert detail_generation_key(7, docs, depth="expanded", context_fingerprint="a") != (
        detail_generation_key(7, docs, depth="expanded", context_fingerprint="b")
    )


def test_article_actual_more_target_prompt_and_semantic_cache_identity():
    ctx = {
        "baseline_text": "Исходная краткая карточка.",
        "context_fingerprint": "a" * 64,
        "expanded_reference_text": "Уже прочитанный содержательный разбор 😀. " * 35,
        "expanded_reference_article_id": 6,
    }
    ctx["expanded_reference_units"] = len(ctx["expanded_reference_text"].encode("utf-16-le")) // 2
    assert reader_target_units("article", ctx) == ctx["expanded_reference_units"] * 2
    assert reader_reference_text("article", ctx) == ctx["expanded_reference_text"]
    assert reader_reference_text("expanded", ctx) == ctx["baseline_text"]
    docs = fact_documents()
    repo = MemoryRepository(docs)
    settings = Settings(_env_file=None, app_env="test")
    facts = build_detail_fact_snapshot(settings, repo, 7, docs, generator=FactGenerator())
    request = build_deep_article_request(
        model="openai/gpt-5.6-luna",
        event={
            **repo.event,
            "_news_reader": {
                "depth": "article",
                "target_units": reader_target_units("article", ctx),
                "baseline_text": ctx["baseline_text"],
                "expanded_reference_text": ctx["expanded_reference_text"],
            },
        },
        snapshot=facts.snapshot,
        documents=docs,
    )
    assert ctx["expanded_reference_text"] in request["messages"][1]["content"]
    assert str(2 * ctx["expanded_reference_units"]) in request["messages"][0]["content"]
    key = detail_generation_key(7, docs, depth="article", reader_context=ctx)
    assert key != detail_generation_key(
        7, docs, depth="article", reader_context={"context_fingerprint": "a" * 64}
    )
    assert key == detail_generation_key(
        7, docs, depth="article", reader_context={**ctx, "retry_authorization": "new-nonce"}
    )
    assert detail_policy_versions("expanded")[1].startswith("news_explainer_ru_v1:")
    assert detail_policy_versions("article")[1].startswith("news_question_ru_v1:")


def test_article_fact_prompt_and_novelty_compare_with_already_shown_more():
    ctx = {
        **context(),
        "expanded_reference_text": "Уже показанные дополнительные сведения.",
        "additional_document_ids": [2],
    }
    seen = []
    guard = SimpleNamespace(llm_transport=lambda transport, stage: transport)
    focused_reader_fact_transport(lambda payload, key: seen.append(payload), guard, ctx, False)(
        {
            "messages": [
                {"role": "system", "content": "Facts."},
                {"role": "user", "content": "Evidence."},
            ]
        },
        "offline",
    )
    assert ctx["expanded_reference_text"] in seen[0]["messages"][1]["content"]
    assert "которого нет в этом разборе" in seen[0]["messages"][0]["content"]


def test_article_repeated_more_facts_stop_before_writer(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    docs = fact_documents()
    repo = MemoryRepository(docs)
    existing = build_detail_fact_snapshot(settings, repo, 7, docs, generator=FactGenerator())
    ctx = {
        **context(),
        "expanded_reference_text": " ".join(f.text for f in existing.snapshot.facts),
        "expanded_reference_units": 1000,
    }
    assert reader_new_fact_ids(existing.snapshot, {2}, ctx["baseline_text"])
    collection = DetailResearchCollection(
        DetailResearchPlanner().plan(repo.event, docs),
        docs,
        SimpleNamespace(valid=True),
        0,
        0,
        0,
        (2,),
        6,
    )

    class Guard:
        def __init__(self, *args, **kwargs):
            pass

        def accounting_repository(self):
            return repo

    monkeypatch.setattr("app.details.research.DurableDetailProvider", Guard)
    writer_calls = []
    with pytest.raises(DetailEvidenceError, match="previously shown reader result"):
        build_detail_article(
            settings,
            repo,
            7,
            depth="article",
            reader_context=ctx,
            research_service=SimpleNamespace(collect=lambda *args, **kwargs: collection),
            fact_generator=FactGenerator(),
            article_generator=SimpleNamespace(
                generate=lambda **kwargs: writer_calls.append(kwargs)
            ),
        )
    assert writer_calls == []


def test_excerpts_are_bounded_without_mutating_or_replacing_full_source_pages():
    text = (
        "Navigation boilerplate. " * 300
        + "Atlas 2 gained a late documented feature."
        + " tail" * 100
    )
    docs = [document(1, "https://openai.com/atlas", "primary", text)]
    excerpts = bounded_reader_fact_documents(docs, {"headline": "Atlas 2 gained a feature"})
    assert len(excerpts[0]["content_text"]) < 2100
    assert "late documented feature" in excerpts[0]["content_text"]
    assert docs[0]["content_text"] == text


def test_reader_requires_new_information_supported_by_additional_pages(tmp_path):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    repo = MemoryRepository(fact_documents())
    facts = build_detail_fact_snapshot(
        settings, repo, 7, fact_documents(), generator=FactGenerator()
    )
    baseline = " ".join(fact.text for fact in facts.snapshot.facts)
    assert reader_new_fact_ids(facts.snapshot, {2}, baseline) == []
    assert reader_new_fact_ids(facts.snapshot, {2}, "OpenAI выпустила Atlas 2.") == [3, 4]
    assert reader_new_fact_ids(facts.snapshot, {99}, "OpenAI выпустила Atlas 2.") == []


@pytest.mark.parametrize("depth,cap", [("expanded", 0.05), ("article", 0.08)])
def test_bounded_full_fact_and_writer_payload_reservations_fit_tier_budget(tmp_path, depth, cap):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    docs = fact_documents()
    repo = MemoryRepository(docs)
    facts = build_detail_fact_snapshot(settings, repo, 7, docs, generator=FactGenerator())
    expanded_docs = docs + [
        document(
            i,
            f"https://reuters.com/atlas-{i}",
            "independent",
            f"Atlas evidence from source {i}. " * 800,
        )
        for i in range(3, 8)
    ]
    fact_request = build_fact_sheet_request(
        model="openai/gpt-5.6-luna",
        event=repo.event,
        documents=bounded_reader_fact_documents(expanded_docs, repo.event),
    )
    reference = (
        "Содержательный уже прочитанный подробный разбор. " * 30 if depth == "article" else ""
    )
    target = len(reference.encode("utf-16-le")) if reference else 1000
    if reference:
        captured = []
        guard = SimpleNamespace(llm_transport=lambda transport, stage: transport)
        focused_reader_fact_transport(
            lambda payload, key: captured.append(payload),
            guard,
            {
                "baseline_text": "Atlas summary.",
                "expanded_reference_text": reference,
                "additional_document_ids": [2, 3, 4, 5, 6, 7],
            },
            False,
        )(fact_request, "offline")
        fact_request = captured[0]
    event = {
        **repo.event,
        "_news_reader": {
            "depth": depth,
            "target_units": target,
            "baseline_text": "Atlas summary.",
            "expanded_reference_text": reference,
        },
    }
    writer_request = build_deep_article_request(
        model="openai/gpt-5.6-luna", event=event, snapshot=facts.snapshot, documents=docs
    )

    def reserve(payload):
        return (
            (
                len(
                    json.dumps(
                        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                )
                + 2048
            )
            * 0.5
            + payload["max_tokens"] * 2
        ) / 1_000_000

    retrieval = 0.0018 + (3 if depth == "expanded" else 6) * 0.0009
    assert retrieval + reserve(fact_request) + reserve(writer_request) <= cap
    assert str(event["_news_reader"]["target_units"]) in writer_request["messages"][0]["content"]


def test_reader_build_returns_new_policy_and_actual_source_coverage(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    docs = fact_documents()
    repo = MemoryRepository(docs)
    ctx = {
        **context(),
        "baseline_document_ids": [1],
        "baseline_source_urls": ["https://openai.com/atlas"],
    }
    collection = DetailResearchCollection(
        DetailResearchPlanner().plan(repo.event, docs),
        docs,
        SimpleNamespace(valid=True),
        1,
        1,
        0.001,
        (2,),
        3,
    )
    service = SimpleNamespace(collect=lambda *args, **kwargs: collection)

    class OfflineGuard:
        def __init__(self, *args, **kwargs):
            pass

        def accounting_repository(self):
            return repo

    monkeypatch.setattr("app.details.research.DurableDetailProvider", OfflineGuard)
    provider = DetailProviderCall(
        model="fake",
        prompt_tokens=100,
        completion_tokens=100,
        reported_cost_usd=0.001,
        accounted_cost_usd=0.001,
    )
    writer = SimpleNamespace(generate=lambda **kwargs: (valid_article(), provider))
    result = build_detail_article(
        settings,
        repo,
        7,
        depth="expanded",
        reader_context=ctx,
        research_service=service,
        fact_generator=FactGenerator(),
        article_generator=writer,
    )
    assert result.depth == "expanded"
    assert (result.schema_version, result.prompt_version) == detail_policy_versions("expanded")
    assert result.provider_response_metadata["actual_additional_sources"] == 1
    assert result.provider_response_metadata["target_additional_sources"] == 3
    assert result.provider_response_metadata["source_target_met"] is False
    assert "Что именно выпустили" not in result.rendered_html


def test_retry_authorization_never_enters_fact_provider_prompt_or_payload():
    ctx = {**context(), "retry_rejected": True, "retry_authorization": "secret-retry-nonce"}
    captured = []
    guard = SimpleNamespace(llm_transport=lambda transport, stage: transport)
    wrapped = focused_reader_fact_transport(
        lambda payload, key: captured.append(payload) or {"ok": True}, guard, ctx, True
    )
    request = {
        "messages": [
            {"role": "system", "content": "Facts only."},
            {"role": "user", "content": "Original evidence."},
        ]
    }
    wrapped(request, "offline")
    raw = json.dumps(captured[0], ensure_ascii=False)
    assert ctx["baseline_text"] in captured[0]["messages"][1]["content"]
    assert "company_event" in raw
    assert "secret-retry-nonce" not in raw and "retry_authorization" not in raw
    assert request["messages"][0]["content"] == "Facts only."


@pytest.mark.parametrize(
    "body",
    [
        "# 403 Forbidden\nThe request was denied by this website. "
        "Please contact its administrator.",
        "Access denied\nYour browser is not permitted to access this resource at this time.",
        "Just a moment...\nEnable JavaScript and cookies to continue using this website.",
        "Checking your browser\nThis verification step is required before the page is accessible.",
        "Verify you are human\nPlease complete this challenge before continuing to the content.",
        "Subscribe to read this article\nCreate a subscriber account to access the content.",
        "Please sign in to continue\nThis content is available only to registered subscribers.",
        "<html><title>Just a moment...</title>"
        "<h1>Enable JavaScript and cookies to continue</h1></html>",
    ],
)
def test_cached_challenge_is_not_an_additional_read_source(tmp_path, body):
    settings = Settings(
        _env_file=None, app_env="test", data_dir=tmp_path, news_reader_max_searches=0
    )
    bad = document(3, "https://reuters.com/atlas-report", "independent", body)
    bad["title"] = "OpenAI Atlas 2 report"
    assert is_full_text_document(bad)  # The legacy body-length rule is unchanged.
    assert reader_error_page(bad)
    repo = MemoryRepository([*fact_documents(), bad])
    client = ReaderClient()
    with pytest.raises(DetailEvidenceError, match="No additional"):
        BoundedDetailResearchService(settings, repo, client=client).collect(
            7, depth="expanded", reader_context=context()
        )
    assert client.search_calls == 0 and client.scrape_urls == []


def test_substantive_article_about_cloudflare_captchas_is_not_an_error_page():
    body = (
        "# Understanding Atlas browser challenges\n\n"
        "Cloudflare sometimes displays 'Just a moment...' or 'Verify you are human'. "
        "The article explains the mechanism of those browser challenges in Atlas, "
        "and compares their effects with access denied responses and paywall prompts.\n\n"
        "## Verify you are human\n\nThis section discusses challenge usability, "
        "not a challenge page."
    )
    doc = document(3, "https://reuters.com/atlas-browser-report", "independent", body)
    doc["title"] = "Cloudflare captchas and Atlas: access denied explained"
    assert not reader_error_page(doc)
    assert reader_document_relevant(doc, MemoryRepository().event, context())


@pytest.mark.parametrize("output", ["baseline_only", "headline_only", "useful_body"])
def test_final_reader_must_reveal_new_fact_in_substantive_body(tmp_path, output):
    settings = Settings(_env_file=None, app_env="test", data_dir=tmp_path)
    docs = fact_documents()
    repo = MemoryRepository(docs)
    original = build_detail_fact_snapshot(
        settings, repo, 7, docs, generator=FactGenerator()
    ).snapshot
    baseline = " ".join(fact.text for fact in original.facts)
    new_text = "Atlas 2 получила дополнительную интеграцию для анализа потоковых событий."
    new_doc = document(3, "https://reuters.com/atlas-integration", "independent", new_text * 20)
    new_fact = DetailFact(
        fact_id=5,
        text=new_text,
        kind="capability",
        document_ids=[3],
        evidence_quotes=[EvidenceQuote(document_id=3, quote=new_text)],
    )
    snapshot = original.model_copy(
        update={
            "facts": [*original.facts, new_fact],
            "document_ids": [1, 2, 3],
            "source_domains": [*original.source_domains, "reuters.com"],
        }
    )
    assert reader_new_fact_ids(snapshot, {3}, baseline) == [5]
    draft = valid_article()
    if output == "headline_only":
        draft = draft.model_copy(
            update={"headline_fact_ids": [1, 5], "used_document_ids": [1, 2, 3]}
        )
    elif output == "useful_body":
        section = draft.sections[0].model_copy(
            update={
                "paragraphs": [
                    *draft.sections[0].paragraphs,
                    DeepArticleParagraph(text=new_text, fact_ids=[5]),
                ]
            }
        )
        draft = draft.model_copy(
            update={"sections": [section, *draft.sections[1:]], "used_document_ids": [1, 2, 3]}
        )
    errors = validate_deep_article(
        draft,
        snapshot=snapshot,
        documents=[*docs, new_doc],
        event=repo.event,
        depth="expanded",
        reader_context={"baseline_text": baseline, "new_fact_ids": [5]},
    )
    if output == "useful_body":
        assert errors == []
    else:
        assert "reader lead/body must include a new verified fact from additional sources" in errors
