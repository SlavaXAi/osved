from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.collectors.ingest import redact_text
from app.collectors.telegram import canonical_http_url
from app.config import Settings
from app.db.repository import Repository
from app.ops.budget import evaluate_budget

SEARCH_PROVIDER = "context.dev"
SEARCH_ENDPOINT = "/web/search"
SCRAPE_ENDPOINT = "/web/scrape/markdown"
SEARCH_RESULT_COST_USD = 0.00009
SCRAPE_COST_USD = 0.0009
DEFAULT_CACHE_DAYS = 30

# Public institutional publishing sites verified 2026-09-15. An official
# document proves what that institution announced, not independent confirmation
# of all of its claims. Social posts and articles on unrelated sites stay out.
DOCUMENTARY_DOMAINS = {
    "esa.int",
    "pm.gc.ca",
    "consilium.europa.eu",
    "ec.europa.eu",
    "eur-lex.europa.eu",
    "whitehouse.gov",
    "congress.gov",
    "govinfo.gov",
    "treasury.gov",
    "ustr.gov",
    "federalregister.gov",
    "federalreserve.gov",
    "commerce.gov",
    "cbp.gov",
    "kremlin.ru",
    "government.ru",
    "cbr.ru",
    "mofcom.gov.cn",
    "gov.cn",
}

PRIMARY_DOMAINS = {
    "anthropic.com",
    "blog.google",
    "deepmind.google",
    "docs.x.ai",
    "github.com",
    "google.com",
    "microsoft.com",
    "nvidia.com",
    "openai.com",
    "qwen.ai",
    "x.ai",
}
INDEPENDENT_DOMAINS = {
    "apnews.com",
    "arstechnica.com",
    "artificialanalysis.ai",
    "arxiv.org",
    "bbc.com",
    "benchmarking.llm-stats.com",
    "bloomberg.com",
    "cnbc.com",
    "economist.com",
    "ft.com",
    "ieee.org",
    "nature.com",
    "nikkei.com",
    "morphllm.com",
    "reuters.com",
    "science.org",
    "scmp.com",
    "semianalysis.com",
    "spectrum.ieee.org",
    "techcrunch.com",
    "techstartups.com",
    "theguardian.com",
    "thenextweb.com",
    "theverge.com",
    "unite.ai",
    "venturebeat.com",
    "wired.com",
    "wsj.com",
}
RADAR_ONLY_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "medium.com",
    "reddit.com",
    "t.me",
    "telegram.me",
    "threads.com",
    "threads.net",
    "twitter.com",
    "x.com",
}

MULTIPART_PUBLIC_SUFFIXES = {
    "co.in",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.hk",
    "com.sg",
    "com.tr",
    "com.tw",
}


class WebResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchResponse:
    query: str
    results: list[dict[str, Any]]
    cost_usd: float


@dataclass(frozen=True)
class ScrapeResponse:
    url: str
    markdown: str
    metadata: dict[str, Any]
    cost_usd: float


@dataclass(frozen=True)
class ResearchStats:
    event_id: int
    query: str
    search_results: int
    pages_scraped: int
    documents_stored: int
    cost_usd: float
    search_error: str | None = None
    search_cost_status: str = "response_reported"
    scrape_errors: int = 0


CommandRunner = Callable[[list[str], Path, Path], dict[str, Any]]


def monid_cli_path() -> str:
    discovered = shutil.which("monid")
    if discovered:
        return discovered
    user_local = Path.home() / ".local/bin/monid"
    if user_local.is_file():
        return str(user_local)
    raise WebResearchError("Monid CLI is not installed")


def run_monid(command: list[str], output_path: Path, working_directory: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    completed = subprocess.run(
        command,
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise WebResearchError(detail[-3000:] or "Monid command failed")
    if not output_path.is_file():
        raise WebResearchError("Monid completed without writing its output file")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WebResearchError("Monid returned a non-object payload")
    return payload


class MonidWebResearchClient:
    def __init__(
        self,
        settings: Settings,
        *,
        runner: CommandRunner = run_monid,
    ) -> None:
        self.settings = settings
        self.runner = runner

    def _output_path(self, operation: str) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        return (
            self.settings.data_dir
            / "research-runs"
            / f"{operation}-{timestamp}-{uuid.uuid4().hex[:8]}.json"
        )

    def search(
        self,
        query: str,
        *,
        freshness: str | None = "last_month",
        num_results: int = 10,
    ) -> SearchResponse:
        body = {
            "query": query,
            "numResults": num_results,
        }
        if freshness is not None:
            body["freshness"] = freshness
        output_path = self._output_path("search")
        command = [
            monid_cli_path(),
            "run",
            "-p",
            SEARCH_PROVIDER,
            "-e",
            SEARCH_ENDPOINT,
            "-i",
            json.dumps(body, separators=(",", ":")),
            "-o",
            str(output_path),
            "-j",
        ]
        payload = self.runner(command, output_path, Path.cwd())
        results = payload.get("results")
        if not isinstance(results, list):
            raise WebResearchError("Search response has no results array")
        normalized = [result for result in results if isinstance(result, dict)]
        return SearchResponse(
            query=str(payload.get("query") or query),
            results=normalized,
            cost_usd=len(normalized) * SEARCH_RESULT_COST_USD,
        )

    def scrape(self, url: str) -> ScrapeResponse:
        query = {
            "url": url,
            "useMainContentOnly": True,
            "includeLinks": True,
            "includeImages": False,
            "maxAgeMs": DEFAULT_CACHE_DAYS * 24 * 60 * 60 * 1000,
        }
        output_path = self._output_path("scrape")
        command = [
            monid_cli_path(),
            "run",
            "-p",
            SEARCH_PROVIDER,
            "-e",
            SCRAPE_ENDPOINT,
            "--query",
            json.dumps(query, separators=(",", ":")),
            "-o",
            str(output_path),
            "-j",
        ]
        payload = self.runner(command, output_path, Path.cwd())
        markdown = payload.get("markdown")
        if payload.get("success") is not True or not isinstance(markdown, str) or not markdown:
            raise WebResearchError(f"Could not scrape {url}")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return ScrapeResponse(
            url=str(payload.get("url") or url),
            markdown=markdown,
            metadata=metadata,
            cost_usd=SCRAPE_COST_USD,
        )


def hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def registrable_domain(url: str) -> str:
    """Return a conservative organization-domain key without a network PSL lookup."""

    host = hostname(url)
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2 or all(label.isdigit() for label in labels):
        return host
    suffix = ".".join(labels[-2:])
    size = 3 if suffix in MULTIPART_PUBLIC_SUFFIXES and len(labels) >= 3 else 2
    return ".".join(labels[-size:])


def domain_matches(host: str, domains: set[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def classify_source(url: str, *, known_primary_urls: set[str] | None = None) -> str:
    canonical = canonical_http_url(url)
    host = hostname(canonical)
    # A saved/redirected link cannot promote a social trigger into research proof.
    if domain_matches(host, RADAR_ONLY_DOMAINS):
        return "radar_only"
    if known_primary_urls and canonical in known_primary_urls:
        return "primary"
    if domain_matches(host, PRIMARY_DOMAINS | DOCUMENTARY_DOMAINS):
        return "primary"
    if domain_matches(host, INDEPENDENT_DOMAINS):
        return "independent"
    return "corroborating"


def infer_event_primary_url(event: dict[str, Any], known_urls: list[str]) -> str | None:
    """A cited article is a research starting point, not automatically a primary source."""

    source_url = str(event.get("source_url") or "")
    if event.get("source_role") == "primary" and source_url:
        return canonical_http_url(source_url)
    canonical_source = canonical_http_url(source_url)
    if (
        canonical_source
        and domain_matches(hostname(canonical_source), DOCUMENTARY_DOMAINS)
        and urlsplit(canonical_source).path.strip("/") not in {"", "en", "fr"}
        and canonical_source in {canonical_http_url(str(url)) for url in known_urls}
    ):
        # Institutional names need not appear in a headline about their own
        # satellite or diplomatic decision. The fetched page still undergoes
        # redirect, relevance, full-text and literal-fact validation below.
        return canonical_source

    external_urls: list[str] = []
    for url in known_urls:
        canonical = canonical_http_url(str(url))
        if not canonical or classify_source(canonical) in {"radar_only", "independent"}:
            continue
        if canonical not in external_urls:
            external_urls.append(canonical)
    if not external_urls:
        return None

    for url in external_urls:
        if brand_matches_url(event, url):
            return url
    return None


GENERIC_BRANDS = {
    "ai",
    "llm",
    "ceo",
    "cto",
    "api",
    "saas",
    "software",
    "open",
    "source",
    "the",
    "new",
    "news",
    "official",
    "model",
    "company",
    "technology",
}
QUERY_STOPWORDS = {
    "the",
    "an",
    "a",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "that",
    "you",
    "your",
    "from",
    "by",
    "its",
    "how",
    "why",
    "news",
    "blog",
    "www",
    "com",
    "https",
    "html",
    "emerges",
    "build",
    "handles",
    "work",
}
BRAND_HOST_SUFFIXES = {
    "",
    "ai",
    "agent",
    "agents",
    "app",
    "hq",
    "labs",
    "tech",
    "robotics",
    "systems",
    "technologies",
    "space",
}
BRAND_HOST_PREFIXES = {"", "get", "use", "try", "meet", "join"}
HEADLINE_BOILERPLATE = {
    "introducing",
    "safety",
    "overview",
    "release",
    "today",
    "announcing",
    "announced",
    "launching",
    "update",
    "report",
    "research",
    "here",
    "welcome",
    "towards",
    "building",
    "meet",
    "benchmark",
    "we",
    "our",
    "this",
    "now",
}


def event_brands(event: dict[str, Any]) -> list[str]:
    entities = event.get("entities") or event.get("entities_json") or {}
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except (TypeError, ValueError):
            entities = {}
    names = []
    if isinstance(entities, dict):
        for key in ("organizations", "products", "people"):
            values = entities.get(key)
            if isinstance(values, list):
                names.extend(str(value) for value in values)
    names.extend(
        token
        for token in re.findall(
            r"\b[A-Z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*\b", str(event.get("headline") or "")
        )
        if token.casefold() not in HEADLINE_BOILERPLATE
    )
    result = []
    for name in names:
        clean = " ".join(name.split()).strip()
        if (
            len(clean) >= 3
            and re.search(r"[A-Za-z]", clean)
            and clean.casefold() not in GENERIC_BRANDS
            and clean not in result
        ):
            result.append(clean)
    # Entity extraction also sees comparisons, navigation and footers. Prefer
    # the actors actually named in the headline over incidental OpenAI/Anthropic
    # mentions; otherwise a Qwen release turns into an unrelated big-tech query.
    headline = str(event.get("headline") or "").casefold()
    headline_names = [
        name
        for name in result
        if re.search(r"(?<![a-z0-9])" + re.escape(name.casefold()) + r"(?![a-z0-9])", headline)
    ]
    if headline_names:
        result = sorted(headline_names, key=lambda name: headline.find(name.casefold()))
    return result[:4]


def _compact_brand(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def brand_matches_url(
    event: dict[str, Any], url: str, *, branded_anchor: str | None = None
) -> bool:
    """Match an event brand to an organization host, never a news article URL slug."""
    if classify_source(url) in {"independent", "radar_only"}:
        return False
    host_brand = _compact_brand(registrable_domain(url).split(".")[0])
    for name in event_brands(event):
        brand = _compact_brand(name)
        if host_brand == brand:
            return True
        if (
            branded_anchor
            and _compact_brand(branded_anchor) == brand
            and any(
                host_brand == prefix + brand + suffix
                for prefix in BRAND_HOST_PREFIXES
                for suffix in BRAND_HOST_SUFFIXES
            )
        ):
            return True
        aliases = {
            "x.ai": ("grok", "xai"),
            "openai.com": ("gpt", "chatgpt"),
            "anthropic.com": ("claude",),
            "qwen.ai": ("qwen",),
            "deepmind.google": ("gemini", "deepmind"),
        }
        if any(
            domain_matches(hostname(url), {domain})
            and any(brand.startswith(alias) for alias in prefixes)
            for domain, prefixes in aliases.items()
        ):
            return True
    return False


def build_research_query(event: dict[str, Any], query: str | None = None) -> str:
    """Compact factual identity query; not a machine translation or a prose headline."""
    headline = str(event.get("headline") or "")
    if query and query.strip() != headline.strip():
        return query.strip()[:240]
    names = event_brands(event)[:2]
    tokens = []
    # English article slugs survive translation even when the stored source
    # item is itself a Russian LLM summary and entity extraction was empty.
    path = urlsplit(str(event.get("source_url") or "")).path.rstrip("/").split("/")[-1]
    material = f"{path} {headline}"
    name_tokens = {_compact_brand(name) for name in names}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}", material):
        token = token.strip(".-").casefold()
        pieces = token.split("-") if "-" in token else [token]
        for part in pieces:
            if (
                len(part) >= 3
                and not part.isdigit()
                and part not in QUERY_STOPWORDS
                and part not in name_tokens
                and part not in tokens
            ):
                tokens.append(part)
    type_terms = {
        "model_release": "model release benchmarks",
        "software_release": "release",
        "company_event": "company launch",
        "research_result": "research results",
        "public_statement": "statement interview",
    }.get(str(event.get("event_type")), "")
    pieces = [*(f'"{name}"' for name in names), *tokens[:7], type_terms]
    result = " ".join(piece for piece in pieces if piece).strip()
    # A translated geopolitical headline may contain no Latin company name.
    # The concrete English story slug is still a better identity than Russian prose.
    return result[:220] if (names or len(tokens) >= 4) and result else (query or headline)[:240]


def documentary_search_query(event: dict[str, Any], query: str) -> str:
    """Search the responsible institution for a policy action, not generic news.

    Domain choice routes retrieval only. It never certifies consequences or the
    truth of an institution's claims; ordinary evidence validation still applies.
    """
    text = " ".join(
        str(event.get(k) or "")
        for k in (
            "headline",
            "source_url",
            "original_text",
        )
    ).lower()
    if not re.search(r"sanction|tariff|legislat|interest.rate|ставк|санкц|пошлин|закон", text):
        return query
    if re.search(r"\btrump\b|\bwhite.house\b|\bcongress\b|трамп|конгресс|белый дом", text):
        return query[:145] + " (site:whitehouse.gov OR site:congress.gov OR site:treasury.gov)"
    if re.search(r"\bfed\b|federal.reserve|\bфрс\b", text):
        return query[:195] + " site:federalreserve.gov"
    if re.search(r"\beu\b|евросоюз|\bес\b", text):
        return query[:160] + " (site:consilium.europa.eu OR site:ec.europa.eu)"
    return query


def is_documentary_index(url: str) -> bool:
    """Catalogs and rolling FAQ indexes are not a document about one event."""
    return domain_matches(hostname(url), DOCUMENTARY_DOMAINS) and urlsplit(url).path.lower().strip(
        "/"
    ) in {
        "",
        "news",
        "newsroom",
        "live",
        "latest",
        "faqs",
        "faqs/updated",
        "most-viewed-bills",
        "congressional-record",
        "index.php/congressional-record",
    }


def relevant_documentary_url(event: dict[str, Any], result: dict[str, Any]) -> bool:
    """Recognize an actual matching institutional page, not every .gov or homepage."""
    url = str(result.get("url") or "")
    if not domain_matches(hostname(url), DOCUMENTARY_DOMAINS) or is_documentary_index(url):
        return False
    if len(urlsplit(url).path.strip("/")) < 8:
        return False
    identity = build_research_query(event)
    tokens = (
        set(re.findall(r"[a-z]{4,}", identity.lower()))
        - QUERY_STOPWORDS
        - {
            "world",
            "release",
            "official",
            "statement",
            "company",
            "launch",
            "government",
        }
    )
    candidate = " ".join(str(result.get(k) or "") for k in ("url", "title", "description"))
    words = set(re.findall(r"[a-z]{4,}", candidate.lower()))
    return len(tokens & words) >= 2


def relevant_search_result(event: dict[str, Any], result: dict[str, Any]) -> bool:
    names = event_brands(event)
    if not names:
        return True  # No reliable identity: do not invent an exclusion decision.
    text = " ".join(str(result.get(key) or "") for key in ("url", "title", "description"))
    return any(
        re.search(r"(?<![a-z0-9])" + re.escape(name.casefold()) + r"(?![a-z0-9])", text.casefold())
        for name in names
    )


def cited_brand_homepage(
    event: dict[str, Any],
    article_url: str,
    markdown: str,
) -> tuple[str, dict[str, str]] | None:
    """Follow at most one explicit brand citation from a reputable article.

    This only authorizes a fetch. A returned page must still identify the brand
    before becoming primary evidence of that company's own statements.
    """
    trusted = classify_source(article_url) in {"primary", "independent"}
    trusted |= event.get("source_role") in {"primary", "local_media"} and canonical_http_url(
        str(event.get("source_url") or "")
    ) == canonical_http_url(article_url)
    if not trusted:
        return None
    names = {_compact_brand(name) for name in event_brands(event)}
    for match in re.finditer(
        r"(?<!!)\[([^\]\n]{1,80})\]\((https?://[^\s)]+)(?:\s+[^)]*)?\)", markdown[:40000]
    ):
        anchor, url = match.groups()
        url = canonical_http_url(url)
        if (
            _compact_brand(anchor) not in names
            or not url
            or registrable_domain(url) == registrable_domain(article_url)
            or not brand_matches_url(event, url, branded_anchor=anchor)
        ):
            continue
        # Avoid tracking links, affiliate redirects, downloads and login flows.
        parts = urlsplit(url)
        if parts.query or re.search(
            r"login|signin|signup|redirect|download|\.(pdf|zip|exe)$", parts.path, re.IGNORECASE
        ):
            continue
        return url, {
            "method": "branded_editorial_citation",
            "cited_by": article_url,
            "anchor": anchor,
        }
    return None


def publisher_from_metadata(url: str, metadata: dict[str, Any]) -> str:
    json_ld = metadata.get("jsonLd")
    if isinstance(json_ld, list):
        for item in json_ld:
            if not isinstance(item, dict):
                continue
            publisher = item.get("publisher")
            if isinstance(publisher, dict) and publisher.get("name"):
                return str(publisher["name"])
    return hostname(url)


def published_at_from_metadata(metadata: dict[str, Any]) -> str | None:
    if metadata.get("publishedTime"):
        return str(metadata["publishedTime"])
    json_ld = metadata.get("jsonLd")
    if isinstance(json_ld, list):
        for item in json_ld:
            if isinstance(item, dict) and item.get("datePublished"):
                return str(item["datePublished"])
    return None


class ResearchEventService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        client: MonidWebResearchClient | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.client = client or MonidWebResearchClient(settings)

    def research(
        self,
        event_id: int,
        *,
        query: str | None = None,
        primary_url: str | None = None,
        freshness: str = "last_month",
        max_pages: int = 2,
    ) -> ResearchStats:
        event = self.repository.event_context(event_id)
        if event is None:
            raise ValueError(f"Unknown event_id: {event_id}")
        explicit_query = bool(query and query.strip() != str(event["headline"]).strip())
        query = build_research_query(event, query)
        expires_at = (datetime.now(UTC) + timedelta(days=DEFAULT_CACHE_DAYS)).isoformat()
        known_urls = self.repository.event_evidence_urls(event_id)
        trusted_explicit_urls = {canonical_http_url(primary_url)} if primary_url else set()
        primary_url = primary_url or infer_event_primary_url(event, known_urls)
        if primary_url:
            known_urls.insert(0, primary_url)
        known_primary_urls = {
            canonical_http_url(url) for url in known_urls if brand_matches_url(event, url)
        }
        if primary_url:
            known_primary_urls.add(canonical_http_url(primary_url))
        if event.get("source_role") == "primary" and event.get("source_url"):
            known_primary_urls.add(canonical_http_url(str(event["source_url"])))
            trusted_explicit_urls.add(canonical_http_url(str(event["source_url"])))

        if not primary_url and not explicit_query:
            query = documentary_search_query(event, query)

        projected = self.repository.current_month_cost() + (10 * SEARCH_RESULT_COST_USD)
        budget = evaluate_budget(
            projected,
            self.settings.budget_warning_usd,
            self.settings.budget_hard_stop_usd,
        )
        if not budget.paid_operations_allowed:
            raise WebResearchError("Monthly hard budget stop before web search")

        search_error = None
        search_cost_status = "response_reported"
        try:
            search = self.client.search(query, freshness=freshness, num_results=10)
        except (WebResearchError, ValueError, subprocess.SubprocessError) as exc:
            search_error = redact_text(str(exc))[:1000]
            # A failed request has no result-count invoice. Reserve the bounded
            # request ceiling explicitly; retain known-source research without retry.
            search_cost_status = "failed_request_reservation_not_invoice"
            search = SearchResponse(query=query, results=[], cost_usd=10 * SEARCH_RESULT_COST_USD)
        self.repository.record_research_query(
            event_id=event_id,
            provider=SEARCH_PROVIDER,
            query=search.query,
            freshness=freshness,
            status="failed" if search_error else "completed",
            results=search.results,
            cost_usd=search.cost_usd,
            expires_at=expires_at,
        )
        self.repository.record_usage(
            "monid/context.dev",
            "web.search",
            cost_usd=search.cost_usd,
            output_units=len(search.results),
            metadata={
                "event_id": event_id,
                "query": search.query,
                "cost_status": search_cost_status,
                "error": search_error,
            },
        )

        stored_document_ids: set[int] = set()
        candidates: list[tuple[str, str]] = []
        for result in search.results:
            url = canonical_http_url(str(result.get("url") or ""))
            if not url or is_documentary_index(url):
                continue
            if (
                not explicit_query
                and url not in known_urls
                and not relevant_search_result(event, result)
            ):
                continue
            evidence_role = classify_source(url, known_primary_urls=known_primary_urls)
            if evidence_role == "primary" and url not in known_primary_urls:
                evidence_role = (
                    "primary"
                    if (brand_matches_url(event, url) or relevant_documentary_url(event, result))
                    else "corroborating"
                )
                if evidence_role == "primary":
                    known_primary_urls.add(url)
            document_id = self.repository.upsert_research_document(
                canonical_url=url,
                url=url,
                title=str(result.get("title") or "") or None,
                publisher=hostname(url),
                source_class=classify_source(url),
                retrieved_at=datetime.now(UTC).isoformat(),
                expires_at=expires_at,
                content_text=str(result.get("description") or "") or None,
                metadata={
                    "content_kind": "search_snippet",
                    "search_query": search.query,
                    "relevance": result.get("relevance"),
                },
            )
            self.repository.attach_research_document(
                event_id=event_id,
                research_document_id=document_id,
                evidence_role=evidence_role,
                relevance={"high": 1.0, "medium": 0.6, "low": 0.2}.get(
                    str(result.get("relevance") or ""),
                    0.4,
                ),
            )
            stored_document_ids.add(document_id)
            candidates.append((url, evidence_role))

        scrape_urls: list[tuple[str, str]] = []
        # Preserve relevance order from SERP. Alphabetically sorting domains
        # previously chose a generic treasury page over the matching White House bill.
        explicit_or_known = [
            primary_url,
            *(url for url, role in candidates if role == "primary"),
            *sorted(known_primary_urls),
        ]
        for candidate_url in explicit_or_known:
            if not candidate_url:
                continue
            canonical = canonical_http_url(candidate_url)
            if canonical and canonical not in {url for url, _ in scrape_urls}:
                scrape_urls.append((canonical, "primary"))
                break
        # A cited article remains worth fetching even when SERP is unavailable
        # or unrelated. It does not thereby become an official source.
        known_articles = [
            (
                canonical_http_url(url),
                (
                    "primary"
                    if canonical_http_url(url) in known_primary_urls
                    else "corroborating"
                    if classify_source(url) == "primary"
                    else classify_source(url)
                ),
            )
            for url in known_urls
            if canonical_http_url(url)
            and classify_source(url) != "radar_only"
            and not is_documentary_index(url)
        ]
        if not scrape_urls and known_articles:
            scrape_urls.append(known_articles[0])
        independent = next(
            (
                url
                for url, source_class in candidates
                if source_class == "independent"
                and url not in {candidate_url for candidate_url, _ in scrape_urls}
            ),
            None,
        )
        if independent:
            scrape_urls.append((independent, "independent"))
        if not scrape_urls:
            scrape_urls.extend(candidates[:1])
        for candidate in [*known_articles, *candidates]:
            if candidate[0] not in {url for url, _ in scrape_urls}:
                scrape_urls.append(candidate)
        scrape_urls = scrape_urls[: max(0, max_pages)]

        pages_scraped = 0
        scrape_errors = 0
        attempts = 0
        attempted_urls: set[str] = set()
        citation_bases: dict[str, dict[str, str]] = {}
        resolved_roles: dict[str, str] = {}
        existing_documents = {
            canonical_http_url(str(doc.get("canonical_url") or doc.get("url") or "")): doc
            for doc in self.repository.event_research_documents(event_id)
        }
        followed_citation = False
        total_cost = search.cost_usd
        while scrape_urls and attempts < max(0, max_pages):
            url, evidence_role = scrape_urls.pop(0)
            if url in attempted_urls:
                continue
            attempted_urls.add(url)
            cached = existing_documents.get(url)
            if cached:
                try:
                    metadata = json.loads(cached.get("metadata_json") or "{}")
                    fresh = datetime.fromisoformat(
                        str(cached["expires_at"]).replace("Z", "+00:00")
                    ) > datetime.now(UTC)
                    recent = datetime.fromisoformat(
                        str(cached["retrieved_at"]).replace("Z", "+00:00")
                    ) > datetime.now(UTC) - timedelta(hours=6)
                except (ValueError, TypeError, KeyError):
                    metadata, fresh, recent = {}, False, False
                if (
                    fresh
                    and recent
                    and metadata.get("content_kind") == "scraped_fulltext"
                    and len(str(cached.get("content_text") or "")) >= 80
                ):
                    # Stored text and provenance stay intact. No paid re-fetch.
                    stored_document_ids.add(int(cached["id"]))
                    continue
            projected = self.repository.current_month_cost() + SCRAPE_COST_USD
            budget = evaluate_budget(
                projected,
                self.settings.budget_warning_usd,
                self.settings.budget_hard_stop_usd,
            )
            if not budget.paid_operations_allowed:
                break
            attempts += 1
            try:
                scraped = self.client.scrape(url)
            except (WebResearchError, ValueError, subprocess.SubprocessError) as exc:
                scrape_errors += 1
                # Keep a bounded failed attempt visible instead of treating it
                # as a free retry. The reservation is explicitly not an invoice.
                self.repository.record_usage(
                    "monid/context.dev",
                    "web.scrape.failed_reservation",
                    cost_usd=SCRAPE_COST_USD,
                    metadata={
                        "event_id": event_id,
                        "url": url,
                        "cost_status": "failed_request_reservation_not_invoice",
                        "error": redact_text(str(exc))[:1000],
                    },
                )
                total_cost += SCRAPE_COST_USD
                continue
            self.repository.record_usage(
                "monid/context.dev",
                "web.scrape.markdown",
                cost_usd=scraped.cost_usd,
                metadata={"event_id": event_id, "url": url},
            )
            pages_scraped += 1
            total_cost += scraped.cost_usd
            canonical_url = canonical_http_url(
                str(scraped.metadata.get("canonicalUrl") or scraped.url)
            )
            if not canonical_url:
                continue
            # A trusted official URL may redirect to its canonical publisher
            # page. Do not fetch it again from SERP and downgrade the same document.
            attempted_urls.add(canonical_url)
            primary_basis = citation_bases.get(url)
            page_identity = " ".join(
                [str(scraped.metadata.get("title") or ""), scraped.markdown[:12000]]
            )
            returned_same_brand = (
                registrable_domain(canonical_url) == registrable_domain(url)
                and relevant_search_result(event, {"title": page_identity})
                and len(scraped.markdown.strip()) >= 80
                and not re.search(
                    r"domain (?:is )?for sale|buy this domain|domain has expired",
                    page_identity,
                    re.IGNORECASE,
                )
            )
            if primary_basis:
                evidence_role = "primary" if returned_same_brand else "corroborating"
            elif (
                evidence_role == "primary"
                and url not in trusted_explicit_urls
                and not returned_same_brand
            ):
                evidence_role = "corroborating"
            if resolved_roles.get(canonical_url) == "primary":
                evidence_role = "primary"
            resolved_roles[canonical_url] = evidence_role
            # The primary label now means an explicit trusted source or a
            # confirmed branded website, not merely the first retrieved URL.
            source_class = evidence_role
            document_id = self.repository.upsert_research_document(
                canonical_url=canonical_url,
                url=scraped.url,
                title=str(scraped.metadata.get("title") or "") or None,
                publisher=publisher_from_metadata(scraped.url, scraped.metadata),
                source_class=source_class,
                published_at=published_at_from_metadata(scraped.metadata),
                retrieved_at=datetime.now(UTC).isoformat(),
                expires_at=expires_at,
                content_text=scraped.markdown,
                metadata={
                    **scraped.metadata,
                    "content_kind": "scraped_fulltext",
                    "primary_basis": primary_basis if evidence_role == "primary" else None,
                },
            )
            self.repository.attach_research_document(
                event_id=event_id,
                research_document_id=document_id,
                evidence_role=evidence_role,
                relevance=1.0,
            )
            stored_document_ids.add(document_id)
            if not followed_citation and attempts < max_pages:
                cited = cited_brand_homepage(event, canonical_url, scraped.markdown)
                if cited and cited[0] not in attempted_urls:
                    followed_citation = True
                    citation_bases[cited[0]] = cited[1]
                    scrape_urls.insert(0, (cited[0], "corroborating"))

        return ResearchStats(
            event_id=event_id,
            query=search.query,
            search_results=len(search.results),
            pages_scraped=pages_scraped,
            documents_stored=len(stored_document_ids),
            cost_usd=total_cost,
            search_error=search_error,
            search_cost_status=search_cost_status,
            scrape_errors=scrape_errors,
        )
