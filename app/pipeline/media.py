from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from app.collectors.telegram import canonical_http_url

MEDIA_POLICY_VERSION = "preview_first_v1"
REJECTED_IMAGE_MARKERS = re.compile(
    r"(?:^|[/_.-])(avatar|default|favicon|icon|logo|placeholder|sprite)(?:[/_.-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MediaDecision:
    mode: str
    preview_url: str | None
    image_url: str | None
    source_document_id: int | None
    checks: dict[str, Any]


def host(url: str) -> str:
    return (urlsplit(url).hostname or "").casefold().removeprefix("www.")


def related_hosts(first: str, second: str) -> bool:
    first_host = host(first)
    second_host = host(second)
    if not first_host or not second_host:
        return False
    return (
        first_host == second_host
        or first_host.endswith(f".{second_host}")
        or second_host.endswith(f".{first_host}")
    )


def metadata_payload(document: dict[str, Any]) -> dict[str, Any]:
    raw = document.get("metadata_json") or document.get("metadata") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def image_value(value: Any) -> tuple[str | None, int | None, int | None]:
    if isinstance(value, str):
        return value, None, None
    if not isinstance(value, dict):
        return None, None, None
    url = value.get("url") or value.get("secure_url") or value.get("contentUrl")
    width = value.get("width")
    height = value.get("height")
    return (
        str(url) if url else None,
        int(width) if isinstance(width, int | float) else None,
        int(height) if isinstance(height, int | float) else None,
    )


def metadata_image_candidates(metadata: dict[str, Any]) -> list[tuple[str, int | None, int | None]]:
    values: list[Any] = []
    for key in ("ogImage", "og:image", "image", "imageUrl", "image_url"):
        if key in metadata:
            value = metadata[key]
            values.extend(value if isinstance(value, list) else [value])
    json_ld = metadata.get("jsonLd")
    if isinstance(json_ld, list):
        for item in json_ld:
            if isinstance(item, dict) and item.get("image"):
                image = item["image"]
                values.extend(image if isinstance(image, list) else [image])

    candidates: list[tuple[str, int | None, int | None]] = []
    for value in values:
        url, width, height = image_value(value)
        canonical = canonical_http_url(url or "")
        if canonical and canonical not in {candidate[0] for candidate in candidates}:
            candidates.append((canonical, width, height))
    return candidates


def validate_image_candidate(
    image_url: str,
    *,
    page_url: str,
    width: int | None,
    height: int | None,
) -> list[str]:
    errors: list[str] = []
    if REJECTED_IMAGE_MARKERS.search(urlsplit(image_url).path):
        errors.append("generic image marker")
    if not related_hosts(image_url, page_url):
        errors.append("image host is unrelated to evidence host")
    if width is not None and height is not None:
        if width < 600 or height < 315:
            errors.append("image is too small")
        ratio = width / height if height else 0
        if not 1.2 <= ratio <= 2.2:
            errors.append("image aspect ratio is unsuitable")
    return errors


def select_preview_first_media(
    documents: list[dict[str, Any]],
    *,
    fallback_urls: list[str],
    preferred_document_ids: list[int] | None = None,
) -> MediaDecision:
    preferred = set(preferred_document_ids or [])
    usable = [document for document in documents if document.get("source_class") != "radar_only"]
    usable.sort(
        key=lambda document: (
            0 if int(document["id"]) in preferred else 1,
            {
                "primary": 0,
                "independent": 1,
                "corroborating": 2,
            }.get(str(document.get("source_class")), 3),
        )
    )
    selected_document: dict[str, Any] | None = None
    preview_url = ""
    for document in usable:
        candidate_url = canonical_http_url(str(document.get("url") or ""))
        if candidate_url:
            selected_document = document
            preview_url = candidate_url
            break
    if not preview_url:
        preview_url = next(
            (canonical_http_url(url) for url in fallback_urls if canonical_http_url(url)),
            "",
        )

    image_url: str | None = None
    image_checks: list[dict[str, Any]] = []
    if selected_document and preview_url:
        for candidate_url, width, height in metadata_image_candidates(
            metadata_payload(selected_document)
        ):
            errors = validate_image_candidate(
                candidate_url,
                page_url=preview_url,
                width=width,
                height=height,
            )
            image_checks.append({"url": candidate_url, "errors": errors})
            if not errors:
                image_url = candidate_url
                break

    return MediaDecision(
        mode="telegram_preview" if preview_url else "text_only",
        preview_url=preview_url or None,
        image_url=image_url,
        source_document_id=int(selected_document["id"]) if selected_document else None,
        checks={
            "policy": "native Telegram preview first; no generated or substitute image",
            "image_candidates": image_checks,
            "safe_og_image_available": image_url is not None,
        },
    )
