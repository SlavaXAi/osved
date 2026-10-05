from app.pipeline.media import select_preview_first_media, validate_image_candidate


def test_preview_first_media_prefers_primary_and_records_safe_og_image() -> None:
    decision = select_preview_first_media(
        [
            {
                "id": 2,
                "url": "https://analysis.example/report",
                "source_class": "independent",
                "metadata_json": "{}",
            },
            {
                "id": 1,
                "url": "https://labs.example/release",
                "source_class": "primary",
                "metadata": {
                    "ogImage": {
                        "url": "https://labs.example/media/release-hero.jpg",
                        "width": 1200,
                        "height": 630,
                    }
                },
            },
        ],
        fallback_urls=["https://t.me/radar/1"],
    )

    assert decision.mode == "telegram_preview"
    assert decision.preview_url == "https://labs.example/release"
    assert decision.image_url == "https://labs.example/media/release-hero.jpg"
    assert decision.source_document_id == 1
    assert decision.checks["safe_og_image_available"] is True


def test_media_layer_rejects_generic_or_unrelated_images_without_substitution() -> None:
    decision = select_preview_first_media(
        [
            {
                "id": 7,
                "url": "https://company.example/news/model",
                "source_class": "primary",
                "metadata": {
                    "ogImage": [
                        "https://company.example/assets/default-logo.png",
                        "https://stock.example/random-ai.jpg",
                    ]
                },
            }
        ],
        fallback_urls=[],
    )

    assert decision.mode == "telegram_preview"
    assert decision.image_url is None
    assert len(decision.checks["image_candidates"]) == 2
    assert decision.checks["safe_og_image_available"] is False


def test_image_dimensions_and_ratio_are_checked_when_metadata_provides_them() -> None:
    assert validate_image_candidate(
        "https://example.com/media/tall.jpg",
        page_url="https://example.com/news",
        width=300,
        height=900,
    ) == ["image is too small", "image aspect ratio is unsuitable"]


def test_hook_document_preference_wins_within_usable_evidence() -> None:
    decision = select_preview_first_media(
        [
            {
                "id": 1,
                "url": "https://company.example/release",
                "source_class": "primary",
            },
            {
                "id": 2,
                "url": "https://benchmark.example/report",
                "source_class": "independent",
            },
        ],
        fallback_urls=[],
        preferred_document_ids=[2],
    )

    assert decision.preview_url == "https://benchmark.example/report"
    assert decision.source_document_id == 2


def test_radar_or_broken_preference_cannot_poison_preview_provenance() -> None:
    documents = [
        {
            "id": 3,
            "url": "https://x.com/radar/status/1",
            "source_class": "radar_only",
        },
        {
            "id": 4,
            "url": "not-an-http-url",
            "source_class": "independent",
        },
        {
            "id": 5,
            "url": "https://company.example/verified-release",
            "source_class": "primary",
        },
    ]

    radar_preference = select_preview_first_media(
        documents,
        fallback_urls=["https://fallback.example/story"],
        preferred_document_ids=[3],
    )
    broken_preference = select_preview_first_media(
        documents,
        fallback_urls=["https://fallback.example/story"],
        preferred_document_ids=[4],
    )

    assert radar_preference.preview_url == "https://company.example/verified-release"
    assert radar_preference.source_document_id == 5
    assert broken_preference.preview_url == "https://company.example/verified-release"
    assert broken_preference.source_document_id == 5
