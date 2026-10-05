from app.pipeline.signal_quality import assess_signal_quality, syndication_identity


def test_many_trigger_repetitions_do_not_become_evidence() -> None:
    rows = [
        {
            "id": index,
            "source_id": f"channel{index}",
            "source_kind": "telegram",
            "source_url": f"https://t.me/channel{index}/1",
            "source_role": "radar",
            "original_source_url": "https://company.example/release",
        }
        for index in range(1, 11)
    ]
    result = assess_signal_quality(rows)
    assert result["spread_score"] == 100
    assert result["source_count"] == 10
    assert result["publisher_family_count"] == 1
    assert result["evidence_status"] == "unverified"
    assert result["requires_verification"]
    assert result["primary_family_count"] == 0
    result = assess_signal_quality(
        [
            {
                "id": 11,
                "source_role": "radar",
                "evidence_role": "primary",
                "source_url": "https://example.org/repeated-claim",
            }
        ]
    )
    assert result["evidence_status"] == "unverified"


def test_primary_and_verified_independent_are_separate_from_spread() -> None:
    primary = {
        "id": 1,
        "source_id": "company",
        "source_kind": "rss",
        "source_url": "https://company.example/release",
        "source_role": "primary",
    }
    opinion = {
        "id": 2,
        "source_id": "reddit",
        "source_kind": "reddit",
        "source_url": "https://reddit.com/r/a/comments/1",
        "evidence_role": "independent",
        "admission_mode": "trigger_only",
        "verified": True,
    }
    result = assess_signal_quality([primary, primary, opinion])
    assert result["mention_count"] == 2
    assert result["evidence_status"] == "primary_only"
    review = {
        "id": 3,
        "source_url": "https://review.example/story",
        "evidence_role": "independent",
        "verified": True,
        "source_kind": "web",
    }
    result = assess_signal_quality([primary, opinion, review])
    assert result["evidence_status"] == "primary_and_independent"
    assert not result["requires_verification"]
    assert result["reference_ids"] == [1, 2, 3]


def test_independent_flag_requires_verification_and_distinct_family() -> None:
    rows = [
        {"id": 1, "source_url": "https://company.example/release", "source_role": "primary"},
        {
            "id": 2,
            "source_url": "https://company.example/blog",
            "evidence_role": "independent",
            "verified": True,
        },
        {"id": 3, "source_url": "https://news.example/story", "evidence_role": "independent"},
    ]
    result = assess_signal_quality(rows)
    assert result["verified_independent_family_count"] == 0
    assert result["evidence_status"] == "primary_only"
    assert assess_signal_quality([])["evidence_status"] == "unverified"
    assert (
        syndication_identity({"original_source_url": "https://example.com/news/item?utm_source=a"})
        == "example.com/news/item"
    )
