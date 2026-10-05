import pytest

from app.research.web import classify_source


@pytest.mark.parametrize(
    "url",
    [
        "https://www.instagram.com/p/example/",
        "https://www.threads.com/@person/post/example",
        "https://www.threads.net/@person/post/example",
        "https://mobile.twitter.com/person/status/123",
        "https://x.com/person/status/123",
    ],
)
def test_social_trigger_cannot_be_promoted_by_saved_primary_url(url: str) -> None:
    assert classify_source(url) == "radar_only"
    assert classify_source(url, known_primary_urls={url}) == "radar_only"


def test_social_domain_boundary_does_not_match_unrelated_brand() -> None:
    assert classify_source("https://x.ai/news/release") == "primary"
    assert classify_source("https://twitter.com.example.org/report") == "corroborating"
