from pathlib import Path

import pytest

from app.collectors.rss import RSSCollector, RSSCollectorError, parse_feed
from app.config import Settings, load_source_registry
from app.db.repository import Repository

PROJECT_ROOT = Path(__file__).parents[1]

RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Official AI News</title>
    <item>
      <title>New agent model released</title>
      <link>https://example.com/news/agent?utm_source=rss</link>
      <guid>release-agent-1</guid>
      <pubDate>Thu, 13 Aug 2026 08:00:00 GMT</pubDate>
      <description><![CDATA[
        A new model can keep focus on long tasks and operate agents.
      ]]></description>
      <content:encoded><![CDATA[The release includes a benchmark report at
      <a href="https://example.com/report?utm_medium=feed">the official report</a>.]]>
      </content:encoded>
    </item>
  </channel>
</rss>
"""

ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Research</title>
  <entry>
    <title>Robotics release</title>
    <id>tag:example.com,2026:robotics</id>
    <updated>2026-08-13T09:00:00Z</updated>
    <link rel="alternate" href="https://example.com/robotics" />
    <summary>New control software for humanoid robots.</summary>
  </entry>
</feed>
"""


def test_parse_rss_and_atom_without_copying_raw_markup() -> None:
    rss_item = parse_feed(RSS_FIXTURE, limit=5)[0]
    atom_item = parse_feed(ATOM_FIXTURE, limit=5)[0]

    assert rss_item.external_id == "release-agent-1"
    assert rss_item.source_url == "https://example.com/news/agent"
    assert "<a href" not in rss_item.original_text
    assert rss_item.outbound_urls == ["https://example.com/report"]
    assert atom_item.external_id == "tag:example.com,2026:robotics"
    assert atom_item.published_at == "2026-08-13T09:00:00+00:00"


def test_parser_rejects_dtd_before_xml_processing() -> None:
    payload = b'<?xml version="1.0"?><!DOCTYPE rss [<!ENTITY x "boom">]><rss>&x;</rss>'
    with pytest.raises(RSSCollectorError, match="forbidden"):
        parse_feed(payload, limit=5)


def test_rss_collection_is_idempotent_and_free(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "telegram-reader.session",
    )
    repository = Repository(settings.database_path)
    repository.migrate()
    repository.sync_sources(load_source_registry(settings.sources_file))
    collector = RSSCollector(settings, repository, fetcher=lambda source: RSS_FIXTURE)

    first = collector.collect(limit_per_source=5)
    second = collector.collect(limit_per_source=5)

    assert first.sources_attempted == 4
    assert first.items_seen == 4
    assert first.items_inserted == 4
    assert first.cost_usd == 0
    assert second.items_inserted == 0
    assert repository.current_month_cost() == 0
