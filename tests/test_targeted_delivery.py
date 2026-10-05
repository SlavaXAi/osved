from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import app.cli as cli_module
import app.telegram_bot.bot as bot_module
from app.config import DeliveryFormat, Settings, load_source_registry
from app.db.repository import Repository
from app.telegram_bot.bot import CardsNotDeliverable, send_cards_by_ids

PROJECT_ROOT = Path(__file__).parents[1]
OWNER_ID = 123


def smoke_cards(repository: Repository, count: int) -> list[dict]:
    cards: list[dict] = []
    for index in range(count):
        card = repository.create_smoke_delivery(
            OWNER_ID,
            f"https://example.com/release-{index}",
        )
        repository.mark_delivery_error(card["delivery_id"], "test retry")
        cards.append(card)
    return cards


def test_deliverable_cards_by_ids_preserves_explicit_order_and_filters_state(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "targeted.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    first, second, sent, invalid = smoke_cards(repository, 4)
    repository.mark_delivery_sent(sent["delivery_id"], 9001)
    with repository.connection() as connection:
        connection.execute(
            "UPDATE cards SET validation_status = 'invalid' WHERE id = ?",
            (invalid["card_id"],),
        )

    rows = repository.deliverable_cards_by_ids(
        OWNER_ID,
        [
            second["card_id"],
            first["card_id"],
            sent["card_id"],
            invalid["card_id"],
            999_999,
            second["card_id"],
        ],
    )

    assert [row["card_id"] for row in rows] == [second["card_id"], first["card_id"]]


def test_send_cards_by_ids_does_not_consume_unselected_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Repository(tmp_path / "sender.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    first, second, unselected = smoke_cards(repository, 3)
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "reader.session",
        allowed_user_id=OWNER_ID,
    )

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class FakeBot:
        def __init__(self) -> None:
            self.session = FakeSession()
            self.preview_urls: list[str] = []

        async def send_message(self, **kwargs):
            self.preview_urls.append(kwargs["link_preview_options"].url)
            return SimpleNamespace(message_id=1000 + len(self.preview_urls))

    fake_bot = FakeBot()
    monkeypatch.setattr(bot_module, "build_bot", lambda _settings: fake_bot)

    result = asyncio.run(
        send_cards_by_ids(
            settings,
            repository,
            [second["card_id"], first["card_id"]],
            delivery_format=DeliveryFormat.REGULAR,
        )
    )

    assert result == {"cards_requested": 2, "cards_seen": 2, "sent": 2, "errors": 0}
    assert fake_bot.preview_urls == [
        "https://example.com/release-1",
        "https://example.com/release-0",
    ]
    assert fake_bot.session.closed is True
    with repository.connection() as connection:
        states = {
            int(row["card_id"]): row["delivery_status"]
            for row in connection.execute(
                "SELECT card_id, delivery_status FROM deliveries"
            ).fetchall()
        }
    assert states[first["card_id"]] == "sent"
    assert states[second["card_id"]] == "sent"
    assert states[unselected["card_id"]] == "failed"


def test_send_cards_by_ids_fails_closed_before_opening_bot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = Repository(tmp_path / "fail-closed.sqlite3")
    repository.migrate()
    repository.sync_sources(
        load_source_registry(PROJECT_ROOT / "tests/fixtures/config/sources.yaml")
    )
    valid, already_sent = smoke_cards(repository, 2)
    repository.mark_delivery_sent(already_sent["delivery_id"], 9002)
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        sources_file=PROJECT_ROOT / "tests/fixtures/config/sources.yaml",
        interests_file=PROJECT_ROOT / "tests/fixtures/config/interests.yaml",
        telegram_session_path=tmp_path / "reader.session",
        allowed_user_id=OWNER_ID,
    )
    monkeypatch.setattr(
        bot_module,
        "build_bot",
        lambda _settings: (_ for _ in ()).throw(AssertionError("must not open bot")),
    )

    try:
        asyncio.run(
            send_cards_by_ids(
                settings,
                repository,
                [valid["card_id"], already_sent["card_id"]],
            )
        )
    except CardsNotDeliverable as exc:
        assert exc.card_ids == [already_sent["card_id"]]
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected CardsNotDeliverable")

    assert repository.deliverable_cards_by_ids(OWNER_ID, [valid["card_id"]])


def test_deliver_cards_cli_preserves_argument_order_and_regular_format(monkeypatch) -> None:
    settings = object()
    repository = object()
    captured: dict = {}

    async def fake_send(
        actual_settings,
        actual_repository,
        card_ids,
        *,
        delivery_format,
    ):
        captured.update(
            settings=actual_settings,
            repository=actual_repository,
            card_ids=card_ids,
            delivery_format=delivery_format,
        )
        return {"cards_requested": 3, "cards_seen": 3, "sent": 3, "errors": 0}

    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))
    monkeypatch.setattr(cli_module, "send_cards_by_ids", fake_send)

    result = CliRunner().invoke(
        cli_module.app,
        ["deliver-cards", "9", "3", "7", "--format", "regular"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "settings": settings,
        "repository": repository,
        "card_ids": [9, 3, 7],
        "delivery_format": DeliveryFormat.REGULAR,
    }
    assert '"sent": 3' in result.output


def test_deliver_candidate_cards_cli_creates_explicit_quarantined_batch(monkeypatch) -> None:
    settings = object()
    repository = object()
    captured: dict = {}

    async def fake_send(
        actual_settings,
        actual_repository,
        *,
        batch_id,
        card_ids,
    ):
        captured.update(
            settings=actual_settings,
            repository=actual_repository,
            batch_id=batch_id,
            card_ids=card_ids,
        )
        return {"cards_seen": 2, "sent": 2, "errors": 0}

    monkeypatch.setattr(cli_module, "bootstrap", lambda: (settings, repository))
    monkeypatch.setattr(cli_module, "send_candidate_batch", fake_send)

    result = CliRunner().invoke(
        cli_module.app,
        ["deliver-candidate-cards", "110", "113"],
    )

    assert result.exit_code == 0, result.output
    assert captured["settings"] is settings
    assert captured["repository"] is repository
    assert captured["batch_id"].startswith("candidate:")
    assert captured["card_ids"] == [110, 113]
    assert '"sent": 2' in result.output
