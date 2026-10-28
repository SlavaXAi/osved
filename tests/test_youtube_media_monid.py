import asyncio
from unittest.mock import AsyncMock, Mock
from urllib.error import HTTPError

import pytest

from app.config import Settings
from app.db.repository import Repository
from app.media import youtube_monid as yt
from app.media.downloads import Downloader, MediaError


@pytest.fixture
def env(tmp_path):
    s = Settings(
        _env_file=None,
        data_dir=tmp_path,
        allowed_user_id=123,
        monid_api_key="test",
        media_youtube_monid_enabled=True,
        budget_hard_stop_usd=20,
    )
    r = Repository(s.database_path)
    r.migrate()
    return s, r


def receipt():
    return {
        "runId": "saved-run",
        "status": "COMPLETED",
        "cost": {"value": 0.005, "currency": "USD"},
        "output": {
            "ok": True,
            "video_id": "abcdefghijk",
            "title": "Example",
            "download_url": "https://mrscraper-files.s3.us-west-2.amazonaws.com/example.mp4",
        },
    }


def catalogue():
    return {"price": {"type": "PER_CALL", "amount": {"value": 0.01, "currency": "USD"}}}


def info():
    return {
        "title": "Example",
        "width": 854,
        "height": 480,
        "duration": 32,
        "vcodec": "h264",
        "acodec": "aac",
        "size": 2_000_000,
        "path": "/example.mp4",
    }


def test_paid_receipt_is_reused_without_a_second_dispatch(env):
    s, r = env
    transport = Mock(side_effect=[catalogue(), receipt()])
    answer = yt.resolve(s, "abcdefghijk", transport=transport)
    denied = Mock(side_effect=AssertionError("Do not pay twice"))
    assert yt.resolve(s, "abcdefghijk", transport=denied) == answer
    assert transport.call_count == 2 and not denied.called
    assert r.current_month_cost() == pytest.approx(0.005)


def test_unknown_post_result_is_not_automatically_replayed(env):
    s, r = env
    transport = Mock(side_effect=[catalogue(), TimeoutError()])
    with pytest.raises(TimeoutError):
        yt.resolve(s, "abcdefghijk", transport=transport)
    with pytest.raises(MediaError, match="media_paid_uncertain"):
        yt.resolve(s, "abcdefghijk", transport=transport)
    assert transport.call_count == 2 and r.current_month_cost() == pytest.approx(0.01)


def test_known_running_request_resumes_by_get_only(env, monkeypatch):
    s, r = env
    monkeypatch.setattr(yt.time, "sleep", lambda _: None)
    transport = Mock(
        side_effect=[catalogue(), {"runId": "saved-run", "status": "RUNNING"}, TimeoutError()]
    )
    with pytest.raises(TimeoutError):
        yt.resolve(s, "abcdefghijk", transport=transport)
    resume = Mock(return_value=receipt())
    assert yt.resolve(s, "abcdefghijk", transport=resume)["status"] == "COMPLETED"
    assert resume.call_count == 1 and resume.call_args.args[1:3] == ("GET", "/v1/runs/saved-run")
    assert r.current_month_cost() == pytest.approx(0.005)


def test_confirmed_rejection_releases_reservation_before_explicit_retry(env):
    s, r = env
    transport = Mock(
        side_effect=[
            catalogue(),
            HTTPError("https://api.monid.ai/v1/run", 402, "Rejected", {}, None),
        ]
    )
    with pytest.raises(MediaError, match="media_budget_blocked"):
        yt.resolve(s, "abcdefghijk", transport=transport)
    assert r.current_month_cost() == 0
    assert (
        yt.resolve(s, "abcdefghijk", transport=Mock(side_effect=[catalogue(), receipt()]))["status"]
        == "COMPLETED"
    )
    assert r.current_month_cost() == pytest.approx(0.005)


def test_price_and_monthly_budget_are_checked_before_paid_dispatch(env):
    s, r = env
    expensive = catalogue()
    expensive["price"]["amount"]["value"] = 0.02
    transport = Mock(return_value=expensive)
    with pytest.raises(MediaError, match="media_budget_blocked"):
        yt.resolve(s, "abcdefghijk", transport=transport)
    assert transport.call_count == 1 and r.current_month_cost() == 0
    r.record_usage("fixture", "existing", cost_usd=19.995)
    transport = Mock(return_value=catalogue())
    with pytest.raises(MediaError, match="media_budget_blocked"):
        yt.resolve(s, "abcdefghijk", transport=transport)
    assert transport.call_count == 1 and r.current_month_cost() == 19.995


def test_monid_micro_dollar_receipts_are_accounted_correctly():
    assert yt.receipt_cost(
        {"billing": {"reportedCost": {"value": 5000, "currency": "USD", "unit": "MICRO_DOLLAR"}}}
    ) == (0.005, False)


def test_monid_workspace_block_is_zero_charge_and_explicit_retry_can_proceed(env):
    s, r = env
    blocked = {
        "runId": "blocked-run",
        "status": "BLOCKED",
        "controls": [{"controlId": "workspace-budget"}],
    }
    assert (
        yt.resolve(s, "abcdefghijk", transport=Mock(side_effect=[catalogue(), blocked]))["status"]
        == "BLOCKED"
    )
    assert r.current_month_cost() == 0
    assert (
        yt.resolve(s, "abcdefghijk", transport=Mock(side_effect=[catalogue(), receipt()]))["status"]
        == "COMPLETED"
    )
    assert r.current_month_cost() == pytest.approx(0.005)


def test_menu_shows_only_resolution_of_real_downloaded_file():
    menu = yt.manifest(info())
    assert set(menu["items"][0]["choices"]) == {"144", "240", "360", "480", "mp3"}
    portrait = {**info(), "width": 480, "height": 854}
    assert set(yt.manifest(portrait)["items"][0]["choices"]) == set(menu["items"][0]["choices"])


@pytest.mark.parametrize(
    "url",
    [
        "http://mrscraper-files.s3.us-west-2.amazonaws.com/a",
        "https://127.0.0.1/a",
        "https://mrscraper-files.s3.us-west-2.amazonaws.com.evil.test/a",
        "https://u:p@mrscraper-files.s3.us-west-2.amazonaws.com/a",
        "file:///etc/passwd",
    ],
)
def test_untrusted_download_destinations_are_rejected(url):
    assert not yt.safe_url(url)


def test_inspection_uses_fallback_only_for_public_youtube_block(env, monkeypatch):
    s, _ = env
    service = Downloader(s)
    service.command = AsyncMock(side_effect=MediaError("youtube_blocked"))
    get = AsyncMock(return_value=info())
    monkeypatch.setattr(yt, "source", get)
    req = {
        "platform": "youtube",
        "external_id": "abcdefghijk",
        "source_url": "https://www.youtube.com/watch?v=abcdefghijk",
    }
    menu = asyncio.run(service.inspect(req))
    assert menu["backend"] == "monid_youtube" and get.await_count == 1
    service.command = AsyncMock(side_effect=MediaError("login_required"))
    with pytest.raises(MediaError, match="login_required"):
        asyncio.run(service.inspect(req))
    assert get.await_count == 1


def test_unavailable_hd_is_not_replaced_by_upscaled_480p(env, monkeypatch):
    s, _ = env
    service = Downloader(s)
    monkeypatch.setattr(yt, "source", AsyncMock(return_value=info()))
    service.command = AsyncMock(side_effect=AssertionError("Never upscale an unavailable quality"))
    req = {"platform": "youtube", "external_id": "abcdefghijk"}
    with pytest.raises(MediaError, match="quality_unavailable"):
        asyncio.run(yt.download(service, req, {"id": 1, "choice": "1080"}))
    assert not service.command.called


def test_existing_local_master_does_not_call_paid_provider(env, monkeypatch):
    s, _ = env
    service = Downloader(s)
    folder = service.root / "youtube" / "abcdefghijk"
    folder.mkdir(parents=True)
    (folder / "source.mp4").write_bytes(b"video")
    saved = {**info(), "size": 5, "created_at": yt.time.time()}
    yt.save(folder / "source.json", saved)
    denied = Mock(side_effect=AssertionError("Cached source must not call Monid"))
    monkeypatch.setattr(yt, "resolve", denied)
    assert asyncio.run(yt.source(service, {"external_id": "abcdefghijk"}))["size"] == 5
    assert not denied.called
