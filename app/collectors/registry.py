from __future__ import annotations

from app.collectors.base import SourceAdapter
from app.collectors.native_adapters import NativeAdapter
from app.config import Settings, SourceConfig
from app.db.repository import Repository


def adapter_id_for(source: SourceConfig, settings: Settings, *, provider: str = "auto") -> str:
    if source.adapter_id != "auto":
        return source.adapter_id
    if source.kind.value == "telegram":
        if provider == "mtproto" or (
            provider == "auto" and settings.credential_status()["telegram_reader"]
        ):
            return "telegram_mtproto"
        return "monid_telegram"
    if source.kind.value == "x":
        return "monid_x_author"
    return source.kind.value


def build_adapter_registry(settings: Settings, repository: Repository) -> dict[str, SourceAdapter]:
    result = {
        key: NativeAdapter(settings, key)
        for key in ("rss", "github", "monid_x_author", "monid_telegram", "telegram_mtproto")
    }
    # Imported here so individual adapters remain optional and the original CLI is independent.
    from app.collectors.platforms import build_platform_adapters

    result.update(build_platform_adapters(settings, repository))
    from app.collectors.browser_adapter import BrowserSourceAdapter
    from app.collectors.browser_queue import BrowserTaskQueue

    result["browser_x"] = BrowserSourceAdapter(BrowserTaskQueue(repository.database_path))
    return result
