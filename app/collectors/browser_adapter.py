"""Optional source adapter for snapshots collected by the local browser bridge."""

from __future__ import annotations

from app.collectors.base import (
    CollectedItem,
    CollectionBatch,
    CollectionRequest,
    SourceCapabilities,
    SourceHealth,
)
from app.collectors.browser_queue import BrowserTaskQueue
from app.config import SourceConfig


class BrowserSourceAdapter:
    def __init__(self, queue: BrowserTaskQueue) -> None:
        self.queue = queue

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(strategies=("for_you",), requires_auth=True, incremental=True)

    def healthcheck(self, source: SourceConfig) -> SourceHealth:
        latest = self.queue.latest_source_task(source.id)
        if latest and latest["error_code"] in {"challenge", "login_required"}:
            return SourceHealth(status="login_required", error=latest["error_code"])
        if latest and latest["status"] in {"staged", "committed", "completed"}:
            return SourceHealth()
        # A new installation must be allowed to enqueue its first task. The
        # collector will report that it is waiting for a visible logged-in tab.
        return SourceHealth()

    def collect(self, request: CollectionRequest) -> CollectionBatch:
        source = request.source
        if source.strategy != "for_you":
            return CollectionBatch(status="failed", error="unsupported_browser_strategy")
        ready = self.queue.ready_results(source.id, limit=1)
        if ready:
            row = ready[0]
            items = [CollectedItem(
                external_id=post["id"], canonical_url=post["url"], text=post["text"],
                published_at=post["published_at"], author=post["author"],
                content_type="post", discovery_mode="for_you", outbound_urls=post["links"],
                raw_payload={
                    "browser_task_id": row["id"], "feed_mode": "for_you",
                    "capture_scope": "visible_posts_only", "admission_mode": "trigger_only",
                },
            ) for post in row["result"]["posts"]]
            # A snapshot is bounded to 80 posts and must be acknowledged as a whole.
            # Do not truncate a durable result if the request limit changed meanwhile.
            return CollectionBatch(
                items=items, status="ok" if items else "empty", next_cursor=row["id"],
                metadata={"browser_task_ids": [row["id"]], "ack_required": True},
            )
        latest = self.queue.latest_source_task(source.id)
        if latest and latest["error_code"] in {"challenge", "login_required"}:
            return CollectionBatch(status="login_required", error=latest["error_code"])
        if not self.queue.has_open_task(source.id):
            interval = max(30, source.poll_interval_minutes) * 60
            slot = int(request.now.timestamp()) // interval
            task_key = f"{source.id}:visible_for_you:{slot}"
            if self.queue.task_exists(task_key):
                return CollectionBatch(status="empty", metadata={
                    "waiting_for": "next_capture_window", "capture_scope": "visible_posts_only",
                })
            self.queue.enqueue(
                source.id, task_key=task_key,
                max_items=max(1, min(request.limit, 80)), ttl_seconds=min(86400, interval * 2),
            )
        return CollectionBatch(status="pending", metadata={
            "waiting_for": "visible_authorized_for_you_tab", "capture_scope": "visible_posts_only",
        })

    def acknowledge(self, batch: CollectionBatch) -> None:
        """Call ONLY after normal idempotent ingest committed successfully."""
        for task_id in batch.metadata.get("browser_task_ids", []):
            self.queue.mark_committed(task_id)
            self.queue.complete(task_id)
