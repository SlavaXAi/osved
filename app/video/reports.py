"""Durable, bounded, on-demand report generation; no implicit paid retries."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.collectors.x_monid import monid_cli_path
from app.config import Settings
from app.db.repository import Repository
from app.llm.accounting import finite_money, resolve_usage_cost
from app.video.report_format import (
    MODEL,
    VERSION,
    create_pdf,
    render_citations,
    request_payload,
    source_payload,
)
from app.video.transcripts import youtube_video_id

TRANSCRIPT_COST = 0.0075
ENDPOINT = "/starvibe/youtube-video-transcript"
MAX_JSON_BYTES = 8_000_000


def report_progress(callback: Callable[[str], None] | None, stage: str) -> None:
    # Cosmetic reporting must never turn a successful paid call into a retry.
    if callback:
        with suppress(Exception):
            callback(stage)


class ReportError(Exception):
    def __init__(self, status: str):
        self.status = status
        super().__init__(status)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".report-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path: Path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReportError("insufficient_context")
    return json.loads(path.read_text(encoding="utf-8"))


def run_monid(command: list[str]):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=55,
        check=False,
        env={**os.environ, "NO_COLOR": "1"},
    )


def model_transport(payload: dict, key: str) -> dict:
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "OSVED-summary-AB/1",
        },
    )
    with urlopen(request, timeout=300) as response:
        raw = response.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ReportError("invalid_output")
    return json.loads(raw)


class VideoReportService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        *,
        monid_runner=run_monid,
        transport=model_transport,
    ):
        self.settings, self.repository = settings, repository
        self.root = settings.data_dir / "video-reports"
        self.monid_runner, self.transport = monid_runner, transport

    def directory(self, video: str) -> Path:
        return self.root / youtube_video_id(video) / VERSION

    def cached_pdf(self, video: str) -> Path | None:
        directory = self.directory(video)
        pdf, audit = directory / "report.pdf", directory / "audit.json"
        if pdf.is_file() and audit.is_file() and pdf.stat().st_size <= 40_000_000:
            try:
                if hashlib.sha256(pdf.read_bytes()).hexdigest() == read_json(audit)["pdf_sha256"]:
                    return pdf
            except (ValueError, KeyError, OSError):
                pass
        return None

    def _reserve(
        self, provider: str, amount: float, video: str, *, operation: str | None = None
    ) -> int:
        with self.repository.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            spent = db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                "WHERE strftime('%Y-%m',created_at)=strftime('%Y-%m','now')"
            ).fetchone()[0]
            if (
                self.settings.video_report_budget_enabled
                and spent + amount > self.settings.budget_hard_stop_usd
            ):
                raise ReportError("budget_blocked")
            row = db.execute(
                "INSERT INTO usage_ledger(provider,operation,cost_usd,metadata_json) "
                "VALUES(?,?,?,?)",
                (
                    provider,
                    operation or "video.pdf." + provider,
                    amount,
                    json.dumps(
                        {
                            "video_id": video,
                            "cost_estimated": True,
                            "billing_basis": "reservation_before_request",
                        }
                    ),
                ),
            )
            return int(row.lastrowid)

    def _settle(self, ledger_id: int, cost: float | None, metadata: dict) -> None:
        if cost is None:
            return  # An uncertain bill retains its reservation, never silently becomes free.
        with self.repository.connection() as db:
            db.execute(
                "UPDATE usage_ledger SET cost_usd=?,metadata_json=? WHERE id=?",
                (cost, json.dumps(metadata, ensure_ascii=False), ledger_id),
            )

    def _model_response(
        self,
        video: str,
        directory: Path,
        request: dict,
        reserve: float,
        *,
        progress: Callable[[str], None] | None,
        operation: str | None = None,
    ) -> tuple[dict, dict]:
        raw, started = directory / "response.json", directory / "llm-started.json"
        if raw.is_file():
            return read_json(raw), read_json(started)
        rejected_usage = {
            "video_id": video,
            "model": MODEL,
            "cost_estimated": False,
            "billing_basis": "rejected_before_dispatch",
            "http_status": 402,
        }
        if started.exists():
            previous = read_json(started)
            failure_path = directory / "llm-errors" / f"{previous['ledger_id']}.json"
            failure = read_json(failure_path) if failure_path.is_file() else {}
            # Only a recorded HTTP 402 proves this attempt was rejected before
            # generation. Timeouts and older unknown attempts stay blocked.
            if not (
                failure.get("ledger_id") == previous["ledger_id"]
                and failure.get("http_status") == 402
                and failure.get("rejected_before_dispatch") is True
            ):
                raise ReportError("uncertain")
            # Also closes a crash between saving the rejection and settling it.
            self._settle(previous["ledger_id"], 0.0, rejected_usage)
        if self.settings.openrouter_api_key is None:
            raise ReportError("disabled")
        ledger_id = self._reserve("openrouter", reserve, video, operation=operation)
        marker = {"ledger_id": ledger_id, "reserve_usd": reserve}
        save_json(directory / "request.json", request)
        save_json(started, marker)
        report_progress(progress, "writing")
        try:
            response = self.transport(request, self.settings.openrouter_api_key.get_secret_value())
        except Exception as exc:
            http_status = exc.code if isinstance(exc, HTTPError) else None
            rejected = http_status == 402
            status = "insufficient_credits" if rejected else "uncertain"
            # Persist diagnostic codes, not exception text, response bodies or
            # headers, which can contain credentials or source content.
            save_json(
                directory / "llm-errors" / f"{ledger_id}.json",
                {
                    **marker,
                    "status": status,
                    "exception_type": type(exc).__name__,
                    "http_status": http_status,
                    "rejected_before_dispatch": rejected,
                    "recorded_at": time.time(),
                },
            )
            if rejected:
                self._settle(ledger_id, 0.0, rejected_usage)
            raise ReportError(status) from exc
        save_json(raw, response)
        return response, marker

    def _transcript(self, video: str) -> dict:
        folder = self.root / video
        output, started, receipt = (
            folder / name
            for name in ("transcript.json", "monid-started.json", "monid-receipt.json")
        )
        if output.is_file():
            return source_payload(read_json(output), video)
        marker = read_json(started) if started.exists() else None
        old = read_json(receipt) if receipt.exists() else {}
        run_id = old.get("runId") or old.get("id")
        if old.get("status") in {"FAILED", "STOPPED", "TIME_OUT", "BLOCKED"}:
            raise ReportError("unavailable" if old["status"] != "BLOCKED" else "provider_blocked")
        if marker and not run_id:
            raise ReportError("uncertain")
        if not marker:
            if self.settings.monid_api_key is None:
                raise ReportError("disabled")
            executable = monid_cli_path()
            ledger_id = self._reserve("monid", TRANSCRIPT_COST, video)
            marker = {"ledger_id": ledger_id, "video_id": video}
            save_json(started, marker)
            command = [
                executable,
                "run",
                "-p",
                "apify",
                "-e",
                ENDPOINT,
                "-i",
                json.dumps(
                    {
                        "youtube_url": f"https://www.youtube.com/watch?v={video}",
                        "include_transcript_text": True,
                    }
                ),
                "-w",
                "30",
                "-o",
                str(output),
                "-j",
            ]
        else:
            command = [monid_cli_path(), "runs", "get", "-r", str(run_id), "-w", "30", "-j"]
        try:
            result = self.monid_runner(command)
            value = json.loads(result.stdout)
            if not isinstance(value, dict):
                raise ValueError("Invalid receipt")
        except Exception as exc:
            raise ReportError("processing" if run_id else "uncertain") from exc
        if run_id and not (value.get("runId") or value.get("id")):
            value["runId"] = run_id
        save_json(receipt, value)
        cost = value.get("cost") or {}
        reported = finite_money(cost.get("value")) if cost.get("currency") == "USD" else None
        state = value.get("status")
        if state == "BLOCKED":
            reported = 0.0
        if state not in {"READY", "RUNNING"}:
            self._settle(
                marker["ledger_id"],
                reported,
                {
                    "video_id": video,
                    "run_id": value.get("runId") or value.get("id"),
                    "cost_estimated": False,
                    "billing_basis": "monid_receipt",
                },
            )
        if state in {"READY", "RUNNING"}:
            raise ReportError("processing")
        if state == "BLOCKED":
            raise ReportError("provider_blocked")
        if state != "COMPLETED" or result.returncode:
            raise ReportError("unavailable")
        if not output.exists() and value.get("output") is not None:
            save_json(output, value["output"])
        if not output.exists():
            raise ReportError("unavailable")
        return source_payload(read_json(output), video)

    def build(self, video: str, *, progress: Callable[[str], None] | None = None) -> Path:
        report_progress(progress, "preparing")
        video = youtube_video_id(video)
        cached = self.cached_pdf(video)
        if cached:
            report_progress(progress, "cache")
            return cached
        directory = self.directory(video)
        directory.mkdir(parents=True, exist_ok=True)
        # The lock covers all report versions for this video, including transcript requests.
        with (directory.parent / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ReportError("processing") from exc
            cached = self.cached_pdf(video)
            if cached:
                report_progress(progress, "cache")
                return cached
            try:
                if (
                    self.settings.video_report_budget_enabled
                    and self.settings.video_report_max_cost_usd < TRANSCRIPT_COST
                ):
                    raise ReportError("budget_blocked")
                if self.settings.openrouter_api_key is None:
                    raise ReportError("disabled")
                report_progress(progress, "transcript")
                source = self._transcript(video)
                save_json(directory / "source.json", source)
                request, reserve = request_payload(source)
                if (
                    self.settings.video_report_budget_enabled
                    and reserve + TRANSCRIPT_COST > self.settings.video_report_max_cost_usd
                ):
                    raise ReportError("insufficient_context")
                response, marker = self._model_response(
                    video, directory, request, reserve, progress=progress
                )
                usage = resolve_usage_cost(
                    response.get("usage"),
                    model=MODEL,
                    price_ceiling_per_million=(0.75, 3.75),
                    reservation_usd=marker["reserve_usd"],
                )
                self._settle(
                    marker["ledger_id"],
                    usage.cost_usd,
                    {"video_id": video, "model": MODEL, **usage.metadata()},
                )
                report_progress(progress, "formatting")
                choice = response.get("choices", [{}])[0]
                original = choice.get("message", {}).get("content")
                if (
                    response.get("model") != MODEL
                    or choice.get("finish_reason") != "stop"
                    or not isinstance(original, str)
                    or not original.strip()
                ):
                    raise ReportError("invalid_output")
                (directory / "original.md").write_text(original, encoding="utf-8")
                rendered = render_citations(original, source)
                (directory / "report.md").write_text(rendered, encoding="utf-8")
                pdf = directory / "report.pdf"
                report_progress(progress, "pdf")
                audit = create_pdf(pdf, original, source)
                assert (directory / "original.md").read_text() == original
                audit.update(
                    input_sha256=source["input_sha256"],
                    generation_count=1,
                    response_id=response.get("id"),
                    usage=usage.metadata(),
                )
                save_json(directory / "audit.json", audit)
                return pdf
            except ReportError:
                raise
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                known = {"unavailable", "insufficient_context", "incomplete_transcript"}
                raise ReportError(str(exc) if str(exc) in known else "invalid_output") from exc

    def cached_tldr(self, video: str) -> str | None:
        from app.video.tldr import VERSION as TLDR_VERSION
        from app.video.tldr import extract_tldr

        video = youtube_video_id(video)
        if self.cached_pdf(video):
            try:
                return extract_tldr(
                    (self.directory(video) / "original.md").read_text(), full_report=True
                )
            except (ValueError, OSError):
                pass
        path = self.root / video / ("tldr-" + TLDR_VERSION) / "result.json"
        if path.is_file():
            try:
                value = read_json(path)
                text = extract_tldr(value["original"])
                if hashlib.sha256(text.encode()).hexdigest() == value["text_sha256"]:
                    return text
            except (ValueError, OSError, KeyError, TypeError):
                pass
        return None

    def build_tldr(self, video: str, *, progress: Callable[[str], None] | None = None) -> str:
        from app.video.tldr import VERSION as TLDR_VERSION
        from app.video.tldr import extract_tldr, tldr_request

        video = youtube_video_id(video)
        report_progress(progress, "preparing")
        cached = self.cached_tldr(video)
        if cached is not None:
            report_progress(progress, "cache")
            return cached
        folder = self.root / video
        directory = folder / ("tldr-" + TLDR_VERSION)
        directory.mkdir(parents=True, exist_ok=True)
        # Both outputs serialize on the SAME video lock and reuse transcript.json.
        with (folder / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ReportError("processing") from exc
            cached = self.cached_tldr(video)
            if cached is not None:
                report_progress(progress, "cache")
                return cached
            try:
                if self.settings.openrouter_api_key is None:
                    raise ReportError("disabled")
                if (
                    self.settings.video_report_budget_enabled
                    and self.settings.video_report_max_cost_usd < TRANSCRIPT_COST
                ):
                    raise ReportError("budget_blocked")
                report_progress(progress, "transcript")
                source = self._transcript(video)
                request, reserve = tldr_request(source)
                if (
                    self.settings.video_report_budget_enabled
                    and reserve + TRANSCRIPT_COST > self.settings.video_report_max_cost_usd
                ):
                    raise ReportError("insufficient_context")
                response, marker = self._model_response(
                    video,
                    directory,
                    request,
                    reserve,
                    progress=progress,
                    operation="video.tldr.openrouter",
                )
                usage = resolve_usage_cost(
                    response.get("usage"),
                    model=MODEL,
                    price_ceiling_per_million=(0.75, 3.75),
                    reservation_usd=marker["reserve_usd"],
                )
                self._settle(
                    marker["ledger_id"],
                    usage.cost_usd,
                    {"video_id": video, "model": MODEL, **usage.metadata()},
                )
                report_progress(progress, "formatting")
                choice = response.get("choices", [{}])[0]
                original = choice.get("message", {}).get("content")
                if response.get("model") != MODEL or choice.get("finish_reason") != "stop":
                    raise ReportError("invalid_output")
                text = extract_tldr(original)
                save_json(
                    directory / "result.json",
                    {
                        "original": original,
                        "input_sha256": source["input_sha256"],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "response_id": response.get("id"),
                        "generation_count": 1,
                    },
                )
                return text
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                known = {"unavailable", "insufficient_context", "incomplete_transcript"}
                raise ReportError(str(exc) if str(exc) in known else "invalid_output") from exc
