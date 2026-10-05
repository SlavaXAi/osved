from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db.repository import Repository
from app.pipeline.editorial import event_organizations


@dataclass(frozen=True)
class BaselineExportResult:
    output_dir: str
    delivered_cards: int
    integrity_check: str
    foreign_key_violations: int
    files: list[str]


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows),
        encoding="utf-8",
    )


def _schema_sql(database_path: Path) -> str:
    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    finally:
        connection.close()
    return "\n\n".join(f"{row[0].rstrip(';')};" for row in rows) + "\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _implementation_fingerprints() -> dict[str, dict[str, Any]]:
    project_root = Path(__file__).parents[2]
    relative_paths = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("config/sources.yaml"),
        Path("config/interests.yaml"),
        Path("config/style-profile.yaml"),
        Path("app/llm/openrouter.py"),
        Path("app/research/facts.py"),
        Path("app/pipeline/selection.py"),
        Path("app/pipeline/ranking.py"),
        Path("app/pipeline/cards.py"),
    ]
    relative_paths.extend(
        path.relative_to(project_root)
        for path in sorted((project_root / "app/db/migrations").glob("*.sql"))
    )
    fingerprints: dict[str, dict[str, Any]] = {}
    for relative_path in relative_paths:
        absolute_path = project_root / relative_path
        if not absolute_path.is_file():
            continue
        fingerprints[str(relative_path)] = {
            "sha256": _file_sha256(absolute_path),
            "bytes": absolute_path.stat().st_size,
        }
    return fingerprints


def export_baseline(
    repository: Repository,
    *,
    output_dir: Path,
    limit: int = 50,
) -> BaselineExportResult:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Baseline directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "snapshot.sqlite3"

    source = sqlite3.connect(f"file:{repository.database_path}?mode=ro", uri=True)
    target = sqlite3.connect(snapshot_path)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()

    with sqlite3.connect(snapshot_path) as connection:
        connection.execute("DELETE FROM runtime_settings")
        connection.commit()

    snapshot = Repository(snapshot_path)
    with sqlite3.connect(snapshot_path) as connection:
        integrity_check = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())

    cards = snapshot.baseline_delivery_rows(limit=limit)
    feedback = snapshot.export_feedback_rows()
    source_yield = snapshot.source_yield_rows()
    status = snapshot.status_counts()

    source_distribution = Counter(str(row["source_name"]) for row in cards)
    organization_distribution: Counter[str] = Counter()
    topic_distribution: Counter[str] = Counter()
    region_distribution: Counter[str] = Counter()
    reaction_distribution: Counter[str] = Counter()
    prompt_versions: Counter[str] = Counter()
    models: Counter[str] = Counter()
    link_counts: list[int] = []
    html_lengths: list[int] = []
    generation_costs: list[float] = []
    research_costs: list[float] = []
    for row in cards:
        organization_distribution.update(event_organizations(row))
        topic_distribution.update(_json_list(row.get("topics_json")))
        region_distribution.update(_json_list(row.get("regions_json")))
        if row.get("final_reaction"):
            reaction_distribution[str(row["final_reaction"])] += 1
        prompt_versions[str(row["prompt_version"])] += 1
        models[str(row.get("model") or "unknown")] += 1
        rendered = str(row["rendered_html"])
        link_counts.append(len(re.findall(r"<a\s+href=", rendered)))
        html_lengths.append(len(rendered))
        generation_costs.append(float(row.get("generation_cost_usd") or 0))
        research_costs.append(float(row.get("research_cost_usd") or 0))

    source_yield_summary = {
        "items": sum(int(row.get("items") or 0) for row in source_yield),
        "accepted_items": sum(int(row.get("accepted_items") or 0) for row in source_yield),
        "events": sum(int(row.get("events") or 0) for row in source_yield),
        "valid_cards": sum(int(row.get("valid_cards") or 0) for row in source_yield),
        "sent": sum(int(row.get("sent") or 0) for row in source_yield),
    }

    metrics = {
        "generated_at": datetime.now(UTC).isoformat(),
        "limit": limit,
        "delivered_cards": len(cards),
        "status": status,
        "source_distribution": dict(source_distribution),
        "organization_distribution": dict(organization_distribution),
        "topic_distribution": dict(topic_distribution),
        "region_distribution": dict(region_distribution),
        "final_reaction_distribution": dict(reaction_distribution),
        "prompt_versions": dict(prompt_versions),
        "models": dict(models),
        "average_rendered_html_length": (
            round(sum(html_lengths) / len(html_lengths), 2) if html_lengths else 0
        ),
        "average_link_count": round(sum(link_counts) / len(link_counts), 2) if link_counts else 0,
        "link_count_distribution": dict(Counter(link_counts)),
        "total_generation_cost_usd": round(sum(generation_costs), 6),
        "total_research_cost_usd": round(sum(research_costs), 6),
        "average_total_cost_per_delivered_card_usd": (
            round((sum(generation_costs) + sum(research_costs)) / len(cards), 6) if cards else 0
        ),
        "source_yield_summary": source_yield_summary,
        "integrity_check": integrity_check,
        "foreign_key_violations": foreign_key_violations,
    }

    _write_json(output_dir / "metrics.json", metrics)
    _write_jsonl(output_dir / "cards.jsonl", cards)
    _write_jsonl(output_dir / "feedback.jsonl", feedback)
    _write_json(output_dir / "source-yield.json", source_yield)
    _write_json(
        output_dir / "implementation.json",
        {
            "fingerprints": _implementation_fingerprints(),
            "prompt_versions_in_corpus": dict(prompt_versions),
            "models_in_corpus": dict(models),
        },
    )
    (output_dir / "schema.sql").write_text(_schema_sql(snapshot_path), encoding="utf-8")
    report = (
        "# V7 Phase 0 baseline\n\n"
        f"Generated: `{metrics['generated_at']}`\n\n"
        f"- Delivered cards in corpus: **{len(cards)}**\n"
        f"- Average HTML length: **{metrics['average_rendered_html_length']}**\n"
        f"- Average links per card: **{metrics['average_link_count']}**\n"
        f"- Average total cost per card: "
        f"**${metrics['average_total_cost_per_delivered_card_usd']}**\n"
        f"- Final reactions: `{json.dumps(dict(reaction_distribution), ensure_ascii=False)}`\n"
        f"- Source distribution: `{json.dumps(dict(source_distribution), ensure_ascii=False)}`\n"
        f"- Organization distribution: "
        f"`{json.dumps(dict(organization_distribution), ensure_ascii=False)}`\n"
        f"- Topic distribution: `{json.dumps(dict(topic_distribution), ensure_ascii=False)}`\n"
        f"- Region distribution: `{json.dumps(dict(region_distribution), ensure_ascii=False)}`\n"
        f"- SQLite integrity: **{integrity_check}**\n"
        f"- Foreign-key violations: **{foreign_key_violations}**\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    artifact_names = [
        "snapshot.sqlite3",
        "metrics.json",
        "cards.jsonl",
        "feedback.jsonl",
        "source-yield.json",
        "implementation.json",
        "schema.sql",
        "report.md",
    ]
    checksums = [f"{_file_sha256(output_dir / name)}  {name}" for name in artifact_names]
    (output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    artifact_names.append("checksums.sha256")
    manifest_name = "manifest.json"
    manifest = {
        "schema_version": "v7_phase0_baseline_v1",
        "private_artifact": True,
        "contains_rendered_cards_and_feedback": True,
        "contains_runtime_settings": False,
        "files": [*artifact_names, manifest_name],
        "checksummed_files": artifact_names[:-1],
    }
    _write_json(output_dir / manifest_name, manifest)
    artifact_names.append(manifest_name)
    return BaselineExportResult(
        output_dir=str(output_dir),
        delivered_cards=len(cards),
        integrity_check=integrity_check,
        foreign_key_violations=foreign_key_violations,
        files=artifact_names,
    )
