from collections import defaultdict
from pathlib import Path

from app.db.repository import Repository
from app.ops.calibration import balanced_research_events


def test_balanced_research_events_round_robins_sources(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "test.sqlite3")
    repository.migrate()
    repository.events_needing_research = (
        lambda limit: [  # type: ignore[method-assign]
            {"id": 1},
            {"id": 2},
            {"id": 3},
            {"id": 4},
        ][:limit]
    )
    source_by_event = {1: "a", 2: "a", 3: "b", 4: "b"}
    repository.event_trigger_sources = lambda event_id: [  # type: ignore[method-assign]
        {"source_id": source_by_event[event_id]}
    ]

    selected = balanced_research_events(repository, limit=4)

    assert [event["id"] for event in selected] == [1, 3, 2, 4]
    counts = defaultdict(int)
    for event in selected[:2]:
        counts[source_by_event[event["id"]]] += 1
    assert counts == {"a": 1, "b": 1}


def test_balanced_research_events_can_filter_to_primary_sources(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "primary.sqlite3")
    repository.migrate()
    repository.events_needing_research = (
        lambda limit: [  # type: ignore[method-assign]
            {"id": 1, "source_role": "radar"},
            {"id": 2, "source_role": "primary"},
        ][:limit]
    )
    repository.event_trigger_sources = lambda event_id: [  # type: ignore[method-assign]
        {"source_id": str(event_id)}
    ]

    selected = balanced_research_events(repository, limit=2, primary_only=True)

    assert [event["id"] for event in selected] == [2]
