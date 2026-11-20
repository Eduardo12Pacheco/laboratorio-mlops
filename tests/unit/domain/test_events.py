"""Domain events for the lifecycle audit log: types, payloads and JSON round-trip."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from mlops_lifecycle_lab.domain.events import EventType, LifecycleEvent

EXPECTED_TYPES = {
    "BOOTSTRAP_COMPLETED",
    "BATCH_VALIDATED",
    "BATCH_QUARANTINED",
    "BATCH_SCORED",
    "LABELS_INGESTED",
    "SNAPSHOT_COMPUTED",
    "RETRAIN_TRIGGERED",
    "CHALLENGER_REGISTERED",
    "GATES_EVALUATED",
    "CHALLENGER_PROMOTED",
    "CHALLENGER_REJECTED",
    "PROMOTION_FORCED",
    "PROBATION_STARTED",
    "PROBATION_PASSED",
    "ROLLBACK_EXECUTED",
}


def make_event() -> LifecycleEvent:
    return LifecycleEvent(
        ts=datetime(2012, 4, 1),
        type=EventType.CHALLENGER_PROMOTED,
        tick=13,
        model_version=2,
        payload={
            "improvement": 0.56,
            "gates": [{"name": "IMPROVEMENT", "passed": True}],
            "alias": "champion",
        },
    )


def test_the_event_type_enum_has_every_documented_type() -> None:
    assert {event_type.value for event_type in EventType} == EXPECTED_TYPES
    assert len(EventType) == 15


def test_event_types_are_string_enums() -> None:
    assert EventType.SNAPSHOT_COMPUTED == "SNAPSHOT_COMPUTED"
    assert f"{EventType.ROLLBACK_EXECUTED}" == "ROLLBACK_EXECUTED"


def test_to_dict_is_json_serialisable_and_uses_iso_timestamps() -> None:
    event = make_event()

    payload = event.to_dict()

    assert payload == {
        "ts": "2012-04-01T00:00:00",
        "type": "CHALLENGER_PROMOTED",
        "tick": 13,
        "model_version": 2,
        "payload": {
            "improvement": 0.56,
            "gates": [{"name": "IMPROVEMENT", "passed": True}],
            "alias": "champion",
        },
    }
    assert json.loads(json.dumps(payload)) == payload


def test_json_round_trip_restores_an_equal_event() -> None:
    event = make_event()

    restored = LifecycleEvent.from_dict(json.loads(json.dumps(event.to_dict())))

    assert restored == event
    assert restored.ts == datetime(2012, 4, 1)
    assert restored.type is EventType.CHALLENGER_PROMOTED


def test_the_payload_defaults_to_an_empty_mapping() -> None:
    event = LifecycleEvent(
        ts=datetime(2012, 1, 1),
        type=EventType.BOOTSTRAP_COMPLETED,
        tick=0,
        model_version=1,
    )

    assert event.payload == {}
    assert event.to_dict()["payload"] == {}


def test_the_payload_mapping_is_copied_on_output() -> None:
    event = make_event()

    first = event.to_dict()["payload"]
    assert isinstance(first, dict)
    first["improvement"] = 0.0

    assert event.to_dict()["payload"]["improvement"] == 0.56


@pytest.mark.parametrize("tick", [-1])
def test_negative_ticks_are_rejected(tick: int) -> None:
    with pytest.raises(ValueError, match="tick"):
        LifecycleEvent(
            ts=datetime(2012, 1, 1),
            type=EventType.SNAPSHOT_COMPUTED,
            tick=tick,
            model_version=1,
        )


def test_non_positive_model_versions_are_rejected() -> None:
    with pytest.raises(ValueError, match="model_version"):
        LifecycleEvent(
            ts=datetime(2012, 1, 1),
            type=EventType.SNAPSHOT_COMPUTED,
            tick=1,
            model_version=0,
        )


def test_unknown_event_types_are_rejected_on_deserialisation() -> None:
    payload = make_event().to_dict()
    payload["type"] = "NOT_AN_EVENT"

    with pytest.raises(ValueError, match="NOT_AN_EVENT"):
        LifecycleEvent.from_dict(payload)


def test_events_are_frozen() -> None:
    event = make_event()

    with pytest.raises(FrozenInstanceError):
        event.tick = 14  # type: ignore[misc]
