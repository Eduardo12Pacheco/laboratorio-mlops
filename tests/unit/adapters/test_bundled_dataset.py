"""Bundled dataset adapter: provenance, raw contract, archive and windows."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset, from_uci_frame
from mlops_lifecycle_lab.domain.errors import ChecksumMismatchError
from mlops_lifecycle_lab.domain.schema import LABEL_COLUMNS, RAW_COLUMNS
from mlops_lifecycle_lab.domain.validation import Verdict, validate_frame
from mlops_lifecycle_lab.domain.windows import TimeWindow, tick_window

DATA_DIR = Path(__file__).resolve().parents[3] / "src/mlops_lifecycle_lab/_data"


@pytest.fixture(scope="module")
def dataset() -> BundledDataset:
    return BundledDataset()


def test_manifest_records_provenance(dataset: BundledDataset) -> None:
    manifest = dataset.manifest()

    assert manifest["doi"] == "10.24432/C5W894"
    assert manifest["license"] == "CC BY 4.0"
    assert "Fanaee-T" in manifest["attribution"]
    assert manifest["uci_zip_sha256"] == (
        "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
    )
    assert manifest["uci_zip_size"] == 279992
    hour = manifest["files"]["hour.csv"]
    assert hour["sha256"] == ("e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f")
    assert hour["size"] == 1156736
    assert hour["rows"] == 17379
    assert hour["start"] == "2011-01-01T00:00:00"
    assert hour["end"] == "2012-12-31T23:00:00"


def test_verify_returns_the_verified_provenance(dataset: BundledDataset) -> None:
    verified = dataset.verify()

    assert verified.sha256 == dataset.manifest()["files"]["hour.csv"]["sha256"]
    assert verified.size == 1156736
    assert verified.rows == 17379
    assert verified.start == datetime(2011, 1, 1)
    assert verified.end == datetime(2012, 12, 31, 23)


def test_raw_frame_matches_the_documented_contract(dataset: BundledDataset) -> None:
    raw = dataset.raw()

    assert list(raw.columns) == [*RAW_COLUMNS, *LABEL_COLUMNS]
    assert len(raw) == 17379
    assert raw["timestamp"].iloc[0] == datetime(2011, 1, 1)
    assert raw["timestamp"].iloc[-1] == datetime(2012, 12, 31, 23)
    assert raw["entity_id"].iloc[0] == 1
    assert raw["entity_id"].iloc[-1] == 17379
    assert raw["timestamp"].is_monotonic_increasing
    assert not raw["timestamp"].duplicated().any()
    assert pd.api.types.is_datetime64_any_dtype(raw["timestamp"])
    for column in ["season", "holiday", "workingday", "weathersit", "casual", "registered", "cnt"]:
        assert pd.api.types.is_integer_dtype(raw[column]), column
    for column in ["temp", "atemp", "hum", "windspeed"]:
        assert pd.api.types.is_float_dtype(raw[column]), column
    assert (raw["cnt"] == raw["casual"] + raw["registered"]).all()
    assert (raw["cnt"] >= 0).all()


def test_full_dataset_validates_as_one_expected_warning(dataset: BundledDataset) -> None:
    report = validate_frame(dataset.raw(), context="training")

    assert report.verdict is Verdict.WARN
    failed = [rule for rule in report.rules if not rule.passed]
    assert [rule.name for rule in failed] == ["hum_positive"]
    assert failed[0].violations == 22
    assert not [rule for rule in report.rules if not rule.passed and rule.severity.value == "ERROR"]


def test_archive_is_strictly_before_the_end(dataset: BundledDataset) -> None:
    raw = dataset.raw()

    archive = dataset.archive(datetime(2011, 6, 1))

    assert len(archive) == 3530
    assert archive["timestamp"].max() < datetime(2011, 6, 1)
    assert len(dataset.archive(datetime(2013, 1, 1))) == 8645
    assert dataset.archive(datetime(2013, 1, 1)).equals(dataset.archive(datetime(2012, 1, 1)))
    assert len(raw) == 17379


def test_window_is_half_open_and_returns_a_copy(dataset: BundledDataset) -> None:
    window = TimeWindow(datetime(2012, 1, 1), datetime(2012, 1, 8))

    batch = dataset.window(window)

    assert len(batch) == 167
    assert batch["timestamp"].min() >= window.start
    assert batch["timestamp"].max() < window.end
    batch.loc[batch.index[0], "temp"] = 0.99
    assert dataset.window(window)["temp"].iloc[0] != 0.99


def test_tick_window_uses_only_the_replay_weeks(dataset: BundledDataset) -> None:
    first = dataset.window(tick_window(1, datetime(2012, 1, 1), days=7))
    last = dataset.window(tick_window(52, datetime(2012, 1, 1), days=7))

    assert first["timestamp"].min() == datetime(2012, 1, 1)
    assert first["timestamp"].max() < datetime(2012, 1, 8)
    assert last["timestamp"].min() == datetime(2012, 12, 23)
    assert last["timestamp"].max() == datetime(2012, 12, 29, 23)


def test_from_uci_frame_derives_timestamp_and_entity_id_without_mutating() -> None:
    uci = pd.DataFrame(
        {
            "instant": [1, 2],
            "dteday": ["2012-01-01", "2012-01-01"],
            "season": [1, 1],
            "yr": [1, 1],
            "mnth": [1, 1],
            "hr": [0, 1],
            "holiday": [0, 0],
            "weekday": [0, 0],
            "workingday": [0, 0],
            "weathersit": [1, 1],
            "temp": [0.24, 0.22],
            "atemp": [0.2879, 0.2727],
            "hum": [0.81, 0.80],
            "windspeed": [0.0, 0.0],
            "casual": [3, 8],
            "registered": [13, 32],
            "cnt": [16, 40],
        }
    )
    before = uci.copy(deep=True)

    raw = from_uci_frame(uci)

    assert list(raw.columns) == [*RAW_COLUMNS, *LABEL_COLUMNS]
    assert raw["timestamp"].tolist() == [datetime(2012, 1, 1, 0), datetime(2012, 1, 1, 1)]
    assert raw["entity_id"].tolist() == [1, 2]
    assert uci.equals(before)


def test_from_uci_frame_requires_the_original_columns() -> None:
    with pytest.raises(KeyError, match="temp"):
        from_uci_frame(pd.DataFrame({"instant": [1], "dteday": ["2012-01-01"], "hr": [0]}))


def test_verify_detects_a_tampered_payload(tmp_path: Path) -> None:
    shutil.copy(DATA_DIR / "manifest.json", tmp_path / "manifest.json")
    tampered_csv = gzip.decompress((DATA_DIR / "hour.csv.gz").read_bytes()).replace(
        b"1,2011-01-01,1,0,1,0,0,6,0,1,0.24",
        b"1,2011-01-01,1,0,1,0,0,6,0,1,0.25",
        1,
    )
    assert hashlib.sha256(tampered_csv).hexdigest() != (
        "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
    )
    (tmp_path / "hour.csv.gz").write_bytes(gzip.compress(tampered_csv, compresslevel=9, mtime=0))

    with pytest.raises(ChecksumMismatchError) as excinfo:
        BundledDataset(tmp_path).verify()

    assert "hour.csv" in str(excinfo.value)
    assert "ChecksumMismatchError" in type(excinfo.value).__name__


def test_manifest_json_is_valid_and_matches_files_on_disk() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    payload = gzip.decompress((DATA_DIR / "hour.csv.gz").read_bytes())

    assert manifest["files"]["hour.csv"]["size"] == len(payload)
    assert manifest["files"]["hour.csv"]["sha256"] == hashlib.sha256(payload).hexdigest()
