"""End-to-end CLI behaviour for the bundled dataset and the validation contract."""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from mlops_lifecycle_lab.entrypoints.cli.app import app

DATA_DIR = Path(__file__).resolve().parents[2] / "src/mlops_lifecycle_lab/_data"
runner = CliRunner()


def combined(result: Result) -> str:
    return result.output + result.stderr


def test_data_verify_reports_the_bundle() -> None:
    result = runner.invoke(app, ["data", "verify"])

    assert result.exit_code == 0, combined(result)
    assert "hour.csv sha256 OK" in result.output
    assert "17379 rows" in result.output


def test_data_info_json_reports_provenance() -> None:
    result = runner.invoke(app, ["data", "info", "--json"])

    assert result.exit_code == 0, combined(result)
    payload = json.loads(result.output)
    assert payload["rows"] == 17379
    assert payload["start"] == "2011-01-01T00:00:00"
    assert payload["end"] == "2012-12-31T23:00:00"
    assert payload["sha256"] == ("e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f")
    assert payload["license"] == "CC BY 4.0"


def test_data_info_human_output_mentions_the_license() -> None:
    result = runner.invoke(app, ["data", "info"])

    assert result.exit_code == 0, combined(result)
    assert "17379 rows" in result.output
    assert "CC BY 4.0" in result.output
    assert "10.24432/C5W894" in result.output


def test_tampered_copy_exits_two_with_checksum_error(tmp_path: Path) -> None:
    tampered = gzip.decompress((DATA_DIR / "hour.csv.gz").read_bytes()).replace(
        b"1,2011-01-01,1,0,1,0,0,6,0,1,0.24",
        b"1,2011-01-01,1,0,1,0,0,6,0,1,0.25",
        1,
    )
    copy = tmp_path / "hour.csv.gz"
    copy.write_bytes(gzip.compress(tampered, compresslevel=9, mtime=0))

    result = runner.invoke(app, ["data", "verify", "--path", str(copy)])

    assert result.exit_code == 2, combined(result)
    assert "ChecksumMismatchError" in combined(result)


def test_validate_bundled_reports_warn_with_one_warning() -> None:
    result = runner.invoke(app, ["validate", "--bundled", "--json"])

    assert result.exit_code == 0, combined(result)
    payload = json.loads(result.output)
    assert payload["verdict"] == "WARN"
    assert payload["failed_rules"] == ["hum_positive"]
    assert payload["rules"]["hum_positive"]["violations"] == 22
    errors = [
        name
        for name, rule in payload["rules"].items()
        if rule["severity"] == "ERROR" and not rule["passed"]
    ]
    assert errors == []


def test_validate_path_accepts_an_original_uci_csv(tmp_path: Path) -> None:
    frame = pd.DataFrame(
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
    path = tmp_path / "hour.csv"
    frame.to_csv(path, index=False)

    result = runner.invoke(app, ["validate", str(path), "--context", "training"])

    assert result.exit_code == 0, combined(result)
    assert "PASS" in result.output


def test_validate_bad_batch_exits_three(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "entity_id": [1],
            "timestamp": [datetime(2012, 1, 1)],
            "season": [9],
            "holiday": [0],
            "workingday": [1],
            "weathersit": [1],
            "temp": [0.2],
            "atemp": [0.2],
            "hum": [0.4],
            "windspeed": [0.1],
        }
    )
    path = tmp_path / "batch.csv"
    frame.to_csv(path, index=False)

    result = runner.invoke(app, ["validate", str(path), "--context", "batch"])

    assert result.exit_code == 3, combined(result)
    assert "season_domain" in combined(result)


def test_validate_requires_exactly_one_input() -> None:
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 2, combined(result)
    assert "--bundled" in combined(result)


def test_data_fetch_rejects_unknown_sources(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["data", "fetch", "--source", "kaggle", "--dest", str(tmp_path / "out")]
    )

    assert result.exit_code == 2, combined(result)
    assert "uci" in combined(result)


def test_data_fetch_with_recorded_fixture_writes_the_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    from mlops_lifecycle_lab.entrypoints.cli.commands import data as data_command

    inner = (
        b"instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,"
        b"temp,atemp,hum,windspeed,casual,registered,cnt\n"
        b"1,2012-01-01,1,1,1,0,0,0,0,1,0.24,0.2879,0.81,0,3,13,16\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("hour.csv", inner)
    zip_payload = buffer.getvalue()

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    import hashlib

    manifest["uci_zip_sha256"] = hashlib.sha256(zip_payload).hexdigest()
    manifest["uci_zip_size"] = len(zip_payload)
    manifest["files"]["hour.csv"] = {
        **manifest["files"]["hour.csv"],
        "sha256": hashlib.sha256(inner).hexdigest(),
        "size": len(inner),
    }

    def fake_download(url: str) -> bytes:
        return zip_payload

    monkeypatch.setattr(data_command, "load_manifest", lambda: manifest)
    monkeypatch.setattr(data_command, "_download_bytes", fake_download)

    result = runner.invoke(app, ["data", "fetch", "--source", "uci", "--dest", str(tmp_path)])

    assert result.exit_code == 0, combined(result)
    assert (tmp_path / "hour.csv").read_bytes() == inner
    assert "hour.csv sha256 OK" in result.output


def test_data_verify_help_mentions_the_path_option() -> None:
    result = runner.invoke(app, ["data", "verify", "--help"])

    assert result.exit_code == 0
    assert "--path" in result.output


def test_attribution_file_is_shipped_with_the_bundle() -> None:
    text = (DATA_DIR / "ATTRIBUTION.txt").read_text(encoding="utf-8")

    assert "CC BY 4.0" in text
    assert "10.24432/C5W894" in text
    assert "Fanaee-T" in text
