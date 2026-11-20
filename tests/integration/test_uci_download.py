"""Optional UCI acquisition adapter: verification happens before anything is written."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from mlops_lifecycle_lab.adapters.data.uci_download import UciDownloader
from mlops_lifecycle_lab.domain.errors import ChecksumMismatchError

DATA_DIR = Path(__file__).resolve().parents[2] / "src/mlops_lifecycle_lab/_data"
HOUR_CSV = (
    b"instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,"
    b"temp,atemp,hum,windspeed,casual,registered,cnt\n"
    b"1,2012-01-01,1,1,1,0,0,0,0,1,0.24,0.2879,0.81,0,3,13,16\n"
)


class RecordingOpener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        return self.payload


def make_zip(hour_csv: bytes = HOUR_CSV) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("hour.csv", hour_csv)
        archive.writestr("Readme.txt", "fixture\n")
    return buffer.getvalue()


def make_manifest(zip_payload: bytes, inner: bytes = HOUR_CSV) -> dict[str, object]:
    base = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    base["uci_zip_sha256"] = hashlib.sha256(zip_payload).hexdigest()
    base["uci_zip_size"] = len(zip_payload)
    files = base["files"]
    assert isinstance(files, dict)
    files["hour.csv"] = {
        **files["hour.csv"],
        "sha256": hashlib.sha256(inner).hexdigest(),
        "size": len(inner),
    }
    return base


def test_fetch_verifies_and_writes_the_inner_csv(tmp_path: Path) -> None:
    zip_payload = make_zip()
    opener = RecordingOpener(zip_payload)
    downloader = UciDownloader(make_manifest(zip_payload), opener)

    written = downloader.fetch(tmp_path / "download")

    assert written == tmp_path / "download" / "hour.csv"
    assert written.read_bytes() == HOUR_CSV
    assert opener.calls == [
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    ]


def test_zip_checksum_mismatch_is_refused_before_writing(tmp_path: Path) -> None:
    zip_payload = make_zip()
    manifest = make_manifest(zip_payload)
    manifest["uci_zip_sha256"] = "0" * 64
    dest = tmp_path / "download"

    with pytest.raises(ChecksumMismatchError, match="zip"):
        UciDownloader(manifest, RecordingOpener(zip_payload)).fetch(dest)

    assert not dest.exists() or not any(dest.iterdir())


def test_inner_checksum_mismatch_is_refused_before_writing(tmp_path: Path) -> None:
    zip_payload = make_zip()
    manifest = make_manifest(zip_payload)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["hour.csv"] = {**files["hour.csv"], "sha256": "0" * 64}
    dest = tmp_path / "download"

    with pytest.raises(ChecksumMismatchError, match=r"hour\.csv"):
        UciDownloader(manifest, RecordingOpener(zip_payload)).fetch(dest)

    assert not dest.exists() or not any(dest.iterdir())


def test_zip_missing_hour_csv_is_rejected(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("day.csv", HOUR_CSV)
    zip_payload = buffer.getvalue()
    dest = tmp_path / "download"

    with pytest.raises(FileNotFoundError, match=r"hour\.csv"):
        UciDownloader(make_manifest(zip_payload), RecordingOpener(zip_payload)).fetch(dest)

    assert not dest.exists() or not any(dest.iterdir())


def test_opener_type_is_a_callable_protocol(tmp_path: Path) -> None:
    zip_payload = make_zip()
    opener: Callable[[str], bytes] = RecordingOpener(zip_payload)
    manifest = make_manifest(zip_payload)

    written = UciDownloader(manifest, opener).fetch(tmp_path)

    assert written.exists()
