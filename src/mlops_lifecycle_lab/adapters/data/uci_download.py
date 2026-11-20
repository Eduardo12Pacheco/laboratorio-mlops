"""Optional acquisition of the pinned UCI archive.

This adapter is never used by tests or by the default runtime path; it exists so a human
can re-fetch the exact archive the bundle came from. Both the outer zip and the inner
`hour.csv` are verified before anything is written to disk.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mlops_lifecycle_lab.domain.errors import ChecksumMismatchError

Opener = Callable[[str], bytes]


class UciDownloader:
    """Download, verify and extract the pinned UCI zip into `dest`."""

    def __init__(self, manifest: Mapping[str, Any], opener: Opener) -> None:
        self._manifest = manifest
        self._opener = opener

    def fetch(self, dest: Path) -> Path:
        """Fetch `hour.csv` from the pinned zip and return the written path.

        Raises:
            ChecksumMismatchError: when either the zip or the extracted CSV does not
                match the pinned sha256. Nothing is written in that case.
            FileNotFoundError: when the archive does not contain `hour.csv`.
        """
        url = str(self._manifest["uci_zip_url"])
        payload = self._opener(url)
        expected_zip = str(self._manifest["uci_zip_sha256"])
        actual_zip = hashlib.sha256(payload).hexdigest()
        if actual_zip != expected_zip:
            raise ChecksumMismatchError("UCI zip", expected=expected_zip, actual=actual_zip)

        expected_zip_size = int(self._manifest["uci_zip_size"])
        if len(payload) != expected_zip_size:
            raise ChecksumMismatchError(
                "UCI zip size", expected=str(expected_zip_size), actual=str(len(payload))
            )

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            try:
                info = archive.getinfo("hour.csv")
            except KeyError as exc:
                raise FileNotFoundError("hour.csv not found in the UCI archive") from exc
            inner = archive.read(info)

        entry = self._manifest["files"]["hour.csv"]
        expected_inner = str(entry["sha256"])
        actual_inner = hashlib.sha256(inner).hexdigest()
        if actual_inner != expected_inner:
            raise ChecksumMismatchError("hour.csv", expected=expected_inner, actual=actual_inner)

        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "hour.csv"
        target.write_bytes(inner)
        return target
