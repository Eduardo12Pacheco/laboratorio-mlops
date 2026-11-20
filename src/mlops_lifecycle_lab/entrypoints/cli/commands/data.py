"""``mll data`` commands: verify the bundle, describe it and optionally re-fetch it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any
from urllib.request import urlopen

import typer

from mlops_lifecycle_lab.adapters.data.bundled_dataset import BundledDataset
from mlops_lifecycle_lab.adapters.data.uci_download import UciDownloader
from mlops_lifecycle_lab.domain.errors import ChecksumMismatchError

data_app = typer.Typer(
    name="data",
    help="Inspect, verify and acquire the bundled UCI bike-sharing dataset.",
    no_args_is_help=True,
)


def load_manifest() -> dict[str, Any]:
    """Read the provenance manifest shipped inside the package."""
    return BundledDataset().manifest()


def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - URL is pinned in the manifest
        payload: bytes = response.read()
        return payload


def _dataset_for(path: Path | None) -> BundledDataset:
    return BundledDataset(path) if path is not None else BundledDataset()


@data_app.command("verify")
def verify(
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Verify a copied .csv.gz instead of the packaged bundle."),
    ] = None,
) -> None:
    """Verify the decompressed sha256 and report the pinned row count."""
    dataset = _dataset_for(path)
    try:
        verified = dataset.verify()
    except ChecksumMismatchError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"hour.csv sha256 OK ({verified.sha256})")
    typer.echo(f"{verified.rows} rows")


@data_app.command("info")
def info(
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Print provenance, license and the pinned checksum."""
    manifest = load_manifest()
    entry = manifest["files"]["hour.csv"]
    payload: dict[str, Any] = {
        "name": manifest["name"],
        "source_url": manifest["source_url"],
        "doi": manifest["doi"],
        "license": manifest["license"],
        "attribution": manifest["attribution"],
        "sha256": entry["sha256"],
        "size": entry["size"],
        "rows": entry["rows"],
        "start": entry["start"],
        "end": entry["end"],
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"{payload['name']} ({payload['rows']} rows)")
    typer.echo(f"window: {payload['start']} -> {payload['end']}")
    typer.echo(f"sha256: {payload['sha256']}")
    typer.echo(f"license: {payload['license']} | DOI: {payload['doi']}")
    typer.echo(f"source: {payload['source_url']}")


@data_app.command("fetch")
def fetch(
    dest: Annotated[Path, typer.Option("--dest", help="Directory that receives hour.csv.")],
    source: Annotated[
        str, typer.Option("--source", help="Acquisition source (only 'uci').")
    ] = "uci",
) -> None:
    """Download the pinned UCI zip, verify zip and CSV, then write `hour.csv`."""
    if source != "uci":
        raise typer.BadParameter("supported source: uci", param_hint="--source")
    manifest = load_manifest()
    downloader = UciDownloader(manifest, _download_bytes)
    try:
        written = downloader.fetch(dest)
    except ChecksumMismatchError as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    actual = hashlib.sha256(written.read_bytes()).hexdigest()
    typer.echo(f"wrote {written}")
    typer.echo(f"hour.csv sha256 OK ({actual})")
