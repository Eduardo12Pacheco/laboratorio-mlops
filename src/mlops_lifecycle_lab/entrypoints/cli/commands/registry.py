"""``mll registry``: inspect and steer the champion/challenger registry (S7, S13)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mlops_lifecycle_lab.application.ports import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS_CHAMPION,
    ModelRegistry,
)
from mlops_lifecycle_lab.bootstrap.container import Container, build_container
from mlops_lifecycle_lab.config import Settings, apply_privacy_defaults
from mlops_lifecycle_lab.domain.errors import DomainError
from mlops_lifecycle_lab.domain.lifecycle import ModelVersion

registry_app = typer.Typer(
    name="registry",
    help="List, inspect, promote or roll back model versions.",
    no_args_is_help=True,
)

_ALIASES = (ALIAS_CHAMPION, ALIAS_PREVIOUS_CHAMPION, ALIAS_CHALLENGER)


def _settings(workspace: Path | None) -> Settings:
    apply_privacy_defaults()
    settings = Settings()
    if workspace is not None:
        settings = settings.model_copy(update={"workspace": workspace})
    return settings


def _container(workspace: Path | None) -> Container:
    return build_container(_settings(workspace))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))
    return "\n".join(lines)


def _aliases_for(registry: ModelRegistry, version: int) -> str:
    names = []
    for alias in _ALIASES:
        found = registry.get_alias(alias)
        if found is not None and found.version == version:
            names.append(alias)
    return ", ".join(names)


def _window(version: ModelVersion) -> str:
    return f"{version.train_window.start.date()} -> {version.train_window.end.date()}"


@registry_app.command("list")
def list_versions(
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Print every version with stage, metrics, training window and aliases."""
    container = _container(workspace)
    try:
        rows = [
            [
                f"v{version.version}",
                version.stage.value,
                _metric(version, "rmse"),
                _metric(version, "wape"),
                _window(version),
                _aliases_for(container.registry, version.version),
            ]
            for version in container.registry.list_versions()
        ]
    finally:
        container.close()
    typer.echo(
        _table(
            ["version", "stage", "rmse", "wape", "train_window", "aliases"],
            rows,
        )
    )


@registry_app.command("show")
def show(
    version: int = typer.Argument(..., min=1),
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Print one version: lineage tags, metrics and aliases."""
    container = _container(workspace)
    try:
        model = container.registry.get_version(version)
        if model is None:
            typer.echo(f"UnknownModelVersionError: v{version} is not registered", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"version: v{model.version}")
        typer.echo(f"stage: {model.stage.value}")
        typer.echo(f"aliases: {_aliases_for(container.registry, version) or '-'}")
        typer.echo(f"train_window: {_window(model)}")
        if model.holdout_window is not None:
            typer.echo(
                "holdout_window: "
                f"{model.holdout_window.start.date()} -> {model.holdout_window.end.date()}"
            )
        typer.echo(f"run_id: {model.run_id}")
        typer.echo(f"data_fingerprint: {model.data_fingerprint}")
        typer.echo(f"trigger_reason: {model.trigger_reason or '-'}")
        typer.echo(
            "metrics: "
            + " ".join(f"{name}={value:.4f}" for name, value in sorted(model.metrics.items()))
        )
        for name, value in sorted(model.tags.items()):
            typer.echo(f"tag {name}: {value}")
    finally:
        container.close()


@registry_app.command("promote")
def promote(
    version: int = typer.Argument(..., min=1),
    force: Annotated[
        bool, typer.Option("--force", help="Bypass the gates (requires --reason).")
    ] = False,
    reason: Annotated[str, typer.Option("--reason", help="Forced-promotion justification.")] = "",
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Promote an approved candidate, or force it with a reason."""
    container = _container(workspace)
    try:
        promoted = container.promotion.execute(version, forced=force, reason=reason or None)
        typer.echo(f"promoted v{promoted.version} to {promoted.stage.value}")
    except (DomainError, ValueError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()


@registry_app.command("rollback")
def rollback(
    reason: Annotated[str, typer.Option("--reason", help="Why the champion is rolled back.")],
    workspace: Annotated[
        Path | None, typer.Option("--workspace", help="Workspace directory (default: .mll).")
    ] = None,
) -> None:
    """Restore the previous champion (S13)."""
    container = _container(workspace)
    try:
        restored = container.rollback.execute(reason)
        typer.echo(f"rolled back to v{restored.version}")
    except (DomainError, ValueError) as exc:
        typer.echo(f"{type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        container.close()


def _metric(version: ModelVersion, name: str) -> str:
    value = version.metrics.get(name)
    return "-" if value is None else f"{value:.4f}"


__all__ = ["registry_app"]
