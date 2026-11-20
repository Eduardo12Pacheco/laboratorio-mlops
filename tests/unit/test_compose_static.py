"""Static contract for the Compose stack (T12, S19).

No Docker daemon is required: the YAML is parsed and inspected so that hardening,
healthchecks, loopback-only ports and dependency conditions are enforced by the default
CI matrix as well as by the ``docker``-marked smoke test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT / "compose.yaml"
ENV_EXAMPLE = ROOT / ".env.example"

HEALTHCHECK_EXEMPT = {"init", "cli"}
HTTP_HEALTHCHECK_SERVICES = {"mlflow", "api", "dashboard"}


def _load() -> dict[str, Any]:
    assert COMPOSE_PATH.is_file(), "compose.yaml is missing"
    document = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _services() -> dict[str, dict[str, Any]]:
    services = _load()["services"]
    assert isinstance(services, dict)
    return services


def test_stack_declares_the_documented_services() -> None:
    assert set(_services()) == {"mlflow", "init", "api", "scheduler", "dashboard", "cli"}
    assert "mll-data" in _load()["volumes"]


def test_every_service_uses_the_repo_image() -> None:
    for name, service in _services().items():
        assert service["build"]["context"] == ".", name
        assert service["build"]["target"] == "runtime", name
        assert service["image"] == "mll:local", name
        assert "latest" not in service["image"], name


def test_every_service_is_hardened_and_non_root() -> None:
    for name, service in _services().items():
        assert service["user"] == "10001:10001", name
        assert service["read_only"] is True, name
        assert "/tmp" in service["tmpfs"], name  # noqa: S108 - intentional tmpfs
        assert service["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in service["security_opt"], name


def test_healthchecks_on_long_running_services() -> None:
    for name, service in _services().items():
        healthcheck = service.get("healthcheck")
        if name in HEALTHCHECK_EXEMPT:
            assert healthcheck is None, name
            continue
        assert healthcheck is not None, name
        assert healthcheck["test"][0] == "CMD", name
        command = " ".join(healthcheck["test"])
        assert "python" in command, name
        if name in HTTP_HEALTHCHECK_SERVICES:
            assert "urllib" in command, name
            assert "http://localhost" in command, name
        else:
            assert "os.kill" in command, name


def test_ports_are_published_on_loopback_only() -> None:
    published: dict[str, tuple[str, str]] = {}
    for name, service in _services().items():
        for mapping in service.get("ports", []):
            assert isinstance(mapping, str), name
            assert mapping.startswith("127.0.0.1:"), mapping
            host_port, container_port = mapping.split(":")[1:]
            published[name] = (host_port, container_port)
    assert published == {
        "mlflow": ("14000", "5000"),
        "api": ("14080", "8000"),
        "dashboard": ("14501", "8501"),
    }


def test_dependency_conditions_match_the_startup_order() -> None:
    services = _services()

    assert services["init"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["init"]["condition"] == "service_completed_successfully"
    assert services["scheduler"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["dashboard"]["depends_on"]["api"]["condition"] == "service_healthy"


def test_mlflow_serves_the_shared_artifact_store() -> None:
    mlflow = _services()["mlflow"]
    command: list[str] = mlflow["command"]
    joined = " ".join(command)

    assert mlflow["entrypoint"] == ["mlflow"]
    assert "--backend-store-uri" in command
    assert "sqlite:////data/mlflow/mlflow.db" in command
    assert "--serve-artifacts" in command
    assert "/data/mlflow/artifacts" in command
    assert "--default-artifact-root" in command
    assert "mlflow-artifacts:/" in command
    assert "--allowed-hosts" in command
    allowed = command[command.index("--allowed-hosts") + 1]
    assert "mlflow:5000" in allowed
    assert "127.0.0.1:14000" in allowed
    assert "--workers" in joined


def test_commands_wire_the_lifecycle_services() -> None:
    services = _services()

    assert services["init"]["command"] == [
        "bootstrap",
        "--if-missing",
        "--at",
        "2012-01-01T00:00:00",
    ]
    assert services["init"]["restart"] == "no"
    assert services["api"]["command"] == [
        "serve",
        "--host",
        "0.0.0.0",  # noqa: S104 - container bind, published on loopback
        "--port",
        "8000",
    ]
    assert services["scheduler"]["command"] == ["scheduler", "start", "--mode", "replay"]
    assert services["dashboard"]["command"] == [
        "dashboard",
        "--host",
        "0.0.0.0",  # noqa: S104 - container bind, published on loopback
        "--port",
        "8501",
    ]
    assert services["cli"]["profiles"] == ["tools"]
    assert services["cli"]["entrypoint"] == ["mll"]


def test_api_forwards_the_optional_admin_key() -> None:
    environment = _services()["api"]["environment"]

    assert environment["MLL_ADMIN_API_KEY"] == "${MLL_ADMIN_API_KEY:-}"


def test_scheduler_drives_the_real_api_with_a_fast_replay() -> None:
    environment = _services()["scheduler"]["environment"]

    assert environment["MLL_SERVING_MODE"] == "http"
    assert environment["MLL_API_URL"] == "http://api:8000"
    assert "MLL_REPLAY_INTERVAL_SECONDS" in environment


def test_shared_environment_is_offline_and_volume_backed() -> None:
    for name, service in _services().items():
        environment = service["environment"]
        assert environment["MLL_WORKSPACE"] == "/data", name
        assert environment["MLL_MLFLOW_TRACKING_URI"] == "http://mlflow:5000", name
        assert environment["MLFLOW_DISABLE_TELEMETRY"] == "true", name
        assert environment["DO_NOT_TRACK"] == "1", name
        assert environment["ITERATIVE_DO_NOT_TRACK"] == "1", name
        assert environment["EVIDENTLY_DISABLE_TELEMETRY"] == "1", name
        assert environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false", name
        assert "mll-data:/data" in service["volumes"], name


def test_env_example_documents_the_operator_switches() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert "COMPOSE_PROJECT_NAME=mlops-lifecycle-lab" in text
    assert "MLL_REPLAY_INTERVAL_SECONDS=" in text
    assert "MLL_ADMIN_API_KEY=" in text
