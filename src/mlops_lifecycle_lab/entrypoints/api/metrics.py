"""Per-application Prometheus metrics registry for the serving API (S18).

A fresh registry per app instance keeps tests isolated (no duplicate collector errors)
and makes `/metrics` contain exactly the recorded series.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
)


class ApiMetrics:
    """Counters, histogram and gauges exposed on `GET /metrics`."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.predictions_total = Counter(
            "mll_predictions_total",
            "Predictions served, by model version",
            ["model_version"],
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            "mll_prediction_latency_seconds",
            "Time spent scoring a serving request",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.model_info = Gauge(
            "mll_model_info",
            "The model currently loaded for serving (always 1)",
            ["version", "stage"],
            registry=self.registry,
        )
        self.champion_version = Gauge(
            "mll_champion_version",
            "Version of the champion currently loaded",
            registry=self.registry,
        )
        self.reload_failures = Counter(
            "mll_model_reload_failures_total",
            "Champion reload attempts that failed and kept the previous model",
            registry=self.registry,
        )

    def record_predictions(self, model_version: int, count: int = 1) -> None:
        """Count served rows for one model version."""
        self.predictions_total.labels(model_version=str(model_version)).inc(count)

    def observe_latency(self, seconds: float) -> None:
        """Record one serving latency observation."""
        self.prediction_latency.observe(seconds)

    def set_champion(self, version: int, stage: str) -> None:
        """Replace the loaded-model series and update the champion version gauge."""
        self.model_info.clear()
        self.model_info.labels(version=str(version), stage=stage).set(1)
        self.champion_version.set(version)

    def record_reload_failure(self) -> None:
        """Count one failed reload that kept the previous model."""
        self.reload_failures.inc()

    def render(self) -> bytes:
        """Render the registry in the Prometheus text exposition format."""
        return generate_latest(self.registry)


__all__ = ["LATENCY_BUCKETS", "ApiMetrics"]
