"""Per-application Prometheus registry: counters, histogram and model gauges."""

from __future__ import annotations

from mlops_lifecycle_lab.entrypoints.api.metrics import ApiMetrics


def test_predictions_are_counted_per_model_version() -> None:
    metrics = ApiMetrics()

    metrics.record_predictions(1, count=3)
    metrics.record_predictions(2, count=1)

    text = metrics.render().decode()
    assert 'mll_predictions_total{model_version="1"} 3.0' in text
    assert 'mll_predictions_total{model_version="2"} 1.0' in text


def test_latency_histogram_and_model_gauges_are_exposed() -> None:
    metrics = ApiMetrics()
    metrics.observe_latency(0.25)

    metrics.set_champion(version=2, stage="CHAMPION")
    metrics.record_reload_failure()

    text = metrics.render().decode()
    assert "mll_prediction_latency_seconds_bucket" in text
    assert "mll_prediction_latency_seconds_count 1.0" in text
    assert 'mll_model_info{stage="CHAMPION",version="2"} 1.0' in text
    assert "mll_champion_version 2.0" in text
    assert "mll_model_reload_failures_total 1.0" in text


def test_new_champion_replaces_the_previous_model_info_series() -> None:
    metrics = ApiMetrics()
    metrics.set_champion(version=1, stage="CHAMPION")
    metrics.set_champion(version=2, stage="CHAMPION")

    text = metrics.render().decode()
    assert 'mll_model_info{stage="CHAMPION",version="1"}' not in text
    assert 'mll_model_info{stage="CHAMPION",version="2"} 1.0' in text
