# syntax=docker/dockerfile:1.7
# Multi-stage image for every Compose service (D16): one locked, non-editable wheel
# installed into a virtualenv, then copied into a slim runtime with only libgomp1.
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer only changes when the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --no-editable

COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm AS runtime

# LightGBM needs the OpenMP runtime (D26); nothing else is required at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 mll \
    && useradd --uid 10001 --gid 10001 --no-create-home \
        --home-dir /home/mll --shell /usr/sbin/nologin mll

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# The named volume is seeded from this directory on first use, so the non-root
# user owns its state without a privileged entrypoint.
RUN mkdir -p /data/mlflow/artifacts \
    && chown -R 10001:10001 /data

ENV PATH=/app/.venv/bin:$PATH \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLL_WORKSPACE=/data \
    MLL_LOG_FORMAT=json \
    MLFLOW_DISABLE_TELEMETRY=true \
    DO_NOT_TRACK=1 \
    ITERATIVE_DO_NOT_TRACK=1 \
    EVIDENTLY_DISABLE_TELEMETRY=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MLFLOW_DISABLE_AGENT_HINT=1

USER 10001:10001

ENTRYPOINT ["mll"]
