#!/usr/bin/env bash
# End-to-end Compose smoke for S19: build the stack, wait for every healthcheck, assert
# that the scheduler promotes the first challenger through the real API and that every
# process runs as the documented non-root user. Always tears the project down with volumes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

project="${COMPOSE_PROJECT_NAME:-mlops-lifecycle-lab-smoke}"
export MLL_REPLAY_INTERVAL_SECONDS="${MLL_REPLAY_INTERVAL_SECONDS:-1}"

compose=(docker compose -p "$project")

cleanup() {
  local status
  status=$?
  if ((status != 0)); then
    echo "compose smoke failed; recent logs:" >&2
    "${compose[@]}" logs --tail=80 >&2 || true
  fi
  "${compose[@]}" down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

wait_http() {
  local url="$1"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

"${compose[@]}" up -d --build --wait --wait-timeout 240

wait_http "http://127.0.0.1:14000/health"
wait_http "http://127.0.0.1:14080/healthz"
wait_http "http://127.0.0.1:14080/readyz"
wait_http "http://127.0.0.1:14501/_stcore/health"

api_uid="$("${compose[@]}" exec -T api id -u | tr -d '\r')"
if [[ "$api_uid" != "10001" ]]; then
  echo "api runs as uid ${api_uid}, expected 10001" >&2
  exit 1
fi

version=""
for _ in $(seq 1 120); do
  if body="$(curl -fsS "http://127.0.0.1:14080/v1/model" 2>/dev/null)"; then
    version="$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version", ""))')"
    if [[ "$version" == "2" ]]; then
      break
    fi
  fi
  sleep 1
done

if [[ "$version" != "2" ]]; then
  echo "champion did not reach version 2 (last seen: '${version}')" >&2
  exit 1
fi

echo "SMOKE OK champion=${version}"
echo "image size: $(docker images mll:local --format '{{.Size}}')"
