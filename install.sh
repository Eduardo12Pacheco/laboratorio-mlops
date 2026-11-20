#!/usr/bin/env bash
# Installer for the mlops-lifecycle-lab lab (L2 Docker Compose stack and L1 uv path).
#
# Exit codes: 0 ok, 1 prerequisites failed, 2 usage error, 3 install failed,
# 4 uninstall aborted. `--check` never builds, starts or removes anything.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${COMPOSE_PROJECT_NAME:-mlops-lifecycle-lab}"
PORTS=(14000 14080 14501)
MIN_DISK_KB=$((4 * 1024 * 1024))

CHECK_ONLY=0
L1_ONLY=0
UNINSTALL=0
ASSUME_YES=0
NO_START=0
FAILURES=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Install the local-first MLOps lab. The default path builds and starts the Docker
Compose stack; --l1 installs the CLI with uv for a Docker-free workflow.

Options:
  --check              Run the prerequisite checks only (no side effects).
  --l1                 Use the Docker-free L1 path (uv, Python 3.12, `mll doctor`).
  --uninstall          Stop the stack and remove its volumes and local images.
  --yes                Do not ask for confirmation on --uninstall.
  --no-start           Prepare the environment but do not build or start the stack.
  --project-name NAME  Compose project name (default: mlops-lifecycle-lab).
  --help, -h           Show this help.

Exit codes: 0 ok, 1 prerequisites failed, 2 usage error, 3 install failed,
4 uninstall aborted.
EOF
}

ok() { printf '[ ok ] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
fail() {
  printf '[FAIL] %s\n' "$*"
  FAILURES=$((FAILURES + 1))
}

usage_error() {
  printf 'error: %s\n\n' "$*" >&2
  usage >&2
  exit 2
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --check) CHECK_ONLY=1 ;;
      --l1) L1_ONLY=1 ;;
      --uninstall) UNINSTALL=1 ;;
      --yes) ASSUME_YES=1 ;;
      --no-start) NO_START=1 ;;
      --project-name)
        (($# >= 2)) || usage_error "--project-name requires a value"
        PROJECT_NAME="$2"
        shift
        ;;
      --project-name=*) PROJECT_NAME="${1#*=}" ;;
      --help | -h)
        usage
        exit 0
        ;;
      *)
        usage_error "unknown option: $1"
        ;;
    esac
    shift
  done
  [[ -n "$PROJECT_NAME" ]] || usage_error "--project-name must not be empty"
  [[ "$PROJECT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] ||
    usage_error "invalid compose project name: $PROJECT_NAME"
  if ((UNINSTALL == 1)) && ((CHECK_ONLY == 1 || L1_ONLY == 1 || NO_START == 1)); then
    usage_error "--uninstall cannot be combined with --check, --l1 or --no-start"
  fi
}

compose() {
  docker compose -p "$PROJECT_NAME" "$@"
}

compose_project_running() {
  local containers
  if containers="$(compose ps --status running -q 2>/dev/null)"; then
    [[ -n "$containers" ]]
    return
  fi
  return 1
}

port_in_use() {
  local port="$1"
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    return 0
  fi
  return 1
}

check_docker() {
  local docker_version compose_version engine_os major

  if ! command -v docker >/dev/null 2>&1; then
    fail "docker not found (install Docker >= 24 with the Compose v2 plugin)"
    return
  fi
  ok "docker found"

  docker_version="$(docker --version 2>/dev/null || true)"
  if [[ "$docker_version" =~ ([0-9]+)\.[0-9]+ ]]; then
    major="${BASH_REMATCH[1]}"
    if ((major >= 24)); then
      ok "docker >= 24 (${docker_version})"
    else
      fail "docker >= 24 is required (found: ${docker_version})"
    fi
  else
    fail "could not determine the docker version"
  fi

  compose_version="$(docker compose version 2>/dev/null || true)"
  if [[ "$compose_version" =~ v?([0-9]+)\.[0-9]+ ]] && ((BASH_REMATCH[1] >= 2)); then
    ok "compose v2 (${compose_version})"
  else
    fail "docker compose v2 is required (found: ${compose_version:-not installed})"
  fi

  if docker info >/dev/null 2>&1; then
    ok "docker daemon reachable"
  else
    fail "docker daemon is not reachable"
  fi

  engine_os="$(docker info --format '{{.OSType}}' 2>/dev/null || true)"
  case "$engine_os" in
    linux) ok "docker engine runs Linux containers" ;;
    "") warn "could not determine the docker engine OS" ;;
    *) fail "docker engine OS is '${engine_os}'; Linux containers are required" ;;
  esac
}

check_ports() {
  local port
  if compose_project_running; then
    ok "host ports are owned by compose project '${PROJECT_NAME}'"
    return
  fi
  for port in "${PORTS[@]}"; do
    if port_in_use "$port"; then
      fail "port ${port} is already in use (stop the process or set COMPOSE_PROJECT_NAME)"
    else
      ok "port ${port} is free"
    fi
  done
}

check_disk() {
  local available_kb
  available_kb="$(df -Pk "$REPO_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}')"
  if [[ -z "$available_kb" ]]; then
    warn "could not determine free disk space"
  elif ((available_kb < MIN_DISK_KB)); then
    fail "at least 4 GB of free disk is required (found: $((available_kb / 1024)) MB)"
  else
    ok "at least 4 GB free disk ($((available_kb / 1024)) MB)"
  fi
}

check_env_file() {
  if [[ -f "$REPO_ROOT/.env" ]]; then
    ok ".env present"
  else
    warn ".env not found (install creates it from .env.example)"
  fi
}

check_l1() {
  if ! command -v uv >/dev/null 2>&1; then
    fail "uv not found (install it from https://docs.astral.sh/uv/)"
    return
  fi
  ok "uv found ($(uv --version 2>/dev/null || echo 'version unknown'))"

  if (cd "$REPO_ROOT" && uv python find 3.12 >/dev/null 2>&1); then
    ok "Python 3.12 resolvable with uv"
  else
    fail "Python 3.12 is not resolvable with uv (run: uv python install 3.12)"
  fi

  if (cd "$REPO_ROOT" && uv run mll doctor >/dev/null 2>&1); then
    ok "mll doctor"
  else
    fail "mll doctor failed (run: uv run mll doctor)"
  fi
}

check_l2() {
  check_docker
  check_ports
  check_disk
  check_env_file
}

run_checks() {
  FAILURES=0
  if ((L1_ONLY == 1)); then
    check_l1
  else
    check_l2
  fi
  ((FAILURES == 0))
}

create_env_file() {
  if [[ -f "$REPO_ROOT/.env" ]]; then
    ok ".env already exists"
  else
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    ok "created .env from .env.example"
  fi
}

print_urls() {
  printf '\nControl Room: http://127.0.0.1:14501\n'
  printf 'API docs:     http://127.0.0.1:14080/docs\n'
  printf 'MLflow:       http://127.0.0.1:14000\n'
}

uninstall() {
  local answer

  if ! command -v docker >/dev/null 2>&1; then
    fail "docker not found"
    exit 1
  fi
  if ((ASSUME_YES == 0)); then
    printf "Remove compose project '%s' with its volumes and local images? [y/N] " "$PROJECT_NAME"
    if ! read -r answer; then
      answer=""
    fi
    case "$answer" in
      y | Y | yes | YES) ;;
      *)
        printf 'uninstall aborted\n'
        exit 4
        ;;
    esac
  fi
  if ! compose down -v --rmi local; then
    fail "docker compose down failed"
    exit 3
  fi
  ok "uninstalled compose project '${PROJECT_NAME}' (.env kept)"
  exit 0
}

install_l1() {
  if ! run_checks; then
    printf '\nL1 prerequisites failed\n' >&2
    exit 1
  fi
  if ! (cd "$REPO_ROOT" && uv tool install --force "$REPO_ROOT"); then
    fail "uv tool install failed"
    exit 3
  fi
  ok "installed the mll CLI with uv"
  printf '\nNext: mll quickstart\n      mll dashboard\n'
  exit 0
}

install_l2() {
  local already_installed=0

  if ! run_checks; then
    printf '\nPrerequisite checks failed; nothing was started.\n' >&2
    exit 1
  fi

  create_env_file

  if ((NO_START == 1)); then
    ok "--no-start requested; the stack was not started"
    print_urls
    exit 0
  fi

  if compose_project_running; then
    already_installed=1
  fi

  if ! compose up -d --build --wait; then
    fail "docker compose up failed"
    exit 3
  fi

  if ((already_installed == 1)); then
    ok "already installed, updated"
  else
    ok "installed compose project '${PROJECT_NAME}'"
  fi
  print_urls
  exit 0
}

main() {
  parse_args "$@"

  if ((UNINSTALL == 1)); then
    uninstall
  fi
  if ((CHECK_ONLY == 1)); then
    if run_checks; then
      printf '\nPrerequisites OK\n'
      exit 0
    fi
    printf '\nPrerequisites FAILED\n' >&2
    exit 1
  fi
  if ((L1_ONLY == 1)); then
    install_l1
  fi
  install_l2
}

main "$@"
