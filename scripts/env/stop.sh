#!/usr/bin/env bash
# ============================================================
# FixProtoGPT — Stop an environment
# ============================================================
# Usage:
#   ./scripts/stop.sh <env>       # Stop a specific environment
#   ./scripts/stop.sh --all       # Stop ALL running environments
#
# Reads the PID from pids/<env>.pid and sends SIGTERM.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_DIR="$PROJECT_ROOT/pids"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

VALID_ENVS="dev qa preprod prod"

usage() {
    echo "Usage: $0 <env|--all>"
    echo "  env : one of $VALID_ENVS"
    echo "  --all : stop all running environments"
    exit 1
}

[[ $# -lt 1 ]] && usage

# ── Stop a single environment ─────────────────────────────────
stop_env() {
    local env_name="$1"
    local pid_file="$PID_DIR/${env_name}.pid"

    if [[ ! -f "$pid_file" ]]; then
        warn "No PID file for '$env_name' — not running (or was stopped manually)."
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")

    echo ""
    info "Stopping '$env_name' (PID $pid)..."

    if kill -0 "$pid" 2>/dev/null; then
        # Send SIGTERM for graceful shutdown
        kill "$pid" 2>/dev/null || true

        # Wait up to 10 seconds for process to exit
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 10 ]]; do
            sleep 1
            waited=$((waited + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            warn "Process did not exit gracefully. Sending SIGKILL..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi

        ok "Environment '$env_name' stopped."
    else
        warn "Process $pid is not running (stale PID file)."
    fi

    rm -f "$pid_file"
}

# ── Handle --all or single environment ────────────────────────
if [[ "$1" == "--all" ]]; then
    echo ""
    echo "============================================================"
    echo "  FixProtoGPT — Stopping ALL environments"
    echo "============================================================"

    stopped=0
    for env_name in $VALID_ENVS; do
        if [[ -f "$PID_DIR/${env_name}.pid" ]]; then
            stop_env "$env_name"
            stopped=$((stopped + 1))
        fi
    done

    if [[ $stopped -eq 0 ]]; then
        info "No environments were running."
    fi

    echo "============================================================"
    echo ""
else
    ENV_NAME="$1"
    if ! echo "$VALID_ENVS" | grep -qw "$ENV_NAME"; then
        err "Invalid environment: '$ENV_NAME'.  Must be one of: $VALID_ENVS"
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  FixProtoGPT — Stopping '$ENV_NAME' environment"
    echo "============================================================"

    stop_env "$ENV_NAME"

    echo "============================================================"
    echo ""
fi
