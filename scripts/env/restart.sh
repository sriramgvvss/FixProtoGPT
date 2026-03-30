#!/usr/bin/env bash
# ============================================================
# FixProtoGPT — Restart an environment
# ============================================================
# Usage:
#   ./scripts/restart.sh <env>             # Flask dev server
#   ./scripts/restart.sh <env> --gunicorn  # Gunicorn
#   ./scripts/restart.sh --all             # Restart all running envs
#
# Equivalent to stop + start.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_DIR="$PROJECT_ROOT/pids"

VALID_ENVS="dev qa preprod prod"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
    echo "Usage: $0 <env|--all> [--gunicorn]"
    echo "  env       : one of $VALID_ENVS"
    echo "  --all     : restart all currently-running environments"
    echo "  --gunicorn: use Gunicorn for the start phase"
    exit 1
}

[[ $# -lt 1 ]] && usage

USE_GUNICORN=""
[[ "${2:-}" == "--gunicorn" ]] && USE_GUNICORN="--gunicorn"

# ── Restart a single environment ──────────────────────────────
restart_env() {
    local env_name="$1"
    "$SCRIPT_DIR/stop.sh" "$env_name"
    "$SCRIPT_DIR/start.sh" "$env_name" $USE_GUNICORN
}

# ── Handle --all or single environment ────────────────────────
if [[ "$1" == "--all" ]]; then
    echo ""
    echo "============================================================"
    echo "  FixProtoGPT — Restarting ALL running environments"
    echo "============================================================"

    restarted=0
    for env_name in $VALID_ENVS; do
        if [[ -f "$PID_DIR/${env_name}.pid" ]]; then
            restart_env "$env_name"
            restarted=$((restarted + 1))
        fi
    done

    if [[ $restarted -eq 0 ]]; then
        info "No environments were running. Nothing to restart."
    fi

    echo "============================================================"
    echo ""
else
    ENV_NAME="$1"
    if ! echo "$VALID_ENVS" | grep -qw "$ENV_NAME"; then
        err "Invalid environment: '$ENV_NAME'.  Must be one of: $VALID_ENVS"
        exit 1
    fi

    restart_env "$ENV_NAME"
fi
