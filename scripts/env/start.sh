#!/usr/bin/env bash
# ============================================================
# FixProtoGPT — Start an environment
# ============================================================
# Usage:
#   ./scripts/start.sh <env>          # Flask dev server
#   ./scripts/start.sh <env> --gunicorn  # Gunicorn (prod-grade)
#
# Environments: dev | qa | preprod | prod
# Default ports: dev=8080  qa=8081  preprod=8082  prod=8083
#
# The process runs in the background; its PID is saved to
# pids/<env>.pid so that stop.sh / restart.sh can manage it.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_DIR="$PROJECT_ROOT/pids"
LOG_BASE="$PROJECT_ROOT/logs"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Validate arguments ───────────────────────────────────────
VALID_ENVS="dev qa preprod prod"

usage() {
    echo "Usage: $0 <env> [--gunicorn]"
    echo "  env: one of $VALID_ENVS"
    echo "  --gunicorn: use Gunicorn instead of Flask dev server"
    exit 1
}

[[ $# -lt 1 ]] && usage
ENV_NAME="$1"
USE_GUNICORN=false
[[ "${2:-}" == "--gunicorn" ]] && USE_GUNICORN=true

if ! echo "$VALID_ENVS" | grep -qw "$ENV_NAME"; then
    err "Invalid environment: '$ENV_NAME'.  Must be one of: $VALID_ENVS"
    exit 1
fi

# ── Check if already running ─────────────────────────────────
mkdir -p "$PID_DIR"
PID_FILE="$PID_DIR/${ENV_NAME}.pid"

if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        err "Environment '$ENV_NAME' is already running (PID $OLD_PID)."
        err "Use  ./scripts/stop.sh $ENV_NAME  first, or  ./scripts/restart.sh $ENV_NAME"
        exit 1
    else
        warn "Stale PID file found (PID $OLD_PID not running). Cleaning up."
        rm -f "$PID_FILE"
    fi
fi

# ── Resolve port (env file → fallback) ───────────────────────
ENV_FILE="$PROJECT_ROOT/config/env/.env.$ENV_NAME"
if [[ -f "$ENV_FILE" ]]; then
    PORT=$(grep -E '^FIXPROTOGPT_PORT=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]')
fi
# Fallback defaults if env file missing or var not set
if [[ -z "${PORT:-}" ]]; then
    case "$ENV_NAME" in
        dev)     PORT=8080 ;;
        qa)      PORT=8081 ;;
        preprod) PORT=8082 ;;
        prod)    PORT=8083 ;;
    esac
fi

# ── Ensure log directory ──────────────────────────────────────
LOG_DIR="$LOG_BASE/$ENV_NAME"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/app_stdout.log"

# ── Banner ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  FixProtoGPT — Starting '$ENV_NAME' environment"
echo "============================================================"
info "Port      : $PORT"
info "Log dir   : logs/$ENV_NAME/"
info "PID file  : pids/$ENV_NAME.pid"

# ── Launch ────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

if $USE_GUNICORN; then
    # Gunicorn — production-grade
    if ! python3 -c "import gunicorn" 2>/dev/null; then
        err "Gunicorn is not installed.  Run:  pip install 'fixprotogpt[prod]'"
        exit 1
    fi

    WORKERS="${FIXPROTOGPT_WORKERS:-4}"
    info "Server    : Gunicorn ($WORKERS workers)"

    FIXPROTOGPT_ENV="$ENV_NAME" \
    FIXPROTOGPT_PORT="$PORT" \
    nohup python3 -m gunicorn \
        "src.api.app:create_app()" \
        --bind "0.0.0.0:$PORT" \
        --workers "$WORKERS" \
        --access-logfile "$LOG_DIR/gunicorn_access.log" \
        --error-logfile "$LOG_DIR/gunicorn_error.log" \
        --pid "$PID_FILE" \
        >> "$STDOUT_LOG" 2>&1 &

    PROC_PID=$!
    # Gunicorn writes its own PID file, but we also capture the launcher PID
    echo "$PROC_PID" > "$PID_FILE"
else
    # Flask dev server
    info "Server    : Flask development server"

    FIXPROTOGPT_ENV="$ENV_NAME" \
    FIXPROTOGPT_PORT="$PORT" \
    nohup python3 -m src.api.app \
        >> "$STDOUT_LOG" 2>&1 &

    PROC_PID=$!
    echo "$PROC_PID" > "$PID_FILE"
fi

# ── Wait briefly to check startup ────────────────────────────
sleep 2

if kill -0 "$PROC_PID" 2>/dev/null; then
    ok "Environment '$ENV_NAME' started successfully (PID $PROC_PID)"
    info "Access at  : http://localhost:$PORT"
    info "Stdout log : $STDOUT_LOG"
else
    err "Process exited immediately. Check $STDOUT_LOG for details."
    rm -f "$PID_FILE"
    tail -20 "$STDOUT_LOG" 2>/dev/null || true
    exit 1
fi

echo "============================================================"
echo ""
