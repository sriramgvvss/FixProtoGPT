#!/usr/bin/env bash
# ============================================================
# FixProtoGPT — Show status of running environments
# ============================================================
# Usage:
#   ./scripts/status.sh           # Show all environments
#   ./scripts/status.sh <env>     # Show a specific environment
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PID_DIR="$PROJECT_ROOT/pids"

# ── Colour helpers ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

VALID_ENVS="dev qa preprod prod"

# ── Port lookup helper ────────────────────────────────────────
default_port() {
    case "$1" in
        dev)     echo 8080 ;;
        qa)      echo 8081 ;;
        preprod) echo 8082 ;;
        prod)    echo 8083 ;;
    esac
}

# ── Show status for one env ───────────────────────────────────
show_env_status() {
    local env_name="$1"
    local pid_file="$PID_DIR/${env_name}.pid"
    local port
    port=$(default_port "$env_name")

    # Try to read actual port from env file
    local env_file="$PROJECT_ROOT/config/env/.env.$env_name"
    if [[ -f "$env_file" ]]; then
        local file_port
        file_port=$(grep -E '^FIXPROTOGPT_PORT=' "$env_file" | cut -d= -f2 | tr -d '[:space:]')
        [[ -n "$file_port" ]] && port="$file_port"
    fi

    if [[ -f "$pid_file" ]]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            # Get process uptime
            local started
            if [[ "$(uname)" == "Darwin" ]]; then
                started=$(ps -o lstart= -p "$pid" 2>/dev/null | xargs) || started="unknown"
            else
                started=$(ps -o lstart= -p "$pid" 2>/dev/null | xargs) || started="unknown"
            fi
            printf "  ${GREEN}●${NC}  %-10s ${GREEN}RUNNING${NC}   PID %-8s  Port %-6s  Started: %s\n" \
                "$env_name" "$pid" "$port" "$started"
        else
            printf "  ${YELLOW}○${NC}  %-10s ${YELLOW}STALE${NC}     PID %-8s  Port %-6s  (process not found)\n" \
                "$env_name" "$pid" "$port"
        fi
    else
        printf "  ${RED}○${NC}  %-10s ${RED}STOPPED${NC}   %13s  Port %-6s\n" \
            "$env_name" "" "$port"
    fi
}

# ── Main ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  FixProtoGPT — Environment Status${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""

if [[ $# -ge 1 ]] && [[ "$1" != "--all" ]]; then
    ENV_NAME="$1"
    if ! echo "$VALID_ENVS" | grep -qw "$ENV_NAME"; then
        echo -e "  ${RED}[ERROR]${NC} Invalid environment: '$ENV_NAME'"
        echo "  Valid: $VALID_ENVS"
        exit 1
    fi
    show_env_status "$ENV_NAME"
else
    for env_name in $VALID_ENVS; do
        show_env_status "$env_name"
    done
fi

echo ""

# ── Quick summary ─────────────────────────────────────────────
running=0
for env_name in $VALID_ENVS; do
    pid_file="$PID_DIR/${env_name}.pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            running=$((running + 1))
        fi
    fi
done

echo -e "  ${CYAN}$running of 4 environments running${NC}"
echo ""
echo -e "${BOLD}============================================================${NC}"
echo ""
