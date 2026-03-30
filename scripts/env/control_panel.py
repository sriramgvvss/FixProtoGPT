#!/usr/bin/env python3
"""
FixProtoGPT — Interactive Environment Control Panel
====================================================

A terminal-based UI for managing FixProtoGPT environments.
Uses ``rich`` for rendering (already a project dependency).

Launch::

    python3 scripts/env/control_panel.py
    # or
    make control-panel

Coding Standards:
    PEP 8, PEP 257 (Google-style), PEP 484
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# ── Resolve paths ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PID_DIR = PROJECT_ROOT / "pids"
ENV_DIR = PROJECT_ROOT / "config" / "env"

# ── Rich imports ──────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.columns import Columns
    from rich.prompt import Prompt, IntPrompt
    from rich import box
except ImportError:
    print("ERROR: 'rich' is required.  Install with:  pip install rich")
    sys.exit(1)

console = Console()

# ── Constants ─────────────────────────────────────────────────
VALID_ENVS = ("dev", "qa", "preprod", "prod")
DEFAULT_PORTS: Dict[str, int] = {
    "dev": 8080,
    "qa": 8081,
    "preprod": 8082,
    "prod": 8083,
}
_DEV_SECRET = "fixprotogpt-dev-secret-change-in-prod"


# ── Helpers ───────────────────────────────────────────────────

def _get_pid(env_name: str) -> Optional[int]:
    """Return the PID for *env_name* if a PID file exists, else ``None``."""
    pid_file = PID_DIR / f"{env_name}.pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _is_running(pid: Optional[int]) -> bool:
    """Return ``True`` if *pid* refers to a live process."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _get_port(env_name: str) -> int:
    """Read port from the env file, falling back to defaults."""
    env_file = ENV_DIR / f".env.{env_name}"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("FIXPROTOGPT_PORT="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    return DEFAULT_PORTS.get(env_name, 8080)


def _get_log_level(env_name: str) -> str:
    """Read log level from the env file."""
    env_file = ENV_DIR / f".env.{env_name}"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("FIXPROTOGPT_LOG_LEVEL="):
                return line.split("=", 1)[1].strip()
    return "DEBUG" if env_name in ("dev", "qa") else "WARNING"


def _get_debug(env_name: str) -> bool:
    """Read debug flag from the env file."""
    env_file = ENV_DIR / f".env.{env_name}"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("FIXPROTOGPT_DEBUG="):
                return line.split("=", 1)[1].strip().lower() in ("true", "1", "yes")
    return env_name in ("dev", "qa")


def _uptime_str(pid: int) -> str:
    """Return a human-readable uptime for *pid*."""
    try:
        # macOS: ps -o etime= gives elapsed time
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        etime = result.stdout.strip()
        if etime:
            return etime
    except Exception:
        pass
    return "?"


def _run_script(script_name: str, *args: str) -> subprocess.CompletedProcess:
    """Run a management script from ``scripts/env/``."""
    script = SCRIPT_DIR / script_name
    cmd = [str(script)] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def _health_check(port: int) -> Optional[dict]:
    """Quick HTTP health check. Returns parsed JSON or None."""
    try:
        import urllib.request
        import json
        url = f"http://localhost:{port}/ops/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _get_admin_key(env_name: Optional[str] = None) -> str:
    """Read the secret key for internal API auth.

    Checks ``FIXPROTOGPT_SECRET_KEY`` env-var first, then the .env
    file for *env_name*, falling back to the dev default.
    """
    key = os.environ.get("FIXPROTOGPT_SECRET_KEY", "")
    if key:
        return key
    if env_name:
        env_file = ENV_DIR / f".env.{env_name}"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("FIXPROTOGPT_SECRET_KEY="):
                    return line.split("=", 1)[1].strip()
    return _DEV_SECRET


def _api_call(port: int, path: str, *, method: str = "GET",
              body: Optional[dict] = None, timeout: int = 10) -> Optional[dict]:
    """Make an authenticated API call to a running environment."""
    try:
        import urllib.request
        import urllib.error
        import json
        url = f"http://localhost:{port}{path}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Admin-Key", _get_admin_key())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as he:
            # Read the response body even on 4xx/5xx
            try:
                return json.loads(he.read().decode())
            except Exception:
                return {"error": f"HTTP {he.code}", "message": f"HTTP {he.code}"}
    except Exception:
        return None


def _get_model_status(port: int) -> Optional[dict]:
    """Fetch model engine status from a running environment."""
    return _api_call(port, "/admin/models")


# ── Local checkpoint scanner (no running server needed) ───────

def _version_to_slug(version: str) -> str:
    """Convert version key to filesystem slug: ``"5.0SP2"`` → ``"fix-5-0sp2"``."""
    return "fix-" + version.lower().replace(".", "-")


MODEL_STORE = PROJECT_ROOT / "model_store"
CONFIG_FILE = PROJECT_ROOT / "config" / "model_config.yaml"


def _load_model_versions() -> tuple:
    """Read FIX version keys from ``config/model_config.yaml``.

    Returns a tuple of version key strings, e.g.
    ``("4.2", "4.4", "5.0SP2", "Latest")``.

    Falls back to scanning ``model_store/checkpoints/`` directory names
    if the config file cannot be parsed.
    """
    try:
        import yaml  # PyYAML is a project dependency
        with open(CONFIG_FILE, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        versions = cfg.get("data", {}).get("fix_versions", {})
        if versions:
            return tuple(versions.keys())
    except Exception:
        pass

    # Fallback: derive from checkpoint directories on disk
    ckpt_root = MODEL_STORE / "checkpoints"
    if ckpt_root.exists():
        found = []
        for d in sorted(ckpt_root.iterdir()):
            if d.is_dir() and d.name.startswith("fix-"):
                # Reverse slug: "fix-5-0sp2" → "5.0SP2" (best-effort)
                raw = d.name[4:].replace("-", ".").upper()
                found.append(raw)
        if found:
            return tuple(found)

    return ()


def _load_version_labels() -> Dict[str, str]:
    """Return ``{"4.2": "FIX 4.2", …}`` from config."""
    try:
        import yaml
        with open(CONFIG_FILE, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("data", {}).get("fix_versions", {})
    except Exception:
        return {}


# Resolved once at import time (cheap YAML parse, no server needed)
_MODEL_VERSIONS: tuple = _load_model_versions()
_VERSION_LABELS: Dict[str, str] = _load_version_labels()

# ── Per-env model preferences (works without a running server) ─

MODEL_PREFS_FILE = MODEL_STORE / "model_prefs.json"


def _read_model_prefs() -> Dict[str, list]:
    """Read ``model_store/model_prefs.json``.

    Returns ``{env_name: [enabled_versions]}`` e.g.
    ``{"dev": ["4.2", "Latest"], "qa": ["4.4"]}``.

    Missing envs default to *all available* versions.
    """
    import json
    try:
        with open(MODEL_PREFS_FILE, "r") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {}


def _write_model_prefs(prefs: Dict[str, list]) -> None:
    """Persist model prefs to ``model_store/model_prefs.json``."""
    import json
    MODEL_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PREFS_FILE, "w") as fh:
        json.dump(prefs, fh, indent=2, sort_keys=True)


def _enabled_versions(env_name: str) -> list:
    """Return the list of model versions enabled for *env_name*.

    If the env has no prefs yet, all versions with checkpoints are enabled.
    """
    prefs = _read_model_prefs()
    if env_name in prefs:
        return prefs[env_name]
    # Default: enable every version that has a checkpoint
    ckpts = _local_checkpoint_status()
    return [v for v, info in ckpts.items() if info["has_checkpoint"]]


def _set_enabled_versions(env_name: str, versions: list) -> None:
    """Set the enabled model list for *env_name* and persist."""
    prefs = _read_model_prefs()
    prefs[env_name] = versions
    _write_model_prefs(prefs)


def _resolve_env_name(env_token: str) -> Optional[str]:
    """Resolve an env token (number or name) to an env name.

    Unlike ``_resolve_env_port`` this does **not** require the server
    to be running — useful for offline model management.
    """
    if env_token.isdigit():
        idx = int(env_token) - 1
        if 0 <= idx < len(VALID_ENVS):
            return VALID_ENVS[idx]
        return None
    if env_token in VALID_ENVS:
        return env_token
    return None


def _local_checkpoint_status() -> Dict[str, dict]:
    """Scan disk for model checkpoints — works without a running server.

    Returns a dict keyed by version with ``has_checkpoint`` and ``path``.
    """
    status: Dict[str, dict] = {}
    for ver in _MODEL_VERSIONS:
        slug = _version_to_slug(ver)
        best = MODEL_STORE / "checkpoints" / slug / "best.pt"
        status[ver] = {
            "has_checkpoint": best.exists(),
            "path": str(best),
        }
    return status


def _any_checkpoint_available() -> bool:
    """Return ``True`` if at least one model checkpoint exists on disk."""
    return any(v["has_checkpoint"] for v in _local_checkpoint_status().values())


def _checkpoint_summary_line() -> str:
    """One-line summary of available checkpoints for display."""
    ckpts = _local_checkpoint_status()
    ready = [v for v, info in ckpts.items() if info["has_checkpoint"]]
    if not ready:
        return "[red]No model checkpoints found[/red]"
    return "[green]" + ", ".join(f"FIX {v}" for v in ready) + "[/green]"


# ── UI Components ─────────────────────────────────────────────

def _build_status_table() -> Table:
    """Build a rich Table showing all environment statuses."""
    table = Table(
        title="Environment Status",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    table.add_column("#", style="dim", width=3, justify="center")
    table.add_column("Environment", style="bold", width=10)
    table.add_column("Status", width=10, justify="center")
    table.add_column("URL", width=26)
    table.add_column("PID", width=8, justify="right")
    table.add_column("Uptime", width=14)
    table.add_column("Debug", width=6, justify="center")
    table.add_column("Log Level", width=10)
    table.add_column("Health", width=8, justify="center")
    table.add_column("Actions", width=36, no_wrap=True)

    for idx, env_name in enumerate(VALID_ENVS, 1):
        pid = _get_pid(env_name)
        running = _is_running(pid)
        port = _get_port(env_name)

        url = f"http://localhost:{port}"

        if running:
            status = Text("● RUNNING", style="bold green")
            url_text = Text(url, style="bold underline blue link " + url)
            pid_str = str(pid)
            uptime = _uptime_str(pid)
            health_data = _health_check(port)
            health = Text("✓ OK", style="green") if health_data else Text("✗ ERR", style="red")
        else:
            status = Text("○ STOPPED", style="dim red")
            url_text = Text(url, style="dim")
            pid_str = "—"
            uptime = "—"
            health = Text("—", style="dim")

        debug = "✓" if _get_debug(env_name) else "✗"
        log_level = _get_log_level(env_name)

        # Build context-sensitive action hints
        acts = Text()
        if running:
            acts.append(f"[{idx}x]", style="bold red")
            acts.append("Stop ", style="red")
            acts.append(f"[{idx}r]", style="bold yellow")
            acts.append("Restart ", style="yellow")
        else:
            acts.append(f"[{idx}s]", style="bold green")
            acts.append("Start ", style="green")
        acts.append(f"[{idx}l]", style="bold blue")
        acts.append("Logs", style="blue")

        table.add_row(
            str(idx),
            env_name,
            status,
            url_text,
            pid_str,
            uptime,
            debug,
            log_level,
            health,
            acts,
        )

    return table


def _build_model_table() -> Panel:
    """Build a Rich table showing model engine status across all environments.

    Works both online (server running — shows live load state) and offline
    (shows checkpoint availability + enabled/disabled preference).
    """
    table = Table(
        title="Model Engines",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title_style="bold white",
        expand=True,
    )
    table.add_column("Env", style="bold", width=8)
    table.add_column("Status", width=8, justify="center")
    for ver in _MODEL_VERSIONS:
        label = _VERSION_LABELS.get(ver, f"FIX {ver}")
        table.add_column(label, width=14, justify="center")

    table.add_column("Actions", width=24, no_wrap=True)

    n_ver = len(_MODEL_VERSIONS)
    ckpts = _local_checkpoint_status()

    for idx, env_name in enumerate(VALID_ENVS, 1):
        pid = _get_pid(env_name)
        running = _is_running(pid)
        enabled = _enabled_versions(env_name)

        # env_status will be refined below once we know model load state
        env_status_running = running

        # Try live data from server; fall back to filesystem
        engines: Optional[dict] = None
        if running:
            port = _get_port(env_name)
            data = _get_model_status(port)
            if data and "engines" in data:
                engines = data["engines"]

        cells = []
        has_any_loaded = False
        has_any_unloaded = False
        has_any_disabled = False
        for ver in _MODEL_VERSIONS:
            is_enabled = ver in enabled
            has_ckpt = ckpts.get(ver, {}).get("has_checkpoint", False)

            if engines:
                # Live server data available
                info = engines.get(ver, {})
                if info.get("loaded"):
                    label, style = "● loaded", "bold green"
                    has_any_loaded = True
                elif has_ckpt and is_enabled:
                    label, style = "○ ready", "yellow"
                    has_any_unloaded = True
                elif has_ckpt and not is_enabled:
                    label, style = "⊘ disabled", "dim yellow"
                    has_any_disabled = True
                else:
                    label, style = "— none", "dim"
            else:
                # Offline — show filesystem + prefs
                if has_ckpt and is_enabled:
                    label, style = "✓ enabled", "green"
                    has_any_unloaded = True
                elif has_ckpt and not is_enabled:
                    label, style = "⊘ disabled", "dim yellow"
                    has_any_disabled = True
                else:
                    label, style = "— none", "dim"
            cells.append(Text(label, style=style))

        # Determine environment status label
        if not env_status_running:
            env_status = Text("○ DOWN", style="dim")
        elif has_any_loaded:
            env_status = Text("● UP", style="bold green")
        else:
            env_status = Text("● DEMO", style="bold yellow")

        acts = Text()
        if has_any_unloaded or has_any_disabled:
            acts.append(f"[{idx}ml]", style="bold green")
            acts.append("Enable ", style="green")
        if has_any_loaded or has_any_unloaded:
            acts.append(f"[{idx}mx]", style="bold red")
            acts.append("Disable", style="red")
        if not acts.plain.strip():
            acts = Text("—", style="dim")
        table.add_row(env_name, env_status, *cells, acts)

    return Panel(table, border_style="magenta", box=box.ROUNDED)


def _build_menu() -> Panel:
    """Build the action menu panel."""
    parts = Text()
    parts.append("Per-env: ", style="bold white")
    parts.append("<#>s", style="bold green")
    parts.append(" Start  ", style="green")
    parts.append("<#>x", style="bold red")
    parts.append(" Stop  ", style="red")
    parts.append("<#>r", style="bold yellow")
    parts.append(" Restart  ", style="yellow")
    parts.append("<#>l", style="bold blue")
    parts.append(" Logs    ", style="blue")
    parts.append("Bulk: ", style="bold white")
    parts.append("sa", style="bold cyan")
    parts.append(" All↑  ", style="cyan")
    parts.append("xa", style="bold cyan")
    parts.append(" All↓  ", style="cyan")
    parts.append("ra", style="bold cyan")
    parts.append(" All↻  ", style="cyan")
    parts.append("h", style="bold cyan")
    parts.append(" Health  ", style="cyan")
    parts.append("m", style="bold magenta")
    parts.append(" Models  ", style="magenta")
    parts.append("q", style="bold cyan")
    parts.append(" Quit  ", style="cyan")
    parts.append("↵", style="bold cyan")
    parts.append(" Refresh", style="cyan")

    model_parts = Text()
    model_parts.append("\nModels: ", style="bold white")
    model_parts.append("<#>ml", style="bold green")
    model_parts.append(" Enable all  ", style="green")
    model_parts.append("<#>mx", style="bold red")
    model_parts.append(" Disable all  ", style="red")
    model_parts.append("<#>ml <ver>", style="bold green")
    model_parts.append(" Enable ver  ", style="green")
    model_parts.append("<#>mx <ver>", style="bold red")
    model_parts.append(" Disable ver", style="red")

    bulk_model = Text()
    bulk_model.append("\nBulk models: ", style="bold white")
    bulk_model.append("mla", style="bold green")
    bulk_model.append(" Enable all everywhere  ", style="green")
    bulk_model.append("mxa", style="bold red")
    bulk_model.append(" Disable all everywhere  ", style="red")
    bulk_model.append("ml <ver>", style="bold green")
    bulk_model.append(" Enable ver everywhere  ", style="green")
    bulk_model.append("mx <ver>", style="bold red")
    bulk_model.append(" Disable ver everywhere", style="red")
    ver_list = " | ".join(_MODEL_VERSIONS) if _MODEL_VERSIONS else "none configured"
    bulk_model.append(f"  [dim](ver = {ver_list})[/dim]")

    combined = Text()
    combined.append_text(parts)
    combined.append_text(model_parts)
    combined.append_text(bulk_model)

    return Panel(
        combined,
        title="Quick Commands  (e.g. [bold]1s[/] = start dev, [bold]mla[/] = enable all models everywhere)",
        border_style="cyan",
        box=box.ROUNDED,
    )





def _auto_load_models(env_name: str) -> bool:
    """After starting an environment, load enabled models that have checkpoints.

    Respects per-env model preferences from ``model_prefs.json``.
    Returns ``True`` if at least one model was loaded successfully.
    """
    port = _get_port(env_name)
    # Give the server a moment to become ready
    console.print(f"  [dim]Waiting for '{env_name}' to accept connections...[/]")
    for _ in range(10):
        time.sleep(1)
        if _health_check(port):
            break
    else:
        console.print(f"  [yellow]⚠ Server not ready yet — skipping auto-load[/]")
        return False

    ckpts = _local_checkpoint_status()
    enabled = _enabled_versions(env_name)
    to_load = [v for v, info in ckpts.items()
               if info["has_checkpoint"] and v in enabled]
    if not to_load:
        console.print(f"  [yellow]⚠ No enabled model checkpoints for '{env_name}'[/]")
        return False

    loaded_count = 0
    console.print(f"  [magenta]Auto-loading {len(to_load)} enabled model(s)...[/]")
    for ver in to_load:
        console.print(f"    [dim]Loading FIX {ver} (~240 MB)...[/]", end="")
        res = _api_call(port, "/admin/models/load", method="POST",
                        body={"version": ver}, timeout=120)
        if res and res.get("loaded"):
            console.print(f"\r    [green]✓ FIX {ver} loaded[/]          ")
            loaded_count += 1
        else:
            msg = res.get("message", "failed") if res else "API call failed (timeout?)"
            console.print(f"\r    [red]✗ FIX {ver}: {msg}[/]          ")

    if loaded_count == 0:
        console.print(
            f"  [bold red]⚠ No models loaded on '{env_name}' "
            f"— running in demo-only mode![/]"
        )
        return False

    console.print(
        f"  [green]✓ {loaded_count}/{len(to_load)} model(s) active on '{env_name}'[/]"
    )
    return True


def _action_start(env_name: str) -> None:
    """Start a single environment.

    Checks for model checkpoints first — warns and asks for confirmation
    if none are found (the server would run in demo-only mode).
    """
    pid = _get_pid(env_name)
    if _is_running(pid):
        console.print(f"  [yellow]⚠ '{env_name}' is already running (PID {pid})[/]")
        return

    # ── Pre-flight: model checkpoint check ─────────────────────
    if not _any_checkpoint_available():
        console.print()
        console.print("  [bold red]⚠ No model checkpoints found on disk.[/]")
        console.print("  [yellow]The server will start in demo-only mode "
                      "(no AI inference, only template responses).[/]")
        console.print("  [dim]Train models first, or place checkpoints in "
                      "model_store/checkpoints/<version>/best.pt[/]")
        console.print()
        try:
            proceed = Prompt.ask(
                "  Start in demo mode anyway?",
                choices=["y", "n"],
                default="n",
            )
        except (KeyboardInterrupt, EOFError):
            return
        if proceed.lower() != "y":
            console.print("  [dim]Start cancelled.[/]")
            return
    else:
        console.print(f"  [dim]Checkpoints found: {_checkpoint_summary_line()}[/]")

    console.print(f"  [cyan]Starting '{env_name}'...[/]")
    result = _run_script("start.sh", env_name)
    if result.returncode == 0:
        console.print(f"  [green]✓ '{env_name}' started successfully[/]")
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                console.print(f"    {line}")
        # Auto-load available models
        _auto_load_models(env_name)
    else:
        console.print(f"  [red]✗ Failed to start '{env_name}'[/]")
        if result.stderr:
            console.print(f"  [dim]{result.stderr.strip()}[/]")
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                console.print(f"    {line}")


def _action_stop(env_name: str) -> None:
    """Stop a single environment."""
    pid = _get_pid(env_name)
    if not _is_running(pid):
        console.print(f"  [yellow]⚠ '{env_name}' is not running[/]")
        return

    console.print(f"  [cyan]Stopping '{env_name}' (PID {pid})...[/]")
    result = _run_script("stop.sh", env_name)
    if result.returncode == 0:
        console.print(f"  [green]✓ '{env_name}' stopped[/]")
    else:
        console.print(f"  [red]✗ Failed to stop '{env_name}'[/]")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"    {line}")


def _action_restart(env_name: str) -> None:
    """Restart a single environment and re-load models."""
    console.print(f"  [cyan]Restarting '{env_name}'...[/]")
    result = _run_script("restart.sh", env_name)
    if result.returncode == 0:
        console.print(f"  [green]✓ '{env_name}' restarted[/]")
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                console.print(f"    {line}")
        # Re-load models after restart
        _auto_load_models(env_name)
    else:
        console.print(f"  [red]✗ Failed to restart '{env_name}'[/]")
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                console.print(f"    {line}")


def _action_start_all() -> None:
    """Start all environments (dev + qa by default; preprod/prod need secret key).

    Checks for model checkpoints before starting — warns if none are
    found and asks for confirmation.
    """
    console.print()

    # ── Pre-flight: model checkpoint check ─────────────────────
    if not _any_checkpoint_available():
        console.print("  [bold red]⚠ No model checkpoints found on disk.[/]")
        console.print("  [yellow]All servers will run in demo-only mode.[/]")
        console.print()
        try:
            proceed = Prompt.ask(
                "  Start all in demo mode anyway?",
                choices=["y", "n"],
                default="n",
            )
        except (KeyboardInterrupt, EOFError):
            return
        if proceed.lower() != "y":
            console.print("  [dim]Start cancelled.[/]")
            return
    else:
        console.print(f"  [dim]Checkpoints found: {_checkpoint_summary_line()}[/]")

    console.print("  [cyan]Starting all environments...[/]")
    started = []
    for env_name in VALID_ENVS:
        pid = _get_pid(env_name)
        if _is_running(pid):
            console.print(f"    [yellow]⚠ '{env_name}' already running — skipping[/]")
            continue
        if env_name in ("preprod", "prod"):
            # Check if secret key is overridden
            secret = os.environ.get("FIXPROTOGPT_SECRET_KEY", "")
            if not secret or secret == "fixprotogpt-dev-secret-change-in-prod":
                console.print(
                    f"    [yellow]⚠ Skipping '{env_name}' — set FIXPROTOGPT_SECRET_KEY first[/]"
                )
                continue
        result = _run_script("start.sh", env_name)
        if result.returncode == 0:
            console.print(f"    [green]✓ '{env_name}' started[/]")
            started.append(env_name)
        else:
            console.print(f"    [red]✗ '{env_name}' failed[/]")

    # Auto-load models on every newly started environment
    for env_name in started:
        _auto_load_models(env_name)


def _action_stop_all() -> None:
    """Stop all running environments."""
    console.print()
    console.print("  [cyan]Stopping all environments...[/]")
    result = _run_script("stop.sh", "--all")
    if result.returncode == 0:
        console.print("  [green]✓ All environments stopped[/]")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"    {line}")


def _action_restart_all() -> None:
    """Restart all running environments and re-load models."""
    console.print()
    console.print("  [cyan]Restarting all running environments...[/]")
    result = _run_script("restart.sh", "--all")
    if result.returncode == 0:
        console.print("  [green]✓ All environments restarted[/]")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            console.print(f"    {line}")

    # Re-load models on every running environment
    for env_name in VALID_ENVS:
        pid = _get_pid(env_name)
        if _is_running(pid):
            _auto_load_models(env_name)


def _action_health() -> None:
    """Run a health check on all running environments."""
    console.print()
    for env_name in VALID_ENVS:
        pid = _get_pid(env_name)
        if not _is_running(pid):
            console.print(f"  [dim]○ {env_name}: not running[/]")
            continue
        port = _get_port(env_name)
        data = _health_check(port)
        if data:
            status = data.get("status", "?")
            env_reported = data.get("env", "?")
            uptime = data.get("uptime_s", "?")
            model_info = data.get("model", {})
            model_status = model_info.get("status", "?")
            demo_mode = model_info.get("demo_mode", model_status != "loaded")
            db_status = data.get("db", {}).get("status", "?")
            mode_tag = "[bold yellow](demo-only)[/]" if demo_mode else "[green](inference)[/]"
            console.print(
                f"  [green]● {env_name}[/]  status={status}  "
                f"uptime={uptime}s  model={model_status}  db={db_status}  {mode_tag}"
            )
        else:
            console.print(f"  [red]● {env_name}[/]  [red]health check failed[/]")
    console.print()


def _action_models() -> None:
    """Display the model engine status table and offer model commands.

    Works regardless of whether environments are running — model
    preferences can be managed offline.
    """
    console.print()
    model_panel = _build_model_table()
    console.print(model_panel)

    console.print()
    console.print("  [bold]Model commands:[/]")
    console.print("    [magenta]l <env#> <version>[/]  Enable a model   (e.g. [bold]l 1 4.4[/])")
    console.print("    [magenta]u <env#> <version>[/]  Disable a model  (e.g. [bold]u 1 Latest[/])")
    console.print("    [magenta]la <env#>[/]           Enable all models on an env")
    console.print("    [magenta]ua <env#>[/]           Disable all models on an env")
    console.print("    [dim]Press Enter to go back[/]")
    console.print()

    try:
        sub = console.input("  [bold]Model cmd:[/] ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if not sub:
        return

    parts = sub.split()
    action = parts[0].lower()

    if action == "l" and len(parts) == 3:
        _model_load_unload(parts[1], parts[2], load=True)
    elif action == "u" and len(parts) == 3:
        _model_load_unload(parts[1], parts[2], load=False)
    elif action == "la" and len(parts) == 2:
        _action_model_load(parts[1])
    elif action == "ua" and len(parts) == 2:
        _model_unload_all(parts[1])
    else:
        console.print("  [yellow]Invalid model command.[/]")


def _resolve_env_port(env_token: str) -> Optional[tuple]:
    """Resolve an env token (number or name) to (env_name, port).

    Returns ``None`` if the env is not running.
    """
    name = _resolve_env_name(env_token)
    if not name:
        return None
    pid = _get_pid(name)
    if not _is_running(pid):
        console.print(f"  [yellow]'{name}' is not running.[/]")
        return None
    return (name, _get_port(name))


def _model_load_unload(env_token: str, version: str, *, load: bool) -> None:
    """Enable/disable a specific model version on an environment.

    Works offline (persists preference) and online (also loads/unloads
    via the API when the server is running).
    """
    env_name = _resolve_env_name(env_token)
    if not env_name:
        console.print("  [yellow]Invalid environment.[/]")
        return

    matched = _match_version(version)
    if not matched:
        console.print(f"  [yellow]Unknown version '{version}'. Use: {', '.join(_MODEL_VERSIONS)}[/]")
        return

    enabled = _enabled_versions(env_name)

    if load:
        # Enable the model
        if matched not in enabled:
            enabled.append(matched)
            _set_enabled_versions(env_name, enabled)
        console.print(f"  [green]✓ FIX {matched} enabled for {env_name}[/]")

        # If server is running, also load it live
        pid = _get_pid(env_name)
        if _is_running(pid):
            port = _get_port(env_name)
            console.print(f"  [cyan]Loading FIX {matched} on running server...[/]")
            result = _api_call(port, "/admin/models/load", method="POST",
                               body={"version": matched}, timeout=120)
            if result and result.get("loaded"):
                console.print(f"  [green]✓ {result.get('message', 'Done')}[/]")
            else:
                msg = result.get("message", "Failed") if result else "API call failed (timeout?)"
                console.print(f"  [red]✗ {msg} (will load on next start)[/]")
        else:
            console.print(f"  [dim]Server not running — model will load on next start.[/]")
    else:
        # Disable the model — guard: at least one must remain enabled
        if matched in enabled:
            if len(enabled) <= 1:
                console.print(
                    f"  [bold red]⚠ Cannot disable FIX {matched} — "
                    f"it is the only enabled model on {env_name}.[/]"
                )
                console.print(
                    "  [yellow]At least one model must remain enabled per environment.[/]"
                )
                return
            enabled.remove(matched)
            _set_enabled_versions(env_name, enabled)
        console.print(f"  [green]✓ FIX {matched} disabled for {env_name}[/]")

        # If server is running, also unload it live
        pid = _get_pid(env_name)
        if _is_running(pid):
            port = _get_port(env_name)
            console.print(f"  [cyan]Unloading FIX {matched} on running server...[/]")
            result = _api_call(port, "/admin/models/unload", method="POST", body={"version": matched})
            if result:
                console.print(f"  [green]✓ {result.get('message', 'Done')}[/]")
            else:
                console.print(f"  [red]✗ API call failed (will apply on next restart)[/]")


def _model_unload_all(env_token: str) -> None:
    """Disable all models on an environment (with confirmation).

    Persists the preference and also unloads live if server is running.
    """
    env_name = _resolve_env_name(env_token)
    if not env_name:
        console.print("  [yellow]Invalid environment.[/]")
        return

    console.print()
    console.print(f"  [bold red]⚠ This will disable ALL models on {env_name}.[/]")
    console.print("  [yellow]The environment will run in demo-only mode (no AI inference).[/]")
    console.print()
    try:
        proceed = Prompt.ask(
            "  Disable all models?",
            choices=["y", "n"],
            default="n",
        )
    except (KeyboardInterrupt, EOFError):
        return
    if proceed.lower() != "y":
        console.print("  [dim]Cancelled.[/]")
        return

    _set_enabled_versions(env_name, [])
    console.print(f"  [green]✓ All models disabled for {env_name}[/]")

    # If server is running, also unload live
    pid = _get_pid(env_name)
    if _is_running(pid):
        port = _get_port(env_name)
        console.print(f"  [cyan]Unloading all models on running server...[/]")
        result = _api_call(port, "/admin/models/unload", method="POST", body={"all": True})
        if result:
            console.print(f"  [green]✓ {result.get('message', 'Done')}[/]")
        else:
            console.print(f"  [red]✗ API call failed (will apply on next restart)[/]")


def _action_model_load(env_num: str, version: Optional[str] = None) -> None:
    """Enable model(s) on environment *env_num*.

    Works offline (persists preference) and online (also loads via API).
    If *version* is given enable just that version, otherwise enable all
    versions that have a checkpoint.
    """
    env_name = _resolve_env_name(env_num)
    if not env_name:
        console.print("  [yellow]Invalid environment.[/]")
        return

    if version:
        matched = _match_version(version)
        if not matched:
            console.print(f"  [yellow]Unknown version '{version}'. Use: {', '.join(_MODEL_VERSIONS)}[/]")
            return
        _model_load_unload(env_num, matched, load=True)
        return

    # No version → enable all versions that have a checkpoint
    ckpts = _local_checkpoint_status()
    to_enable = [v for v, info in ckpts.items() if info["has_checkpoint"]]

    if not to_enable:
        console.print(f"  [yellow]No model checkpoints found on disk.[/]")
        return

    enabled = _enabled_versions(env_name)
    newly_enabled = [v for v in to_enable if v not in enabled]

    if not newly_enabled:
        console.print(f"  [dim]All available models already enabled on {env_name}.[/]")
        return

    # Persist preferences
    updated = list(enabled) + newly_enabled
    _set_enabled_versions(env_name, updated)

    for ver in newly_enabled:
        console.print(f"  [green]✓ FIX {ver} enabled for {env_name}[/]")

    # If server running, also load live
    pid = _get_pid(env_name)
    if _is_running(pid):
        port = _get_port(env_name)
        for ver in newly_enabled:
            console.print(f"  [cyan]Loading FIX {ver} on running server...[/]")
            result = _api_call(port, "/admin/models/load", method="POST",
                               body={"version": ver}, timeout=120)
            if result and result.get("loaded"):
                console.print(f"    [green]✓ {result.get('message', 'Done')}[/]")
            else:
                msg = result.get("message", "Failed") if result else "API call failed (timeout?)"
                console.print(f"    [red]✗ {msg}[/]")
    else:
        console.print(f"  [dim]Server not running — models will load on next start.[/]")


def _action_model_unload(env_num: str, version: Optional[str] = None) -> None:
    """Disable model(s) on environment *env_num*.

    Works offline (persists preference) and online (also unloads via API).
    If *version* is given disable just that version (guarded: cannot
    disable the last enabled model). Otherwise disable all (with confirmation).
    """
    env_name = _resolve_env_name(env_num)
    if not env_name:
        console.print("  [yellow]Invalid environment.[/]")
        return

    if version:
        matched = _match_version(version)
        if not matched:
            console.print(f"  [yellow]Unknown version '{version}'. Use: {', '.join(_MODEL_VERSIONS)}[/]")
            return
        _model_load_unload(env_num, matched, load=False)
        return

    # No version → disable all (with confirmation)
    _model_unload_all(env_num)


# ── Bulk model actions (across ALL environments) ──────────────────

def _action_bulk_model_enable_all() -> None:
    """Enable all models (with checkpoints) on every environment."""
    ckpts = _local_checkpoint_status()
    available = [v for v, info in ckpts.items() if info["has_checkpoint"]]
    if not available:
        console.print("  [yellow]No model checkpoints found on disk.[/]")
        return

    console.print(f"  [magenta]Enabling {len(available)} model(s) on all {len(VALID_ENVS)} environments...[/]")
    for env_name in VALID_ENVS:
        enabled = _enabled_versions(env_name)
        newly = [v for v in available if v not in enabled]
        if newly:
            _set_enabled_versions(env_name, list(enabled) + newly)
            for v in newly:
                console.print(f"    [green]✓ FIX {v} enabled on {env_name}[/]")
        else:
            console.print(f"    [dim]{env_name} — already all enabled[/]")

        # If server running, load live
        pid = _get_pid(env_name)
        if _is_running(pid):
            port = _get_port(env_name)
            for v in newly:
                result = _api_call(port, "/admin/models/load", method="POST",
                                   body={"version": v}, timeout=120)
                if result and result.get("loaded"):
                    console.print(f"      [green]↳ loaded live on {env_name}[/]")


def _action_bulk_model_disable_all() -> None:
    """Disable all models on every environment (with confirmation)."""
    console.print()
    console.print("  [bold red]⚠ This will disable ALL models on ALL environments.[/]")
    console.print("  [yellow]All environments will run in demo-only mode.[/]")
    console.print()
    try:
        proceed = Prompt.ask("  Disable all models everywhere?", choices=["y", "n"], default="n")
    except (KeyboardInterrupt, EOFError):
        return
    if proceed.lower() != "y":
        console.print("  [dim]Cancelled.[/]")
        return

    for env_name in VALID_ENVS:
        _set_enabled_versions(env_name, [])
        console.print(f"  [green]✓ All models disabled on {env_name}[/]")

        pid = _get_pid(env_name)
        if _is_running(pid):
            port = _get_port(env_name)
            result = _api_call(port, "/admin/models/unload", method="POST", body={"all": True})
            if result:
                console.print(f"    [green]↳ unloaded live on {env_name}[/]")


def _action_bulk_model_version(version: str, *, load: bool) -> None:
    """Enable or disable a specific model version on ALL environments."""
    matched = _match_version(version)
    if not matched:
        console.print(f"  [yellow]Unknown version '{version}'. Use: {', '.join(_MODEL_VERSIONS)}[/]")
        return

    action_word = "Enabling" if load else "Disabling"
    console.print(f"  [magenta]{action_word} FIX {matched} on all {len(VALID_ENVS)} environments...[/]")

    for env_name in VALID_ENVS:
        enabled = _enabled_versions(env_name)

        if load:
            if matched not in enabled:
                enabled.append(matched)
                _set_enabled_versions(env_name, enabled)
            console.print(f"    [green]✓ FIX {matched} enabled on {env_name}[/]")

            pid = _get_pid(env_name)
            if _is_running(pid):
                port = _get_port(env_name)
                result = _api_call(port, "/admin/models/load", method="POST",
                                   body={"version": matched}, timeout=120)
                if result and result.get("loaded"):
                    console.print(f"      [green]↳ loaded live[/]")
        else:
            if matched in enabled:
                if len(enabled) <= 1:
                    console.print(
                        f"    [red]⚠ Skipping {env_name} — FIX {matched} is the only enabled model[/]"
                    )
                    continue
                enabled.remove(matched)
                _set_enabled_versions(env_name, enabled)
            console.print(f"    [green]✓ FIX {matched} disabled on {env_name}[/]")

            pid = _get_pid(env_name)
            if _is_running(pid):
                port = _get_port(env_name)
                result = _api_call(port, "/admin/models/unload", method="POST", body={"version": matched})
                if result:
                    console.print(f"      [green]↳ unloaded live[/]")


def _match_version(user_input: str) -> Optional[str]:
    """Case-insensitive match of user input to a known version key."""
    for ver in _MODEL_VERSIONS:
        if user_input.lower() == ver.lower():
            return ver
    return None


def _action_logs_for(env_name: str) -> None:
    """Tail the last N lines of a log file for *env_name*."""
    log_dir = PROJECT_ROOT / "logs" / env_name
    log_files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    if not log_files:
        console.print(f"  [yellow]No log files found in logs/{env_name}/[/]")
        return

    console.print()
    for idx, lf in enumerate(log_files, 1):
        size_kb = lf.stat().st_size / 1024
        console.print(f"  [bold]{idx}[/]) {lf.name}  [dim]({size_kb:.1f} KB)[/]")

    console.print(f"  [bold]{len(log_files) + 1}[/]) [dim]Back[/]")
    console.print()

    try:
        choice = Prompt.ask(
            "  Select log file",
            choices=[str(i) for i in range(1, len(log_files) + 2)],
            default=str(len(log_files) + 1),
        )
    except (KeyboardInterrupt, EOFError):
        return

    idx = int(choice) - 1
    if idx >= len(log_files):
        return

    selected = log_files[idx]

    try:
        lines = IntPrompt.ask("  Lines to show", default=30)
    except (KeyboardInterrupt, EOFError):
        lines = 30

    console.print()
    console.rule(f"[bold]{selected.name}[/] (last {lines} lines)")
    try:
        result = subprocess.run(
            ["tail", f"-{lines}", str(selected)],
            capture_output=True, text=True, timeout=5,
        )
        console.print(result.stdout if result.stdout else "[dim]  (empty)[/]")
    except Exception as e:
        console.print(f"  [red]Error reading log: {e}[/]")
    console.rule()


# ── Main Loop ─────────────────────────────────────────────────

def _banner() -> None:
    """Print the application banner."""
    console.clear()
    banner_text = Text()
    banner_text.append("  FixProtoGPT ", style="bold white on blue")
    banner_text.append("  Environment Control Panel  ", style="bold cyan")
    console.print()
    console.print(Panel(banner_text, box=box.DOUBLE, border_style="blue", expand=True))


def main() -> None:
    """Interactive control panel main loop."""
    while True:
        _banner()

        # Show model engine status first (models before environments)
        model_panel = _build_model_table()
        console.print(model_panel)
        console.print()

        # Show environment status
        table = _build_status_table()
        console.print(table)
        console.print()

        # Show menu
        menu = _build_menu()
        console.print(menu)
        console.print()

        # Get user input — accepts inline commands like 1s, 2x, 3r, 4l, 1ml, 1mx 4.4
        try:
            raw_cmd = console.input("  [bold]Command:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]Goodbye![/]\n")
            break

        if not raw_cmd:
            continue  # refresh

        # Split into base command + optional argument (for version specs)
        cmd_parts = raw_cmd.split(None, 1)
        cmd = cmd_parts[0].lower()
        cmd_arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else None

        # ── Global / bulk commands ────────────────────────────
        if cmd in ("q", "quit", "0"):
            console.print("\n  [dim]Goodbye![/]\n")
            break
        elif cmd == "sa":
            _action_start_all()
        elif cmd == "xa":
            _action_stop_all()
        elif cmd == "ra":
            _action_restart_all()
        elif cmd == "h":
            _action_health()
        elif cmd == "m":
            _action_models()
        elif cmd == "mla":
            _action_bulk_model_enable_all()
        elif cmd == "mxa":
            _action_bulk_model_disable_all()
        elif cmd == "ml" and cmd_arg:
            _action_bulk_model_version(cmd_arg, load=True)
        elif cmd == "mx" and cmd_arg:
            _action_bulk_model_version(cmd_arg, load=False)

        # ── Per-env model commands: <env#>ml / <env#>mx [version] ─
        elif cmd[0].isdigit() and len(cmd) >= 3 and cmd[1:3] == "ml":
            env_num = cmd[0]
            version_arg = cmd[3:].strip() or cmd_arg
            _action_model_load(env_num, version_arg)

        elif cmd[0].isdigit() and len(cmd) >= 3 and cmd[1:3] == "mx":
            env_num = cmd[0]
            version_arg = cmd[3:].strip() or cmd_arg
            _action_model_unload(env_num, version_arg)

        # ── Per-env commands: <env#><action> ──────────────────
        elif len(cmd) == 2 and cmd[0].isdigit():
            env_idx = int(cmd[0]) - 1
            action = cmd[1]
            if 0 <= env_idx < len(VALID_ENVS):
                env_name = VALID_ENVS[env_idx]
                if action == "s":
                    _action_start(env_name)
                elif action == "x":
                    _action_stop(env_name)
                elif action == "r":
                    _action_restart(env_name)
                elif action == "l":
                    _action_logs_for(env_name)
                else:
                    console.print(f"  [yellow]Unknown action '{action}'. Use s/x/r/l[/]")
            else:
                console.print(f"  [yellow]Invalid env #. Use 1-{len(VALID_ENVS)}[/]")

        else:
            console.print("  [yellow]Unknown command. See quick commands above.[/]")

        # Pause before refresh
        console.print()
        try:
            console.input("  Press [bold]Enter[/] to continue ")
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()
