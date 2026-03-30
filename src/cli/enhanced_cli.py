"""
Module: src.cli.enhanced_cli
=============================

Enhanced CLI for FixProtoGPT.

Modern slash-command interface with FIX version selection,
automatic parameter tuning, and Rich terminal rendering.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils import paths
from src.utils.paths import load_config
from src.core.version_registry import list_installed

console = Console()

# ── Load settings from config/model_config.yaml ──────────────────
_cfg = load_config()
_inference_cfg = _cfg.get("inference", {})
_data_cfg = _cfg.get("data", {})

DEFAULT_TEMPERATURE: float = _inference_cfg.get("temperature", 0.8)
DEFAULT_MAX_TOKENS: int = _inference_cfg.get("max_new_tokens", 256)

# fix_versions: {"4.2": "FIX 4.2", ...} from config
_KNOWN_VERSIONS: Dict[str, str] = _data_cfg.get("fix_versions", {})


def _discover_versions() -> List[str]:
    """Return FIX version keys that have checkpoint or tokenizer on disk.

    Uses the canonical ``version_registry.list_installed()`` so that
    CLI, API, and UI all see the same set of available versions.
    """
    installed = list_installed()
    if installed:
        return installed
    # Fallback: always offer the active version from config
    return [paths.active_version()]


class FixProtoGPTCLI:
    """Modern slash-command CLI for FixProtoGPT."""

    def __init__(self) -> None:
        """Initialise CLI with empty engine, tokenizer, and version."""
        self.engine = None
        self.tokenizer = None
        self.fix_version: str = paths.active_version()

    # ── Model loading ─────────────────────────────────────────────

    def _warn_model_unavailable(self) -> None:
        """Print a short model-unavailable notice with available alternatives."""
        available = [
            v for v in list_installed()
            if v.get("has_model") and v["version"] != self.fix_version
        ]
        console.print(
            f"[yellow]⚠ No model loaded for FIX {self.fix_version} — using demo mode.[/yellow]"
        )
        if available:
            names = ", ".join(v["label"] for v in available)
            console.print(
                f"[cyan]  Available models: {names}[/cyan]  "
                f"[dim](use [bold]/version[/bold] to switch)[/dim]"
            )

    def load_model(self, version: Optional[str] = None) -> bool:
        """Load the inference engine for a specific FIX version.

        Args:
            version: FIX version key (e.g. ``"4.4"``). Uses
                     ``self.fix_version`` when ``None``.

        Returns:
            ``True`` if the full model loaded; ``False`` for demo mode.
        """
        ver = version or self.fix_version
        model_path = paths.best_model(ver)
        tokenizer_path = paths.tokenizer_dir(ver)

        if not model_path.exists():
            console.print(f"[yellow]Model not found for FIX {ver}. Running in demo mode.[/yellow]")
            # Show available alternatives
            available = [
                v for v in list_installed()
                if v.get("has_model") and v["version"] != ver
            ]
            if available:
                names = ", ".join(v["label"] for v in available)
                console.print(
                    f"[cyan]Available models: {names}[/cyan]\n"
                    f"[dim]Use [bold]/version[/bold] to switch.[/dim]"
                )
            try:
                from src.core.tokenizer import FixProtocolTokenizer
                self.tokenizer = FixProtocolTokenizer()
                self.engine = None
                return False
            except Exception:
                return False

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Loading FIX {ver} model...", total=None)

                from src.inference.generate import FixProtoGPTInference
                self.engine = FixProtoGPTInference(
                    str(model_path),
                    str(tokenizer_path),
                    fix_version=ver,
                )
                progress.update(task, completed=True)

            console.print(f"[green]Model loaded successfully (FIX {ver})[/green]")
            self.fix_version = ver
            return True

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            self.engine = None
            return False

    # ── Demo fallback helpers ─────────────────────────────────────

    def _demo_generate(self, prompt: str) -> str:
        """Build a demo FIX message from NL — same logic as the API.

        Parses *prompt* for symbol, side, quantity, order type, and price,
        then assembles a realistic FIX message using the current version's
        BeginString and ApplVerID.
        """
        import random as _rand
        from src.core.version_registry import get_version_info
        from src.data.symbol_resolver import resolve_symbol

        ver_info = get_version_info(self.fix_version)
        begin = ver_info.begin_string if ver_info else "FIXT.1.1"
        appl = f"|1128={ver_info.appl_ver_id}" if ver_info and ver_info.appl_ver_id else ""

        low = prompt.lower()

        # ── Symbol
        symbol = "AAPL"
        resolved = resolve_symbol(prompt)
        if resolved:
            symbol = resolved
        else:
            import re
            sym_match = re.search(r'\b([A-Z]{1,5})\b', prompt)
            if sym_match:
                symbol = sym_match.group(1)

        # ── Side
        if "sell" in low or "short" in low:
            side_code = "2"
        else:
            side_code = "1"

        # ── Qty
        qty = "100"
        qty_match = __import__("re").search(r'(\d+)\s*(?:shares?|lots?|units?)', low)
        if qty_match:
            qty = qty_match.group(1)

        # ── Order type & price
        if "limit" in low:
            ord_type = "2"
        elif "stop" in low:
            ord_type = "3"
        else:
            ord_type = "1"

        price_tag = ""
        price_match = __import__("re").search(r'(?:at|price|@)\s*\$?([\d.]+)', low)
        if price_match and ord_type != "1":
            price_tag = f"|44={price_match.group(1)}"

        # ── Message type
        if "cancel" in low:
            msg_type = "F"
        elif "market data" in low or "mdreq" in low:
            msg_type = "V"
        elif "amend" in low or "replace" in low:
            msg_type = "G"
        else:
            msg_type = "D"

        oid = f"ORD{_rand.randint(10000, 99999)}"

        if msg_type == "F":
            return f"8={begin}|35=F|49=SENDER|56=TARGET|11=CANCEL001|41={oid}|55={symbol}|54={side_code}{appl}|"
        if msg_type == "V":
            return f"8={begin}|35=V|49=SENDER|56=TARGET|262=MDREQ001|263=1|264=0|55={symbol}{appl}|"
        if msg_type == "G":
            return f"8={begin}|35=G|49=SENDER|56=TARGET|11={oid}|41=ORDER123|55={symbol}|54={side_code}|38={qty}|40={ord_type}{price_tag}{appl}|"

        return f"8={begin}|9=178|35={msg_type}|49=SENDER|56=TARGET|11={oid}|55={symbol}|54={side_code}|38={qty}|40={ord_type}{price_tag}{appl}|"

    # ── Banner & version picker ───────────────────────────────────

    def show_banner(self) -> None:
        """Display the welcome banner."""
        banner = Panel(
            "[bold]FixProtoGPT v1.0.0[/bold]\n"
            "AI-Powered FIX Protocol Assistant\n\n"
            '[dim]Type [bold cyan]/help[/bold cyan] for available commands, '
            'or just type a prompt to generate.[/dim]',
            border_style="cyan",
            expand=False,
            padding=(1, 4),
        )
        console.print(banner)

    def select_version(self) -> str:
        """Prompt the user to pick a FIX version at startup.

        Returns:
            The chosen version key (e.g. ``"5.0SP2"``).
        """
        available = _discover_versions()

        console.print("\n[bold cyan]Select FIX Version[/bold cyan]")
        t = Table(box=box.ROUNDED, show_header=False)
        t.add_column("#", style="cyan", width=4)
        t.add_column("Version", style="white")
        t.add_column("Status", style="dim")

        for idx, ver in enumerate(available, 1):
            label = _KNOWN_VERSIONS.get(ver, f"FIX {ver}")
            has_model = paths.best_model(ver).exists()
            status = "[green]model ready[/green]" if has_model else "[yellow]demo only[/yellow]"
            t.add_row(str(idx), label, status)

        console.print(t)

        while True:
            choice = Prompt.ask(
                "[bold]Select version[/bold]",
                default="1",
            ).strip()

            if choice.isdigit() and 1 <= int(choice) <= len(available):
                return available[int(choice) - 1]

            # Allow typing the version string directly
            for ver in available:
                if choice.lower() == ver.lower():
                    return ver

            console.print("[red]Invalid selection. Try again.[/red]")

    # ── Help ──────────────────────────────────────────────────────

    @staticmethod
    def show_help() -> None:
        """Print the slash-command reference card."""
        t = Table(
            title="Commands",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        t.add_column("Command", style="cyan", min_width=30)
        t.add_column("Description", style="white")

        t.add_row("/generate <prompt>", "Generate a FIX message from natural language")
        t.add_row("<prompt>  [dim](no slash)[/dim]", "Same as /generate — just type your prompt")
        t.add_row("/validate <fix_msg>", "Validate a FIX message")
        t.add_row("/explain <fix_msg>", "Explain a FIX message field-by-field")
        t.add_row("/complete <partial_msg>", "Complete a partial FIX message")
        t.add_row("/examples", "Show example prompts and messages")
        t.add_row("/version", "Switch FIX version")
        t.add_row("/models", "Show model availability for all versions")
        t.add_row("/status", "Show system and model information")
        t.add_row("/help", "Show this help")
        t.add_row("/exit  or  /quit", "Exit FixProtoGPT")

        console.print(t)

    # ── User commands ─────────────────────────────────────────────

    def generate_fix(self, prompt: str) -> None:
        """Generate a FIX message from *prompt*."""
        if not prompt:
            console.print("[yellow]Usage: /generate <prompt>[/yellow]")
            console.print("[dim]  Example: /generate Buy 100 shares of AAPL at market price[/dim]")
            return

        with console.status("[bold green]Generating...", spinner="dots"):
            if self.engine:
                try:
                    result = self.engine.natural_language_to_fix(prompt)
                except (AttributeError, NotImplementedError):
                    outputs = self.engine.generate(
                        prompt,
                        temperature=DEFAULT_TEMPERATURE,
                        max_new_tokens=DEFAULT_MAX_TOKENS,
                        num_samples=1,
                    )
                    result = outputs[0]
            else:
                # Demo fallback — same NL-aware logic as the API
                result = self._demo_generate(prompt)

        if not self.engine:
            self._warn_model_unavailable()

        console.print(Panel(
            result,
            title=f"[green]Generated FIX Message (FIX {self.fix_version})[/green]",
            border_style="green",
            expand=False,
        ))

    def validate_fix(self, fix_msg: str) -> None:
        """Validate a FIX message."""
        if not fix_msg:
            console.print("[yellow]Usage: /validate <fix_message>[/yellow]")
            return

        with console.status("[bold green]Validating...", spinner="dots"):
            if self.engine:
                result = self.engine.validate_fix_message(fix_msg)
            else:
                from src.core.tokenizer import FixProtocolTokenizer
                if not self.tokenizer:
                    self.tokenizer = FixProtocolTokenizer()

                fields = self.tokenizer.parse_fix_message(fix_msg)
                required = {'8', '9', '35', '49', '56', '10'}
                present = {f['tag'] for f in fields}
                missing = required - present

                result = {
                    'valid': len(missing) == 0,
                    'fields': fields,
                    'missing_required_fields': list(missing),
                    'num_fields': len(fields),
                }

        if not self.engine:
            self._warn_model_unavailable()

        if result['valid']:
            console.print(Panel(
                f"[green]Valid FIX Message[/green]\n\nFields: {result['num_fields']}",
                border_style="green",
            ))
        else:
            console.print(Panel(
                f"[red]Invalid FIX Message[/red]\n\n"
                f"Missing required: {', '.join(result['missing_required_fields'])}",
                border_style="red",
            ))

        if result['fields']:
            table = Table(title="Message Fields", box=box.ROUNDED)
            table.add_column("Tag", style="cyan")
            table.add_column("Name", style="yellow")
            table.add_column("Value", style="green")
            for field in result['fields'][:20]:
                table.add_row(field['tag'], field['name'], field['value'])
            console.print(table)

    def explain_fix(self, fix_msg: str) -> None:
        """Explain a FIX message field-by-field."""
        if not fix_msg:
            console.print("[yellow]Usage: /explain <fix_message>[/yellow]")
            return

        with console.status("[bold green]Analyzing...", spinner="dots"):
            fallback = "Unable to explain this message."
            if self.engine:
                result = self.engine.explain_fix_message(fix_msg)
            else:
                result = None
                # Demo fallback — use scraper KB enrichment (same as API)
                try:
                    from src.data.scraper import FIXProtocolScraper
                    from src.core.tokenizer import FixProtocolTokenizer
                    from src.inference.explainer import build_explain_summary
                    from src.utils.fix_enrichment import enrich_parsed_fields, extract_msg_type_info

                    if not self.tokenizer:
                        self.tokenizer = FixProtocolTokenizer()
                    parsed = self.tokenizer.parse_fix_message(fix_msg)
                    enriched = enrich_parsed_fields(
                        parsed,
                        FIXProtocolScraper.FIELDS,
                        FIXProtocolScraper.ENUMERATIONS,
                        full=True,
                    )
                    msg_info = extract_msg_type_info(enriched, FIXProtocolScraper.MESSAGE_TYPES)
                    summary_text = build_explain_summary(enriched, msg_info)
                    result = {
                        "summary": summary_text,
                        "model_insight": {
                            "source": "unavailable",
                            "nl_interpretation": "",
                            "msg_type_knowledge": "",
                        },
                        "fields": enriched,
                        "message_type": msg_info,
                    }
                except Exception:
                    # Ultimate fallback — plain text
                    if not self.tokenizer:
                        from src.core.tokenizer import FixProtocolTokenizer
                        self.tokenizer = FixProtocolTokenizer()
                    fields = self.tokenizer.parse_fix_message(fix_msg)
                    fallback = "FIX Message Breakdown:\n\n"
                    for field in fields[:10]:
                        fallback += f"Tag {field['tag']} ({field['name']}): {field['value']}\n"

        if not self.engine:
            self._warn_model_unavailable()

        if result and isinstance(result, dict):
            # Summary panel
            summary = str(result.get("summary", ""))
            if summary:
                console.print(Panel(
                    summary,
                    title="[cyan]Message Explanation[/cyan]",
                    border_style="cyan",
                    expand=False,
                ))

            # Model insight
            insight = result.get("model_insight", {})
            if isinstance(insight, dict):
                nl = insight.get("nl_interpretation", "")
                if nl:
                    console.print(f"  [bold]Model Interpretation:[/bold] {nl}")

            # Fields table
            fields = result.get("fields", [])
            if fields:
                table = Table(
                    title="Field Breakdown",
                    show_lines=True,
                    expand=False,
                )
                table.add_column("Tag", style="bold yellow", width=6)
                table.add_column("Name", style="cyan", min_width=14)
                table.add_column("Value", style="green", min_width=10)
                table.add_column("Meaning", style="white", min_width=20, max_width=60)
                for f in fields:
                    tag = str(f.get("tag", ""))
                    name = str(f.get("name", ""))
                    value = str(f.get("value", ""))
                    meaning = str(f.get("value_meaning") or f.get("explanation", ""))
                    table.add_row(tag, name, value, meaning)
                console.print(table)
        else:
            console.print(Panel(
                fallback,
                title="[cyan]Message Explanation[/cyan]",
                border_style="cyan",
                expand=False,
            ))

    def complete_fix(self, partial: str) -> None:
        """Complete a partial FIX message."""
        if not partial:
            console.print("[yellow]Usage: /complete <partial_fix_message>[/yellow]")
            return

        with console.status("[bold green]Completing...", spinner="dots"):
            if self.engine:
                completed = self.engine.complete_fix_message(partial)
            else:
                completed = partial + "54=1|38=100|40=2|44=150.00|59=0|10=123|"

        if not self.engine:
            self._warn_model_unavailable()

        console.print(Panel(
            completed,
            title="[green]Completed Message[/green]",
            border_style="green",
            expand=False,
        ))

    @staticmethod
    def show_examples() -> None:
        """Show example prompts and FIX messages."""
        console.print()
        examples = [
            ("Natural Language Prompts", [
                "Buy 100 shares of AAPL at market price",
                "Sell 50 shares of GOOGL at limit price $150",
                "Create a market data request for MSFT",
                "Cancel order ORDER123",
            ]),
            ("FIX Messages (for /validate, /explain, /complete)", [
                "8=FIXT.1.1|9=178|35=D|55=AAPL|54=1|38=100|40=2|44=150.50|1128=9|",
                "8=FIXT.1.1|35=8|55=GOOGL|54=2|38=50|150=2|39=2|1128=9|",
                "8=FIXT.1.1|35=V|262=MDREQ001|263=1|55=MSFT|1128=9|",
            ]),
        ]
        for category, items in examples:
            console.print(f"[bold yellow]{category}:[/bold yellow]")
            for i, item in enumerate(items, 1):
                console.print(f"  {i}. [dim]{item}[/dim]")
            console.print()

    def show_status(self) -> None:
        """Show system and model information."""
        model_exists = paths.best_model(self.fix_version).exists()
        tokenizer_exists = paths.tokenizer_dir(self.fix_version).exists()

        t = Table(box=box.ROUNDED, show_header=False, title="System Status")
        t.add_column("Property", style="cyan")
        t.add_column("Value", style="yellow")

        t.add_row("App Version", "1.0.0")
        t.add_row("FIX Version", self.fix_version)
        t.add_row("Model", "Loaded" if self.engine else ("Available" if model_exists else "Not Found"))
        t.add_row("Tokenizer", "Available" if tokenizer_exists else "Not Found")
        t.add_row("Mode", "Production" if self.engine else "Demo")
        t.add_row("Temperature", str(DEFAULT_TEMPERATURE))
        t.add_row("Max Tokens", str(DEFAULT_MAX_TOKENS))

        console.print(t)

    def switch_version(self) -> None:
        """Interactive version switcher."""
        new_ver = self.select_version()
        if new_ver == self.fix_version and self.engine is not None:
            console.print(f"[dim]Already on FIX {new_ver}.[/dim]")
            return

        self.engine = None  # Force reload
        self.fix_version = new_ver
        self.load_model(new_ver)

    def show_models(self) -> None:
        """Display model availability for every known FIX version."""
        versions = list_installed()
        t = Table(
            title="Model Availability",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        t.add_column("Version", style="white")
        t.add_column("Model", style="white")
        t.add_column("Data", style="white")
        t.add_column("", style="dim")

        for v in versions:
            label = v.get("label", v["version"])
            has_model = v.get("has_model", False)
            has_data = v.get("has_data", False)
            model_status = "[green]ready[/green]" if has_model else "[red]not trained[/red]"
            data_status = "[green]yes[/green]" if has_data else "[yellow]no[/yellow]"
            marker = "← active" if v["version"] == self.fix_version else ""
            t.add_row(label, model_status, data_status, marker)

        console.print(t)

    # ══════════════════════════════════════════════════════════════
    #  Main Loop — slash-command REPL
    # ══════════════════════════════════════════════════════════════

    def _parse_input(self, raw: str):
        """Parse user input into (command, argument).

        Returns:
            A ``(command, argument)`` tuple.  *command* is lower-cased
            without the leading ``/``.  For bare prompts (no slash),
            command is ``"generate"`` and argument is the full input.
        """
        text = raw.strip()
        if not text:
            return ("", "")

        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd = parts[0][1:].lower()  # strip leading /
            arg = parts[1] if len(parts) > 1 else ""
            return (cmd, arg)

        # No slash → default to generate
        return ("generate", text)

    def run(self) -> None:
        """Run the interactive slash-command REPL."""
        self.show_banner()

        # ── Version selection ─────────────────────────────────────
        self.fix_version = self.select_version()
        self.load_model(self.fix_version)

        mode_label = "[green]production[/green]" if self.engine else "[yellow]demo[/yellow]"
        console.print(
            f"\n[bold]FIX {self.fix_version}[/bold] | {mode_label} | "
            f"Type [bold cyan]/help[/bold cyan] for commands\n"
        )

        # ── REPL ──────────────────────────────────────────────────
        while True:
            try:
                raw = console.input(
                    f"[bold cyan]fixprotogpt[/bold cyan] [dim](FIX {self.fix_version})[/dim] > "
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[cyan]Goodbye![/cyan]\n")
                break

            cmd, arg = self._parse_input(raw)

            if not cmd:
                continue

            try:
                if cmd == "generate":
                    self.generate_fix(arg)
                elif cmd == "validate":
                    self.validate_fix(arg)
                elif cmd == "explain":
                    self.explain_fix(arg)
                elif cmd == "complete":
                    self.complete_fix(arg)
                elif cmd == "examples":
                    self.show_examples()
                elif cmd == "version":
                    self.switch_version()
                    mode_label = "[green]production[/green]" if self.engine else "[yellow]demo[/yellow]"
                    console.print(
                        f"\n[bold]FIX {self.fix_version}[/bold] | {mode_label}\n"
                    )
                elif cmd == "models":
                    self.show_models()
                elif cmd == "status":
                    self.show_status()
                elif cmd == "help":
                    self.show_help()
                elif cmd in ("exit", "quit"):
                    console.print("[cyan]Goodbye![/cyan]\n")
                    break
                else:
                    console.print(
                        f"[red]Unknown command:[/red] /{cmd}\n"
                        f"[dim]Type /help for available commands.[/dim]"
                    )
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    cli = FixProtoGPTCLI()
    cli.run()


if __name__ == "__main__":
    main()
