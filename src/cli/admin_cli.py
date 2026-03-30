"""
Module: src.cli.admin_cli
=========================

Admin CLI for FixProtoGPT.

Slash-command REPL for spec ingestion, client overlay management,
and training operations.  Launch standalone::

    python -m src.cli.admin_cli

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
from typing import List

# Add project root to path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.data.ingest.normalizer import (
    ingest_file as _ingest_file,
    ingest_directory as _ingest_directory,
    load_canonical,
    specs_to_training_lines,
    canonical_json_path,
)
from src.data.ingest.client_overlay import (
    list_clients,
    create_client,
    delete_client,
    save_uploaded_file,
    ingest_client_specs,
    build_client_training_data,
    get_client_stats,
    load_client_canonical,
)

console = Console()


class AdminCLI:
    """Slash-command admin REPL for FixProtoGPT."""

    # ── Banner & help ─────────────────────────────────────────────

    @staticmethod
    def show_banner() -> None:
        """Display the admin welcome banner."""
        banner = Panel(
            "[bold]FixProtoGPT Admin CLI[/bold]\n"
            "Spec ingestion, client management, and training\n\n"
            '[dim]Type [bold cyan]/help[/bold cyan] for available commands.[/dim]',
            border_style="cyan",
            expand=False,
            padding=(1, 4),
        )
        console.print(banner)

    @staticmethod
    def show_help() -> None:
        """Print the slash-command reference card."""
        t = Table(
            title="Admin Commands",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        t.add_column("Command", style="cyan", min_width=35)
        t.add_column("Description", style="white")

        # Specs
        t.add_row("[bold yellow]Spec Management[/bold yellow]", "")
        t.add_row("/ingest-file <path>", "Ingest a single spec file")
        t.add_row("/ingest-dir <path>", "Ingest all spec files in a directory")
        t.add_row("/specs", "View base spec stats")
        t.add_row("", "")

        # Clients
        t.add_row("[bold yellow]Client Management[/bold yellow]", "")
        t.add_row("/clients", "List all clients")
        t.add_row("/create-client <id>", "Create a new client overlay")
        t.add_row("/delete-client <id>", "Delete a client and all its data")
        t.add_row("/upload <client_id>", "Upload spec files to a client")
        t.add_row("/ingest-client <client_id>", "Parse uploaded specs into canonical format")
        t.add_row("/build-data <client_id>", "Build client training data")
        t.add_row("/client-stats <client_id>", "View detailed client stats")
        t.add_row("", "")

        # Training
        t.add_row("[bold yellow]Training[/bold yellow]", "")
        t.add_row("/finetune", "Fine-tune base model from feedback")
        t.add_row("/finetune-client <client_id>", "Fine-tune a client model")
        t.add_row("/training-status", "Show training status and checkpoints")
        t.add_row("", "")

        # General
        t.add_row("/help", "Show this help")
        t.add_row("/exit  or  /quit", "Exit admin CLI")

        console.print(t)

    # ── Input parsing ─────────────────────────────────────────────

    @staticmethod
    def _parse_input(raw: str):
        """Parse user input into ``(command, argument)``."""
        text = raw.strip()
        if not text:
            return ("", "")

        if text.startswith("/"):
            parts = text.split(None, 1)
            cmd = parts[0][1:].lower()
            arg = parts[1] if len(parts) > 1 else ""
            return (cmd, arg)

        return ("unknown", text)

    # ══════════════════════════════════════════════════════════════
    #  Spec Management
    # ══════════════════════════════════════════════════════════════

    def ingest_spec_file(self, file_path: str) -> None:
        """Ingest a single spec file into the base canonical store."""
        if not file_path:
            console.print("[yellow]Usage: /ingest-file <path>[/yellow]")
            console.print("[dim]  Supported: .pdf  .docx  .xml  .xsd  .csv  .tsv  .xlsx[/dim]")
            return

        fp = Path(file_path).expanduser().resolve()

        if not fp.is_file():
            console.print(f"[red]File not found: {fp}[/red]")
            return

        with console.status(f"[bold green]Ingesting {fp.name}...", spinner="dots"):
            specs = _ingest_file(fp)

        console.print(Panel(
            f"[green]Ingested {len(specs)} records from {fp.name}[/green]",
            border_style="green",
        ))

    def ingest_spec_directory(self, dir_path: str) -> None:
        """Ingest all supported files in a directory."""
        if not dir_path:
            console.print("[yellow]Usage: /ingest-dir <path>[/yellow]")
            return

        dp = Path(dir_path).expanduser().resolve()

        if not dp.is_dir():
            console.print(f"[red]Not a directory: {dp}[/red]")
            return

        with console.status(f"[bold green]Ingesting {dp}...", spinner="dots"):
            specs = _ingest_directory(dp)

        console.print(Panel(
            f"[green]Ingested {len(specs)} records from {dp}[/green]",
            border_style="green",
        ))

    def view_base_specs(self) -> None:
        """Display statistics about the base canonical spec store."""
        try:
            specs = load_canonical()
        except Exception:
            console.print("[yellow]No base specs ingested yet.[/yellow]")
            return

        if not specs:
            console.print("[yellow]No base specs ingested yet.[/yellow]")
            return

        from collections import Counter
        kind_counts = Counter(s.kind.value for s in specs)

        t = Table(title="Base Canonical Specs", box=box.ROUNDED)
        t.add_column("Kind", style="cyan")
        t.add_column("Count", style="yellow", justify="right")
        for kind, count in sorted(kind_counts.items()):
            t.add_row(kind, str(count))
        t.add_row("[bold]Total[/bold]", f"[bold]{len(specs)}[/bold]")
        console.print(t)

        cpath = canonical_json_path()
        console.print(f"\n[dim]Stored at: {cpath}[/dim]")

    # ══════════════════════════════════════════════════════════════
    #  Client Management
    # ══════════════════════════════════════════════════════════════

    def list_all_clients(self) -> None:
        """Show all registered clients."""
        clients = list_clients()
        if not clients:
            console.print("[yellow]No clients registered yet.[/yellow]")
            return

        t = Table(title="Registered Clients", box=box.ROUNDED)
        t.add_column("#", style="dim", width=4)
        t.add_column("Client ID", style="cyan")
        t.add_column("Uploads", style="yellow", justify="right")
        t.add_column("Specs", style="yellow", justify="right")
        t.add_column("Model", style="green")

        for idx, cid in enumerate(clients, 1):
            try:
                stats = get_client_stats(cid)
                model_status = "yes" if stats.get("has_checkpoint") else "no"
                t.add_row(
                    str(idx), cid,
                    str(stats.get("upload_count", 0)),
                    str(stats.get("spec_count", 0)),
                    model_status,
                )
            except Exception:
                t.add_row(str(idx), cid, "?", "?", "?")

        console.print(t)

    def create_new_client(self, client_id: str) -> None:
        """Create a new client overlay."""
        if not client_id:
            console.print("[yellow]Usage: /create-client <client_id>[/yellow]")
            return

        if not client_id.replace("-", "").replace("_", "").isalnum():
            console.print("[red]Invalid client ID. Use alphanumeric, dashes, underscores.[/red]")
            return

        existing = list_clients()
        if client_id in existing:
            console.print(f"[yellow]Client '{client_id}' already exists.[/yellow]")
            return

        overlay_dir = create_client(client_id)
        console.print(Panel(
            f"[green]Created client '{client_id}'[/green]\n\n"
            f"Overlay dir: {overlay_dir}",
            border_style="green",
        ))

    def delete_existing_client(self, client_id: str) -> None:
        """Delete a client and all its data."""
        if not client_id:
            console.print("[yellow]Usage: /delete-client <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        if not Confirm.ask(f"[red]Delete client '{client_id}' and ALL its data?[/red]"):
            console.print("[dim]Cancelled.[/dim]")
            return

        deleted = delete_client(client_id)
        if deleted:
            console.print(f"[green]Deleted client '{client_id}'.[/green]")
        else:
            console.print(f"[yellow]Nothing to delete for '{client_id}'.[/yellow]")

    def upload_client_specs(self, client_id: str) -> None:
        """Upload spec files from local disk to a client's overlay."""
        if not client_id:
            console.print("[yellow]Usage: /upload <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        console.print("[dim]Enter file paths (comma-separated or one per prompt). Empty to finish.[/dim]")
        uploaded: List[str] = []

        while True:
            raw = Prompt.ask("File path(s)", default="")
            if not raw:
                break

            for part in raw.split(","):
                fp = Path(part.strip()).expanduser().resolve()
                if not fp.is_file():
                    console.print(f"[red]  Not found: {fp}[/red]")
                    continue

                data = fp.read_bytes()
                save_uploaded_file(client_id, fp.name, data)
                uploaded.append(fp.name)
                console.print(f"[green]  {fp.name}[/green] ({len(data):,} bytes)")

        if uploaded:
            console.print(Panel(
                f"[green]Uploaded {len(uploaded)} file(s) for client '{client_id}'[/green]",
                border_style="green",
            ))
        else:
            console.print("[dim]No files uploaded.[/dim]")

    def ingest_client(self, client_id: str) -> None:
        """Parse all uploaded specs for a client into canonical format."""
        if not client_id:
            console.print("[yellow]Usage: /ingest-client <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        with console.status(f"[bold green]Ingesting specs for {client_id}...", spinner="dots"):
            specs = ingest_client_specs(client_id)

        if specs:
            console.print(Panel(
                f"[green]Ingested {len(specs)} records for '{client_id}'[/green]",
                border_style="green",
            ))
        else:
            console.print("[yellow]No specs parsed. Upload files first.[/yellow]")

    def build_client_training_data(self, client_id: str) -> None:
        """Merge base + overlay and write client training data."""
        if not client_id:
            console.print("[yellow]Usage: /build-data <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        with console.status(f"[bold green]Building training data for {client_id}...", spinner="dots"):
            out = build_client_training_data(client_id)

        lines = sum(1 for _ in open(out))
        console.print(Panel(
            f"[green]Training data ready[/green]\n\n"
            f"Lines: {lines:,}\nPath : {out}",
            border_style="green",
        ))

    def view_client_stats(self, client_id: str) -> None:
        """Display detailed stats for a client."""
        if not client_id:
            console.print("[yellow]Usage: /client-stats <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        stats = get_client_stats(client_id)

        t = Table(title=f"Client: {client_id}", box=box.ROUNDED, show_header=False)
        t.add_column("Property", style="cyan")
        t.add_column("Value", style="yellow")

        t.add_row("Overlay Dir", stats.get("overlay_dir", "\u2014"))
        t.add_row("Uploaded Files", str(stats.get("upload_count", 0)))
        if stats.get("upload_files"):
            for fname in stats["upload_files"]:
                t.add_row("", f"  {fname}")
        t.add_row("Canonical Specs", str(stats.get("spec_count", 0)))
        t.add_row("Training Lines", f"{stats.get('training_lines', 0):,}")
        t.add_row("Has Checkpoint", "yes" if stats.get("has_checkpoint") else "no")

        console.print(t)

    # ══════════════════════════════════════════════════════════════
    #  Training
    # ══════════════════════════════════════════════════════════════

    def finetune_base(self) -> None:
        """Trigger a fine-tuning run on the base model from feedback."""
        from src.training.finetune import FineTuner, FinetuneConfig

        max_steps = int(Prompt.ask("Max steps", default="500"))
        lr = float(Prompt.ask("Learning rate", default="1e-4"))
        min_pairs = int(Prompt.ask("Min new pairs", default="5"))

        config = FinetuneConfig(
            max_steps=max_steps,
            learning_rate=lr,
            min_new_pairs=min_pairs,
        )

        ft = FineTuner(config=config)

        pre = ft.preflight()
        if not pre["ready"]:
            console.print("[yellow]Preflight check failed:[/yellow]")
            for reason in pre["reasons"]:
                console.print(f"  [yellow]- {reason}[/yellow]")
            if not Confirm.ask("Proceed anyway?"):
                return

        console.print(f"\n[dim]New training pairs: {pre['new_pairs']}[/dim]")
        if not Confirm.ask("Start fine-tuning?"):
            return

        with console.status("[bold green]Fine-tuning in progress...", spinner="dots"):
            result = ft.run()

        self._show_finetune_result(result)

    def finetune_client(self, client_id: str) -> None:
        """Trigger a client-specific fine-tuning run."""
        if not client_id:
            console.print("[yellow]Usage: /finetune-client <client_id>[/yellow]")
            return

        clients = list_clients()
        if client_id not in clients:
            console.print(f"[red]Client '{client_id}' not found.[/red]")
            return

        from src.training.finetune import FineTuner, FinetuneConfig

        max_steps = int(Prompt.ask("Max steps", default="500"))
        lr = float(Prompt.ask("Learning rate", default="1e-4"))

        config = FinetuneConfig(
            max_steps=max_steps,
            learning_rate=lr,
            client_id=client_id,
        )

        ft = FineTuner(config=config)

        stats = get_client_stats(client_id)
        if stats.get("training_lines", 0) == 0:
            console.print("[yellow]No training data. Run /build-data first.[/yellow]")
            if not Confirm.ask("Build training data now?"):
                return
            with console.status("[bold green]Building training data...", spinner="dots"):
                build_client_training_data(client_id)

        if not Confirm.ask(f"Start fine-tuning for client '{client_id}'?"):
            return

        with console.status(f"[bold green]Fine-tuning {client_id}...", spinner="dots"):
            result = ft.run_client(client_id)

        self._show_finetune_result(result, client_id)

    def _show_finetune_result(self, result, client_id: str = None) -> None:
        """Pretty-print a FinetuneResult."""
        label = f"client '{client_id}'" if client_id else "base model"

        if result.success:
            t = Table(
                title=f"Fine-tune Complete \u2014 {label}",
                box=box.ROUNDED, show_header=False,
            )
            t.add_column("Metric", style="cyan")
            t.add_column("Value", style="green")
            t.add_row("Steps", str(result.steps_trained))
            t.add_row("Final Loss", f"{result.final_loss:.4f}")
            t.add_row("Best Val Loss", f"{result.best_val_loss:.4f}")
            t.add_row("Checkpoint", result.checkpoint_path)
            t.add_row("Duration", f"{result.duration_secs:.1f}s")
            console.print(t)
        else:
            console.print(Panel(
                f"[red]Fine-tuning failed for {label}[/red]\n\n{result.error}",
                border_style="red",
            ))

    def training_status(self) -> None:
        """Show training status from the log file."""
        try:
            from src.training.training_status import parse_log, list_checkpoints
        except ImportError:
            console.print("[red]Training status module not available.[/red]")
            return

        log_data = parse_log()
        if log_data is None:
            console.print("[yellow]No training log found.[/yellow]")
        else:
            t = Table(title="Latest Training Run", box=box.ROUNDED, show_header=False)
            t.add_column("Property", style="cyan")
            t.add_column("Value", style="yellow")

            if log_data.get("device"):
                t.add_row("Device", log_data["device"])
            if log_data.get("model_params"):
                t.add_row("Parameters", log_data["model_params"])

            steps = log_data.get("steps", [])
            if steps:
                latest = steps[-1]
                t.add_row("Latest Step", str(latest.get("step", "?")))
                t.add_row("Train Loss", str(latest.get("loss", "?")))

            val_evals = log_data.get("val_evals", [])
            if val_evals:
                latest_val = val_evals[-1]
                t.add_row("Val Loss", str(latest_val.get("val_loss", "?")))

            console.print(t)

        try:
            ckpts = list_checkpoints()
            if ckpts:
                console.print("\n[bold]Checkpoints:[/bold]")
                for ck in ckpts[-5:]:
                    console.print(f"  {ck}")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  Main REPL
    # ══════════════════════════════════════════════════════════════

    def run(self) -> None:
        """Run the slash-command admin REPL."""
        self.show_banner()

        while True:
            try:
                raw = console.input(
                    "[bold magenta]admin[/bold magenta] > "
                )
            except (EOFError, KeyboardInterrupt):
                console.print("\n[cyan]Exiting admin mode.[/cyan]\n")
                break

            cmd, arg = self._parse_input(raw)

            if not cmd:
                continue

            try:
                # Spec management
                if cmd == "ingest-file":
                    self.ingest_spec_file(arg)
                elif cmd == "ingest-dir":
                    self.ingest_spec_directory(arg)
                elif cmd == "specs":
                    self.view_base_specs()

                # Client management
                elif cmd == "clients":
                    self.list_all_clients()
                elif cmd == "create-client":
                    self.create_new_client(arg)
                elif cmd == "delete-client":
                    self.delete_existing_client(arg)
                elif cmd == "upload":
                    self.upload_client_specs(arg)
                elif cmd == "ingest-client":
                    self.ingest_client(arg)
                elif cmd == "build-data":
                    self.build_client_training_data(arg)
                elif cmd == "client-stats":
                    self.view_client_stats(arg)

                # Training
                elif cmd == "finetune":
                    self.finetune_base()
                elif cmd == "finetune-client":
                    self.finetune_client(arg)
                elif cmd == "training-status":
                    self.training_status()

                # General
                elif cmd == "help":
                    self.show_help()
                elif cmd in ("exit", "quit"):
                    console.print("[cyan]Exiting admin mode.[/cyan]\n")
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
    """Standalone admin entry point."""
    admin = AdminCLI()
    admin.run()


if __name__ == "__main__":
    main()
