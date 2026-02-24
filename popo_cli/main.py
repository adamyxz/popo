"""Popo CLI - Interactive REPL interface."""

from rich.console import Console
from rich.prompt import Prompt
import typer

app = typer.Typer(
    name="popo",
    help="An interactive CLI tool built with typer and rich",
    add_completion=False,
)

console = Console()


def handle_hi_command() -> None:
    """Handle the /hi command."""
    console.print("[bold green]hi[/bold green]", emoji=True)


def handle_panel_command(args: str = "") -> None:
    """Handle the /panel command."""
    import shlex

    try:
        parts = shlex.split(args) if args else []
        coin = parts[0] if len(parts) > 0 else "BTC"
        interval = parts[1] if len(parts) > 1 else "5m"

        # Validate
        valid_coins = ["BTC", "ETH", "SOL"]
        valid_intervals = ["5m", "15m"]

        if coin.upper() not in valid_coins:
            console.print(f"[red]Invalid coin: {coin}[/red]")
            console.print(f"[dim]Valid coins: {', '.join(valid_coins)}[/dim]")
            return

        if interval.lower() not in valid_intervals:
            console.print(f"[red]Invalid interval: {interval}[/red]")
            console.print(f"[dim]Valid intervals: {', '.join(valid_intervals)}[/dim]")
            return

        console.print(f"[cyan]Starting panel: {coin.upper()} {interval.lower()}[/cyan]")
        console.print("[dim]Press Ctrl+C to exit panel[/dim]\n")

        from .panel import run_panel
        run_panel(coin.upper(), interval.lower())

    except KeyboardInterrupt:
        console.print("\n[yellow]Panel stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting panel: {e}[/red]")


def handle_command(command: str) -> bool:
    """Handle a user command.

    Returns True if should continue, False if should exit.
    """
    command = command.strip()

    if not command:
        return True

    if command == "/hi":
        handle_hi_command()
    elif command == "/exit" or command == "/quit":
        return False
    elif command == "/help":
        show_help()
    elif command.startswith("/panel"):
        parts = command.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        handle_panel_command(args)
        return True  # Continue after panel exits
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Type [bold]/help[/bold] for available commands")

    return True


def show_help() -> None:
    """Show available commands."""
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

  [bold]/hi[/bold]     - Say hi
  [bold]/panel[/bold] [coin] [interval] - Start trading panel (e.g., /panel BTC 5m)
  [bold]/help[/bold]   - Show this help message
  [bold]/exit[/bold]   - Exit the CLI
  [bold]/quit[/bold]   - Exit the CLI

[bold cyan]Panel Options:[/bold cyan]
  [dim]Coins: BTC, ETH, SOL[/dim]
  [dim]Intervals: 5m, 15m[/dim]

[bold cyan]Examples:[/bold cyan]
  [dim]/panel BTC 5m[/dim]     - Bitcoin 5-minute panel
  [dim]/panel ETH 15m[/dim]    - Ethereum 15-minute panel
  [dim]/panel SOL 5m[/dim]     - Solana 5-minute panel
"""
    console.print(help_text)


def repl() -> None:
    """Run the interactive REPL loop."""
    console.print("[bold cyan]Welcome to Popo CLI![/bold cyan]")
    console.print("Type [bold]/help[/bold] for available commands, [bold]/exit[/bold] to quit\n")

    while True:
        try:
            user_input = Prompt.ask(
                "[bold blue]>[/bold blue]",
                console=console,
            )

            should_continue = handle_command(user_input)
            if not should_continue:
                console.print("[yellow]Goodbye![/yellow]")
                break

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


@app.command()
def start() -> None:
    """Start the interactive CLI."""
    repl()


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
