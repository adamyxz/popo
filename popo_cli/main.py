"""Popo CLI - Interactive REPL interface."""

from rich.console import Console
from rich.prompt import Prompt
import typer
import asyncio

app = typer.Typer(
    name="popo",
    help="An interactive CLI tool built with typer and rich",
    add_completion=False,
)

console = Console()


@app.command()
def test_snapshot() -> None:
    """Test saving a snapshot to database."""
    async def _test():
        from .db import get_db
        import time
        console.print("[cyan]Testing snapshot save...[/cyan]")
        try:
            db = await get_db()

            # Create test display data
            test_data = {
                "marketSlug": "test-market-" + str(int(time.time())),
                "title": "Test Market - Will BTC go up?",
                "priceToBeat": 95000.00,
                "currentPriceLine": "[green]CURRENT PRICE: $95,234.50 (+$234.50)[/green]",
                "polyHeader": "UP [green]53c[/green] | DOWN [red]47c[/red]",
                "predictValue": "[green]LONG[/green] 65% / [red]SHORT[/red] 35%",
                "signal": "BUY UP",
                "signalStats": "UPx3 DOWNx1",
                "heikenAshi": "[green]GREEN x5[/green]",
                "rsi": "[green]58.3 ↑[/green]",
                "macd": "[green]bullish (expanding)[/green]",
                "delta": "[green]+0.23% | +0.67%[/green]",
                "vwap": "[green]95,100 (0.14%) | slope: UP[/green]",
                "liquidity": 250000.00,
                "timeLeft": "████████░░ 80%",
                "binanceSpot": "BTC (Binance): $95,235",
                "signalsHistory": [
                    {
                        "direction": "up",
                        "up_price": 0.5299999999999999,
                        "down_price": 0.47000000000000003,
                        "remaining_time": 240,
                        "timestamp": "2026-02-25T10:00:00+00:00"
                    },
                    {
                        "direction": "up",
                        "up_price": 0.5100000000000001,
                        "down_price": 0.48999999999999994,
                        "remaining_time": 180,
                        "timestamp": "2026-02-25T10:05:00+00:00"
                    }
                ]
            }

            await db.save_snapshot(test_data, "BTC", "5m")
            console.print("[green]✓[/green] Test snapshot saved!")

            # Query it back
            snapshots = await db.get_snapshots(limit=1)
            if snapshots:
                console.print(f"[green]✓[/green] Retrieved snapshot: {snapshots[0]['market_slug']}")
            else:
                console.print("[yellow]Warning: No snapshots found after save[/yellow]")

            await db.close()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(_test())


@app.command()
def init_db() -> None:
    """Initialize database tables and add missing columns."""
    async def _init():
        from .db import get_db
        console.print("[cyan]Initializing database...[/cyan]")
        try:
            db = await get_db()

            # Add columns for existing tables
            async with db.pool.acquire() as conn:
                try:
                    await conn.execute("""
                        ALTER TABLE snapshots
                        ADD COLUMN IF NOT EXISTS signals_history JSONB;
                    """)
                    console.print("[green]✓[/green] Added signals_history column")
                except Exception as e:
                    console.print(f"[yellow]Note: {e}[/yellow]")

                try:
                    await conn.execute("""
                        ALTER TABLE snapshots
                        ADD COLUMN IF NOT EXISTS vwap_slope VARCHAR(20);
                    """)
                    console.print("[green]✓[/green] Added vwap_slope column")
                except Exception as e:
                    console.print(f"[yellow]Note: {e}[/yellow]")

            console.print("[green]✓[/green] Database connected and tables updated!")
            console.print("[dim]Table 'snapshots' is ready to store data[/dim]")

            # Test query
            async with db.pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM snapshots")
                console.print(f"[dim]Current snapshot count: {count}[/dim]")

            await db.close()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(_init())


@app.command()
def snapshots(market: str = typer.Option(None, "--market", "-m", help="Filter by market slug"),
              limit: int = typer.Option(10, "--limit", "-l", help="Number of snapshots to show")) -> None:
    """Show saved snapshots."""
    async def _show():
        from .db import get_db
        try:
            db = await get_db()
            rows = await db.get_snapshots(market_slug=market, limit=limit)

            if not rows:
                console.print("[dim]No snapshots found[/dim]")
                await db.close()
                return

            from rich.table import Table

            table = Table(title=f"Snapshots {f'(market: {market})' if market else ''}")
            table.add_column("ID", style="dim")
            table.add_column("Market")
            table.add_column("Coin")
            table.add_column("Price to Beat")
            table.add_column("Current Price")
            table.add_column("Signal")
            table.add_column("Time")

            for row in rows:
                table.add_row(
                    str(row["id"]),
                    row.get("market_slug", "-")[:30] + "..." if len(row.get("market_slug", "")) > 30 else row.get("market_slug", "-"),
                    row.get("coin", "-"),
                    f"${row.get('price_to_beat', 0):.2f}" if row.get("price_to_beat") else "-",
                    f"${row.get('current_price', 0):.2f}" if row.get("current_price") else "-",
                    row.get("signal", "-"),
                    row.get("timestamp", "-").strftime("%H:%M:%S") if row.get("timestamp") else "-"
                )

            console.print(table)
            console.print(f"\n[dim]Total: {len(rows)} snapshot(s)[/dim]")

            await db.close()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(_show())


@app.command()
def migrate_snapshots() -> None:
    """Migrate existing snapshots to new schema format."""
    async def _migrate():
        from .db import get_db
        from decimal import Decimal, ROUND_HALF_UP
        console.print("[cyan]Migrating snapshots to new schema...[/cyan]")
        try:
            db = await get_db()

            async with db.pool.acquire() as conn:
                # Fetch all snapshots that need migration
                rows = await conn.fetch("""
                    SELECT id, rsi, vwap, signals_history, price_to_beat, current_price,
                           ptb_delta, liquidity, binance_spot, poly_prices
                    FROM snapshots
                """)

                migrated = 0
                for row in rows:
                    updates = []
                    values = []
                    param_count = 1

                    # Migrate RSI - remove arrow
                    if row["rsi"]:
                        import re
                        rsi_clean = re.sub(r'\s*[↑↓-]\s*', '', str(row["rsi"]))
                        rsi_clean = db._remove_rich_tags(rsi_clean)
                        # Extract just the number
                        rsi_number = rsi_clean.split()[0] if rsi_clean else None
                        if rsi_number and rsi_number != row["rsi"]:
                            updates.append(f"rsi = ${param_count}")
                            values.append(rsi_number)
                            param_count += 1

                    # Migrate VWAP - extract slope
                    if row["vwap"] and not row.get("vwap_slope"):
                        import re
                        vwap_raw = db._remove_rich_tags(str(row["vwap"]))
                        # Extract slope
                        vwap_slope = ""
                        if "slope:" in vwap_raw:
                            slope_part = vwap_raw.split("slope:")[1].strip()
                            vwap_slope = slope_part.split()[0] if slope_part else ""

                        if vwap_slope:
                            updates.append(f"vwap_slope = ${param_count}")
                            values.append(vwap_slope)
                            param_count += 1

                    # Fix precision for numeric columns
                    for col in ['price_to_beat', 'current_price', 'ptb_delta', 'liquidity', 'binance_spot']:
                        if row.get(col) is not None:
                            try:
                                value = Decimal(str(row[col]))
                                rounded = value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                                updates.append(f"{col} = ${param_count}")
                                values.append(rounded)
                                param_count += 1
                            except:
                                pass

                    # Fix poly_prices precision
                    if row.get("poly_prices"):
                        import json
                        try:
                            prices = json.loads(row["poly_prices"]) if isinstance(row["poly_prices"], str) else row["poly_prices"]
                            cleaned = {}
                            for key, value in prices.items():
                                if value is not None:
                                    dec_value = Decimal(str(value))
                                    cleaned[key] = float(dec_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                            if cleaned != prices:
                                updates.append(f"poly_prices = ${param_count}")
                                values.append(json.dumps(cleaned))
                                param_count += 1
                        except:
                            pass

                    # Clean signals_history prices to 2 decimal places
                    if row["signals_history"]:
                        import json
                        try:
                            signals_history = json.loads(row["signals_history"])
                            cleaned = []
                            for sig in signals_history:
                                up_price = sig.get("up_price")
                                down_price = sig.get("down_price")

                                cleaned_sig = {
                                    "direction": sig.get("direction"),
                                    "up_price": float(Decimal(str(up_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) if up_price is not None else None,
                                    "down_price": float(Decimal(str(down_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) if down_price is not None else None,
                                    "remaining_time": sig.get("remaining_time"),
                                    "timestamp": sig.get("timestamp")
                                }
                                cleaned.append(cleaned_sig)

                            updates.append(f"signals_history = ${param_count}")
                            values.append(json.dumps(cleaned))
                            param_count += 1
                        except:
                            pass

                    if updates:
                        values.append(row["id"])
                        await conn.execute(f"""
                            UPDATE snapshots
                            SET {', '.join(updates)}
                            WHERE id = ${param_count}
                        """, *values)
                        migrated += 1

                console.print(f"[green]✓[/green] Migrated {migrated} snapshot(s)")

            await db.close()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(_migrate())


@app.command()
def cleanup_db() -> None:
    """Remove deprecated columns (time_left, raw_data) from database."""
    async def _cleanup():
        from .db import get_db
        console.print("[yellow]Warning: This will permanently remove time_left and raw_data columns.[/yellow]")
        console.print("[dim]These columns contain data that will be deleted.[/dim]")

        from rich.prompt import Confirm
        if not Confirm.ask("Do you want to continue?"):
            console.print("[dim]Cleanup cancelled.[/dim]")
            return

        console.print("[cyan]Cleaning up database...[/cyan]")
        try:
            db = await get_db()

            async with db.pool.acquire() as conn:
                # Drop time_left column
                try:
                    await conn.execute("""
                        ALTER TABLE snapshots
                        DROP COLUMN IF EXISTS time_left;
                    """)
                    console.print("[green]✓[/green] Dropped time_left column")
                except Exception as e:
                    console.print(f"[yellow]Note: {e}[/yellow]")

                # Drop raw_data column
                try:
                    await conn.execute("""
                        ALTER TABLE snapshots
                        DROP COLUMN IF EXISTS raw_data;
                    """)
                    console.print("[green]✓[/green] Dropped raw_data column")
                except Exception as e:
                    console.print(f"[yellow]Note: {e}[/yellow]")

            console.print("[green]✓[/green] Database cleanup completed!")

            await db.close()
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(_cleanup())


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

[bold cyan]CLI Commands:[/bold cyan]
  [dim]popo start[/dim]              - Start interactive CLI
  [dim]popo init-db[/dim]            - Initialize database tables
  [dim]popo test-snapshot[/dim]      - Test saving a snapshot
  [dim]popo snapshots[/dim]          - Show saved snapshots
  [dim]popo snapshots -m MARKET_SLUG[/dim] - Filter by market
  [dim]popo migrate-snapshots[/dim]  - Migrate existing snapshots to new schema
  [dim]popo cleanup-db[/dim]         - Remove deprecated columns (time_left, raw_data)
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


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Main entry point for the CLI."""
    # If no subcommand is provided, start the REPL
    if ctx.invoked_subcommand is None:
        repl()


if __name__ == "__main__":
    app()
