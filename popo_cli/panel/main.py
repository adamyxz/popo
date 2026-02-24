"""Main panel logic - real-time trading analysis."""

import asyncio
import time
import re
import sys
import warnings
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table

# Suppress websockets library internal errors
warnings.filterwarnings("ignore", message=".*recv_messages.*")

# Monkey patch to fix websockets compatibility issue
try:
    import websockets.asyncio.connection
    original_connection_lost = websockets.asyncio.connection.ClientConnection.connection_lost

    def patched_connection_lost(self, exc):
        """Patched connection_lost that handles missing recv_messages."""
        try:
            return original_connection_lost(self, exc)
        except AttributeError:
            # recv_messages doesn't exist, just skip
            pass

    websockets.asyncio.connection.ClientConnection.connection_lost = patched_connection_lost
except Exception:
    pass  # If patching fails, continue without it

from .config import get_config
from .utils import (
    format_number, format_pct, fmt_time_left, fmt_progress_bar, get_candle_window_timing,
    async_sleep_ms, price_to_beat_from_polymarket_market, format_signed_delta,
    narrative_from_sign, narrative_from_rsi, narrative_from_slope, format_prob_pct,
    parse_price_to_beat, extract_numeric_from_market
)
from .indicators import (
    compute_session_vwap, compute_vwap_series, compute_rsi, sma, slope_last,
    compute_macd, compute_heiken_ashi, count_consecutive
)
from .engines import detect_regime, score_direction, apply_time_awareness, compute_edge, decide
from .data import (
    fetch_klines, fetch_last_price, fetch_chainlink_btc_usd, fetch_polymarket_snapshot,
    BinanceTradeStream, start_polymarket_chainlink_price_stream, start_chainlink_price_stream
)

# Import trading service
from ..trading import get_trading_service, PlaceOrderRequest, OrderInfo


class TradingState:
    """Trading state for panel."""

    def __init__(self):
        self.cart_amount = 0  # Current cart amount (only one direction)
        self.cart_direction = None  # "up" or "down" or None
        self.open_orders: List[OrderInfo] = []
        self.market_id = None
        self.up_token_id = None
        self.down_token_id = None
        self.refresh_trigger = 0  # Increment to trigger refresh

    def add_to_cart(self, direction: str, amount: int = 1):
        """Add amount to cart. If direction changes, clear previous amount."""
        if self.cart_direction is None:
            # First selection
            self.cart_direction = direction
            self.cart_amount = amount
        elif self.cart_direction == direction:
            # Same direction, add amount
            self.cart_amount += amount
        else:
            # Different direction, clear and set new
            self.cart_direction = direction
            self.cart_amount = amount
        self.refresh_trigger += 1  # Trigger refresh

    def clear_cart(self):
        """Clear the cart (after placing order)."""
        self.cart_amount = 0
        self.cart_direction = None
        self.refresh_trigger += 1  # Trigger refresh

    async def place_order(self, trading_service):
        """Place order based on cart contents."""
        if not trading_service or not trading_service.is_configured():
            return "Trading service not configured"

        if self.cart_direction is None or self.cart_amount <= 0:
            return "Cart is empty (use arrow keys to add amount)"

        if not (self.market_id and (self.up_token_id or self.down_token_id)):
            return "Waiting for market data..."

        direction = self.cart_direction
        amount = self.cart_amount
        token_id = self.up_token_id if direction == "up" else self.down_token_id

        request = PlaceOrderRequest(
            market_id=self.market_id,
            token_id=token_id,
            direction=direction,
            side="BUY",
            amount_in_dollars=float(amount),
        )

        result = await trading_service.place_order(request)

        if result.success:
            # Clear cart after successful order
            self.clear_cart()
            # Refresh open orders
            try:
                self.open_orders = await trading_service.get_open_orders()
            except Exception:
                pass
            return f"Order placed: {result.order_id}"
        else:
            return f"Order failed: {result.error}"

    async def close_all(self, trading_service, console: Console):
        """Close all positions."""
        if not trading_service or not trading_service.is_configured():
            console.print("[yellow]Trading service not configured[/yellow]")
            return

        if not self.open_orders:
            console.print("[dim]No open orders to close[/dim]")
            return

        console.print(f"[cyan]Closing {len(self.open_orders)} order(s)...[/cyan]")
        results = await trading_service.close_all_orders()
        for result in results:
            if result.success:
                console.print(f"[green]✓ Closed {result.order_id}[/green]")
            else:
                console.print(f"[red]✗ Failed to close {result.order_id}: {result.error}[/red]")

        # Refresh open orders
        try:
            self.open_orders = await trading_service.get_open_orders()
        except Exception:
            pass


def count_vwap_crosses(closes: List[float], vwap_series: List[Optional[float]], lookback: int) -> Optional[int]:
    """Count VWAP crosses in lookback period."""
    if len(closes) < lookback or len(vwap_series) < lookback:
        return None

    crosses = 0
    for i in range(len(closes) - lookback + 1, len(closes)):
        prev = closes[i - 1] - (vwap_series[i - 1] or 0)
        cur = closes[i] - (vwap_series[i] or 0)
        if prev == 0:
            continue
        if (prev > 0 and cur < 0) or (prev < 0 and cur > 0):
            crosses += 1

    return crosses


class PanelDisplay:
    """Panel display manager."""

    def __init__(self, console: Console):
        self.console = console

    def render_trading_panel(self, trading_state: 'TradingState') -> Panel:
        """Render the trading panel at bottom."""
        # Cart display
        if trading_state.cart_direction == "up":
            cart_color = "green"
            cart_label = "UP"
            cart_amount_str = f"[bold {cart_color}]${trading_state.cart_amount}[/bold {cart_color}]"
        elif trading_state.cart_direction == "down":
            cart_color = "red"
            cart_label = "DOWN"
            cart_amount_str = f"[bold {cart_color}]${trading_state.cart_amount}[/bold {cart_color}]"
        else:
            cart_color = "dim"
            cart_label = "EMPTY"
            cart_amount_str = "[dim]$0[/dim]"

        # Count open orders by direction (only for current market)
        if trading_state.market_id:
            up_orders = [o for o in trading_state.open_orders
                        if o.direction == "up" and o.status in ["matched", "open"] and o.market_id == trading_state.market_id]
            down_orders = [o for o in trading_state.open_orders
                         if o.direction == "down" and o.status in ["matched", "open"] and o.market_id == trading_state.market_id]
        else:
            up_orders = []
            down_orders = []

        up_pnl = sum(o.pnl or 0 for o in up_orders)
        down_pnl = sum(o.pnl or 0 for o in down_orders)

        up_pnl_str = f"[green]+${up_pnl:.2f}[/green]" if up_pnl > 0 else f"[red]${up_pnl:.2f}[/red]" if up_pnl < 0 else "[dim]$0.00[/dim]"
        down_pnl_str = f"[green]+${down_pnl:.2f}[/green]" if down_pnl > 0 else f"[red]${down_pnl:.2f}[/red]" if down_pnl < 0 else "[dim]$0.00[/dim]"

        # Build trading panel content
        content = f"""[bold cyan]━━━ Trading Panel ━━━[/bold cyan]

[dim]━━━ Cart ━━━[/dim]
  Direction: [{cart_color}]{cart_label}[/{cart_color}]
  Amount: {cart_amount_str}

[dim]━━━ Open Positions ━━━[/dim]
[bold green]UP:[/bold green] {len(up_orders)} orders | P&L: {up_pnl_str}
[bold red]DOWN:[/bold red] {len(down_orders)} orders | P&L: {down_pnl_str}

[dim]━━━ Controls ━━━[/dim]
[dim]← UP | → DOWN | Enter: Buy | Space: Sell All | q: Quit[/dim]"""

        return Panel(content, border_style="bright_yellow", padding=(0, 1))

    def render(self, data: Dict[str, Any]) -> Panel:
        """Render the panel display with new layout."""
        # Parse data
        title = data.get("title", "-")
        market_slug = data.get("marketSlug", "-")

        # Price to beat
        price_to_beat = data.get("priceToBeat")
        ptb_text = f"[bold white on red] ${format_number(price_to_beat, 2)} [/bold white on red]" if price_to_beat else "[dim]—[/dim]"

        # Current price with delta
        current_price_line = data.get("currentPriceLine", "-")

        # Time left
        time_left = data.get("timeLeft", "-")

        # Parse prediction values
        predict_value = data.get("predictValue", "-")
        up_prob = "—"
        down_prob = "—"

        predict_match = re.search(r'LONG[^\d]*(\d+)%', predict_value)
        if predict_match:
            up_prob = f"{predict_match.group(1)}%"

        down_match = re.search(r'SHORT[^\d]*(\d+)%', predict_value)
        if down_match:
            down_prob = f"{down_match.group(1)}%"

        # Parse signal stats
        signal_stats = data.get("signalStats", "UPx0 DOWNx0")
        up_signals = "0"
        down_signals = "0"

        signal_match = re.search(r'UPx(\d+)', signal_stats)
        if signal_match:
            up_signals = signal_match.group(1)

        down_match = re.search(r'DOWNx(\d+)', signal_stats)
        if down_match:
            down_signals = down_match.group(1)

        # Get Polymarket prices
        poly_header_data = data.get("polyHeader", "")
        up_price = "—c"
        down_price = "—c"

        poly_match = re.search(r'UP[^\d]*(\d+)c', poly_header_data)
        if poly_match:
            up_price = f"{poly_match.group(1)}c"

        down_match = re.search(r'DOWN[^\d]*(\d+)c', poly_header_data)
        if down_match:
            down_price = f"{down_match.group(1)}c"

        # Signal and indicators
        signal = data.get("signal", "NO TRADE")
        signal_color = "green" if "UP" in signal else "red" if "DOWN" in signal else "dim"

        # Create UP panel (green)
        up_content = f"""[bold white on green]      UP      [/bold white on green]

[bold green]Price:[/bold green] {up_price}
[bold green]Probability:[/bold green] {up_prob}
[bold green]Signals:[/bold green] {up_signals}x"""

        up_panel = Panel(
            up_content,
            border_style="green",
            padding=(1, 2)
        )

        # Create DOWN panel (red)
        down_content = f"""[bold white on red]     DOWN     [/bold white on red]

[bold red]Price:[/bold red] {down_price}
[bold red]Probability:[/bold red] {down_prob}
[bold red]Signals:[/bold red] {down_signals}x"""

        down_panel = Panel(
            down_content,
            border_style="red",
            padding=(1, 2)
        )

        # Create prediction boxes row
        prediction_boxes = Columns([up_panel, down_panel], equal=True)

        # Build indicators section
        indicators_text = f"""[bold cyan]Indicators[/bold cyan]
  [bold]Signal:[/bold] [{signal_color}]{signal}[/{signal_color}] | {signal_stats}
  [bold]Heiken Ashi:[/bold] {data.get("heikenAshi", "-")}
  [bold]RSI:[/bold] {data.get("rsi", "-")}
  [bold]MACD:[/bold] {data.get("macd", "-")}
  [bold]Delta 1/3:[/bold] {data.get("delta", "-")}
  [bold]VWAP:[/bold] {data.get("vwap", "-")}"""

        # Add liquidity if available
        if data.get("liquidity"):
            indicators_text += f"\n  [bold]Liquidity:[/bold] {format_number(data.get('liquidity'), 0)}"

        # Add Binance spot
        indicators_text += f"\n  [bold]Binance Spot:[/bold] {data.get('binanceSpot', '-')}"
        indicators_text += "\n\n[dim]created by Adam[/dim]"

        # Build complete layout
        top_text = f"""[bold cyan]{title}[/bold cyan]
[dim]Market: {market_slug}[/dim]

[bold]PRICE TO BEAT:[/bold] {ptb_text}

{current_price_line}

[bold yellow]TIME LEFT:[/bold yellow] {time_left}"""

        # Combine all in a Group
        from rich.console import Group
        return Panel(
            Group(
                top_text,
                prediction_boxes,
                indicators_text
            ),
            title="[bold]Popo Panel[/bold]",
            border_style="bright_blue",
            padding=(0, 1)
        )


async def run_panel_async(coin: str = "BTC", interval: str = "5m"):
    """Run panel async loop with trading support."""
    console = Console()
    display = PanelDisplay(console)

    config = get_config(coin, interval)

    # Initialize trading service
    trading_service = None
    trading_state = TradingState()
    try:
        trading_service = await get_trading_service()
        if trading_service.is_configured():
            console.print("[green]✓[/green] Trading service initialized")
            # Load existing open orders
            try:
                trading_state.open_orders = await trading_service.get_open_orders()
                if trading_state.open_orders:
                    console.print(f"[dim]Found {len(trading_state.open_orders)} open order(s)[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load open orders: {e}[/yellow]")
        else:
            console.print("[yellow]Warning: Trading service not configured[/yellow]")
            console.print("[dim]Set POLYMARKET_PRIVATE_KEY in .env to enable trading[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to initialize trading service: {e}[/yellow]")
        trading_service = None

    # Initialize state
    prev_spot_price = None
    prev_current_price = None
    last_valid_current_price = None  # Cache last valid price for fallback
    price_to_beat_state = {"slug": None, "value": None, "setAtMs": None}
    previous_market_end_info = {"slug": None, "endMs": None}
    price_history: List[Dict[str, Any]] = []
    signal_stats = {"marketSlug": None, "upSignals": 0, "downSignals": 0}
    last_saved_market_end: Optional[int] = None
    last_saved_market_slug: Optional[str] = None
    saved_market_slugs: set = set()
    _last_market_data: Optional[Dict[str, Any]] = None
    signals_history: List[Dict[str, Any]] = []  # Track signals for current market
    consecutive_price_failures = 0  # Track consecutive failures
    last_valid_poly_data: Optional[Dict[str, Any]] = None  # Cache last valid Polymarket data
    poly_fetch_failures = 0
    ws_connection_warnings_shown = False  # Track if we've warned about WS connections

    # Create HTTP session
    import aiohttp
    session = aiohttp.ClientSession()

    # Initialize database
    from ..db import get_db, close_db
    db = None
    try:
        db = await get_db()
        # Test the connection
        async with db.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        console.print("[green]✓[/green] Database connected - snapshots will be saved automatically")
    except Exception as e:
        console.print(f"[yellow]Warning: Database connection failed: {e}[/yellow]")
        console.print("[dim]Snapshots will not be saved. Run 'popo init-db' to set up the database.[/dim]")
        db = None

    # Start WebSocket streams (matching JS version)
    # 1. Binance trade stream
    binance_stream = BinanceTradeStream(config["symbol"])
    asyncio.create_task(binance_stream.start())

    # 2. Polymarket Chainlink price stream
    polymarket_live_stream = start_polymarket_chainlink_price_stream(
        ws_url=config["polymarket"].get("live_data_ws_url", ""),
        symbol_includes=config["coin"].lower(),
        on_update=None
    )
    asyncio.create_task(polymarket_live_stream.start())

    # 3. Chainlink price stream via Polygon WebSocket
    chainlink_wss_urls = config["chainlink"].get("polygon_wss_urls", [])
    if config["chainlink"].get("polygon_wss_url"):
        chainlink_wss_urls.append(config["chainlink"]["polygon_wss_url"])

    chainlink_stream = None
    if chainlink_wss_urls and config["chainlink"].get("aggregator"):
        # Remove duplicates and filter
        seen = set()
        unique_urls = []
        for url in chainlink_wss_urls:
            url = str(url).strip()
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)

        if unique_urls:
            chainlink_stream = start_chainlink_price_stream(
                wss_urls=unique_urls,
                aggregator=config["chainlink"]["aggregator"],
                decimals=8,
                on_update=None
            )
            asyncio.create_task(chainlink_stream.start())

    try:
        console.clear()
        live = Live(console=console, refresh_per_second=10)  # Higher refresh rate
        live.start()

        # Track last refresh trigger to detect changes
        last_refresh_trigger = trading_state.refresh_trigger

        # Create input task for key detection
        async def key_handler():
            """Handle keyboard input in background - single character mode."""
            import sys
            import tty
            import termios
            import os
            import select

            fd = sys.stdin.fileno()
            old_settings = None

            try:
                # Save old settings
                old_settings = termios.tcgetattr(fd)
                tty.setcbreak(fd)

                while True:
                    # Check if data available
                    if select.select([fd], [], [], 0)[0]:
                        try:
                            # Read one character
                            ch = os.read(fd, 1).decode('utf-8', errors='ignore')

                            # Handle arrow keys (escape sequence)
                            if ch == '\x1b':  # ESC
                                # Wait a bit for next chars
                                await asyncio.sleep(0.005)
                                if select.select([fd], [], [], 0)[0]:
                                    ch2 = os.read(fd, 1).decode('utf-8', errors='ignore')
                                    if ch2 == '[':
                                        await asyncio.sleep(0.005)
                                        if select.select([fd], [], [], 0)[0]:
                                            ch3 = os.read(fd, 1).decode('utf-8', errors='ignore')
                                            if ch3 == 'D':  # Left arrow = UP
                                                old_dir = trading_state.cart_direction
                                                trading_state.add_to_cart("up", 1)
                                                if old_dir and old_dir != "up":
                                                    console.print("[yellow]← Switched to UP, cart cleared[/yellow]")
                                                console.print(f"[green]← UP Cart: ${trading_state.cart_amount}[/green]")
                                            elif ch3 == 'C':  # Right arrow = DOWN
                                                old_dir = trading_state.cart_direction
                                                trading_state.add_to_cart("down", 1)
                                                if old_dir and old_dir != "down":
                                                    console.print("[yellow]→ Switched to DOWN, cart cleared[/yellow]")
                                                console.print(f"[red]→ DOWN Cart: ${trading_state.cart_amount}[/red]")
                                continue

                            # Handle other keys
                            if ch in ('\r', '\n'):  # Enter
                                result = await trading_state.place_order(trading_service)
                                if result:
                                    if "Order placed" in result:
                                        console.print(f"[green]✓ {result}[/green]")
                                    else:
                                        console.print(f"[yellow]{result}[/yellow]")
                            elif ch == ' ':  # Space
                                await trading_state.close_all(trading_service, console)
                            elif ch == '\x03' or ch == 'q':  # Ctrl+C or q
                                raise KeyboardInterrupt

                        except (OSError, IOError):
                            break

                    await asyncio.sleep(0.02)  # Small sleep to avoid busy loop

            finally:
                # Restore terminal settings
                if old_settings:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Start key handler as background task
        key_task = asyncio.create_task(key_handler())

        try:
            while True:
                try:
                    # Check if trading panel needs immediate refresh
                    current_refresh_trigger = trading_state.refresh_trigger
                    needs_immediate_refresh = (current_refresh_trigger != last_refresh_trigger)
                    if needs_immediate_refresh:
                        last_refresh_trigger = current_refresh_trigger

                    # Get WebSocket prices (matching JS: wsTick, polymarketWsTick, chainlinkWsTick)
                    # Get WebSocket prices (matching JS: wsTick, polymarketWsTick, chainlinkWsTick)
                    ws_tick = binance_stream.get_last()
                    ws_price = ws_tick.get("price")

                    polymarket_ws_tick = polymarket_live_stream.get_last()
                    polymarket_ws_price = polymarket_ws_tick.get("price")

                    chainlink_ws_tick = chainlink_stream.get_last() if chainlink_stream else {"price": None, "updatedAt": None}
                    chainlink_ws_price = chainlink_ws_tick.get("price")

                    # Fetch data
                    timing = get_candle_window_timing(config["candle_window_minutes"])

                    # Determine Chainlink source with priority (matching JS logic)
                    # Priority: Polymarket WS > Chainlink WS > Chainlink HTTP
                    if polymarket_ws_price is not None:
                        chainlink_data = {
                            "price": polymarket_ws_price,
                            "updatedAt": polymarket_ws_tick.get("updatedAt"),
                            "source": "polymarket_ws"
                        }
                    elif chainlink_ws_price is not None:
                        chainlink_data = {
                            "price": chainlink_ws_price,
                            "updatedAt": chainlink_ws_tick.get("updatedAt"),
                            "source": "chainlink_ws"
                        }
                    else:
                        # Fall back to HTTP
                        try:
                            chainlink_data = await fetch_chainlink_btc_usd(config["chainlink"])
                            if chainlink_data.get("price") is None:
                                console.print(f"[yellow]Warning: Chainlink HTTP returned None price[/yellow]")
                        except Exception as e:
                            console.print(f"[yellow]Warning: Chainlink fetch failed: {e}[/yellow]")
                            chainlink_data = {"price": None, "updatedAt": None, "source": "error"}

                    # Log price availability for debugging
                    if chainlink_data.get("price") is None:
                        console.print(f"[dim]Debug: polymarket_ws_price={polymarket_ws_price}, chainlink_ws_price={chainlink_ws_price}[/dim]")

                        # Warn about WebSocket connections once
                        if not ws_connection_warnings_shown:
                            console.print("[yellow]Warning: WebSocket prices unavailable - check connections[/yellow]")
                            console.print("[dim]  Will retry with HTTP fallback...[/dim]")
                            ws_connection_warnings_shown = True
                    else:
                        # Reset warning flag when we get data
                        ws_connection_warnings_shown = False

                    # Fetch Binance data (with error handling)
                    try:
                        klines_1m = await fetch_klines(session, config["symbol"], "1m", 240, config["binance_base_url"])
                        klines_5m = await fetch_klines(session, config["symbol"], "5m", 200, config["binance_base_url"])
                        last_price = await fetch_last_price(session, config["symbol"], config["binance_base_url"])
                    except Exception as e:
                        console.print(f"[yellow]Warning: Binance fetch failed: {e}[/yellow]")
                        klines_1m = []
                        klines_5m = []
                        last_price = None

                    # Fetch Polymarket (with error handling and caching)
                    try:
                        poly = await fetch_polymarket_snapshot(
                            session,
                            config["polymarket"],
                            config["polymarket"].get("market_slug"),
                            config["polymarket"].get("series_id"),
                            config["polymarket"].get("auto_select_latest", True),
                            config["gamma_base_url"],
                            config["clob_base_url"]
                        )
                        if poly.get("ok"):
                            last_valid_poly_data = poly
                            poly_fetch_failures = 0
                        else:
                            poly_fetch_failures += 1
                            console.print(f"[yellow]Warning: Polymarket snapshot failed: {poly.get('reason', 'unknown')}[/yellow]")
                            # Use cached data if available and failures < 5
                            if last_valid_poly_data and poly_fetch_failures < 5:
                                console.print("[dim]Using cached Polymarket data[/dim]")
                                poly = last_valid_poly_data
                    except Exception as e:
                        poly_fetch_failures += 1
                        console.print(f"[yellow]Warning: Polymarket fetch failed: {e}[/yellow]")
                        # Use cached data if available
                        if last_valid_poly_data and poly_fetch_failures < 5:
                            console.print("[dim]Using cached Polymarket data[/dim]")
                            poly = last_valid_poly_data
                        else:
                            import traceback
                            console.print(f"[dim]{traceback.format_exc()[-300:]}[/dim]")
                            poly = {"ok": False, "reason": "fetch_error"}

                    # Process data
                    candles = klines_1m
                    closes = [c["close"] for c in candles if c["close"] is not None]

                    if not closes:
                        await async_sleep_ms(config["poll_interval_ms"])
                        continue

                    # Compute indicators
                    vwap = compute_session_vwap(candles)
                    vwap_series = compute_vwap_series(candles)
                    vwap_now = vwap_series[-1] if vwap_series else None

                    lookback = config["vwap_slope_lookback_minutes"]
                    if vwap_series and len(vwap_series) >= lookback:
                        vwap_slope = (vwap_now - vwap_series[-lookback]) / lookback if vwap_now is not None else None
                    else:
                        vwap_slope = None

                    vwap_dist = (last_price - vwap_now) / vwap_now if vwap_now is not None and last_price is not None else None

                    rsi_now = compute_rsi(closes, config["rsi_period"])
                    rsi_series = []
                    for i in range(len(closes)):
                        sub = closes[:i + 1]
                        r = compute_rsi(sub, config["rsi_period"])
                        if r is not None:
                            rsi_series.append(r)

                    rsi_ma = sma(rsi_series, config["rsi_ma_period"]) if rsi_series else None
                    rsi_slope = slope_last(rsi_series, 3) if rsi_series else None

                    macd = compute_macd(closes, config["macd_fast"], config["macd_slow"], config["macd_signal"])

                    ha = compute_heiken_ashi(candles)
                    consec = count_consecutive(ha) if ha else {"color": None, "count": 0}

                    vwap_cross_count = count_vwap_crosses(closes, vwap_series or [], 20) if closes and vwap_series else None

                    volume_recent = sum(c["volume"] or 0 for c in candles[-20:]) if candles else 0
                    volume_avg = sum(c["volume"] or 0 for c in candles[-120:]) / 6 if len(candles) >= 120 else None

                    failed_vwap_reclaim = False
                    if vwap_now is not None and vwap_series and len(vwap_series) >= 3 and len(candles) >= 2:
                        failed_vwap_reclaim = (
                            candles[-1]["close"] < vwap_now and
                            candles[-2]["close"] > (vwap_series[-2] or 0)
                        )

                    # Detect regime
                    regime_info = detect_regime(
                        last_price, vwap_now, vwap_slope, vwap_cross_count,
                        volume_recent, volume_avg
                    )

                    # Score direction
                    scored = score_direction(
                        price=last_price,
                        vwap=vwap_now,
                        vwap_slope=vwap_slope,
                        rsi=rsi_now,
                        rsi_slope=rsi_slope,
                        macd=macd,
                        heiken_color=consec.get("color"),
                        heiken_count=consec.get("count"),
                        failed_vwap_reclaim=failed_vwap_reclaim
                    )

                    # Time awareness
                    settlement_ms = None
                    if poly.get("ok") and poly.get("market", {}).get("endDate"):
                        from datetime import datetime, timezone
                        try:
                            settlement_ms = int(datetime.fromisoformat(
                                poly["market"]["endDate"].replace("Z", "+00:00")
                            ).timestamp() * 1000)
                        except:
                            pass

                    settlement_left_min = (settlement_ms - int(time.time() * 1000)) / 60_000 if settlement_ms else None
                    time_left_min = settlement_left_min if settlement_ms is not None else timing["remainingMinutes"]

                    time_aware = apply_time_awareness(scored["rawUp"], time_left_min, config["candle_window_minutes"])

                    # Compute edge and decision
                    market_up = poly.get("prices", {}).get("up") if poly.get("ok") else None
                    market_down = poly.get("prices", {}).get("down") if poly.get("ok") else None

                    edge = compute_edge(
                        time_aware["adjustedUp"],
                        time_aware["adjustedDown"],
                        market_up,
                        market_down
                    )

                    rec = decide(
                        time_left_min if time_left_min is not None else config["candle_window_minutes"],
                        edge.get("edgeUp"),
                        edge.get("edgeDown"),
                        time_aware["adjustedUp"],
                        time_aware["adjustedDown"]
                    )

                    # Update signal stats
                    market_slug = poly.get("market", {}).get("slug", "") if poly.get("ok") else ""
                    if rec.get("action") == "ENTER" and signal_stats.get("marketSlug") != market_slug:
                        signal_stats = {"marketSlug": market_slug, "upSignals": 0, "downSignals": 0}

                    if rec.get("action") == "ENTER":
                        if rec.get("side") == "UP":
                            signal_stats["upSignals"] += 1
                            # Record signal history
                            signals_history.append({
                                "direction": "up",
                                "up_price": market_up,
                                "down_price": market_down,
                                "remaining_time": int(time_left_min * 60) if time_left_min is not None else None,
                                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
                            })
                        if rec.get("side") == "DOWN":
                            signal_stats["downSignals"] += 1
                            # Record signal history
                            signals_history.append({
                                "direction": "down",
                                "up_price": market_up,
                                "down_price": market_down,
                                "remaining_time": int(time_left_min * 60) if time_left_min is not None else None,
                                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
                            })

                    # KEY: Current price from Chainlink (matching JS: currentPrice = chainlink?.price ?? null)
                    current_price = chainlink_data.get("price")
                    current_price_timestamp = chainlink_data.get("updatedAt")

                    # Fallback to last valid price if current is None (with limit)
                    if current_price is None:
                        if last_valid_current_price is not None and consecutive_price_failures < 10:
                            current_price = last_valid_current_price
                            consecutive_price_failures += 1
                            if consecutive_price_failures == 1:
                                console.print("[yellow]Using cached current price due to fetch failure[/yellow]")
                        else:
                            consecutive_price_failures += 1
                            if consecutive_price_failures % 5 == 1:  # Log every 5th failure
                                console.print(f"[red]Error: No current price available for {consecutive_price_failures} cycles[/red]")
                    else:
                        # Update cache when we have valid data
                        if last_valid_current_price != current_price:
                            last_valid_current_price = float(current_price)
                        consecutive_price_failures = 0

                    # Spot price from Binance (matching JS: spotPrice = wsPrice ?? lastPrice)
                    spot_price = ws_price if ws_price is not None else last_price

                    # Record price history (matching JS logic)
                    if current_price is not None and current_price_timestamp is not None:
                        price_history.append({"price": float(current_price), "timestampMs": current_price_timestamp})
                        if len(price_history) > 1000:
                            price_history = price_history[-1000:]

                    # Handle market switching and price to beat
                    explicit_price_to_beat = None
                    if poly.get("ok") and poly.get("market"):
                        explicit_price_to_beat = price_to_beat_from_polymarket_market(poly["market"])
                        # Debug logging
                        if explicit_price_to_beat is None:
                            console.print(f"[dim]Debug: No explicit price_to_beat found in market data for {market_slug}[/dim]")
                            console.print(f"[dim]  Market keys: {list(poly.get('market', {}).keys())[:10]}[/dim]")

                    # Market switch detection
                    if market_slug and price_to_beat_state["slug"] != market_slug:
                        # Reset signal stats on market switch
                        signal_stats = {"marketSlug": market_slug, "upSignals": 0, "downSignals": 0}
                        signals_history = []  # Reset signals history on market switch

                        # Inherit price from previous market end
                        inherited_price = None
                        if previous_market_end_info.get("endMs") and price_history:
                            target_ms = previous_market_end_info["endMs"]
                            closest = None
                            min_diff = float('inf')

                            for entry in price_history:
                                diff = abs(entry["timestampMs"] - target_ms)
                                if diff < min_diff:
                                    min_diff = diff
                                    closest = entry["price"]

                            if closest is not None and min_diff < 60_000:
                                inherited_price = closest

                        price_to_beat_state = {
                            "slug": market_slug,
                            "value": inherited_price,
                            "setAtMs": int(time.time() * 1000) if inherited_price is not None else None
                        }
                        previous_market_end_info = {"slug": None, "endMs": None}

                    # Latch current price if no explicit or inherited price (matching JS logic)
                    market_start_ms = None
                    if poly.get("ok") and poly.get("market", {}).get("eventStartTime"):
                        from datetime import datetime
                        try:
                            market_start_ms = int(datetime.fromisoformat(
                                poly["market"]["eventStartTime"].replace("Z", "+00:00")
                            ).timestamp() * 1000)
                        except:
                            pass

                    if (price_to_beat_state["slug"] == market_slug and
                        price_to_beat_state["value"] is None and
                        explicit_price_to_beat is None and
                        current_price is not None):
                        now_ms = int(time.time() * 1000)
                        ok_to_latch = market_start_ms is None or now_ms >= market_start_ms
                        if ok_to_latch:
                            price_to_beat_state["value"] = float(current_price)
                            price_to_beat_state["setAtMs"] = now_ms

                    price_to_beat = explicit_price_to_beat or (
                        price_to_beat_state["value"] if price_to_beat_state["slug"] == market_slug else None
                    )

                    # Final fallback: if still None and we have current_price, use it as price_to_beat
                    if price_to_beat is None and current_price is not None and price_to_beat_state["value"] is None:
                        # Only do this for the first time for this market
                        if price_to_beat_state["slug"] != market_slug:
                            price_to_beat = float(current_price)
                            price_to_beat_state = {
                                "slug": market_slug,
                                "value": float(current_price),
                                "setAtMs": int(time.time() * 1000)
                            }
                            console.print(f"[cyan]Setting initial price_to_beat to current price: ${current_price:.2f}[/cyan]")

                    # Debug logging for price to beat
                    if price_to_beat is None:
                        console.print(f"[dim]Debug price_to_beat: explicit={explicit_price_to_beat}, state_slug={price_to_beat_state['slug']}, market_slug={market_slug}, state_value={price_to_beat_state['value']}[/dim]")

                    # Format output
                    p_long = time_aware.get("adjustedUp")
                    p_short = time_aware.get("adjustedDown")

                    predict_value = f"[green]LONG[/green] {format_prob_pct(p_long, 0)} / [red]SHORT[/red] {format_prob_pct(p_short, 0)}"

                    signal = f"BUY {rec.get('side')}" if rec.get("action") == "ENTER" else "NO TRADE"
                    signal_stats_str = f"UPx{signal_stats['upSignals']} DOWNx{signal_stats['downSignals']}"

                    # Heiken Ashi
                    ha_narrative = consec.get("color", "NEUTRAL").upper()
                    heiken_str = f"{consec.get('color', '-').upper()} x{consec.get('count', 0)}"
                    ha_color = "green" if ha_narrative == "LONG" else "red" if ha_narrative == "SHORT" else "dim"
                    heiken_line = f"[{ha_color}]{heiken_str}[/{ha_color}]"

                    # RSI
                    rsi_arrow = "↓" if rsi_slope and rsi_slope < 0 else "↑" if rsi_slope and rsi_slope > 0 else "-"
                    rsi_str = f"{format_number(rsi_now, 1)} {rsi_arrow}"
                    rsi_narrative = narrative_from_slope(rsi_slope)
                    rsi_color = "green" if rsi_narrative == "LONG" else "red" if rsi_narrative == "SHORT" else "dim"
                    rsi_line = f"[{rsi_color}]{rsi_str}[/{rsi_color}]"

                    # MACD
                    if macd:
                        hist = macd.get("hist")
                        hist_delta = macd.get("histDelta")
                        if hist is not None:
                            if hist < 0:
                                macd_label = "bearish" + (" (expanding)" if hist_delta and hist_delta < 0 else "")
                            else:
                                macd_label = "bullish" + (" (expanding)" if hist_delta and hist_delta > 0 else "")
                        else:
                            macd_label = "-"
                    else:
                        macd_label = "-"

                    macd_narrative = narrative_from_sign(macd.get("hist") if macd else None)
                    macd_color = "green" if macd_narrative == "LONG" else "red" if macd_narrative == "SHORT" else "dim"
                    macd_line = f"[{macd_color}]{macd_label}[/{macd_color}]"

                    # Delta
                    last_close = candles[-1]["close"] if candles else None
                    close_1m_ago = candles[-2]["close"] if len(candles) >= 2 else None
                    close_3m_ago = candles[-4]["close"] if len(candles) >= 4 else None

                    delta_1m = (last_close - close_1m_ago) if last_close and close_1m_ago else None
                    delta_3m = (last_close - close_3m_ago) if last_close and close_3m_ago else None

                    delta_1_narrative = narrative_from_sign(delta_1m)
                    delta_3_narrative = narrative_from_sign(delta_3m)
                    delta_1_color = "green" if delta_1_narrative == "LONG" else "red" if delta_1_narrative == "SHORT" else "dim"
                    delta_3_color = "green" if delta_3_narrative == "LONG" else "red" if delta_3_narrative == "SHORT" else "dim"

                    delta_str = f"[{delta_1_color}]{format_signed_delta(delta_1m, last_close)}[/{delta_1_color}] | [{delta_3_color}]{format_signed_delta(delta_3m, last_close)}[/{delta_3_color}]"

                    # VWAP
                    vwap_color = "green" if vwap_dist and vwap_dist > 0 else "red" if vwap_dist and vwap_dist < 0 else "dim"
                    vwap_slope_label = "UP" if vwap_slope and vwap_slope > 0 else "DOWN" if vwap_slope and vwap_slope < 0 else "FLAT"
                    vwap_str = f"{format_number(vwap_now, 0)} ({format_pct(vwap_dist, 2)}) | slope: {vwap_slope_label}"
                    vwap_line = f"[{vwap_color}]{vwap_str}[/{vwap_color}]"

                    # Current price - color based on comparison with priceToBeat
                    current_price_color = "green" if price_to_beat and current_price and current_price > price_to_beat else "red" if price_to_beat and current_price and current_price < price_to_beat else "white"

                    # Calculate delta from price to beat
                    ptb_delta = None
                    if current_price is not None and price_to_beat is not None:
                        ptb_delta = current_price - price_to_beat

                    ptb_delta_color = "green" if ptb_delta and ptb_delta > 0 else "red" if ptb_delta and ptb_delta < 0 else "dim"
                    ptb_delta_text = ""
                    if ptb_delta is not None:
                        sign = "+" if ptb_delta > 0 else "-" if ptb_delta < 0 else ""
                        ptb_delta_text = f"[{ptb_delta_color}]{sign}${abs(ptb_delta):.2f}[/{ptb_delta_color}]"
                    else:
                        ptb_delta_text = "[dim]-[/dim]"

                    # Update trading state with market info
                    if poly.get("ok") and poly.get("market"):
                        trading_state.market_id = poly["market"].get("id", "")
                        # Get token IDs from poly response (already parsed by fetch_polymarket_snapshot)
                        if poly.get("tokens"):
                            trading_state.up_token_id = poly["tokens"].get("upTokenId")
                            trading_state.down_token_id = poly["tokens"].get("downTokenId")

                    # Update open orders P&L based on current prices
                    # Filter orders: only show orders for current market
                    up_orders_pnl = []
                    down_orders_pnl = []
                    if trading_service and trading_state.open_orders and trading_state.market_id:
                        for order in trading_state.open_orders:
                            # Skip orders that don't belong to current market
                            if order.market_id != trading_state.market_id:
                                continue

                            if order.status in ["matched", "open"]:
                                current_price_for_order = market_up if order.direction == "up" else market_down
                                if order.price and current_price_for_order:
                                    # P&L = (current_price - entry_price) * shares
                                    pnl = (current_price_for_order - order.price) * (order.shares or 0)
                                    pnl_percentage = ((current_price_for_order / order.price) - 1) * 100 if order.price > 0 else 0
                                    order.pnl = pnl
                                    order.pnl_percentage = pnl_percentage

                                    order_dict = {
                                        "order_id": order.order_id,
                                        "price": order.price,
                                        "shares": order.shares,
                                        "pnl": pnl,
                                        "pnl_percentage": pnl_percentage,
                                    }
                                    if order.direction == "up":
                                        up_orders_pnl.append(order_dict)
                                    else:
                                        down_orders_pnl.append(order_dict)

                    # Build display data
                    display_data = {
                        "title": poly.get("market", {}).get("question", "-") if poly.get("ok") else "-",
                        "marketSlug": market_slug,
                        "predictValue": predict_value,
                        "signal": signal,
                        "signalStats": signal_stats_str,
                        "heikenAshi": heiken_line,
                        "rsi": rsi_line,
                        "macd": macd_line,
                        "delta": delta_str,
                        "vwap": vwap_line,
                        "liquidity": poly.get("market", {}).get("liquidityNum") or poly.get("market", {}).get("liquidity") if poly.get("ok") else None,
                        "priceToBeat": price_to_beat,
                        "currentPriceLine": f"[{current_price_color}]CURRENT PRICE: ${format_number(current_price, 2)} ({ptb_delta_text})[/{current_price_color}]",
                        "polyHeader": f"UP [green]{format_number(market_up * 100, 0) if market_up else 0}c[/green] | DOWN [red]{format_number(market_down * 100, 0) if market_down else 0}c[/red]",
                        "timeLeft": fmt_progress_bar(time_left_min, config["candle_window_minutes"]),
                        "binanceSpot": f"BTC (Binance): ${format_number(spot_price, 0)}",
                    }

                    # Render main panel and trading panel
                    main_panel = display.render(display_data)
                    trading_panel = display.render_trading_panel(trading_state)

                    # Update display with both panels stacked
                    from rich.console import Group
                    live.update(Group(main_panel, trading_panel), refresh=True)  # Force refresh

                    # Check for market switch and save previous market if ended
                    # Only save if: 1) we have previous market data, 2) market slug actually changed, 3) previous slug is valid
                    if (_last_market_data is not None and
                        _last_market_data.get("marketSlug") and
                        _last_market_data.get("marketSlug") != market_slug and
                        market_slug):  # Current slug is also valid (not empty)

                        # Market switched - reset trading cart
                        old_cart_amount = trading_state.cart_amount
                        old_cart_direction = trading_state.cart_direction
                        trading_state.clear_cart()
                        trading_state.refresh_trigger += 1

                        if old_cart_amount > 0:
                            console.print(f"[yellow]Market switched - cart cleared ({old_cart_direction.upper()} ${old_cart_amount})[/yellow]")

                        prev_slug = _last_market_data.get("marketSlug")
                        prev_end_str = _last_market_data.get("_endDate", "")

                        # Check if previous market ended
                        if prev_end_str:
                            from datetime import datetime
                            try:
                                prev_settlement_ms = int(datetime.fromisoformat(prev_end_str.replace("Z", "+00:00")).timestamp() * 1000)
                                now_ms = int(time.time() * 1000)
                                prev_ended = prev_settlement_ms <= now_ms

                                if prev_ended and prev_slug not in saved_market_slugs:
                                    if db:
                                        try:
                                            await db.save_snapshot(_last_market_data, coin, interval)
                                            saved_market_slugs.add(prev_slug)
                                            console.print(f"[green]✓[/green] Snapshot saved for: [cyan]{prev_slug[:40]}...[/cyan]")

                                            # Store previous market end info for inheritance
                                            previous_market_end_info = {
                                                "slug": prev_slug,
                                                "endMs": prev_settlement_ms
                                            }
                                        except Exception as e:
                                            console.print(f"[yellow]Warning: Failed to save snapshot for {prev_slug[:30]}: {e}[/yellow]")
                                    else:
                                        console.print(f"[dim]Skipping snapshot save (database not connected)[/dim]")
                            except Exception as e:
                                console.print(f"[dim]Error processing market switch: {e}[/dim]")

                    # Store current market data for next iteration (add endDate and signals history for later use)
                    # ONLY update if we have a valid market slug
                    if market_slug:
                        current_market_data = display_data.copy()
                        current_market_data["_endDate"] = poly.get("market", {}).get("endDate") if poly.get("ok") else None
                        current_market_data["signalsHistory"] = signals_history.copy()  # Include signals history
                        _last_market_data = current_market_data

                    # Update previous values
                    prev_spot_price = spot_price
                    prev_current_price = current_price

                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    import traceback
                    traceback.print_exc()

                # Dynamic sleep: short if cart changed, normal otherwise
                if trading_state.refresh_trigger != last_refresh_trigger:
                    await asyncio.sleep(0.05)  # Fast refresh after key press
                else:
                    await async_sleep_ms(config["poll_interval_ms"])

        finally:
            live.stop()
            # Cancel key handler task
            key_task.cancel()
            try:
                await key_task
            except asyncio.CancelledError:
                pass

    finally:
        await session.close()
        # Close WebSocket streams
        await binance_stream.close()
        await polymarket_live_stream.close()
        if chainlink_stream:
            await chainlink_stream.close()
        # Close database
        if db:
            try:
                await close_db()
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to close database: {e}[/yellow]")


def run_panel(coin: str = "BTC", interval: str = "5m"):
    """Run panel (sync wrapper)."""
    try:
        asyncio.run(run_panel_async(coin, interval))
    except KeyboardInterrupt:
        print("\n[yellow]Panel stopped.[/yellow]")
