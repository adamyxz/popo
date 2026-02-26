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
from rich.box import ROUNDED
from collections import deque
from datetime import datetime

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
    async_sleep_ms, format_signed_delta,
    narrative_from_sign, narrative_from_rsi, narrative_from_slope, format_prob_pct
)
from .indicators import (
    compute_session_vwap, compute_vwap_series, compute_rsi, sma, slope_last,
    compute_macd, compute_heiken_ashi, count_consecutive
)
from .engines import (
    detect_regime, compute_edge, decide, decide_stabilized,
    predict_candle_direction, predict_next_candle_direction,
    calculate_order_book_imbalance, SignalStabilizer
)
from .data import (
    fetch_klines, fetch_last_price, fetch_chainlink_btc_usd, fetch_polymarket_snapshot,
    BinanceTradeStream, BinanceKlineStream, BinanceDepthStream,
    start_polymarket_chainlink_price_stream, start_chainlink_price_stream
)
from .candlestick import CandlestickBuilder, format_candlestick_display
from .llm_predictor import LLMPredictor

# Import trading service
from ..trading import get_trading_service, PlaceOrderRequest, OrderInfo


class LogBuffer:
    """Thread-safe log buffer for panel display."""

    def __init__(self, max_lines: int = 50):
        self.logs = deque(maxlen=max_lines)
        self._max_lines = max_lines
        self.update_trigger = 0  # Increment to trigger refresh when new logs arrive

    def log(self, message: str, level: str = "info"):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            "timestamp": timestamp,
            "message": message,
            "level": level
        })
        self.update_trigger += 1  # Trigger refresh

    def debug(self, message: str):
        self.log(message, "debug")

    def info(self, message: str):
        self.log(message, "info")

    def warning(self, message: str):
        self.log(message, "warning")

    def error(self, message: str):
        self.log(message, "error")

    def success(self, message: str):
        self.log(message, "success")

    def get_logs(self) -> List[dict]:
        """Get all logs."""
        return list(self.logs)

    def clear(self):
        """Clear all logs."""
        self.logs.clear()


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

    async def place_order(self, trading_service, log_buffer: 'LogBuffer'):
        """Place order based on cart contents."""
        # Debug: Log state
        log_buffer.debug(f"place_order called: direction={self.cart_direction}, amount=${self.cart_amount}")

        if not trading_service or not trading_service.is_configured():
            log_buffer.warning("Trading service not configured")
            return "Trading service not configured"

        if self.cart_direction is None or self.cart_amount <= 0:
            log_buffer.debug(f"Cart is empty (direction={self.cart_direction}, amount={self.cart_amount})")
            return "Cart is empty (use arrow keys to add amount)"

        if not (self.market_id and (self.up_token_id or self.down_token_id)):
            log_buffer.debug(f"Waiting for market data (market_id={self.market_id})")
            return "Waiting for market data..."

        direction = self.cart_direction
        amount = self.cart_amount
        token_id = self.up_token_id if direction == "up" else self.down_token_id

        log_buffer.info(f"Placing {direction.upper()} order: ${amount}")

        request = PlaceOrderRequest(
            market_id=self.market_id,
            token_id=token_id,
            direction=direction,
            side="BUY",
            amount_in_dollars=float(amount),
        )

        result = await trading_service.place_order(request)
        if result.success:
            log_buffer.success(f"Order placed: {result.order_id}")
            # Clear cart after successful order
            self.clear_cart()
            # Refresh open orders
            try:
                self.open_orders = await trading_service.get_open_orders()
            except Exception:
                pass
            return f"✓ 订单成功: {result.order_id}"
        else:
            log_buffer.error(f"Order failed: {result.error}")
            # Error message already contains user-friendly text
            return f"✗ {result.error}"

    async def close_all(self, trading_service, log_buffer: 'LogBuffer'):
        """Close all positions."""
        log_buffer.info(f"Closing {len(self.open_orders)} order(s)...")

        if not trading_service or not trading_service.is_configured():
            log_buffer.warning("Trading service not configured")
            return

        if not self.open_orders:
            log_buffer.info("No open orders to close")
            return

        try:
            # Add timeout to prevent hanging
            import asyncio
            results = await asyncio.wait_for(
                trading_service.close_all_orders(),
                timeout=30.0  # 30 second timeout
            )
            log_buffer.debug(f"close_all_orders returned {len(results)} results")

            for result in results:
                if result.success:
                    log_buffer.success(f"Closed {result.order_id}")
                else:
                    log_buffer.error(f"Failed to close {result.order_id}: {result.error}")
        except asyncio.TimeoutError:
            log_buffer.error("Close operation timed out after 30 seconds")
        except Exception as e:
            log_buffer.error(f"Error closing orders: {e}")
            import traceback
            traceback.print_exc()

        # Refresh open orders
        try:
            self.open_orders = await trading_service.get_open_orders()
        except Exception as e:
            log_buffer.debug(f"Failed to refresh open orders: {e}")


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

        # Build trading panel content with extra width padding for 60% ratio
        content = f"""[bold cyan]━━━ Trading Panel ━━━[/bold cyan]

[dim]━━━ Cart ━━━[/dim]
  Direction: [{cart_color}]{cart_label}[/{cart_color}]
  Amount: {cart_amount_str}

[dim]━━━ Open Positions ━━━[/dim]
[bold green]UP:[/bold green] {len(up_orders)} orders | P&L: {up_pnl_str}
[bold red]DOWN:[/bold red] {len(down_orders)} orders | P&L: {down_pnl_str}

[dim]━━━ Controls ━━━[/dim]
[dim]← UP | → DOWN | Enter: Buy | Space: Sell All | q: Quit[/dim]"""

        trading_content = content
        trading_panel = Panel(trading_content, border_style="bright_yellow", padding=(0, 2), title="[bold]Trading[/bold]", expand=True)
        return trading_panel

    def render_log_panel(self, log_buffer: 'LogBuffer') -> Panel:
        """Render the log panel."""
        logs = log_buffer.get_logs()

        if not logs:
            log_content = "[dim]Waiting for logs...[/dim]\n[dim]Panel will log trading activity here[/dim]"
        else:
            # Build log lines with color coding based on level
            # Show most recent logs first (newest at top)
            log_lines = []
            for log_entry in reversed(logs):  # Show newest first
                msg = log_entry["message"]
                level = log_entry["level"]
                ts = log_entry["timestamp"]

                # Strip existing markup and re-color based on level
                # Remove existing rich tags for cleaner display
                clean_msg = msg
                for tag in ["green", "red", "yellow", "cyan", "dim", "bold", "white"]:
                    clean_msg = clean_msg.replace(f"[{tag}]", "").replace(f"[/{tag}]", "")

                # Color based on level
                if level == "error":
                    colored_msg = f"[red]{clean_msg}[/red]"
                elif level == "warning":
                    colored_msg = f"[yellow]{clean_msg}[/yellow]"
                elif level == "success":
                    colored_msg = f"[green]{clean_msg}[/green]"
                elif level == "debug":
                    colored_msg = f"[dim]{clean_msg}[/dim]"
                else:
                    colored_msg = f"[white]{clean_msg}[/white]"

                log_lines.append(f"[dim]{ts}[/dim] {colored_msg}")

            # Show last 25 logs (already reversed, so newest first)
            log_content = "\n".join(log_lines[:25])

        return Panel(log_content, border_style="bright_blue", padding=(0, 1), title="[bold cyan]📋 System Logs[/bold cyan]", subtitle="[dim]Real-time[/dim]", expand=True, height=28)

    def render_trading_with_logs(self, trading_state: 'TradingState', log_buffer: 'LogBuffer'):
        """Render trading panel and log panel side by side (60/40 ratio)."""
        from rich.table import Table

        trading_panel = self.render_trading_panel(trading_state)
        log_panel = self.render_log_panel(log_buffer)

        # Use Table to achieve fixed 60/40 ratio
        table = Table(show_header=False, show_edge=False, expand=True, pad_edge=False)
        table.add_column("trading", ratio=6, width=None)
        table.add_column("logs", ratio=4, width=None)
        table.add_row(trading_panel, log_panel)

        return table

    def render(self, data: Dict[str, Any], candlestick_panel: Optional[Panel] = None) -> Panel:
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

        # Get Expected Values (EV)
        ev_up = data.get("evUp")
        ev_down = data.get("evDown")

        # Format EV display
        ev_up_str = f"{ev_up*100:+.1f}%" if ev_up is not None else "N/A"
        ev_down_str = f"{ev_down*100:+.1f}%" if ev_down is not None else "N/A"

        # Color code EV
        ev_up_color = "green" if ev_up is not None and ev_up > 0 else "red" if ev_up is not None and ev_up < 0 else "dim"
        ev_down_color = "green" if ev_down is not None and ev_down > 0 else "red" if ev_down is not None and ev_down < 0 else "dim"

        # Signal and indicators
        signal = data.get("signal", "NO TRADE")
        signal_color = "green" if "UP" in signal else "red" if "DOWN" in signal else "dim"

        # Determine if there's an active signal for each direction
        # Active signal: signal text contains the direction AND signal count > 0
        has_up_signal = "UP" in signal and int(up_signals) > 0
        has_down_signal = "DOWN" in signal and int(down_signals) > 0

        # Create UP panel (green)
        # Active signal: solid filled style; No signal: outline style
        if has_up_signal:
            # Solid filled style when active
            up_content = f"""[bold white on green]      UP      [/bold white on green]

[bold white on green]Price:[/bold white on green] {up_price}
[bold white on green]Probability:[/bold white on green] {up_prob}
[bold white on green]EV:[/bold white on green] [{ev_up_color}]{ev_up_str}[/{ev_up_color}]
[bold white on green]Signals:[/bold white on green] {up_signals}x"""
            up_panel = Panel(
                up_content,
                border_style="green",
                style="on green",  # Solid background
                padding=(1, 2)
            )
        else:
            # Outline style when no signal
            up_content = f"""[bold green]      UP      [/bold green]

[bold green]Price:[/bold green] {up_price}
[bold green]Probability:[/bold green] {up_prob}
[bold green]EV:[/bold green] [{ev_up_color}]{ev_up_str}[/{ev_up_color}]
[bold green]Signals:[/bold green] {up_signals}x"""
            up_panel = Panel(
                up_content,
                border_style="green",
                padding=(1, 2)
            )

        # Create DOWN panel (red)
        if has_down_signal:
            # Solid filled style when active
            down_content = f"""[bold white on red]     DOWN     [/bold white on red]

[bold white on red]Price:[/bold white on red] {down_price}
[bold white on red]Probability:[/bold white on red] {down_prob}
[bold white on red]EV:[/bold white on red] [{ev_down_color}]{ev_down_str}[/{ev_down_color}]
[bold white on red]Signals:[/bold white on red] {down_signals}x"""
            down_panel = Panel(
                down_content,
                border_style="red",
                style="on red",  # Solid background
                padding=(1, 2)
            )
        else:
            # Outline style when no signal
            down_content = f"""[bold red]     DOWN     [/bold red]

[bold red]Price:[/bold red] {down_price}
[bold red]Probability:[/bold red] {down_prob}
[bold red]EV:[/bold red] [{ev_down_color}]{ev_down_str}[/{ev_down_color}]
[bold red]Signals:[/bold red] {down_signals}x"""
            down_panel = Panel(
                down_content,
                border_style="red",
                padding=(1, 2)
            )

        # Create prediction boxes row with candlestick and next candle prediction
        # Get next candle prediction data
        next_candle_pred = data.get("nextCandleDirection", {})
        next_candle_up_prob = next_candle_pred.get("up")
        next_candle_down_prob = next_candle_pred.get("down")
        next_candle_confidence = next_candle_pred.get("confidence")
        next_candle_dominant = next_candle_pred.get("dominant_signal", "N/A")

        # Create next candle prediction panel (yellow border)
        next_candle_panel = None
        if next_candle_up_prob is not None and next_candle_down_prob is not None:
            next_up_pct = next_candle_up_prob * 100
            next_down_pct = next_candle_down_prob * 100

            # Determine dominant signal color
            if next_candle_dominant == "MACD":
                dominant_color = "cyan"
            elif next_candle_dominant == "RSI":
                dominant_color = "magenta"
            elif next_candle_dominant == "VWAP":
                dominant_color = "blue"
            elif next_candle_dominant == "Heiken":
                dominant_color = "green"
            elif next_candle_dominant == "OB":
                dominant_color = "yellow"
            else:
                dominant_color = "dim"

            # Confidence color
            conf_color = "green" if next_candle_confidence and next_candle_confidence > 0.5 else "yellow" if next_candle_confidence and next_candle_confidence > 0.3 else "red"

            next_candle_content = f"""[bold yellow]NEXT CANDLE[/bold yellow]

[bold green]UP:[/bold green] {next_up_pct:.0f}%
[bold red]DOWN:[/bold red] {next_down_pct:.0f}%

[dim]Confidence:[/dim] [{conf_color}]{next_candle_confidence*100:.0f}%[/{conf_color}]
[dim]Dominant:[/dim] [{dominant_color}]{next_candle_dominant}[/{dominant_color}]"""

            next_candle_panel = Panel(
                next_candle_content,
                title="[bold]NEXT[/bold]",
                border_style="yellow",
                padding=(1, 2)
            )

        # Create LLM prediction panel (compact to avoid layout issues)
        llm_panel = None
        llm_pred = data.get("llmPrediction", {})
        llm_status = llm_pred.get("status", "waiting")
        llm_direction = llm_pred.get("direction")
        llm_confidence = llm_pred.get("confidence", 0)
        llm_reasoning = llm_pred.get("reasoning", "")
        llm_timestamp = llm_pred.get("timestamp", "")

        if llm_status == "waiting":
            llm_content = "[dim]Analyzing market...[/dim]"
        elif llm_status == "predicting":
            llm_content = "[yellow]Predicting...[/yellow]"
        elif llm_status == "disabled":
            llm_content = "[red]× No API key[/red]"
        else:  # success or error
            # Format direction display with color
            if llm_direction == "UP":
                dir_color = "green"
                dir_display = f"[{dir_color}]▲ UP[/{dir_color}]"
            elif llm_direction == "DOWN":
                dir_color = "red"
                dir_display = f"[{dir_color}]▼ DOWN[/{dir_color}]"
            else:
                dir_color = "dim"
                dir_display = "[dim]—[/dim]"

            # Show full reasoning (up to ~250 chars for ~6 lines of text at ~40 chars per line)
            # Don't truncate - let the panel handle text wrapping
            reasoning_display = llm_reasoning

            # Layout: direction + confidence on first line, reasoning below (auto-wraps)
            # Color matches direction: UP=green, DOWN=red
            llm_content = f"{dir_display} [{dir_color}]{llm_confidence:.0f}%[/{dir_color}]\n{reasoning_display}"

        llm_panel = Panel(
            llm_content,
            title="[bold]LLM[/bold]",
            border_style="magenta" if llm_status != "disabled" else "dim",
            padding=(0, 1),
            width=50  # Wider panel to show more text per line
        )

        # Build columns - LLM panel goes LEFT of candlestick
        # Don't use equal=True to prevent layout distortion
        panels_list = [up_panel, down_panel]
        if llm_panel:
            panels_list.append(llm_panel)
        if candlestick_panel:
            panels_list.append(candlestick_panel)
        if next_candle_panel:
            panels_list.append(next_candle_panel)

        prediction_boxes = Columns(panels_list, equal=False)

        # Build indicators section - compact multi-column table layout
        # Get candle direction prediction
        candle_pred = data.get("candleDirection", {})
        candle_up_prob = candle_pred.get("up")
        candle_down_prob = candle_pred.get("down")
        candle_signals = candle_pred.get("signals", {})

        # Candle direction probability summary
        candle_pred_str = "-"
        if candle_up_prob is not None and candle_down_prob is not None:
            up_pct = candle_up_prob * 100
            down_pct = candle_down_prob * 100
            candle_pred_str = f"[green]{up_pct:.0f}%[/green]/[red]{down_pct:.0f}%[/red]"

        # Build compact indicators table with 4 columns (no width constraints, let it flow)
        indicators_table = Table(show_header=False, box=None, padding=(0, 1))
        indicators_table.add_column("[dim][/dim]", style="dim")
        indicators_table.add_column("")
        indicators_table.add_column("[dim][/dim]", style="dim")
        indicators_table.add_column("")

        # Row 1: Signal & Candle Direction
        signal_full = f"[{signal_color}]{signal}[/{signal_color}] | {signal_stats}"
        indicators_table.add_row(
            f"[bold]Signal[/bold]", signal_full,
            f"[bold]Candle Dir[/bold]", candle_pred_str
        )

        # Row 2: Heiken Ashi & RSI
        indicators_table.add_row(
            f"[bold]Heiken[/bold]", str(data.get('heikenAshi', '-')),
            f"[bold]RSI[/bold]", str(data.get('rsi', '-'))
        )

        # Row 3: MACD & Delta
        indicators_table.add_row(
            f"[bold]MACD[/bold]", str(data.get('macd', '-')),
            f"[bold]Delta[/bold]", str(data.get('delta', '-'))
        )

        # Row 4: VWAP & Liquidity
        vwap_text = str(data.get('vwap', '-'))
        liquidity_text = str(format_number(data.get('liquidity'), 0)) if data.get('liquidity') else "-"
        indicators_table.add_row(
            f"[bold]VWAP[/bold]", vwap_text,
            f"[bold]Liquidity[/bold]", liquidity_text
        )

        # Build candle signals table (4 columns, no fixed width)
        candle_signals_table = None
        if candle_signals:
            candle_signals_table = Table(show_header=False, box=None, padding=(0, 1))
            candle_signals_table.add_column("[dim][/dim]", style="dim")
            candle_signals_table.add_column("")
            candle_signals_table.add_column("[dim][/dim]", style="dim")
            candle_signals_table.add_column("")

            price_sig = candle_signals.get("priceSignal")
            vol_press = candle_signals.get("volumePressure")
            ob_imbal = candle_signals.get("obImbalance")
            time_wt = candle_signals.get("timeWeight")
            elapsed = candle_signals.get("elapsedRatio")
            trend_sig = candle_signals.get("trendSignal")

            # Row 1: Price Signal & Volume Pressure
            price_str = "-"
            vol_str = "-"
            if price_sig is not None:
                price_color = "green" if price_sig > 0 else "red" if price_sig < 0 else "dim"
                price_str = f"[{price_color}]{price_sig:+.2f}[/{price_color}]"
            if vol_press is not None:
                vol_color = "green" if vol_press > 0 else "red" if vol_press < 0 else "dim"
                vol_str = f"[{vol_color}]{vol_press:+.2f}[/{vol_color}]"

            candle_signals_table.add_row(
                f"[bold]Price Sig[/bold]", price_str,
                f"[bold]Vol Press[/bold]", vol_str
            )

            # Row 2: OB Imbalance & Time Weight & Progress
            ob_str = "-"
            time_str = "-"
            elapsed_str = "-"
            if ob_imbal is not None:
                ob_color = "green" if ob_imbal > 0 else "red" if ob_imbal < 0 else "dim"
                ob_str = f"[{ob_color}]{ob_imbal:+.2f}[/{ob_color}]"
            if time_wt is not None:
                time_str = f"{time_wt:.2f}"
            if elapsed is not None:
                elapsed_pct = elapsed * 100
                elapsed_str = f"{elapsed_pct:.0f}%"

            candle_signals_table.add_row(
                f"[bold]OB Imbal[/bold]", ob_str,
                f"[bold]Time Wt[/bold]", time_str
            )

            # Row 3: Progress & Trend
            trend_str = "-"
            if trend_sig is not None:
                trend_color = "green" if trend_sig > 0 else "red" if trend_sig < 0 else "dim"
                trend_str = f"[{trend_color}]{trend_sig:+.2f}[/{trend_color}]"

            candle_signals_table.add_row(
                f"[bold]Progress[/bold]", elapsed_str,
                f"[bold]Trend[/bold]", trend_str
            )

            # Row 4: Confidence (NEW)
            confidence = candle_pred.get("confidence")
            if confidence is not None:
                conf_pct = confidence * 100
                conf_color = "green" if confidence > 0.5 else "yellow" if confidence > 0.3 else "red"
                conf_str = f"[{conf_color}]{conf_pct:.0f}%[/{conf_color}]"
                candle_signals_table.add_row(
                    f"[bold]Confidence[/bold]", conf_str,
                    "", ""
                )

        # Build real-time trading table (6 columns for compact display)
        realtime_table = Table(show_header=True, box=None, title="[bold cyan]Real-Time Data[/bold cyan]", padding=(0, 1))
        realtime_table.add_column("[dim]M1[/dim]", style="dim")
        realtime_table.add_column("[dim]V1[/dim]", justify="right")
        realtime_table.add_column("[dim]M2[/dim]", style="dim")
        realtime_table.add_column("[dim]V2[/dim]", justify="right")
        realtime_table.add_column("[dim]M3[/dim]", style="dim")
        realtime_table.add_column("[dim]V3[/dim]", justify="right")

        # Current candle info
        current_kline = data.get("currentKline")
        candle_change_str = "-"
        candle_range_str = "-"
        candle_vol_str = "-"

        if current_kline and current_kline.get("open") and current_kline.get("close"):
            open_p = current_kline.get("open")
            close_p = current_kline.get("close")
            high_p = current_kline.get("high")
            low_p = current_kline.get("low")
            volume = current_kline.get("volume")

            if close_p and open_p:
                change_pct = ((close_p - open_p) / open_p) * 100
                change_color = "green" if change_pct > 0 else "red" if change_pct < 0 else "dim"
                candle_change_str = f"[{change_color}]{change_pct:+.2f}%[/{change_color}]"

            if high_p and low_p:
                range_pct = ((high_p - low_p) / low_p) * 100
                candle_range_str = f"{range_pct:.2f}%"

            if volume:
                candle_vol_str = f"{format_number(volume, 3)}"

        # Volume stats
        vol_stats = data.get("volStats")
        buy_ratio_str = "-"
        buy_vol_str = "-"
        sell_vol_str = "-"
        trade_count_str = "-"

        if vol_stats:
            buy_vol = vol_stats.get("buyVolume", 0)
            sell_vol = vol_stats.get("sellVolume", 0)
            total_vol = vol_stats.get("totalVolume", 0)
            trade_count = vol_stats.get("tradeCount", 0)

            if buy_vol and sell_vol and total_vol:
                buy_ratio = (buy_vol / total_vol) * 100
                sell_ratio = (sell_vol / total_vol) * 100
                buy_ratio_str = f"[green]{buy_ratio:.0f}%[/green]/[red]{sell_ratio:.0f}%[/red]"
                buy_vol_str = f"[green]{format_number(buy_vol, 2)}[/green]"
                sell_vol_str = f"[red]{format_number(sell_vol, 2)}[/red]"

            if trade_count:
                trade_count_str = f"{trade_count}"

        # Order book info
        depth = data.get("depthSnapshot")
        bid_ask_str = "-"
        bid_vol_str = "-"
        ask_vol_str = "-"

        if depth:
            bid_vol = depth.get("bidVolume", 0)
            ask_vol = depth.get("askVolume", 0)

            if bid_vol and ask_vol:
                total_ob = bid_vol + ask_vol
                bid_ratio = (bid_vol / total_ob) * 100
                ask_ratio = (ask_vol / total_ob) * 100
                bid_ask_str = f"[green]{bid_ratio:.0f}%[/green]/[red]{ask_ratio:.0f}%[/red]"
                bid_vol_str = f"[green]{format_number(bid_vol, 1)}[/green]"
                ask_vol_str = f"[red]{format_number(ask_vol, 1)}[/red]"

        # Add rows to real-time table (6 columns)
        realtime_table.add_row(
            "Chg", candle_change_str,
            "B/S", buy_ratio_str,
            "B/A", bid_ask_str
        )

        realtime_table.add_row(
            "Rng", candle_range_str,
            "BuyVol", buy_vol_str,
            "BidVol", bid_vol_str
        )

        realtime_table.add_row(
            "Vol", candle_vol_str,
            "SellVol", sell_vol_str,
            "AskVol", ask_vol_str
        )

        if trade_count_str != "-":
            realtime_table.add_row(
                "Trades", trade_count_str,
                "", "", "", ""
            )

        # Build complete layout
        top_text = f"""[bold cyan]{title}[/bold cyan]
[dim]Market: {market_slug}[/dim]

[bold]PRICE TO BEAT:[/bold] {ptb_text}

{current_price_line}

[bold yellow]TIME LEFT:[/bold yellow] {time_left}"""

        # Combine all in a Group
        from rich.console import Group

        # Build the indicators group
        indicators_group = [indicators_table]
        if candle_signals_table:
            indicators_group.append(candle_signals_table)
        indicators_group.append(realtime_table)
        indicators_group.append(f"[dim]created by Adam[/dim]")

        return Panel(
            Group(
                top_text,
                prediction_boxes,
                *indicators_group
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
    log_buffer = LogBuffer(max_lines=50)

    # Add initial log
    log_buffer.info(f"Popo Panel starting: {coin} {interval}")

    try:
        trading_service = await get_trading_service()
        if trading_service.is_configured():
            log_buffer.success("Trading service initialized")
            # Load existing open orders
            try:
                trading_state.open_orders = await trading_service.get_open_orders()
                if trading_state.open_orders:
                    log_buffer.info(f"Found {len(trading_state.open_orders)} open order(s)")
            except Exception as e:
                log_buffer.warning(f"Failed to load open orders: {e}")
        else:
            log_buffer.warning("Trading service not configured - Set POLYMARKET_PRIVATE_KEY in .env to enable")
    except Exception as e:
        log_buffer.warning(f"Failed to initialize trading service: {e}")
        trading_service = None

    # Initialize state
    prev_spot_price = None
    prev_current_price = None
    last_valid_current_price = None  # Cache last valid price for fallback
    previous_snapshot_current_price = None  # Store the last saved snapshot's current_price
    price_to_beat_initialized = False  # Track if price_to_beat has been initialized
    price_history: List[Dict[str, Any]] = []
    signal_stats = {"marketSlug": None, "upSignals": 0, "downSignals": 0}

    # Signal stabilizers for each market (keyed by market slug)
    signal_stabilizers: Dict[str, SignalStabilizer] = {}

    # Track current candle ID to detect new candles
    current_candle_id: Optional[str] = None
    last_saved_market_end: Optional[int] = None
    last_saved_market_slug: Optional[str] = None
    saved_market_slugs: set = set()
    _last_market_data: Optional[Dict[str, Any]] = None
    signals_history: List[Dict[str, Any]] = []  # Track signals for current market
    consecutive_price_failures = 0  # Track consecutive failures
    last_valid_poly_data: Optional[Dict[str, Any]] = None  # Cache last valid Polymarket data
    poly_fetch_failures = 0
    ws_connection_warnings_shown = False  # Track if we've warned about WS connections

    # Health check state
    last_health_check = 0
    health_warning_shown = {"polymarket": False, "chainlink": False}

    # Candlestick chart builder (uses Chainlink price)
    # K线周期与市场周期保持一致 (5m 或 15m)
    candlestick_builder = CandlestickBuilder(interval_seconds=config["candle_window_minutes"] * 60)

    # Initialize LLM predictor
    llm_predictor = LLMPredictor(config)
    if llm_predictor.is_enabled():
        log_buffer.success("LLM predictor enabled - DeepSeek V3")

    # LLM prediction state
    llm_prediction_state: Optional[Dict[str, Any]] = {"status": "waiting", "marketSlug": None}

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
        log_buffer.success("Database connected - snapshots will be saved automatically")

        # Get price_to_beat from the most recent snapshot's current_price
        try:
            previous_snapshot_current_price = await db.get_last_snapshot_current_price()
            if previous_snapshot_current_price is not None:
                log_buffer.success(f"Price to beat initialized: ${previous_snapshot_current_price:.2f}")
        except Exception as e:
            log_buffer.warning(f"Failed to get last snapshot price: {e}")
            previous_snapshot_current_price = None
    except Exception as e:
        log_buffer.warning(f"Database connection failed: {e}")
        log_buffer.info("Snapshots will not be saved. Run 'popo init-db' to set up the database.")
        db = None
        previous_snapshot_current_price = None

    # Start WebSocket streams (matching JS version)
    # 1. Binance K-line stream (for current candle data)
    kline_interval = "1m"  # Match candle window
    binance_kline_stream = BinanceKlineStream(config["symbol"], kline_interval)
    asyncio.create_task(binance_kline_stream.start())

    # 2. Binance depth stream (for order book imbalance)
    binance_depth_stream = BinanceDepthStream(config["symbol"], "100ms")
    asyncio.create_task(binance_depth_stream.start())

    # 3. Binance trade stream (for aggressive buy/sell classification)
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
        last_log_trigger = log_buffer.update_trigger

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
                                                    log_buffer.warning("Switched to UP, cart cleared")
                                                log_buffer.info(f"UP Cart: ${trading_state.cart_amount}")
                                            elif ch3 == 'C':  # Right arrow = DOWN
                                                old_dir = trading_state.cart_direction
                                                trading_state.add_to_cart("down", 1)
                                                if old_dir and old_dir != "down":
                                                    log_buffer.warning("Switched to DOWN, cart cleared")
                                                log_buffer.info(f"DOWN Cart: ${trading_state.cart_amount}")
                                continue

                            # Handle other keys
                            if ch in ('\r', '\n'):  # Enter
                                result = await trading_state.place_order(trading_service, log_buffer)
                            elif ch == ' ':  # Space
                                await trading_state.close_all(trading_service, log_buffer)
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
                    # Check if trading panel needs immediate refresh (cart change or new logs)
                    current_refresh_trigger = trading_state.refresh_trigger
                    current_log_trigger = log_buffer.update_trigger
                    needs_immediate_refresh = (current_refresh_trigger != last_refresh_trigger or
                                              current_log_trigger != last_log_trigger)
                    if needs_immediate_refresh:
                        last_refresh_trigger = current_refresh_trigger
                        last_log_trigger = current_log_trigger

                    # Get WebSocket prices (matching JS: wsTick, polymarketWsTick, chainlinkWsTick)
                    ws_tick = binance_stream.get_last()
                    ws_price = ws_tick.get("price")

                    polymarket_ws_tick = polymarket_live_stream.get_last()
                    polymarket_ws_price = polymarket_ws_tick.get("price")

                    chainlink_ws_tick = chainlink_stream.get_last() if chainlink_stream else {"price": None, "updatedAt": None}
                    chainlink_ws_price = chainlink_ws_tick.get("price")

                    # Check WebSocket connection health periodically
                    current_time = time.time()
                    if current_time - last_health_check > 30:
                        last_health_check = current_time

                        # Check Polymarket stream health
                        if not polymarket_live_stream.is_healthy(timeout_seconds=90):
                            status = polymarket_live_stream.get_connection_status()
                            if not health_warning_shown["polymarket"]:
                                log_buffer.warning(f"Polymarket WebSocket unhealthy (no data for 90s) - connected={status['connected']}, attempts={status['connect_attempts']}")
                                health_warning_shown["polymarket"] = True
                        else:
                            health_warning_shown["polymarket"] = False

                        # Check Chainlink stream health
                        if chainlink_stream and not chainlink_stream.is_healthy(timeout_seconds=300):
                            status = chainlink_stream.get_connection_status()
                            if not health_warning_shown["chainlink"]:
                                log_buffer.warning(f"Chainlink WebSocket unhealthy (no data for 5min) - connected={status['connected']}, subscribed={status.get('subscribed', False)}, attempts={status['connect_attempts']}")
                                health_warning_shown["chainlink"] = True
                        else:
                            health_warning_shown["chainlink"] = False

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
                                log_buffer.warning("Chainlink HTTP returned None price")
                        except Exception as e:
                            log_buffer.warning(f"Chainlink fetch failed: {e}")
                            chainlink_data = {"price": None, "updatedAt": None, "source": "error"}

                    # Log price availability for debugging
                    if chainlink_data.get("price") is None:
                        # Warn about WebSocket connections once
                        if not ws_connection_warnings_shown:
                            log_buffer.warning("WebSocket prices unavailable - will retry with HTTP fallback")
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
                        log_buffer.warning(f"Binance fetch failed: {e}")
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
                            log_buffer.warning(f"Polymarket snapshot failed: {poly.get('reason', 'unknown')}")
                            # Use cached data if available and failures < 5
                            if last_valid_poly_data and poly_fetch_failures < 5:
                                log_buffer.debug("Using cached Polymarket data")
                                poly = last_valid_poly_data
                    except Exception as e:
                        poly_fetch_failures += 1
                        log_buffer.warning(f"Polymarket fetch failed: {e}")
                        # Use cached data if available
                        if last_valid_poly_data and poly_fetch_failures < 5:
                            log_buffer.debug("Using cached Polymarket data")
                            poly = last_valid_poly_data
                        else:
                            import traceback
                            log_buffer.debug(f"Polymarket error: {traceback.format_exc()[-200:]}")
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

                    # Predict current candle direction using volume-weighted signals (PRIMARY)
                    # Get current kline data
                    current_kline = binance_kline_stream.get_current_kline()
                    candle_direction_pred = None
                    next_candle_direction_pred = None
                    depth_snapshot = None
                    vol_stats = None
                    elapsed_ratio = None  # Initialize outside the if block

                    if current_kline and current_kline.get("open") and current_kline.get("close"):
                        # Calculate elapsed ratio within current candle
                        open_time = current_kline.get("openTime", 0)
                        close_time = current_kline.get("closeTime", 0)
                        now_ms = int(time.time() * 1000)

                        if close_time > open_time:
                            elapsed_ratio = (now_ms - open_time) / (close_time - open_time)
                            elapsed_ratio = max(0, min(1, elapsed_ratio))
                        else:
                            elapsed_ratio = None

                        # Get order book snapshot
                        depth_snapshot = binance_depth_stream.get_depth_snapshot(levels=5)

                        # Sync trade stream with depth data for classification
                        best_bid = depth_snapshot.get("bids", [])[0][0] if depth_snapshot.get("bids") else None
                        best_ask = depth_snapshot.get("asks", [])[0][0] if depth_snapshot.get("asks") else None
                        binance_stream.set_best_bid_ask(best_bid, best_ask)

                        # Reset trade tracking if new candle started
                        if binance_stream.current_candle_start_ts != open_time:
                            binance_stream.reset_candle_tracking(open_time)

                        # Detect new candle for stabilizer reset
                        new_candle_id = f"{config['symbol']}_{open_time}"
                        if current_candle_id != new_candle_id:
                            current_candle_id = new_candle_id
                            # Reset all stabilizers on new candle
                            for stabilizer in signal_stabilizers.values():
                                stabilizer.reset()

                        # Get candle volume stats
                        vol_stats = binance_stream.get_candle_volume_stats()

                        # Run candle direction prediction
                        # Use recent closes for volatility adaptation (last 20-30 candles)
                        recent_closes = closes[-30:] if len(closes) >= 30 else closes

                        candle_direction_pred = predict_candle_direction(
                            open_price=current_kline.get("open"),
                            current_price=current_kline.get("close"),
                            elapsed_ratio=elapsed_ratio,
                            window_minutes=config["candle_window_minutes"],  # Pass window duration
                            buy_volume=vol_stats.get("buyVolume"),
                            sell_volume=vol_stats.get("sellVolume"),
                            total_volume=vol_stats.get("totalVolume"),
                            bid_volume=depth_snapshot.get("bidVolume"),
                            ask_volume=depth_snapshot.get("askVolume"),
                            prev_closes=recent_closes,  # Pass historical data for volatility
                        )

                    # Predict next candle direction using technical indicators
                    # This can be calculated even without current_kline data
                    next_candle_direction_pred = predict_next_candle_direction(
                        rsi=rsi_now,
                        rsi_slope=rsi_slope,
                        macd=macd,
                        vwap=vwap_now,
                        vwap_slope=vwap_slope,
                        heiken_color=consec.get("color"),
                        heiken_count=consec.get("count"),
                        bid_volume=depth_snapshot.get("bidVolume") if depth_snapshot else None,
                        ask_volume=depth_snapshot.get("askVolume") if depth_snapshot else None,
                        prev_closes=closes[-50:] if len(closes) >= 50 else closes,
                        regime=regime_info.get("regime"),
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

                    # Use candle direction prediction as primary probability signal
                    # Fallback to neutral if prediction unavailable
                    if candle_direction_pred:
                        raw_model_up = candle_direction_pred.get("up", 0.5)
                        raw_model_down = candle_direction_pred.get("down", 0.5)
                    else:
                        raw_model_up = 0.5
                        raw_model_down = 0.5

                    # Get or create signal stabilizer for this market
                    market_slug = poly.get("market", {}).get("slug", "") if poly.get("ok") else ""
                    if market_slug and market_slug not in signal_stabilizers:
                        signal_stabilizers[market_slug] = SignalStabilizer(
                            window_minutes=config["candle_window_minutes"]
                        )

                    # Apply signal stabilization if we have a valid market
                    if market_slug and market_slug in signal_stabilizers:
                        stabilizer = signal_stabilizers[market_slug]

                        # Create raw prediction dict for stabilizer
                        raw_pred = {
                            "up": raw_model_up,
                            "down": raw_model_down,
                            "signals": candle_direction_pred.get("signals", {}) if candle_direction_pred else {},
                            "confidence": candle_direction_pred.get("confidence", 0) if candle_direction_pred else 0
                        }

                        # Process through stabilizer
                        stabilized = stabilizer.process(raw_pred, elapsed_ratio)

                        # Use stabilized probabilities
                        model_up = stabilized["up"]
                        model_down = stabilized["down"]
                        signal_confirmed = stabilized["confirmed"]
                        signal_action = stabilized["action"]

                        # For display: show raw vs stabilized
                        stabilized_display = f"{stabilized['up']:.1%}" if stabilized.get("smoothed") else None
                    else:
                        # No stabilizer, use raw values
                        model_up = raw_model_up
                        model_down = raw_model_down
                        signal_confirmed = False
                        signal_action = None
                        stabilized_display = None

                    # Compute edge and decision (use stabilized probabilities)
                    market_up = poly.get("prices", {}).get("up") if poly.get("ok") else None
                    market_down = poly.get("prices", {}).get("down") if poly.get("ok") else None

                    edge = compute_edge(
                        model_up,
                        model_down,
                        market_up,
                        market_down
                    )

                    # Use stabilized decision engine
                    rec = decide_stabilized(
                        time_left_min if time_left_min is not None else config["candle_window_minutes"],
                        edge.get("edgeUp"),
                        edge.get("edgeDown"),
                        model_up,
                        model_down,
                        window_minutes=config["candle_window_minutes"],
                        market_up=market_up,
                        market_down=market_down,
                        ev_up=edge.get("expectedValueUp"),
                        ev_down=edge.get("expectedValueDown"),
                        signal_confirmed=signal_confirmed,
                        signal_action=signal_action
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
                                log_buffer.warning("Using cached current price due to fetch failure")
                        else:
                            consecutive_price_failures += 1
                            if consecutive_price_failures % 5 == 1:  # Log every 5th failure
                                log_buffer.error(f"No current price available for {consecutive_price_failures} cycles")
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

                        # Update candlestick builder with Chainlink price
                        candlestick_builder.add_price(float(current_price), current_price_timestamp)

                    # Price to beat: use previous snapshot's current_price, or initialize once with current price
                    if previous_snapshot_current_price is not None:
                        price_to_beat = previous_snapshot_current_price
                    elif not price_to_beat_initialized and current_price is not None:
                        # One-time initialization: use current price as initial price_to_beat
                        price_to_beat = float(current_price)
                        previous_snapshot_current_price = price_to_beat  # Store it for subsequent use
                        price_to_beat_initialized = True
                        log_buffer.success(f"Initial price_to_beat set: ${current_price:.2f}")
                    elif previous_snapshot_current_price is not None:
                        # Already initialized, use the stored value
                        price_to_beat = previous_snapshot_current_price
                    else:
                        price_to_beat = None
                        if not price_to_beat_initialized:
                            log_buffer.error("No price available for price_to_beat")

                    # Format output - use candle direction prediction
                    p_long = model_up if candle_direction_pred else 0.5
                    p_short = model_down if candle_direction_pred else 0.5

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

                    # === LLM PREDICTION TRIGGER ===
                    # Trigger prediction once when a new market starts (handled by market switch logic below)
                    # The prediction will be triggered immediately when market_slug changes

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
                        "marketId": poly.get("market", {}).get("id", "") if poly.get("ok") else "",
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
                        "evUp": edge.get("expectedValueUp"),
                        "evDown": edge.get("expectedValueDown"),
                        "timeLeft": fmt_progress_bar(time_left_min, config["candle_window_minutes"]),
                        "binanceSpot": f"BTC (Binance): ${format_number(spot_price, 0)}",
                        "candleDirection": candle_direction_pred or {},
                        "nextCandleDirection": next_candle_direction_pred or {},
                        # Add real-time trading stats
                        "currentKline": current_kline,
                        "depthSnapshot": depth_snapshot if current_kline and current_kline.get("open") else None,
                        "volStats": vol_stats if current_kline and current_kline.get("open") else None,
                        # Add LLM prediction state
                        "llmPrediction": llm_prediction_state,
                        # Add klines for LLM
                        "klines": candles,
                        # Additional data for LLM
                        "currentPrice": current_price,
                        "sessionOpenPrice": candles[0].get("open") if candles else current_price,
                        "polymarketUpOdds": market_up if market_up else 0,
                        "polymarketDownOdds": market_down if market_down else 0,
                        "candleStartTime": timing.get("startMs"),
                    }

                    # === TRIGGER LLM PREDICTION FOR NEW MARKET ===
                    # Trigger prediction when:
                    # 1. LLM is enabled
                    # 2. We have candle data
                    # 3. Status is "waiting", "disabled", or None (meaning need to predict)
                    # 4. Either this is the first market, or the market slug changed
                    if llm_predictor.is_enabled() and candles and llm_prediction_state.get("status") in ["waiting", None, "disabled"]:
                        current_llm_market = llm_prediction_state.get("marketSlug")
                        should_predict = current_llm_market != market_slug

                        if should_predict:
                            # Update state to predicting
                            llm_prediction_state["status"] = "predicting"
                            llm_prediction_state["direction"] = None
                            llm_prediction_state["confidence"] = 0
                            llm_prediction_state["reasoning"] = ""
                            llm_prediction_state["marketSlug"] = market_slug

                            # Get session open price (first candle's open price in the window)
                            session_open_price = candles[0].get("open") if candles else current_price

                            # Build data for LLM (only raw market data, no system predictions)
                            llm_input_data = {
                                "title": config["coin"],
                                "currentPrice": current_price,
                                "priceToBeat": price_to_beat,
                                "sessionOpenPrice": session_open_price,
                                "vwap": vwap_now,
                                "vwapSlope": vwap_slope,
                                "rsi": rsi_now,
                                "macd": {
                                    "value": macd.get("value") if macd else None,
                                    "signal": macd.get("signal") if macd else None,
                                    "histogram": macd.get("hist") if macd else None,
                                },
                                "timeLeft": fmt_progress_bar(time_left_min if time_left_min is not None else config["candle_window_minutes"], config["candle_window_minutes"]),
                                "candleStartTime": timing.get("startMs"),
                                "klines": candles,  # Historical klines for CSV format
                            }

                            # Run prediction asynchronously (non-blocking)
                            async def run_llm_prediction(state_dict: dict):
                                result = await llm_predictor.predict(llm_input_data)
                                state_dict.clear()
                                state_dict.update(result)

                            asyncio.create_task(run_llm_prediction(llm_prediction_state))
                            log_buffer.info(f"LLM prediction triggered for market: {market_slug[:40] if market_slug else 'unknown'}...")

                    # Render main panel and trading panel
                    # Build candlestick chart panel
                    candlestick_candles = candlestick_builder.get_display_candles()
                    candlestick_panel = format_candlestick_display(candlestick_candles, current_price) if candlestick_candles else None

                    main_panel = display.render(display_data, candlestick_panel)
                    trading_with_logs = display.render_trading_with_logs(trading_state, log_buffer)

                    # Update display with both panels stacked
                    from rich.console import Group
                    live.update(Group(main_panel, trading_with_logs), refresh=True)  # Force refresh

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

                        # Reset signal stats for new market
                        old_up_signals = signal_stats["upSignals"]
                        old_down_signals = signal_stats["downSignals"]
                        signal_stats = {"marketSlug": market_slug, "upSignals": 0, "downSignals": 0}

                        # Clear signals history for new market
                        old_signals_count = len(signals_history)
                        signals_history.clear()

                        # Reset LLM prediction state for new market
                        llm_prediction_state.clear()
                        llm_prediction_state.update({"status": "waiting", "marketSlug": None})
                        llm_predictor.reset()
                        log_buffer.info("LLM prediction state reset for market switch")

                        # Log market switch with signal stats reset
                        if old_up_signals > 0 or old_down_signals > 0:
                            log_buffer.info(f"Signal stats reset: UPx{old_up_signals} DOWNx{old_down_signals} ({old_signals_count} total)")

                        if old_cart_amount > 0:
                            log_buffer.warning(f"Market switched - cart cleared ({old_cart_direction.upper()} ${old_cart_amount})")

                        prev_slug = _last_market_data.get("marketSlug")
                        prev_end_str = _last_market_data.get("_endDate", "")
                        prev_market_id = _last_market_data.get("marketId", "")

                        # Check if previous market ended and save snapshot
                        if prev_end_str:
                            from datetime import datetime
                            try:
                                prev_settlement_ms = int(datetime.fromisoformat(prev_end_str.replace("Z", "+00:00")).timestamp() * 1000)
                                now_ms = int(time.time() * 1000)
                                prev_ended = prev_settlement_ms <= now_ms

                                if prev_ended and prev_slug not in saved_market_slugs:
                                    # Close any open orders for the ended market
                                    if prev_market_id and trading_service:
                                        try:
                                            result = await trading_service.close_orders_for_market(prev_market_id)
                                            if result.get("success") and result.get("count", 0) > 0:
                                                count = result["count"]
                                                log_buffer.warning(f"Market ended - closed {count} unsold order(s)")
                                                # Clear open orders display
                                                trading_state.open_orders = []
                                        except Exception as e:
                                            log_buffer.debug(f"Failed to close orders for ended market: {e}")

                                    if db:
                                        try:
                                            await db.save_snapshot(_last_market_data, coin, interval)
                                            saved_market_slugs.add(prev_slug)
                                            log_buffer.success(f"Snapshot saved: {prev_slug[:40]}...")

                                            # Store the current_price from the saved snapshot for next inheritance
                                            saved_current_price = _last_market_data.get("currentPriceLine")
                                            if saved_current_price:
                                                # Extract numeric price from the display string
                                                import re
                                                match = re.search(r'\$?([\d,]+\.?\d*)', saved_current_price)
                                                if match:
                                                    try:
                                                        previous_snapshot_current_price = float(match.group(1).replace(',', ''))
                                                        log_buffer.debug(f"Stored current_price: ${previous_snapshot_current_price:.2f} for next market")
                                                    except ValueError:
                                                        pass
                                        except Exception as e:
                                            log_buffer.warning(f"Failed to save snapshot: {e}")
                            except Exception as e:
                                log_buffer.debug(f"Error processing market switch: {e}")

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
                    log_buffer.error(f"Error: {e}")
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
                log_buffer.warning(f"Failed to close database: {e}")


def run_panel(coin: str = "BTC", interval: str = "5m"):
    """Run panel (sync wrapper)."""
    try:
        asyncio.run(run_panel_async(coin, interval))
    except KeyboardInterrupt:
        pass  # Clean exit
