"""Simple candlestick chart using Chainlink real-time price."""

import time
from typing import List, Dict, Any, Optional
from rich.panel import Panel
from rich.text import Text


class CandlestickBuilder:
    """Build candlesticks from real-time price updates."""

    def __init__(self, interval_seconds: int = 60):
        """Initialize builder with candle interval in seconds."""
        self.interval_seconds = interval_seconds
        self.candles: List[Dict[str, Any]] = []
        self.current_candle: Optional[Dict[str, Any]] = None
        self.current_candle_start: Optional[int] = None

    def add_price(self, price: float, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        """Add price and update candles. Returns completed candle if any."""
        # Calculate which candle this price belongs to
        timestamp_s = timestamp_ms // 1000
        candle_start = (timestamp_s // self.interval_seconds) * self.interval_seconds

        # If this is a new candle
        if self.current_candle_start is None or candle_start > self.current_candle_start:
            # Save previous candle if exists
            completed_candle = None
            if self.current_candle is not None:
                self.candles.append(self.current_candle)
                completed_candle = self.current_candle
                # Keep only last 10 candles
                if len(self.candles) > 10:
                    self.candles.pop(0)

            # Start new candle
            self.current_candle_start = candle_start
            self.current_candle = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "startTime": candle_start * 1000,
            }
            return completed_candle

        # Update current candle
        if self.current_candle is not None:
            self.current_candle["high"] = max(self.current_candle["high"], price)
            self.current_candle["low"] = min(self.current_candle["low"], price)
            self.current_candle["close"] = price

        return None

    def get_display_candles(self) -> List[Dict[str, Any]]:
        """Get candles for display (completed + current)."""
        result = self.candles.copy()
        if self.current_candle is not None:
            result.append(self.current_candle)
        return result


class CandlestickChart:
    """Render candlestick chart in terminal with proper wicks and bodies."""

    def __init__(self, height: int = 8):
        """Initialize chart with height in rows."""
        self.height = height

    def render(self, candles: List[Dict[str, Any]], current_price: Optional[float] = None) -> Panel:
        """Render candles as a panel with wicks (上下影线) and bodies."""
        if not candles:
            content = "[dim]等待数据...[/dim]"
            return Panel(content, title="[bold]K线[/bold]", border_style="cyan", padding=(0, 1))

        # Calculate price range
        all_highs = [c["high"] for c in candles if c.get("high") is not None]
        all_lows = [c["low"] for c in candles if c.get("low") is not None]

        if not all_highs or not all_lows:
            content = "[dim]无数据[/dim]"
            return Panel(content, title="[bold]K线[/bold]", border_style="cyan", padding=(0, 1))

        min_price = min(all_lows)
        max_price = max(all_highs)

        # Add padding to range
        price_range = max_price - min_price
        if price_range == 0:
            price_range = max_price * 0.005
        padding = price_range * 0.1
        min_price -= padding
        max_price += padding
        price_range = max_price - min_price

        # Precompute for each candle: which row is body, which is wick
        # For each candle, we need to know: upper_wick_rows, body_rows, lower_wick_rows
        candle_data = []
        for candle in candles:
            o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
            is_green = c >= o

            # Body top and bottom (price)
            body_top = max(o, c)
            body_bottom = min(o, c)

            # Convert to row positions
            def price_to_row(p: float) -> int:
                progress = (p - min_price) / price_range
                return self.height - 1 - int(progress * (self.height - 1))

            h_row = price_to_row(h)
            l_row = price_to_row(l)
            body_top_row = price_to_row(body_top)
            body_bottom_row = price_to_row(body_bottom)

            candle_data.append({
                "is_green": is_green,
                "h_row": h_row,
                "l_row": l_row,
                "body_top_row": body_top_row,
                "body_bottom_row": body_bottom_row,
            })

        # Build chart rows
        rows = []
        for row in range(self.height):
            # Calculate price for this row (for label)
            row_progress = 1 - (row / (self.height - 1))
            row_price = min_price + (price_range * row_progress)

            # Build this row's candles
            line = ""
            for cd in candle_data:
                is_green = cd["is_green"]
                h_row = cd["h_row"]
                l_row = cd["l_row"]
                body_top_row = cd["body_top_row"]
                body_bottom_row = cd["body_bottom_row"]

                # Determine character based on position
                if row < h_row or row > l_row:
                    # Outside candle range
                    line += " "
                elif body_top_row <= row <= body_bottom_row:
                    # Body (实体)
                    if is_green:
                        line += "[green]█[/green]"
                    else:
                        line += "[red]█[/red]"
                else:
                    # Wick (影线) - use thin vertical line
                    if is_green:
                        line += "[green]│[/green]"
                    else:
                        line += "[red]│[/red]"

            rows.append((row_price, line))

        # Build content with price labels
        content_lines = []
        for i, (row_price, line) in enumerate(rows):
            if i == 0:
                label = f"[cyan]{row_price:.0f}[/cyan]"
            elif i == len(rows) - 1:
                label = f"[cyan]{row_price:.0f}[/cyan]"
            else:
                label = "     "
            content_lines.append(f"{label} {line}")

        # Add current price line at bottom
        if current_price is not None:
            content_lines.append(f"[bold yellow]● ${current_price:.2f}[/bold yellow]")

        content = "\n".join(content_lines)

        return Panel(content, title="[bold]K线[/bold]", border_style="cyan", padding=(0, 1))


def format_candlestick_display(candles: List[Dict[str, Any]], current_price: Optional[float] = None, height: int = 8) -> Panel:
    """Format candlestick chart for display."""
    chart = CandlestickChart(height=height)
    return chart.render(candles, current_price)
