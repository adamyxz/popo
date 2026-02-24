"""Main panel logic - real-time trading analysis."""

import asyncio
import time
import re
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns

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
        indicators = f"""[bold cyan]Indicators[/bold cyan]
  [bold]Signal:[/bold] [{signal_color}]{signal}[/{signal_color}] | {signal_stats}
  [bold]Heiken Ashi:[/bold] {data.get("heikenAshi", "-")}
  [bold]RSI:[/bold] {data.get("rsi", "-")}
  [bold]MACD:[/bold] {data.get("macd", "-")}
  [bold]Delta 1/3:[/bold] {data.get("delta", "-")}
  [bold]VWAP:[/bold] {data.get("vwap", "-")}"""

        # Add liquidity if available
        liquidity = data.get("liquidity")
        if liquidity:
            indicators += f"\n  [bold]Liquidity:[/bold] {format_number(liquidity, 0)}"

        # Add Binance spot
        indicators += f"\n  [bold]Binance Spot:[/bold] {data.get('binanceSpot', '-')}"

        # Build complete layout as renderables
        # Top section
        top_text = f"""[bold cyan]{title}[/bold cyan]
[dim]Market: {market_slug}[/dim]

[bold]PRICE TO BEAT:[/bold] {ptb_text}

{current_price_line}

[bold yellow]TIME LEFT:[/bold yellow] {time_left}"""

        # Indicators section
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

        # Create panels for text sections
        top_panel_content = Panel(top_text, border_style="bright_blue", padding=(0, 1))
        indicators_panel_content = Panel(indicators_text, border_style="bright_blue", padding=(0, 1))

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
    """Run panel async loop."""
    console = Console()
    display = PanelDisplay(console)

    config = get_config(coin, interval)

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
        live = Live(console=console, refresh_per_second=1)
        live.start()

        try:
            while True:
                try:
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

                    # Update display
                    panel = display.render(display_data)
                    live.update(panel)

                    # Check for market switch and save previous market if ended
                    # Only save if: 1) we have previous market data, 2) market slug actually changed, 3) previous slug is valid
                    if (_last_market_data is not None and
                        _last_market_data.get("marketSlug") and
                        _last_market_data.get("marketSlug") != market_slug and
                        market_slug):  # Current slug is also valid (not empty)

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

                await async_sleep_ms(config["poll_interval_ms"])

        finally:
            live.stop()

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
