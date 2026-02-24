"""Utility functions."""

import time
from typing import Optional


def clamp(x: Optional[float], min_val: float, max_val: float) -> Optional[float]:
    """Clamp value between min and max."""
    if x is None:
        return None
    return max(min_val, min(max_val, x))


def sleep_ms(ms: int):
    """Sleep for milliseconds (sync)."""
    time.sleep(ms / 1000)


async def async_sleep_ms(ms: int):
    """Sleep for milliseconds (async)."""
    import asyncio
    await asyncio.sleep(ms / 1000)


def format_number(x: Optional[float], digits: int = 0) -> str:
    """Format number with fixed digits."""
    if x is None or x != x:  # NaN check
        return "-"
    return f"{x:.{digits}f}"


def format_pct(x: Optional[float], digits: int = 2) -> str:
    """Format as percentage."""
    if x is None or x != x:
        return "-"
    return f"{x * 100:.{digits}f}%"


def get_candle_window_timing(window_minutes: int) -> dict:
    """Get timing info for candle window."""
    now_ms = int(time.time() * 1000)
    window_ms = window_minutes * 60_000
    start_ms = (now_ms // window_ms) * window_ms
    end_ms = start_ms + window_ms
    elapsed_ms = now_ms - start_ms
    remaining_ms = end_ms - now_ms

    return {
        "startMs": start_ms,
        "endMs": end_ms,
        "elapsedMs": elapsed_ms,
        "remainingMs": remaining_ms,
        "elapsedMinutes": elapsed_ms / 60_000,
        "remainingMinutes": remaining_ms / 60_000
    }


def fmt_time_left(mins: Optional[float]) -> str:
    """Format time left as MM:SS."""
    if mins is None:
        return "--:--"
    total_seconds = max(0, int(mins * 60))
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"


def fmt_et_time() -> str:
    """Format current ET time."""
    from datetime import datetime, timezone, timedelta
    try:
        et_tz = timezone(timedelta(hours=-5))
        return datetime.now(et_tz).strftime("%H:%M:%S")
    except:
        return "--:--:--"


def get_btc_session() -> str:
    """Get current BTC trading session."""
    from datetime import datetime, timezone, timedelta
    try:
        utc_tz = timezone.utc
        h = datetime.now(utc_tz).hour

        in_asia = 0 <= h < 8
        in_europe = 7 <= h < 16
        in_us = 13 <= h < 22

        if in_europe and in_us:
            return "Europe/US overlap"
        if in_asia and in_europe:
            return "Asia/Europe overlap"
        if in_asia:
            return "Asia"
        if in_europe:
            return "Europe"
        if in_us:
            return "US"
        return "Off-hours"
    except:
        return "Unknown"


def parse_price_to_beat(market: dict) -> Optional[float]:
    """Parse price to beat from market question text."""
    import re
    text = str(market.get("question") or market.get("title") or "")
    if not text:
        return None

    m = re.search(r'price\s*to\s*beat[^\d$]*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)', text, re.IGNORECASE)
    if not m:
        return None

    raw = m.group(1).replace(",", "")
    try:
        n = float(raw)
        return n if __import__('math').isfinite(n) else None
    except ValueError:
        return None


def extract_numeric_from_market(market: dict) -> Optional[float]:
    """Extract numeric value from market data."""
    direct_keys = [
        "priceToBeat", "price_to_beat", "strikePrice", "strike_price",
        "strike", "threshold", "thresholdPrice", "threshold_price",
        "targetPrice", "target_price", "referencePrice", "reference_price"
    ]

    for k in direct_keys:
        v = market.get(k)
        if v is not None:
            try:
                n = float(v)
                if __import__('math').isfinite(n) and 1000 < n < 2_000_000:
                    return n
            except (TypeError, ValueError):
                continue

    return None


def price_to_beat_from_polymarket_market(market: dict) -> Optional[float]:
    """Get price to beat from Polymarket market data."""
    n = extract_numeric_from_market(market)
    if n is not None:
        return n
    return parse_price_to_beat(market)


def format_signed_delta(delta: Optional[float], base: Optional[float]) -> str:
    """Format signed delta with USD and percentage."""
    if delta is None or base is None or base == 0:
        return "--"
    sign = "+" if delta > 0 else "-" if delta < 0 else ""
    pct = (abs(delta) / abs(base)) * 100
    return f"{sign}${abs(delta):.2f}, {sign}{pct:.2f}%"


def narrative_from_sign(x: Optional[float]) -> str:
    """Get narrative from sign."""
    if x is None or not __import__('math').isfinite(x) or x == 0:
        return "NEUTRAL"
    return "LONG" if x > 0 else "SHORT"


def narrative_from_rsi(rsi: Optional[float]) -> str:
    """Get narrative from RSI."""
    if rsi is None or not __import__('math').isfinite(rsi):
        return "NEUTRAL"
    if rsi >= 55:
        return "LONG"
    if rsi <= 45:
        return "SHORT"
    return "NEUTRAL"


def narrative_from_slope(slope: Optional[float]) -> str:
    """Get narrative from slope."""
    if slope is None or not __import__('math').isfinite(slope) or slope == 0:
        return "NEUTRAL"
    return "LONG" if slope > 0 else "SHORT"


def format_prob_pct(p: Optional[float], digits: int = 0) -> str:
    """Format probability as percentage."""
    if p is None or not __import__('math').isfinite(p):
        return "-"
    return f"{p * 100:.{digits}f}%"
