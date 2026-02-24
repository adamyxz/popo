"""VWAP (Volume Weighted Average Price) indicator."""

from typing import List, Dict, Any, Optional


def compute_session_vwap(candles: List[Dict[str, Any]]) -> Optional[float]:
    """Compute VWAP for a session."""
    if not isinstance(candles, list) or len(candles) == 0:
        return None

    pv = 0.0  # Price * Volume
    v = 0.0   # Volume

    for candle in candles:
        high = candle.get("high")
        low = candle.get("low")
        close = candle.get("close")
        volume = candle.get("volume")

        if None in (high, low, close, volume):
            continue

        tp = (high + low + close) / 3
        pv += tp * volume
        v += volume

    if v == 0:
        return None

    return pv / v


def compute_vwap_series(candles: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Compute VWAP series (cumulative)."""
    series = []

    for i in range(len(candles)):
        sub = candles[:i + 1]
        series.append(compute_session_vwap(sub))

    return series
