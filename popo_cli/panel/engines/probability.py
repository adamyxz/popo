"""Probability scoring engine."""

from typing import Dict, Any, Optional


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def score_direction(
    price: Optional[float] = None,
    vwap: Optional[float] = None,
    vwap_slope: Optional[float] = None,
    rsi: Optional[float] = None,
    rsi_slope: Optional[float] = None,
    macd: Optional[Dict[str, Any]] = None,
    heiken_color: Optional[str] = None,
    heiken_count: Optional[int] = None,
    failed_vwap_reclaim: Optional[bool] = None
) -> Dict[str, Any]:
    """Score direction based on indicators."""
    up = 1
    down = 1

    if price is not None and vwap is not None:
        if price > vwap:
            up += 2
        if price < vwap:
            down += 2

    if vwap_slope is not None:
        if vwap_slope > 0:
            up += 2
        if vwap_slope < 0:
            down += 2

    if rsi is not None and rsi_slope is not None:
        if rsi > 55 and rsi_slope > 0:
            up += 2
        if rsi < 45 and rsi_slope < 0:
            down += 2

    if macd is not None:
        hist = macd.get("hist")
        hist_delta = macd.get("histDelta")
        macd_val = macd.get("macd")

        if hist is not None and hist_delta is not None:
            expanding_green = hist > 0 and hist_delta > 0
            expanding_red = hist < 0 and hist_delta < 0
            if expanding_green:
                up += 2
            if expanding_red:
                down += 2

        if macd_val is not None:
            if macd_val > 0:
                up += 1
            if macd_val < 0:
                down += 1

    if heiken_color is not None and heiken_count is not None:
        if heiken_color == "green" and heiken_count >= 2:
            up += 1
        if heiken_color == "red" and heiken_count >= 2:
            down += 1

    if failed_vwap_reclaim is True:
        down += 3

    raw_up = up / (up + down)

    return {"upScore": up, "downScore": down, "rawUp": raw_up}


def apply_time_awareness(
    raw_up: float,
    remaining_minutes: Optional[float],
    window_minutes: int
) -> Dict[str, Any]:
    """Apply time decay to probability."""
    if remaining_minutes is None:
        remaining_minutes = window_minutes

    time_decay = clamp(remaining_minutes / window_minutes, 0, 1)
    adjusted_up = clamp(0.5 + (raw_up - 0.5) * time_decay, 0, 1)

    return {
        "timeDecay": time_decay,
        "adjustedUp": adjusted_up,
        "adjustedDown": 1 - adjusted_up
    }
