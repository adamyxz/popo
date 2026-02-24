"""Heiken Ashi indicator."""

from typing import List, Dict, Any, Tuple


def compute_heiken_ashi(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute Heiken Ashi candles."""
    if not isinstance(candles, list) or len(candles) == 0:
        return []

    ha = []

    for i in range(len(candles)):
        c = candles[i]
        ha_close = (c["open"] + c["high"] + c["low"] + c["close"]) / 4

        if i > 0:
            prev = ha[i - 1]
            ha_open = (prev["open"] + prev["close"]) / 2
        else:
            ha_open = (c["open"] + c["close"]) / 2

        ha_high = max(c["high"], ha_open, ha_close)
        ha_low = min(c["low"], ha_open, ha_close)

        is_green = ha_close >= ha_open
        body = abs(ha_close - ha_open)

        ha.append({
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "isGreen": is_green,
            "body": body
        })

    return ha


def count_consecutive(ha_candles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count consecutive same-color Heiken Ashi candles."""
    if not isinstance(ha_candles, list) or len(ha_candles) == 0:
        return {"color": None, "count": 0}

    last = ha_candles[-1]
    target = "green" if last.get("isGreen", False) else "red"

    count = 0
    for i in range(len(ha_candles) - 1, -1, -1):
        c = ha_candles[i]
        color = "green" if c.get("isGreen", False) else "red"
        if color != target:
            break
        count += 1

    return {"color": target, "count": count}
