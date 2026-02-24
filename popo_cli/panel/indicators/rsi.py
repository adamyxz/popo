"""RSI (Relative Strength Index) indicator."""

from typing import List, Optional


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def compute_rsi(closes: List[float], period: int) -> Optional[float]:
    """Compute RSI."""
    if not isinstance(closes, list) or len(closes) < period + 1:
        return None

    gains = 0.0
    losses = 0.0

    for i in range(len(closes) - period, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        diff = cur - prev

        if diff > 0:
            gains += diff
        else:
            losses += -diff

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return clamp(rsi, 0, 100)


def sma(values: List[float], period: int) -> Optional[float]:
    """Compute Simple Moving Average."""
    if not isinstance(values, list) or len(values) < period:
        return None

    slice_vals = values[-period:]
    return sum(slice_vals) / period


def slope_last(values: List[float], points: int) -> Optional[float]:
    """Compute slope over last N points."""
    if not isinstance(values, list) or len(values) < points:
        return None

    slice_vals = values[-points:]
    first = slice_vals[0]
    last = slice_vals[-1]

    return (last - first) / (points - 1)
