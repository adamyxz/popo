"""MACD (Moving Average Convergence Divergence) indicator."""

from typing import List, Dict, Any, Optional


def ema(values: List[float], period: int) -> Optional[float]:
    """Compute Exponential Moving Average."""
    if not isinstance(values, list) or len(values) < period:
        return None

    k = 2 / (period + 1)
    prev = values[0]

    for i in range(1, len(values)):
        prev = values[i] * k + prev * (1 - k)

    return prev


def compute_macd(closes: List[float], fast: int, slow: int, signal: int) -> Optional[Dict[str, Any]]:
    """Compute MACD."""
    if not isinstance(closes, list) or len(closes) < slow + signal:
        return None

    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)

    if fast_ema is None or slow_ema is None:
        return None

    macd_line = fast_ema - slow_ema

    # Compute MACD series for signal line
    macd_series = []
    for i in range(len(closes)):
        sub = closes[:i + 1]
        f = ema(sub, fast)
        s = ema(sub, slow)
        if f is None or s is None:
            continue
        macd_series.append(f - s)

    signal_line = ema(macd_series, signal)
    if signal_line is None:
        return None

    hist = macd_line - signal_line

    # Compute previous histogram
    prev_hist = None
    if len(macd_series) >= signal + 1:
        prev_macd_series = macd_series[:-1]
        prev_signal = ema(prev_macd_series, signal)
        if prev_signal is not None:
            prev_hist = macd_series[-1] - prev_signal

    hist_delta = None if prev_hist is None else hist - prev_hist

    return {
        "macd": macd_line,
        "signal": signal_line,
        "hist": hist,
        "histDelta": hist_delta
    }
