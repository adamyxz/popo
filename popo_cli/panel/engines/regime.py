"""Regime detection engine."""

from typing import Dict, Any, Optional


def detect_regime(
    price: Optional[float],
    vwap: Optional[float],
    vwap_slope: Optional[float],
    vwap_cross_count: Optional[int],
    volume_recent: Optional[float],
    volume_avg: Optional[float]
) -> Dict[str, Any]:
    """Detect market regime."""
    if price is None or vwap is None or vwap_slope is None:
        return {"regime": "CHOP", "reason": "missing_inputs"}

    above = price > vwap

    low_volume = (
        volume_recent is not None and
        volume_avg is not None and
        volume_recent < 0.6 * volume_avg
    )

    if low_volume and abs((price - vwap) / vwap) < 0.001:
        return {"regime": "CHOP", "reason": "low_volume_flat"}

    if above and vwap_slope > 0:
        return {"regime": "TREND_UP", "reason": "price_above_vwap_slope_up"}

    if not above and vwap_slope < 0:
        return {"regime": "TREND_DOWN", "reason": "price_below_vwap_slope_down"}

    if vwap_cross_count is not None and vwap_cross_count >= 3:
        return {"regime": "RANGE", "reason": "frequent_vwap_cross"}

    return {"regime": "RANGE", "reason": "default"}
