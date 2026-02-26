"""Decision engines for panel module."""

from .regime import detect_regime
from .edge import compute_edge, decide, decide_stabilized
from .candle_direction import (
    predict_candle_direction,
    predict_next_candle_direction,
    classify_aggressive_trades,
    calculate_order_book_imbalance
)
from .signal_stabilizer import (
    SignalStabilizer,
    stabilize_prediction
)

__all__ = [
    "detect_regime",
    "compute_edge",
    "decide",
    "decide_stabilized",
    "predict_candle_direction",
    "predict_next_candle_direction",
    "classify_aggressive_trades",
    "calculate_order_book_imbalance",
    "SignalStabilizer",
    "stabilize_prediction",
]

# DEPRECATED: Old probability scoring system
# The system now uses predict_candle_direction as the unified probability engine
# from .probability import score_direction, apply_time_awareness

