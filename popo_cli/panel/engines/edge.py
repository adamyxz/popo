"""Edge computation and decision engine."""

from typing import Dict, Any, Optional


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def compute_edge(
    model_up: float,
    model_down: float,
    market_yes: Optional[float],
    market_no: Optional[float]
) -> Dict[str, Any]:
    """Compute edge between model and market."""
    if market_yes is None or market_no is None:
        return {
            "marketUp": None,
            "marketDown": None,
            "edgeUp": None,
            "edgeDown": None
        }

    total = market_yes + market_no
    market_up = (market_yes / total) if total > 0 else None
    market_down = (market_no / total) if total > 0 else None

    edge_up = model_up - market_up if market_up is not None else None
    edge_down = model_down - market_down if market_down is not None else None

    return {
        "marketUp": clamp(market_up, 0, 1) if market_up is not None else None,
        "marketDown": clamp(market_down, 0, 1) if market_down is not None else None,
        "edgeUp": edge_up,
        "edgeDown": edge_down
    }


def decide(
    remaining_minutes: float,
    edge_up: Optional[float],
    edge_down: Optional[float],
    model_up: Optional[float] = None,
    model_down: Optional[float] = None
) -> Dict[str, Any]:
    """Make trading decision."""
    if remaining_minutes > 10:
        phase = "EARLY"
    elif remaining_minutes > 5:
        phase = "MID"
    else:
        phase = "LATE"

    threshold = {"EARLY": 0.05, "MID": 0.1, "LATE": 0.2}[phase]
    min_prob = {"EARLY": 0.55, "MID": 0.6, "LATE": 0.65}[phase]

    if edge_up is None or edge_down is None:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": "missing_market_data"
        }

    best_side = "UP" if edge_up > edge_down else "DOWN"
    best_edge = edge_up if best_side == "UP" else edge_down
    best_model = model_up if best_side == "UP" else model_down

    if best_edge < threshold:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": f"edge_below_{threshold}"
        }

    if best_model is not None and best_model < min_prob:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": f"prob_below_{min_prob}"
        }

    if best_edge >= 0.2:
        strength = "STRONG"
    elif best_edge >= 0.1:
        strength = "GOOD"
    else:
        strength = "OPTIONAL"

    return {
        "action": "ENTER",
        "side": best_side,
        "phase": phase,
        "strength": strength,
        "edge": best_edge
    }
