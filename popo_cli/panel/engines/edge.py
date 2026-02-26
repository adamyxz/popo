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
            "edgeDown": None,
            "expectedValueUp": None,
            "expectedValueDown": None
        }

    total = market_yes + market_no
    market_up = (market_yes / total) if total > 0 else None
    market_down = (market_no / total) if total > 0 else None

    edge_up = model_up - market_up if market_up is not None else None
    edge_down = model_down - market_down if market_down is not None else None

    # Calculate expected value (EV) considering odds
    # EV = (model_prob * payoff) - (1 - model_prob) * cost
    # For binary market: if market_price = 0.60, payoff = (1 - 0.60) / 0.60 = 0.67
    ev_up = None
    ev_down = None

    if market_up is not None and market_up > 0:
        # If we buy UP at market_up price
        # Win: get (1 - market_up) profit
        # Lose: lose market_up
        payoff_up = (1 - market_up) / market_up  # Decimal odds
        ev_up = (model_up * payoff_up) - ((1 - model_up) * 1)

    if market_down is not None and market_down > 0:
        payoff_down = (1 - market_down) / market_down
        ev_down = (model_down * payoff_down) - ((1 - model_down) * 1)

    return {
        "marketUp": clamp(market_up, 0, 1) if market_up is not None else None,
        "marketDown": clamp(market_down, 0, 1) if market_down is not None else None,
        "edgeUp": edge_up,
        "edgeDown": edge_down,
        "expectedValueUp": ev_up,
        "expectedValueDown": ev_down
    }


def decide(
    remaining_minutes: float,
    edge_up: Optional[float],
    edge_down: Optional[float],
    model_up: Optional[float] = None,
    model_down: Optional[float] = None,
    window_minutes: int = 5,
    market_up: Optional[float] = None,
    market_down: Optional[float] = None,
    ev_up: Optional[float] = None,
    ev_down: Optional[float] = None
) -> Dict[str, Any]:
    """
    Make trading decision with odds awareness.

    Key insight: Late in the candle, even if probability edge exists,
    the odds may be too poor to justify the trade.

    Example:
    - 30 seconds left
    - market_up = 0.95 (already very confident)
    - model_up = 0.98 (slightly better)
    - edge = 0.03 (3%)
    - BUT: buying at 0.95 means risking $0.95 to win $0.05
    - Expected value may still be positive, but risk/reward is terrible

    Args:
        remaining_minutes: Time left until candle closes
        edge_up: Model probability - market probability for UP
        edge_down: Model probability - market probability for DOWN
        model_up: Model's UP probability
        model_down: Model's DOWN probability
        window_minutes: Candle window duration (5 or 15 minutes)
        market_up: Market UP price (for odds checking)
        market_down: Market DOWN price (for odds checking)
        ev_up: Expected value for UP trade
        ev_down: Expected value for DOWN trade

    Returns:
        Decision dict with action, side, phase, strength, etc.
    """
    # Determine phase based on window_minutes
    if window_minutes == 5:
        # 5-minute candles
        if remaining_minutes > 3:
            phase = "EARLY"
        elif remaining_minutes > 1.5:
            phase = "MID"
        else:
            phase = "LATE"
    elif window_minutes == 15:
        # 15-minute candles
        if remaining_minutes > 10:
            phase = "EARLY"
        elif remaining_minutes > 5:
            phase = "MID"
        else:
            phase = "LATE"
    else:
        # Fallback for other window sizes
        ratio = remaining_minutes / window_minutes
        if ratio > 0.6:
            phase = "EARLY"
        elif ratio > 0.3:
            phase = "MID"
        else:
            phase = "LATE"

    if edge_up is None or edge_down is None:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": "missing_market_data"
        }

    # Determine best side
    best_side = "UP" if edge_up > edge_down else "DOWN"
    best_edge = edge_up if best_side == "UP" else edge_down
    best_model = model_up if best_side == "UP" else model_down
    best_market = market_up if best_side == "UP" else market_down
    best_ev = ev_up if best_side == "UP" else ev_down

    # LATE CHASE GUARD:
    # In the final seconds, avoid buying already expensive outcomes even when
    # edge/EV look positive. This protects against late fills with poor
    # risk/reward when the candle direction is already mostly decided.
    if window_minutes > 0:
        remaining_ratio = remaining_minutes / window_minutes
    else:
        remaining_ratio = 1.0

    if best_market is not None and best_model is not None:
        # Final 12% of window (~36s for 5m, ~108s for 15m): no chasing >=85c.
        if remaining_ratio <= 0.12 and best_market >= 0.85 and best_model >= 0.75:
            return {
                "action": "NO_TRADE",
                "side": None,
                "phase": phase,
                "reason": f"late_chase_block_price_{best_market:.2f}_prob_{best_model:.2f}"
            }

        # Final 10% of window: lock out near-certain directional chasing.
        if remaining_ratio <= 0.10 and best_model >= 0.97:
            return {
                "action": "NO_TRADE",
                "side": None,
                "phase": phase,
                "reason": f"late_extreme_prob_block_prob_{best_model:.2f}"
            }

    # ODDS AWARENESS: Check if odds are too poor to justify trade
    # Especially important in LATE phase when prices are skewed
    if best_market is not None:
        # If market price is too high, poor risk/reward
        if best_market > 0.90:
            # Market is already very confident (>90%)
            # Need significant edge and positive EV to trade
            min_edge_for_skewed = 0.08  # Need 8% edge
            min_ev = 0.05  # Need at least 5% EV

            if best_edge < min_edge_for_skewed:
                return {
                    "action": "NO_TRADE",
                    "side": None,
                    "phase": phase,
                    "reason": f"odds_too_poor_price_{best_market:.2f}_edge_{best_edge:.2f}"
                }

            if best_ev is not None and best_ev < min_ev:
                return {
                    "action": "NO_TRADE",
                    "side": None,
                    "phase": phase,
                    "reason": f"ev_too_low_{best_ev:.2f}_price_{best_market:.2f}"
                }

        elif best_market > 0.80:
            # Market is quite confident (>80%)
            # Still need good edge
            min_edge_for_skewed = 0.06  # Need 6% edge
            min_ev = 0.03  # Need at least 3% EV

            if best_edge < min_edge_for_skewed:
                return {
                    "action": "NO_TRADE",
                    "side": None,
                    "phase": phase,
                    "reason": f"odds_too_poor_price_{best_market:.2f}_edge_{best_edge:.2f}"
                }

            if best_ev is not None and best_ev < min_ev:
                return {
                    "action": "NO_TRADE",
                    "side": None,
                    "phase": phase,
                    "reason": f"ev_too_low_{best_ev:.2f}_price_{best_market:.2f}"
                }

    # PHASE-BASED THRESHOLDS
    # Late phase requires higher edge due to:
    # 1. Less time for new information
    # 2. Market prices already skew (poor odds)
    # 3. Higher execution risk
    if phase == "EARLY":
        threshold = 0.05
        min_prob = 0.55
    elif phase == "MID":
        threshold = 0.08
        min_prob = 0.60
    else:  # LATE
        threshold = 0.10  # Raised from 0.12
        min_prob = 0.62  # Slightly lower to allow trades

    # Apply basic thresholds
    if best_edge < threshold:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": f"edge_below_{threshold:.2f}"
        }

    if best_model is not None and best_model < min_prob:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": f"prob_below_{min_prob}"
        }

    # Expected value check (if available)
    if best_ev is not None and best_ev < 0:
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": phase,
            "reason": f"negative_ev_{best_ev:.2f}"
        }

    # Determine strength
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
        "edge": best_edge,
        "expectedValue": best_ev,
        "marketPrice": best_market
    }


def decide_stabilized(
    remaining_minutes: float,
    edge_up: Optional[float],
    edge_down: Optional[float],
    model_up: Optional[float] = None,
    model_down: Optional[float] = None,
    window_minutes: int = 5,
    market_up: Optional[float] = None,
    market_down: Optional[float] = None,
    ev_up: Optional[float] = None,
    ev_down: Optional[float] = None,
    signal_confirmed: bool = False,
    signal_action: Optional[str] = None
) -> Dict[str, Any]:
    """Make trading decision with signal stabilization awareness.

    This is an enhanced version of decide() that integrates with SignalStabilizer.
    It will only recommend ENTER when:
    1. All edge/odds/probability thresholds are met (via decide())
    2. AND the signal has been confirmed by the stabilizer

    Args:
        remaining_minutes: Time left until candle closes
        edge_up: Model probability - market probability for UP
        edge_down: Model probability - market probability for DOWN
        model_up: Model's UP probability
        model_down: Model's DOWN probability
        window_minutes: Candle window duration (5 or 15 minutes)
        market_up: Market UP price (for odds checking)
        market_down: Market DOWN price (for odds checking)
        ev_up: Expected value for UP trade
        ev_down: Expected value for DOWN trade
        signal_confirmed: Whether the signal has been confirmed by stabilizer
        signal_action: The current action from stabilizer ("BUY UP", "BUY DOWN", "NO TRADE", "WAIT X")

    Returns:
        Decision dict with action, side, phase, strength, etc.
    """
    # IMPORTANT: Check stabilizer signal action first
    # If stabilizer says "NO TRADE", respect it regardless of edge
    if signal_action == "NO TRADE":
        return {
            "action": "NO_TRADE",
            "side": None,
            "phase": "LATE" if remaining_minutes < 2 else "MID",
            "strength": None,
            "reason": "stabilizer_no_trade_zone",
            "edge": None,
            "expectedValue": None
        }

    # First, run the standard decision logic
    base_decision = decide(
        remaining_minutes=remaining_minutes,
        edge_up=edge_up,
        edge_down=edge_down,
        model_up=model_up,
        model_down=model_down,
        window_minutes=window_minutes,
        market_up=market_up,
        market_down=market_down,
        ev_up=ev_up,
        ev_down=ev_down
    )

    # If base decision is already NO_TRADE, return it
    if base_decision["action"] == "NO_TRADE":
        return base_decision

    # Check if signal is confirmed
    if not signal_confirmed:
        # Signal not yet confirmed, return WAIT
        return {
            "action": "NO_TRADE",
            "side": base_decision["side"],  # Still indicate which side we're watching
            "phase": base_decision["phase"],
            "strength": "PENDING",
            "reason": f"signal_not_confirmed_{signal_action}",
            "waiting_for": signal_action,
            "edge": base_decision["edge"],
            "expectedValue": base_decision.get("expectedValue"),
            "marketPrice": base_decision.get("marketPrice")
        }

    # Signal is confirmed and all thresholds met, return ENTER
    return {
        "action": "ENTER",
        "side": base_decision["side"],
        "phase": base_decision["phase"],
        "strength": base_decision["strength"],
        "edge": base_decision["edge"],
        "expectedValue": base_decision.get("expectedValue"),
        "marketPrice": base_decision.get("marketPrice"),
        "signal_confirmed": True
    }
