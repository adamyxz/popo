"""Predict current candle closing direction using volume-weighted signals.

This module predicts whether the current forming candle will close above or below
its open price (green/red), using:
1. Current candle internal data (price position, volume profile)
2. Order book imbalance
3. Trade flow (aggressive buy/sell volume)
4. Recent candle context (trend analysis)
5. Confidence-based probability smoothing
6. Adaptive volatility-based sensitivity
"""

from typing import Dict, Any, Optional, List
from .probability import clamp


def calculate_atr(prices: List[float], period: int = 14) -> float:
    """Calculate Average True Range for volatility estimation.

    Args:
        prices: List of closing prices
        period: ATR period (default 14)

    Returns:
        ATR value as a percentage of average price
    """
    if len(prices) < period + 1:
        # Fallback to standard deviation if not enough data
        if len(prices) < 2:
            return 0.005  # 0.5% default
        avg_price = sum(prices) / len(prices)
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        return (std_dev / avg_price) if avg_price > 0 else 0.005

    true_ranges = []
    for i in range(1, len(prices)):
        high_low = abs(prices[i] - prices[i-1])
        true_ranges.append(high_low / prices[i-1])

    atr = sum(true_ranges[-period:]) / min(period, len(true_ranges))
    return atr


def calculate_volatility_adjustment(
    prev_closes: Optional[List[float]],
    default_volatility: float = 0.005
) -> float:
    """Calculate price signal sensitivity based on historical volatility.

    Higher volatility → lower sensitivity (avoid overfitting noise)
    Lower volatility → higher sensitivity (capture small movements)

    Args:
        prev_closes: Recent closing prices
        default_volatility: Default if no data (0.5%)

    Returns:
        Sensitivity multiplier (typically 0.5 to 2.0)
    """
    if not prev_closes or len(prev_closes) < 3:
        return 1.0  # Default sensitivity

    # Calculate recent volatility
    atr_pct = calculate_atr(prev_closes, period=min(14, len(prev_closes)))

    # Base sensitivity on 0.5% volatility = 1.0x
    # If volatility is 1.0%, sensitivity = 0.5x
    # If volatility is 0.25%, sensitivity = 2.0x
    base_volatility = 0.005  # 0.5%
    sensitivity = base_volatility / max(atr_pct, 0.001)

    # Clamp to reasonable range
    return clamp(sensitivity, 0.3, 3.0)


def predict_candle_direction(
    # Current candle data
    open_price: Optional[float] = None,
    current_price: Optional[float] = None,
    elapsed_ratio: Optional[float] = None,  # 0.0 to 1.0
    window_minutes: int = 5,  # Candle window (5 or 15 minutes)

    # PRICE REALITY CHECK: Override unrealistic late-stage predictions
    # When time is short and price has moved significantly, respect the price action
    enable_price_reality_check: bool = True,  # Enable/disable the check

    # Volume data
    buy_volume: Optional[float] = None,  # Aggressive buy volume in current candle
    sell_volume: Optional[float] = None,  # Aggressive sell volume in current candle
    total_volume: Optional[float] = None,  # Total volume in current candle

    # Order book
    bid_volume: Optional[float] = None,  # Total bid volume (top 5 levels)
    ask_volume: Optional[float] = None,  # Total ask volume (top 5 levels)

    # Recent context (optional)
    prev_closes: Optional[list[float]] = None,  # Last N candles' closes
) -> Dict[str, Any]:
    """
    Predict if current candle will close above (green) or below (red) open.

    Core logic:
    1. Price position: Where is current price relative to open? (30%)
    2. Volume pressure: Buy vs sell volume ratio (40% - PRIMARY)
    3. Order book imbalance: Bid vs ask depth (30%)
    4. Trend context: Consecutive candle directions (10% - NEW)
    5. Time weight: Progressive (more weight as candle progresses)
    6. Confidence smoothing: Avoid overconfident predictions on weak signals

    Args:
        window_minutes: Candle window duration (5 or 15 minutes).
                      Used to calculate absolute time-based probability bounds.

    Returns:
        {
            "up": float,  # Probability candle closes green
            "down": float,  # Probability candle closes red
            "signals": { ... },  # Component signals for debugging
            "confidence": float  # Overall signal strength (0-1)
        }
    """
    up_score = 0.0
    down_score = 0.0
    signals = {}

    # Calculate adaptive sensitivity based on historical volatility
    volatility_adjustment = calculate_volatility_adjustment(prev_closes)
    signals["volatilityAdjustment"] = volatility_adjustment

    # 1. Price position signal (0.25 weight max - REDUCED from 0.3) - LESS SENSITIVE
    if open_price is not None and current_price is not None:
        price_change_pct = (current_price - open_price) / open_price

        # Adaptive normalization: use volatility to determine sensitivity
        # CHANGED: More conservative baseline - 1.0% change = 0.33 signal (was 0.5%)
        # CHANGED: Reduced base_sensitivity from 200 to 100
        # CHANGED: Tighter clamp on volatility_adjustment
        clamped_volatility = clamp(volatility_adjustment, 0.5, 1.5)  # Limit adjustment range
        base_sensitivity = 100.0  # 1 / 0.01 = 100 for 1.0% baseline (was 200)
        raw_signal = price_change_pct * base_sensitivity * clamped_volatility
        price_signal = clamp(raw_signal, -0.8, 0.8)  # Reduced from -1.0/1.0

        if price_signal > 0:
            up_score += price_signal * 0.25  # Reduced from 0.3
        else:
            down_score += abs(price_signal) * 0.25  # Reduced from 0.3

        signals["priceSignal"] = price_signal
        signals["priceChangePct"] = price_change_pct

    # 2. Volume pressure signal (0.15 weight max - FURTHER REDUCED from 0.25)
    # Heavily reduced to prevent signal whipsaw from instantaneous volume changes
    # Volume is the NOISIEST signal - should have minimal weight
    if buy_volume is not None and sell_volume is not None:
        vol_sum = buy_volume + sell_volume
        if vol_sum > 0:
            # Volume imbalance: +1 for all buy, -1 for all sell
            vol_pressure = (buy_volume - sell_volume) / vol_sum

            if vol_pressure > 0:
                up_score += vol_pressure * 0.15  # Reduced from 0.25 (was 0.4)
            else:
                down_score += abs(vol_pressure) * 0.15  # Reduced from 0.25

            signals["volumePressure"] = vol_pressure

    # 3. Order book imbalance (0.35 weight max - INCREASED from 0.3)
    # OB is more stable than volume - give it slightly more weight
    if bid_volume is not None and ask_volume is not None:
        ob_sum = bid_volume + ask_volume
        if ob_sum > 0:
            # OB imbalance: +1 for bid-heavy, -1 for ask-heavy
            ob_imbalance = (bid_volume - ask_volume) / ob_sum

            if ob_imbalance > 0:
                up_score += ob_imbalance * 0.35  # Increased from 0.3
            else:
                down_score += abs(ob_imbalance) * 0.35  # Increased from 0.3

            signals["obImbalance"] = ob_imbalance

    # 4. Trend context signal (0.20 weight max - INCREASED from 0.1)
    # Use consecutive candle directions to add momentum context
    # Trend is more stable than instantaneous signals - increased weight
    if prev_closes is not None and len(prev_closes) >= 3:
        recent_changes = []
        for i in range(1, len(prev_closes)):
            if prev_closes[i] > 0 and prev_closes[i-1] > 0:
                change = (prev_closes[i] - prev_closes[i-1]) / prev_closes[i-1]
                recent_changes.append(change)

        if recent_changes:
            # Count consecutive ups/downs
            consecutive_up = sum(1 for c in recent_changes if c > 0)
            consecutive_down = sum(1 for c in recent_changes if c < 0)

            # Trend signal: add weight for momentum (increased from 0.1 to 0.2)
            trend_weight = 0.20
            if consecutive_up >= 2:
                trend_strength = min(consecutive_up / len(recent_changes), 1.0)
                up_score += trend_weight * trend_strength
                signals["trendSignal"] = trend_strength
            elif consecutive_down >= 2:
                trend_strength = min(consecutive_down / len(recent_changes), 1.0)
                down_score += trend_weight * trend_strength
                signals["trendSignal"] = -trend_strength

    # 5. Time weight: Progressive
    # - First 50% of candle: weight ramps from 0.2 to 1.0
    # - After 50%: weight stays at 1.0 (signals are reliable)
    time_weight = 0.2  # Minimum weight even at start
    if elapsed_ratio is not None:
        if elapsed_ratio < 0.5:
            # Ramp from 0.2 to 1.0 in first half
            time_weight = 0.2 + (elapsed_ratio / 0.5) * 0.8
        else:
            time_weight = 1.0
    else:
        time_weight = 1.0  # Default to full weight if no time data

    signals["timeWeight"] = time_weight
    if elapsed_ratio is not None:
        signals["elapsedRatio"] = elapsed_ratio

    # Apply time weight
    up_score *= time_weight
    down_score *= time_weight

    # Calculate raw signal strength (based on weighted scores, NOT final ratio)
    # We use the absolute score magnitude, not the ratio
    total_weighted_score = abs(up_score) + abs(down_score)

    # Max possible weighted score depends on time_weight:
    # UPDATED WEIGHTS:
    # At full weight (1.0): max = 0.25 (price) + 0.15 (volume) + 0.35 (OB) + 0.20 (trend) = 0.95
    # At min weight (0.2): max = 0.95 * 0.2 = 0.19
    max_possible_at_time = 0.95 * time_weight

    # IMPORTANT: Signal strength should measure directional conviction, not total activity
    # Use the net score (difference between up and down) relative to max possible
    # This prevents weak bidirectional signals from appearing confident
    net_score = max(up_score, down_score) - min(up_score, down_score)
    signal_strength = net_score / max_possible_at_time if max_possible_at_time > 0 else 0

    # Convert to probability
    if total_weighted_score == 0:
        # No signal, return neutral
        return {
            "up": 0.5,
            "down": 0.5,
            "signals": signals,
            "confidence": 0.0
        }

    raw_up = up_score / total_weighted_score

    # 6. Confidence-based probability smoothing - VERY AGGRESSIVE
    # Weak signals should pull VERY strongly toward neutral (0.5)
    # The idea: if total signal strength is low, don't be too confident
    # CHANGED: Increased thresholds and made pull even stronger
    if signal_strength < 0.20:
        # Extremely weak signal: overwhelming pull to neutral (10% signal, 90% neutral)
        raw_up = raw_up * 0.10 + 0.5 * 0.90
    elif signal_strength < 0.35:
        # Very weak signal: very strong pull to neutral (20% signal, 80% neutral)
        raw_up = raw_up * 0.20 + 0.5 * 0.80
    elif signal_strength < 0.55:
        # Weak signal: strong pull to neutral (40% signal, 60% neutral)
        raw_up = raw_up * 0.40 + 0.5 * 0.60

    # 7. Time-based probability bounds (FIXED: use absolute time, not ratio)
    # Early in the candle, don't allow extreme probabilities
    if elapsed_ratio is not None:
        # Calculate elapsed time in minutes
        elapsed_minutes = elapsed_ratio * window_minutes

        if elapsed_minutes < 1.0:
            # First 1 minute: keep probability between 35-65%
            raw_up = clamp(raw_up, 0.35, 0.65)
        elif elapsed_minutes < 2.0:
            # 1-2 minutes: keep probability between 25-75%
            raw_up = clamp(raw_up, 0.25, 0.75)
        elif elapsed_minutes < window_minutes * 0.5:
            # 2 minutes to halfway: keep probability between 15-85%
            raw_up = clamp(raw_up, 0.15, 0.85)
        elif elapsed_minutes < window_minutes * 0.75:
            # Halfway to 75%: keep probability between 10-90%
            raw_up = clamp(raw_up, 0.10, 0.90)

    # Final clamp to ensure valid probability
    raw_up = clamp(raw_up, 0.05, 0.95)

    # PRICE REALITY CHECK: Override unrealistic late-stage predictions
    # When time is short and price has moved significantly, respect the price action
    if enable_price_reality_check:
        if elapsed_ratio is not None and open_price is not None and current_price is not None:
            elapsed_minutes = elapsed_ratio * window_minutes
            price_change_pct = (current_price - open_price) / open_price

            # In the last 1.5 minutes, if price has moved >0.08%, strongly bias toward actual direction
            # Lowered threshold from 0.15% to catch smaller but decisive moves
            if elapsed_minutes >= (window_minutes - 1.5):  # Last 1.5 minutes
                threshold = 0.0008  # 0.08% move (lowered to catch user's scenario)

                if price_change_pct > threshold:
                    # Price is clearly UP, override to at least 70% UP
                    raw_up = max(raw_up, 0.70)
                    signals["priceRealityCheck"] = "late_up_override"
                elif price_change_pct < -threshold:
                    # Price is clearly DOWN, override to at most 30% UP (70% DOWN)
                    raw_up = min(raw_up, 0.30)
                    signals["priceRealityCheck"] = "late_down_override"

            # In the last 3 minutes, if price has moved >0.15%, moderate bias
            elif elapsed_minutes >= (window_minutes - 3.0):  # Last 3 minutes
                threshold = 0.0015  # 0.15% move

                if price_change_pct > threshold:
                    # Price is clearly UP, bias toward UP
                    raw_up = max(raw_up, 0.60)
                    signals["priceRealityCheck"] = "late_up_bias"
                elif price_change_pct < -threshold:
                    # Price is clearly DOWN, bias toward DOWN
                    raw_up = min(raw_up, 0.40)
                    signals["priceRealityCheck"] = "late_down_bias"

    return {
        "up": raw_up,
        "down": 1 - raw_up,
        "signals": signals,
        "confidence": clamp(signal_strength, 0, 1)
    }


def classify_aggressive_trades(trades: list[Dict[str, Any]], bid_price: float, ask_price: float) -> Dict[str, float]:
    """
    Classify trades as aggressive buy or sell based on execution price.

    Aggressive buy: Trade at or above ask
    Aggressive sell: Trade at or below bid

    Args:
        trades: List of trade dicts with 'price' and 'quantity'
        bid_price: Current best bid price
        ask_price: Current best ask price

    Returns:
        {"buyVolume": float, "sellVolume": float}
    """
    buy_vol = 0.0
    sell_vol = 0.0

    for trade in trades:
        price = trade.get("price", 0)
        qty = trade.get("qty", 0)

        if price >= ask_price:
            buy_vol += qty
        elif price <= bid_price:
            sell_vol += qty

    return {"buyVolume": buy_vol, "sellVolume": sell_vol}


def calculate_order_book_imbalance(depth_snapshot: Dict[str, Any], levels: int = 5) -> Dict[str, float]:
    """
    Calculate order book imbalance from depth snapshot.

    Args:
        depth_snapshot: Order book snapshot with 'bids' and 'asks'
        levels: Number of levels to consider

    Returns:
        {"bidVolume": float, "askVolume": float, "imbalance": float}
    """
    bids = depth_snapshot.get("bids", [])[:levels]
    asks = depth_snapshot.get("asks", [])[:levels]

    bid_vol = sum(float(b[1]) for b in bids if len(b) > 1)
    ask_vol = sum(float(a[1]) for a in asks if len(a) > 1)

    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0

    return {
        "bidVolume": bid_vol,
        "askVolume": ask_vol,
        "imbalance": imbalance
    }


def predict_next_candle_direction(
    # Technical indicators
    rsi: Optional[float] = None,
    rsi_slope: Optional[float] = None,
    macd: Optional[Dict[str, Any]] = None,
    vwap: Optional[float] = None,
    vwap_slope: Optional[float] = None,
    heiken_color: Optional[str] = None,
    heiken_count: Optional[int] = None,

    # Order book
    bid_volume: Optional[float] = None,
    ask_volume: Optional[float] = None,

    # Historical context
    prev_closes: Optional[list[float]] = None,

    # Market regime (for dynamic weight adjustment)
    regime: Optional[str] = None,  # "TREND_UP", "TREND_DOWN", "RANGE", "CHOP"
) -> Dict[str, Any]:
    """
    Predict the NEXT candle's closing direction using technical indicators.

    Unlike predict_candle_direction() which uses real-time candle data,
    this function predicts the upcoming candle that hasn't opened yet.

    Core logic with DYNAMIC weights based on market regime:
    1. MACD trend signal (20-35% depending on regime)
    2. RSI momentum (15-30% depending on regime)
    3. VWAP trend (15-25% depending on regime)
    4. Heiken Ashi pattern (10-20%)
    5. Order book imbalance (10-15%)
    6. Price momentum (10% - from recent closes)

    Dynamic weight adjustment:
    - TREND_UP/TREND_DOWN: Increase MACD and Heiken Ashi weight (trend following)
    - RANGE: Increase RSI and OB weight (mean reversion)
    - CHOP: Reduce all weights, stay neutral

    Args:
        rsi: RSI value (0-100)
        rsi_slope: RSI slope (positive/negative trend)
        macd: Dict with 'hist', 'histDelta', 'macd' values
        vwap: VWAP value
        vwap_slope: VWAP slope direction
        heiken_color: Current Heiken Ashi color ("green" or "red")
        heiken_count: Consecutive Heiken Ashi count
        bid_volume: Total bid volume
        ask_volume: Total ask volume
        prev_closes: Recent closing prices for momentum
        regime: Market regime for dynamic weight adjustment

    Returns:
        {
            "up": float,  # Probability next candle closes green
            "down": float,  # Probability next candle closes red
            "signals": { ... },  # Component signals
            "confidence": float,  # Overall signal strength
            "dominant_signal": str  # Name of strongest signal
        }
    """
    up_score = 0.0
    down_score = 0.0
    signals = {}

    # Determine dynamic weights based on regime
    if regime == "TREND_UP" or regime == "TREND_DOWN":
        # Strong trend: favor trend-following indicators
        macd_weight = 0.35
        rsi_weight = 0.15
        vwap_weight = 0.20
        heiken_weight = 0.15
        ob_weight = 0.10
        momentum_weight = 0.05
    elif regime == "RANGE":
        # Range-bound: favor mean-reversion indicators
        macd_weight = 0.20
        rsi_weight = 0.30
        vwap_weight = 0.15
        heiken_weight = 0.10
        ob_weight = 0.15
        momentum_weight = 0.10
    elif regime == "CHOP":
        # Choppy market: reduce conviction
        macd_weight = 0.15
        rsi_weight = 0.20
        vwap_weight = 0.15
        heiken_weight = 0.10
        ob_weight = 0.10
        momentum_weight = 0.05
    else:
        # Default balanced weights
        macd_weight = 0.25
        rsi_weight = 0.20
        vwap_weight = 0.18
        heiken_weight = 0.12
        ob_weight = 0.15
        momentum_weight = 0.10

    signals["regime"] = regime or "UNKNOWN"
    signals["weights"] = {
        "macd": macd_weight,
        "rsi": rsi_weight,
        "vwap": vwap_weight,
        "heiken": heiken_weight,
        "ob": ob_weight,
        "momentum": momentum_weight
    }

    # Track signal strengths for dominant signal detection
    signal_strengths = {}

    # 1. MACD Trend Signal
    if macd is not None:
        hist = macd.get("hist")
        hist_delta = macd.get("histDelta")
        macd_val = macd.get("macd")

        macd_signal = 0.0
        if hist is not None:
            # Histogram direction is primary
            if hist > 0:
                macd_signal += 0.5
                if hist_delta is not None and hist_delta > 0:
                    macd_signal += 0.3  # Expanding green
            elif hist < 0:
                macd_signal -= 0.5
                if hist_delta is not None and hist_delta < 0:
                    macd_signal -= 0.3  # Expanding red

        if macd_val is not None:
            # MACD above/below zero line
            if macd_val > 0:
                macd_signal += 0.2
            else:
                macd_signal -= 0.2

        if macd_signal > 0:
            up_score += macd_signal * macd_weight
        else:
            down_score += abs(macd_signal) * macd_weight

        signals["macdSignal"] = macd_signal
        signal_strengths["MACD"] = abs(macd_signal)

    # 2. RSI Momentum Signal
    if rsi is not None:
        rsi_signal = 0.0

        # RSI level interpretation
        if rsi > 60:
            # Bullish zone
            rsi_signal += 0.3
            if rsi > 70:
                rsi_signal += 0.2  # Strong bullish
        elif rsi < 40:
            # Bearish zone
            rsi_signal -= 0.3
            if rsi < 30:
                rsi_signal -= 0.2  # Strong bearish

        # RSI slope (momentum)
        if rsi_slope is not None:
            if rsi_slope > 0:
                rsi_signal += 0.2
            else:
                rsi_signal -= 0.2

        if rsi_signal > 0:
            up_score += rsi_signal * rsi_weight
        else:
            down_score += abs(rsi_signal) * rsi_weight

        signals["rsiSignal"] = rsi_signal
        signal_strengths["RSI"] = abs(rsi_signal)

    # 3. VWAP Trend Signal
    if vwap_slope is not None:
        vwap_signal = 0.0

        if vwap_slope > 0:
            vwap_signal = 0.5
            # Stronger slope = stronger signal
            vwap_signal = min(vwap_signal * abs(vwap_slope) * 100, 1.0)
        elif vwap_slope < 0:
            vwap_signal = -0.5
            vwap_signal = max(vwap_signal * abs(vwap_slope) * 100, -1.0)

        if vwap_signal > 0:
            up_score += vwap_signal * vwap_weight
        else:
            down_score += abs(vwap_signal) * vwap_weight

        signals["vwapSignal"] = vwap_signal
        signal_strengths["VWAP"] = abs(vwap_signal)

    # 4. Heiken Ashi Pattern Signal
    if heiken_color is not None:
        heiken_signal = 0.0

        if heiken_color == "green":
            heiken_signal = 0.5
            # Consecutive greens add confidence
            if heiken_count is not None and heiken_count >= 2:
                heiken_signal += 0.3 * min(heiken_count / 5, 1.0)
        elif heiken_color == "red":
            heiken_signal = -0.5
            if heiken_count is not None and heiken_count >= 2:
                heiken_signal -= 0.3 * min(heiken_count / 5, 1.0)

        if heiken_signal > 0:
            up_score += heiken_signal * heiken_weight
        else:
            down_score += abs(heiken_signal) * heiken_weight

        signals["heikenSignal"] = heiken_signal
        signal_strengths["Heiken"] = abs(heiken_signal)

    # 5. Order Book Imbalance Signal
    if bid_volume is not None and ask_volume is not None:
        ob_total = bid_volume + ask_volume
        if ob_total > 0:
            ob_imbalance = (bid_volume - ask_volume) / ob_total
            ob_signal = ob_imbalance

            if ob_signal > 0:
                up_score += ob_signal * ob_weight
            else:
                down_score += abs(ob_signal) * ob_weight

            signals["obSignal"] = ob_signal
            signal_strengths["OB"] = abs(ob_signal)

    # 6. Price Momentum Signal (from recent closes)
    if prev_closes is not None and len(prev_closes) >= 3:
        momentum_signal = 0.0

        # Calculate recent price changes
        changes = []
        for i in range(1, len(prev_closes)):
            if prev_closes[i] > 0 and prev_closes[i-1] > 0:
                change = (prev_closes[i] - prev_closes[i-1]) / prev_closes[i-1]
                changes.append(change)

        if changes:
            # Count consecutive ups/downs
            consecutive_up = sum(1 for c in changes if c > 0)
            consecutive_down = sum(1 for c in changes if c < 0)

            # Momentum strength based on consistency
            if consecutive_up > consecutive_down:
                momentum_signal = min(consecutive_up / len(changes), 1.0)
            elif consecutive_down > consecutive_up:
                momentum_signal = -min(consecutive_down / len(changes), 1.0)

        if momentum_signal > 0:
            up_score += momentum_signal * momentum_weight
        else:
            down_score += abs(momentum_signal) * momentum_weight

        signals["momentumSignal"] = momentum_signal
        signal_strengths["Momentum"] = abs(momentum_signal)

    # Calculate total weighted score
    total_score = up_score + down_score

    if total_score == 0:
        return {
            "up": 0.5,
            "down": 0.5,
            "signals": signals,
            "confidence": 0.0,
            "dominant_signal": "None"
        }

    # Convert to probability
    raw_up = up_score / total_score

    # Confidence-based probability smoothing
    # Calculate max possible score
    max_possible = macd_weight + rsi_weight + vwap_weight + heiken_weight + ob_weight + momentum_weight
    signal_strength = total_score / max_possible

    # Apply smoothing based on confidence
    if signal_strength < 0.25:
        raw_up = raw_up * 0.2 + 0.5 * 0.8
    elif signal_strength < 0.45:
        raw_up = raw_up * 0.4 + 0.5 * 0.6

    # Clamp to valid range
    raw_up = clamp(raw_up, 0.15, 0.85)  # Slightly tighter than current candle

    # Determine dominant signal
    dominant_signal = "None"
    if signal_strengths:
        dominant_signal = max(signal_strengths, key=signal_strengths.get)

    return {
        "up": raw_up,
        "down": 1 - raw_up,
        "signals": signals,
        "confidence": clamp(signal_strength, 0, 1),
        "dominant_signal": dominant_signal
    }
