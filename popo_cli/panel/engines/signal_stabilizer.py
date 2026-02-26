"""Signal stabilization module to reduce noise and prevent signal whipsaw.

This module implements:
1. EMA-based signal smoothing
2. Signal confirmation mechanism (persistence threshold)
3. Reduced volume pressure impact
4. No-trade zone for weak signals
"""

from typing import Dict, Any, Optional, List
from collections import deque
import time


class SignalStabilizer:
    """Smooth and validate trading signals to prevent whipsaw."""

    # Configuration - CONSERVATIVE SETTINGS to reduce whipsaw
    EMA_ALPHA = 0.15  # EMA smoothing factor (lower = more smoothing) - reduced from 0.3
    CONFIRMATION_SECONDS = 35  # Minimum seconds signal must persist - increased from 20
    CONFIRMATION_COUNT = 5  # Minimum consecutive signals in same direction - increased from 3
    NO_TRADE_ZONE = (0.40, 0.60)  # Probability range where we don't trade - widened from (0.45, 0.55)

    # Direction lock: once confirmed, requires stronger signal to reverse
    DIRECTION_LOCK_ENABLED = True
    REVERSAL_THRESHOLD_MULTIPLIER = 1.5  # Need 1.5x stronger signal to reverse confirmed direction

    def __init__(self, window_minutes: int = 5):
        """Initialize signal stabilizer.

        Args:
            window_minutes: Candle window duration for timing calculations
        """
        self.window_minutes = window_minutes

        # Signal history for EMA smoothing
        self.ema_up = 0.5  # Start neutral
        self.ema_down = 0.5

        # Signal confirmation tracking
        self.signal_history: deque = deque(maxlen=10)  # Last 10 signals
        self.current_direction: Optional[str] = None  # "UP", "DOWN", or None
        self.direction_start_time: Optional[float] = None
        self.consecutive_count = 0

        # Confirmed signal (only changes when confirmation criteria met)
        self.confirmed_signal: Optional[str] = None  # "BUY UP", "BUY DOWN", or "NO TRADE"

        # Direction lock: prevent premature reversal of confirmed signals
        self.locked_direction: Optional[str] = None  # "UP" or "DOWN" once confirmed
        self.lock_start_time: Optional[float] = None
        self.lock_strength_at_lock: float = 0.0  # Signal strength when direction was locked

    def _apply_ema_smoothing(self, raw_up: float, raw_down: float) -> tuple[float, float]:
        """Apply exponential moving average to smooth signal transitions.

        Formula: EMA = α * new_value + (1 - α) * previous_EMA

        CHANGES:
        - Extreme signal threshold changed to <=10% or >=90% (stricter)
        - Strong signal threshold changed to <=20% or >=80% (stricter)
        - Strong signals now use 0.4 alpha (slower than before 0.6)

        Args:
            raw_up: Raw UP probability from prediction engine
            raw_down: Raw DOWN probability from prediction engine

        Returns:
            Smoothed (up, down) probabilities
        """
        # Extreme signals: bypass smoothing entirely - trust the prediction
        # Changed from 15%/85% to 10%/90% for more aggressive filtering
        if raw_up <= 0.10 or raw_up >= 0.90:
            # Use raw value directly for extreme signals
            self.ema_up = raw_up
            self.ema_down = raw_down
            return raw_up, raw_down

        # Strong signals: use faster alpha but still conservative
        # Changed from 25%/75% to 20%/80%
        # Changed alpha from 0.6 to 0.4 for more smoothing
        alpha = self.EMA_ALPHA
        if raw_up <= 0.20 or raw_up >= 0.80:
            alpha = 0.4  # Reduced from 0.6 for more smoothing

        # Apply EMA to both probabilities
        self.ema_up = alpha * raw_up + (1 - alpha) * self.ema_up
        self.ema_down = alpha * raw_down + (1 - alpha) * self.ema_down

        # Renormalize to ensure sum = 1
        total = self.ema_up + self.ema_down
        if total > 0:
            self.ema_up /= total
            self.ema_down /= total

        return self.ema_up, self.ema_down

    def _check_no_trade_zone(self, up_prob: float) -> bool:
        """Check if probability is in the no-trade zone.

        Args:
            up_prob: UP probability

        Returns:
            True if in no-trade zone
        """
        lower, upper = self.NO_TRADE_ZONE
        return lower <= up_prob <= upper

    def _update_confirmation_tracking(self, raw_up: float, raw_down: float):
        """Update signal confirmation tracking.

        A signal is only confirmed when:
        1. It has been consistent for CONFIRMATION_SECONDS
        2. OR it has appeared CONFIRMATION_COUNT times consecutively

        DIRECTION LOCK: Once a direction is confirmed, it requires a stronger
        reverse signal to change direction (prevents whipsaw).

        Args:
            raw_up: Raw UP probability
            raw_down: Raw DOWN probability
        """
        current_time = time.time()

        # Determine current raw direction (wider threshold for NO_TRADE_ZONE)
        # Use NO_TRADE_ZONE bounds: if up < 0.40 -> DOWN, if up > 0.60 -> UP
        new_direction = None
        signal_strength = 0.0

        if raw_up > 0.60:
            new_direction = "UP"
            signal_strength = raw_up - 0.5  # Distance from neutral
        elif raw_up < 0.40:
            new_direction = "DOWN"
            signal_strength = 0.5 - raw_up  # Distance from neutral
        else:
            new_direction = "NEUTRAL"
            signal_strength = 0.0

        # DIRECTION LOCK: Check if we should resist direction change
        if self.DIRECTION_LOCK_ENABLED and self.locked_direction is not None:
            if new_direction != self.locked_direction and new_direction != "NEUTRAL":
                # Attempting to reverse - check if signal is strong enough
                # Need 1.5x the strength that caused the original lock
                required_strength = self.lock_strength_at_lock * self.REVERSAL_THRESHOLD_MULTIPLIER

                if signal_strength < required_strength:
                    # Signal not strong enough to reverse - keep locked direction
                    new_direction = self.locked_direction

        # Add to history
        self.signal_history.append({
            "direction": new_direction,
            "timestamp": current_time,
            "up": raw_up,
            "down": raw_down,
            "strength": signal_strength
        })

        # Check if direction changed
        if new_direction != self.current_direction:
            # Direction changed, reset tracking
            self.current_direction = new_direction
            self.direction_start_time = current_time
            self.consecutive_count = 1
        else:
            # Same direction, increment counter
            self.consecutive_count += 1

        # Update direction lock when signal is confirmed
        if new_direction in ["UP", "DOWN"] and signal_strength > 0:
            if self.locked_direction != new_direction:
                # New direction being locked
                self.locked_direction = new_direction
                self.lock_start_time = current_time
                self.lock_strength_at_lock = signal_strength

    def _is_signal_confirmed(self) -> tuple[bool, Optional[str]]:
        """Check if current signal meets confirmation criteria.

        Returns:
            (is_confirmed, signal_direction)
        """
        if not self.current_direction or self.current_direction == "NEUTRAL":
            return False, "NO TRADE"

        # Check time-based confirmation
        if self.direction_start_time is not None:
            elapsed = time.time() - self.direction_start_time
            if elapsed >= self.CONFIRMATION_SECONDS:
                return True, f"BUY {self.current_direction}"

        # Check count-based confirmation
        if self.consecutive_count >= self.CONFIRMATION_COUNT:
            return True, f"BUY {self.current_direction}"

        # Not confirmed yet
        return False, f"WAIT {self.current_direction}"

    def process(
        self,
        raw_prediction: Dict[str, Any],
        elapsed_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process a raw prediction and return stabilized signal.

        Args:
            raw_prediction: Raw prediction from candle_direction engine
                {
                    "up": float,
                    "down": float,
                    "signals": {...},
                    "confidence": float
                }
            elapsed_ratio: Optional elapsed time ratio (0.0 to 1.0)

        Returns:
            Stabilized prediction with confirmation status:
            {
                "up": float,  # Smoothed UP probability
                "down": float,  # Smoothed DOWN probability
                "raw_up": float,  # Original UP probability
                "raw_down": float,  # Original DOWN probability
                "signals": {...},  # Original signals
                "confidence": float,  # Original confidence
                "smoothed": True,  # Flag indicating smoothing was applied
                "confirmed": bool,  # Whether signal is confirmed
                "action": str,  # "BUY UP", "BUY DOWN", "NO TRADE", or "WAIT X"
                "confirmation_progress": {  # Progress toward confirmation
                    "direction": str,
                    "elapsed_seconds": float,
                    "consecutive_count": int,
                    "time_progress": float,  # 0 to 1
                    "count_progress": float  # 0 to 1
                }
            }
        """
        raw_up = raw_prediction.get("up", 0.5)
        raw_down = raw_prediction.get("down", 0.5)

        # Step 1: Apply EMA smoothing
        smoothed_up, smoothed_down = self._apply_ema_smoothing(raw_up, raw_down)

        # Step 2: Check no-trade zone (use smoothed probabilities)
        in_no_trade_zone = self._check_no_trade_zone(smoothed_up)

        if in_no_trade_zone:
            # In no-trade zone, reset confirmation
            self.current_direction = "NEUTRAL"
            self.direction_start_time = None
            self.consecutive_count = 0
            self.confirmed_signal = "NO TRADE"

            return {
                "up": smoothed_up,
                "down": smoothed_down,
                "raw_up": raw_up,
                "raw_down": raw_down,
                "signals": raw_prediction.get("signals", {}),
                "confidence": raw_prediction.get("confidence", 0),
                "smoothed": True,
                "confirmed": True,
                "action": "NO TRADE",
                "reason": "in_no_trade_zone",
                "confirmation_progress": None
            }

        # Step 3: Update confirmation tracking (use raw signals for responsiveness)
        self._update_confirmation_tracking(raw_up, raw_down)

        # Step 4: Check if signal is confirmed
        is_confirmed, action = self._is_signal_confirmed()

        # Update confirmed signal if confirmed
        if is_confirmed:
            self.confirmed_signal = action

        # Calculate confirmation progress
        progress = None
        if self.current_direction and self.current_direction != "NEUTRAL":
            elapsed_time = 0
            if self.direction_start_time:
                elapsed_time = time.time() - self.direction_start_time

            time_progress = min(elapsed_time / self.CONFIRMATION_SECONDS, 1.0)
            count_progress = min(self.consecutive_count / self.CONFIRMATION_COUNT, 1.0)

            # Overall progress is max of time and count
            overall_progress = max(time_progress, count_progress)

            progress = {
                "direction": self.current_direction,
                "elapsed_seconds": elapsed_time,
                "consecutive_count": self.consecutive_count,
                "time_progress": time_progress,
                "count_progress": count_progress,
                "overall_progress": overall_progress
            }

        return {
            "up": smoothed_up,
            "down": smoothed_down,
            "raw_up": raw_up,
            "raw_down": raw_down,
            "signals": raw_prediction.get("signals", {}),
            "confidence": raw_prediction.get("confidence", 0),
            "smoothed": True,
            "confirmed": is_confirmed,
            "action": self.confirmed_signal if is_confirmed else action,
            "confirmation_progress": progress
        }

    def reset(self):
        """Reset stabilizer state (useful for new candle)."""
        self.ema_up = 0.5
        self.ema_down = 0.5
        self.signal_history.clear()
        self.current_direction = None
        self.direction_start_time = None
        self.consecutive_count = 0
        self.confirmed_signal = None
        # Reset direction lock
        self.locked_direction = None
        self.lock_start_time = None
        self.lock_strength_at_lock = 0.0


def stabilize_prediction(
    raw_prediction: Dict[str, Any],
    stabilizer: Optional[SignalStabilizer] = None,
    elapsed_ratio: Optional[float] = None
) -> Dict[str, Any]:
    """Convenience function to stabilize a raw prediction.

    Args:
        raw_prediction: Raw prediction from candle_direction engine
        stabilizer: SignalStabilizer instance (will create if None)
        elapsed_ratio: Optional elapsed time ratio

    Returns:
        Stabilized prediction
    """
    if stabilizer is None:
        stabilizer = SignalStabilizer()

    return stabilizer.process(raw_prediction, elapsed_ratio)
