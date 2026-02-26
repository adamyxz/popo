"""LLM-based market prediction using DeepSeek API."""

import os
import json
import time
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime


class LLMPredictor:
    """LLM-based market predictor using DeepSeek API.

    This predictor provides an independent analysis source that operates
    on raw market data only, without seeing the system's predictions.
    This allows for comparison between system analysis and LLM analysis.
    """

    def __init__(self, config: dict):
        """Initialize LLM predictor.

        Args:
            config: Configuration dictionary containing deepseek settings
        """
        self.api_key = config.get("deepseek", {}).get("api_key", "")
        self.base_url = config.get("deepseek", {}).get("base_url", "https://api.deepseek.com/v1")
        self.model = config.get("deepseek", {}).get("model", "deepseek-v3")
        self.enabled = bool(self.api_key)

        # Track last prediction result
        self._last_prediction_result: Optional[Dict[str, Any]] = None

        # Import httpx lazily (only when enabled)
        self._httpx = None
        if self.enabled:
            try:
                import httpx
                self._httpx = httpx
            except ImportError:
                self.enabled = False
                print("[yellow]Warning: httpx not installed. LLM prediction disabled.[/yellow]")

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from markdown code blocks.

        Args:
            content: Raw content that may contain markdown code blocks

        Returns:
            Clean JSON string
        """
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]  # Remove ```

        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```

        return content.strip()

    def is_enabled(self) -> bool:
        """Check if LLM predictor is enabled."""
        return self.enabled and self._httpx is not None

    def format_klines_csv(self, klines: List[Dict[str, Any]], limit: int = 10) -> str:
        """Format klines as CSV for LLM consumption.

        Args:
            klines: List of kline dictionaries
            limit: Maximum number of klines to include

        Returns:
            CSV formatted string
        """
        if not klines:
            return "No historical data available"

        # Get most recent klines (up to limit)
        recent_klines = klines[-limit:] if len(klines) > limit else klines

        # CSV header
        lines = ["timestamp,open,high,low,close,volume"]

        # Add each kline
        for k in recent_klines:
            ts = k.get("closeTime", k.get("timestamp", ""))
            o = k.get("open", 0)
            h = k.get("high", 0)
            l = k.get("low", 0)
            c = k.get("close", 0)
            v = k.get("volume", 0)
            lines.append(f"{ts},{o:.2f},{h:.2f},{l:.2f},{c:.2f},{v:.4f}")

        return "\n".join(lines)

    def build_prompt(self, data: Dict[str, Any]) -> str:
        """Build the prompt for LLM prediction.

        IMPORTANT: This method only uses raw market data and technical indicators.
        It deliberately excludes the system's prediction results to maintain
        independence between system analysis and LLM analysis.

        Args:
            data: Panel data dictionary

        Returns:
            Formatted prompt string
        """
        # Extract key information
        coin = str(data.get("title", "BTC"))
        open_price = self._safe_float(data.get("sessionOpenPrice"))

        # Technical indicators (computed from historical data)
        vwap = self._safe_float(data.get("vwap"))
        vwap_slope = self._safe_float(data.get("vwapSlope"))
        rsi = self._safe_float(data.get("rsi", 50))
        macd = data.get("macd", {}) or {}
        macd_value = self._safe_float(macd.get("value"))
        macd_signal = self._safe_float(macd.get("signal"))
        macd_histogram = self._safe_float(macd.get("histogram"))

        # Time info
        time_left = str(data.get("timeLeft", "-"))
        candle_start = str(data.get("candleStartTime", "") or "")

        # Historical kline data (raw market data)
        klines = data.get("klines", []) or []
        klines_csv = self.format_klines_csv(klines, limit=10)

        # Format values safely (handle None cases)
        open_price_str = f"${open_price:.2f}" if open_price is not None else "N/A"

        vwap_str = f"${vwap:.2f}" if vwap is not None else "N/A"
        vwap_slope_str = f"{vwap_slope:.6f}" if vwap_slope is not None else "N/A"
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"
        macd_val_str = f"{macd_value:.6f}" if macd_value is not None else "N/A"
        macd_sig_str = f"{macd_signal:.6f}" if macd_signal is not None else "N/A"
        macd_hist_str = f"{macd_histogram:.6f}" if macd_histogram is not None else "N/A"

        prompt = f"""You are an independent cryptocurrency trading analyst. Your task is to analyze historical market data and predict whether the current candle will close UP or DOWN relative to its opening price.

# MARKET CONTEXT
- Asset: {coin}
- Candle Open Price: {open_price_str}

# TIME CONTEXT
- Candle Window Start: {candle_start if candle_start else 'N/A'}
- Time Remaining: {time_left}
- Current Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

# TECHNICAL INDICATORS (Computed from Historical Price Data)
- VWAP: {vwap_str}
- VWAP Slope: {vwap_slope_str}
- RSI (14): {rsi_str}
- MACD: {macd_val_str}
- MACD Signal: {macd_sig_str}
- MACD Histogram: {macd_hist_str}

# HISTORICAL PRICE DATA (Last 10 candles)
```csv
{klines_csv}
```

# YOUR TASK
Based ONLY on the historical price data, technical indicators, and historical patterns provided above, independently predict:
1. Direction: Will the candle close UP or DOWN relative to the opening price?
2. Confidence: Your confidence level (0-100%)
3. Reasoning: Brief explanation of your analysis (max 2 sentences)

# RESPONSE FORMAT
Respond ONLY with valid JSON in this exact format:
{{"direction": "UP" or "DOWN", "confidence": 0-100, "reasoning": "your brief explanation"}}

Note: You are providing an independent analysis based on historical data. Focus on patterns, trends, and market structure from the historical candles and technical indicators.
"""

        return prompt

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert a value to float.

        Args:
            value: Any value

        Returns:
            Float or None if conversion fails
        """
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM prediction for current market state.

        Args:
            data: Panel data dictionary containing raw market data

        Returns:
            Prediction result dict with keys: direction, confidence, reasoning, timestamp
        """
        if not self.is_enabled():
            return {
                "direction": None,
                "confidence": 0,
                "reasoning": "LLM prediction disabled (no API key)",
                "timestamp": None,
                "status": "disabled"
            }

        try:
            prompt = self.build_prompt(data)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an independent cryptocurrency trading analyst. Always respond with valid JSON only. Format: {\"direction\": \"UP\" or \"DOWN\", \"confidence\": 0-100, \"reasoning\": \"brief explanation\"}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }

            async with self._httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )

                # Print error details for debugging
                if response.status_code != 200:
                    error_detail = response.text
                    return {
                        "direction": None,
                        "confidence": 0,
                        "reasoning": f"HTTP {response.status_code}: {error_detail[:200]}",
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "error"
                    }

                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")

                # Extract JSON from markdown code blocks if present
                content = self._extract_json_from_response(content)

                # Parse JSON response
                prediction = json.loads(content)

                # Validate and normalize
                direction = prediction.get("direction", "").upper()
                if direction not in ["UP", "DOWN"]:
                    direction = None

                confidence = max(0, min(100, prediction.get("confidence", 0)))
                reasoning = prediction.get("reasoning", "No reasoning provided")

                result_data = {
                    "direction": direction,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "success"
                }

                self._last_prediction_result = result_data
                return result_data

        except Exception as e:
            return {
                "direction": None,
                "confidence": 0,
                "reasoning": f"Error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error"
            }

    def get_last_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the last prediction result.

        Returns:
            Last prediction result dict or None
        """
        return self._last_prediction_result

    def reset(self) -> None:
        """Reset LLM predictor state for new market.

        Clears the last prediction result, allowing a new prediction
        to be made when the market starts.
        """
        self._last_prediction_result = None
