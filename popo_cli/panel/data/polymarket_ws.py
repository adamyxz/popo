"""Polymarket WebSocket for Chainlink prices."""

import asyncio
import json
import time
from typing import Optional, Dict, Any, Callable


def safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def normalize_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Normalize payload to dict."""
    if not payload:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        return safe_json_parse(payload)
    return None


def to_finite_number(x: Any) -> Optional[float]:
    """Convert to finite number or None."""
    try:
        n = float(x) if not isinstance(x, str) else float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


class PolymarketChainlinkStream:
    """Polymarket Chainlink price stream via WebSocket."""

    def __init__(
        self,
        ws_url: str,
        symbol_includes: str = "btc",
        on_update: Optional[Callable] = None
    ):
        self.ws_url = ws_url
        self.symbol_includes = symbol_includes.lower()
        self.on_update = on_update
        self.last_price: Optional[float] = None
        self.last_updated_at: Optional[int] = None
        self._closed = False
        self._ws = None
        self._task = None
        self._reconnect_ms = 500

        # Connection health tracking
        self._connected = False
        self._last_data_time = None  # Last time we received valid data
        self._connect_attempt_count = 0
        self._last_alert_time = 0

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed or not self.ws_url:
            return

        import websockets

        self._connect_attempt_count += 1

        try:
            async with websockets.connect(
                self.ws_url,
                close_timeout=10,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=20,   # Wait for pong response
            ) as ws:
                self._ws = ws
                self._connected = True
                self._reconnect_ms = 500

                # Subscribe
                subscribe_msg = {
                    "action": "subscribe",
                    "subscriptions": [
                        {"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}
                    ]
                }
                await ws.send(json.dumps(subscribe_msg))

                # Listen for messages
                async for message in ws:
                    if self._closed:
                        break

                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    if msg.get("topic") != "crypto_prices_chainlink":
                        continue

                    payload = normalize_payload(msg.get("payload")) or {}
                    symbol = str(payload.get("symbol") or payload.get("pair") or payload.get("ticker", "")).lower()

                    if self.symbol_includes and self.symbol_includes not in symbol:
                        continue

                    price = to_finite_number(
                        payload.get("value") or
                        payload.get("price") or
                        payload.get("current") or
                        payload.get("data")
                    )

                    if price is None:
                        continue

                    # Handle timestamp
                    raw_timestamp = to_finite_number(payload.get("timestamp") or payload.get("updatedAt"))
                    updated_at_ms = self.last_updated_at

                    if raw_timestamp is not None:
                        ts_str = str(int(raw_timestamp))
                        if len(ts_str) >= 16:
                            updated_at_ms = int(raw_timestamp / 1000)
                        elif len(ts_str) >= 13:
                            updated_at_ms = int(raw_timestamp)
                        else:
                            updated_at_ms = int(raw_timestamp * 1000)

                    self.last_price = price
                    self.last_updated_at = updated_at_ms or self.last_updated_at
                    self._last_data_time = time.time()

                    if self.on_update:
                        self.on_update({
                            "price": self.last_price,
                            "updatedAt": self.last_updated_at,
                            "source": "polymarket_ws"
                        })

                # Connection closed normally (outside async for)
                self._connected = False

        except Exception as e:
            self._connected = False
            if not self._closed:
                # Alert on reconnect failures (but not on every attempt to avoid spam)
                now = time.time()
                if now - self._last_alert_time > 60:  # Alert at most once per minute
                    print(f"[yellow]Polymarket WebSocket connection failed (attempt {self._connect_attempt_count}), reconnecting in {self._reconnect_ms}ms...[/yellow]")
                    self._last_alert_time = now

                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(30000, int(self._reconnect_ms * 1.5))
                self._task = asyncio.create_task(self._connect())

    async def start(self):
        """Start the stream."""
        if not self.ws_url:
            return
        self._task = asyncio.create_task(self._connect())

    def get_last(self) -> Dict[str, Any]:
        """Get last received data."""
        return {
            "price": self.last_price,
            "updatedAt": self.last_updated_at,
            "source": "polymarket_ws"
        }

    def is_healthy(self, timeout_seconds: int = 60) -> bool:
        """Check if the stream is healthy (receiving data within timeout)."""
        if self._last_data_time is None:
            return False
        return (time.time() - self._last_data_time) < timeout_seconds

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information."""
        return {
            "connected": self._connected,
            "has_price": self.last_price is not None,
            "last_data_age_seconds": time.time() - self._last_data_time if self._last_data_time else None,
            "connect_attempts": self._connect_attempt_count,
            "reconnect_delay_ms": self._reconnect_ms
        }

    async def close(self):
        """Close the stream."""
        self._closed = True
        self._connected = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def start_polymarket_chainlink_price_stream(
    ws_url: str,
    symbol_includes: str = "btc",
    on_update: Optional[Callable] = None
) -> PolymarketChainlinkStream:
    """Start a Polymarket Chainlink price stream."""
    return PolymarketChainlinkStream(ws_url, symbol_includes, on_update)
