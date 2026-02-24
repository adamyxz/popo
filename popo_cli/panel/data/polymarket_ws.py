"""Polymarket WebSocket for Chainlink prices."""

import asyncio
import json
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

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed or not self.ws_url:
            return

        import websockets

        try:
            async with websockets.connect(
                self.ws_url,
                close_timeout=10
            ) as ws:
                self._ws = ws
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

                    if self.on_update:
                        self.on_update({
                            "price": self.last_price,
                            "updatedAt": self.last_updated_at,
                            "source": "polymarket_ws"
                        })

        except Exception:
            if not self._closed:
                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(10000, int(self._reconnect_ms * 1.5))
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

    async def close(self):
        """Close the stream."""
        self._closed = True
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
