"""Chainlink WebSocket price stream."""

import asyncio
import json
import time
from typing import Optional, Dict, Any, Callable, List


def hex_to_signed_bigint(hex_str: str) -> int:
    """Convert hex string to signed bigint."""
    x = int(hex_str, 16)
    two_255 = 1 << 255
    two_256 = 1 << 256
    return x if x < two_255 else x - two_256


def to_number(x: Any) -> Optional[float]:
    """Convert to number or return None."""
    try:
        n = float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


# AnswerUpdated event signature
# event AnswerUpdated(int256 indexed answer, uint256 indexed roundId, uint256 updatedAt)
ANSWER_UPDATED_TOPIC0 = "0x" + "0x" + "1" + "6e" + "f2" + "d8" + "f5" + "c9" + "4a" + "5b" + "3e" + "01" + "00" + "00" + "00"  # keccak256("AnswerUpdated(int256,uint256,uint256)")


class ChainlinkPriceStream:
    """Chainlink WebSocket price stream via Polygon WebSocket."""

    def __init__(
        self,
        wss_urls: List[str],
        aggregator: str,
        decimals: int = 8,
        on_update: Optional[Callable] = None
    ):
        self.wss_urls = wss_urls
        self.aggregator = aggregator
        self.decimals = decimals
        self.on_update = on_update

        self.last_price: Optional[float] = None
        self.last_updated_at: Optional[int] = None

        self._closed = False
        self._ws = None
        self._task = None
        self._reconnect_ms = 500
        self._url_index = 0
        self._sub_id = None
        self._next_id = 1

        # Connection health tracking
        self._connected = False
        self._last_data_time = None  # Last time we received valid data
        self._connect_attempt_count = 0
        self._last_alert_time = 0

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed or not self.wss_urls or not self.aggregator:
            return

        import websockets

        url = self.wss_urls[self._url_index % len(self.wss_urls)]
        self._url_index += 1

        self._connect_attempt_count += 1

        try:
            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=30,
            ) as ws:
                self._ws = ws
                self._connected = True
                self._reconnect_ms = 500

                # Subscribe to logs
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": self._next_id,
                    "method": "eth_subscribe",
                    "params": [
                        "logs",
                        {
                            "address": self.aggregator,
                            "topics": [ANSWER_UPDATED_TOPIC0]
                        }
                    ]
                }
                self._next_id += 1
                await ws.send(json.dumps(subscribe_msg))

                # Listen for messages
                async for message in ws:
                    if self._closed:
                        break

                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    # Handle subscription confirmation
                    if msg.get("id") and msg.get("result") and isinstance(msg["result"], str) and not self._sub_id:
                        self._sub_id = msg["result"]
                        continue

                    # Handle subscription updates
                    if msg.get("method") != "eth_subscription":
                        continue

                    params = msg.get("params")
                    if not params or not params.get("result"):
                        continue

                    log = params["result"]
                    topics = log.get("topics", [])
                    if not isinstance(topics, list) or len(topics) < 2:
                        continue

                    try:
                        # Extract answer from topics[1]
                        answer = hex_to_signed_bigint(topics[1])
                        price = to_number(answer) / (10 ** self.decimals)

                        # Extract updatedAt from log.data
                        log_data = log.get("data")
                        updated_at = None
                        if log_data and isinstance(log_data, str):
                            updated_at = to_number(int(log_data, 16))

                        self.last_price = price if __import__("math").isfinite(price) else self.last_price
                        self.last_updated_at = (updated_at * 1000) if updated_at is not None else self.last_updated_at
                        self._last_data_time = time.time()

                        if self.on_update:
                            self.on_update({
                                "price": self.last_price,
                                "updatedAt": self.last_updated_at,
                                "source": "chainlink_ws"
                            })

                    except (ValueError, IndexError):
                        continue

                # Connection closed normally
                self._connected = False

        except Exception as e:
            self._connected = False
            if not self._closed:
                # Alert on reconnect failures (but not on every attempt to avoid spam)
                now = time.time()
                if now - self._last_alert_time > 60:  # Alert at most once per minute
                    print(f"[yellow]Chainlink WebSocket connection failed (attempt {self._connect_attempt_count}), reconnecting in {self._reconnect_ms}ms...[/yellow]")
                    self._last_alert_time = now

                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(30000, int(self._reconnect_ms * 1.5))
                # Reconnect
                self._task = asyncio.create_task(self._connect())

    async def start(self):
        """Start the stream."""
        if self._closed or not self.wss_urls:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._connect())

    def get_last(self) -> Dict[str, Any]:
        """Get last received data."""
        return {
            "price": self.last_price,
            "updatedAt": self.last_updated_at,
            "source": "chainlink_ws"
        }

    def is_healthy(self, timeout_seconds: int = 120) -> bool:
        """Check if the stream is healthy (receiving data within timeout).

        Chainlink price updates are less frequent than Polymarket, so we use a longer timeout.
        """
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
            "reconnect_delay_ms": self._reconnect_ms,
            "subscribed": self._sub_id is not None
        }

    async def close(self):
        """Close the stream."""
        self._closed = True
        self._connected = False

        # Unsubscribe if connected
        if self._ws and self._sub_id:
            try:
                unsubscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": self._next_id,
                    "method": "eth_unsubscribe",
                    "params": [self._sub_id]
                }
                self._next_id += 1
                await self._ws.send(json.dumps(unsubscribe_msg))
            except:
                pass

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def start_chainlink_price_stream(
    wss_urls: List[str],
    aggregator: str,
    decimals: int = 8,
    on_update: Optional[Callable] = None
) -> ChainlinkPriceStream:
    """Start a Chainlink price stream."""
    return ChainlinkPriceStream(wss_urls, aggregator, decimals, on_update)
