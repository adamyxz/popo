"""Binance data fetching."""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional


def to_number(x: Any) -> Optional[float]:
    """Convert to number or return None."""
    try:
        n = float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


async def fetch_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int, base_url: str = "https://api.binance.com") -> List[Dict[str, Any]]:
    """Fetch klines from Binance."""
    url = f"{base_url}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, params=params, timeout=timeout) as response:
            if not response.ok:
                return []
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return []

    return [
        {
            "openTime": int(k[0]),
            "open": to_number(k[1]),
            "high": to_number(k[2]),
            "low": to_number(k[3]),
            "close": to_number(k[4]),
            "volume": to_number(k[5]),
            "closeTime": int(k[6])
        }
        for k in data
    ]


async def fetch_last_price(session: aiohttp.ClientSession, symbol: str, base_url: str = "https://api.binance.com") -> Optional[float]:
    """Fetch last price from Binance."""
    url = f"{base_url}/api/v3/ticker/price"
    params = {"symbol": symbol}

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, params=params, timeout=timeout) as response:
            if not response.ok:
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return None

    return to_number(data.get("price"))


class BinanceTradeStream:
    """Binance WebSocket trade stream."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.lower()
        self.last_price: Optional[float] = None
        self.last_ts: Optional[int] = None
        self._closed = False
        self._ws = None
        self._task = None
        self._reconnect_ms = 500

    def _build_ws_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed:
            return

        import websockets
        url = self._build_ws_url()

        try:
            async with websockets.connect(url) as ws:
                self._ws = ws
                self._reconnect_ms = 500

                async for message in ws:
                    if self._closed:
                        break

                    try:
                        import json
                        msg = json.loads(message)
                        p = to_number(msg.get("p"))
                        if p is not None:
                            self.last_price = p
                            self.last_ts = __import__("time").time_ns() // 1_000_000
                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception as e:
            if not self._closed:
                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(10000, int(self._reconnect_ms * 1.5))
                # Reconnect
                self._task = asyncio.create_task(self._connect())

    async def start(self):
        """Start the stream."""
        self._task = asyncio.create_task(self._connect())

    def get_last(self) -> Dict[str, Any]:
        """Get last received data."""
        return {"price": self.last_price, "ts": self.last_ts}

    async def close(self):
        """Close the stream."""
        self._closed = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


def start_binance_trade_stream(symbol: str = "BTCUSDT") -> BinanceTradeStream:
    """Start a Binance trade stream (non-async version for compatibility)."""
    stream = BinanceTradeStream(symbol)
    # Note: In async context, you'd call await stream.start()
    # For now, we return the stream object and start it when needed
    return stream
