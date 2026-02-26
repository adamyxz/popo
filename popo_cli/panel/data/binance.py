"""Binance data fetching."""

import aiohttp
import asyncio
import websockets
from typing import List, Dict, Any, Optional


def to_number(x: Any) -> Optional[float]:
    """Convert to number or return None."""
    try:
        n = float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


class BinanceKlineStream:
    """Binance WebSocket K-line stream."""

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m"):
        self.symbol = symbol.lower()
        self.interval = interval
        self.current_kline: Optional[Dict[str, Any]] = None
        self.last_closed_kline: Optional[Dict[str, Any]] = None
        self._closed = False
        self._task = None
        self._reconnect_ms = 500

    def _build_ws_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol}@kline_{self.interval}"

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed:
            return

        url = self._build_ws_url()

        try:
            async with websockets.connect(url) as ws:
                self._reconnect_ms = 500

                async for message in ws:
                    if self._closed:
                        break

                    try:
                        import json
                        msg = json.loads(message)
                        k = msg.get("k", {})

                        if k:
                            self.current_kline = {
                                "openTime": k.get("t"),
                                "open": to_number(k.get("o")),
                                "high": to_number(k.get("h")),
                                "low": to_number(k.get("l")),
                                "close": to_number(k.get("c")),
                                "volume": to_number(k.get("v")),
                                "closeTime": k.get("T"),
                                "isClosed": k.get("x", False),
                                "quoteVolume": to_number(k.get("q")),
                                "trades": k.get("n"),
                            }

                            # When candle closes, save as last_closed
                            if self.current_kline.get("isClosed"):
                                self.last_closed_kline = self.current_kline.copy()

                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception:
            if not self._closed:
                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(10000, int(self._reconnect_ms * 1.5))
                self._task = asyncio.create_task(self._connect())

    async def start(self):
        """Start the stream."""
        self._task = asyncio.create_task(self._connect())

    def get_current_kline(self) -> Optional[Dict[str, Any]]:
        """Get current forming kline."""
        return self.current_kline

    def get_last_closed_kline(self) -> Optional[Dict[str, Any]]:
        """Get last closed kline."""
        return self.last_closed_kline

    async def close(self):
        """Close the stream."""
        self._closed = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


class BinanceDepthStream:
    """Binance WebSocket depth stream (order book)."""

    def __init__(self, symbol: str = "BTCUSDT", update_speed: str = "100ms"):
        self.symbol = symbol.lower()
        self.update_speed = update_speed  # 100ms, 1000ms
        self.bids: List[tuple[float, float]] = []  # (price, quantity)
        self.asks: List[tuple[float, float]] = []
        self.last_update_ts: Optional[int] = None
        self._closed = False
        self._task = None
        self._reconnect_ms = 500
        self._last_sync = None  # Last local orderbook state

    def _build_ws_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol}@depth@{self.update_speed}"

    def _apply_update(self, data: Dict[str, Any]):
        """Apply depth update to local order book."""
        # For simplicity, we'll just store the raw bids/asks from updates
        # In production, you'd want to maintain a proper local orderbook
        new_bids = data.get("b", [])
        new_asks = data.get("a", [])

        if new_bids:
            # Update bids - for simplicity, replace
            self.bids = [(to_number(p), to_number(q)) for p, q in new_bids if to_number(q) and to_number(q) > 0]

        if new_asks:
            # Update asks
            self.asks = [(to_number(p), to_number(q)) for p, q in new_asks if to_number(q) and to_number(q) > 0]

        # Sort bids descending, asks ascending
        self.bids.sort(key=lambda x: x[0], reverse=True)
        self.asks.sort(key=lambda x: x[0])

        self.last_update_ts = __import__("time").time_ns() // 1_000_000

    async def _connect(self):
        """Connect to WebSocket."""
        if self._closed:
            return

        url = self._build_ws_url()

        try:
            async with websockets.connect(url) as ws:
                self._reconnect_ms = 500

                async for message in ws:
                    if self._closed:
                        break

                    try:
                        import json
                        msg = json.loads(message)
                        self._apply_update(msg)

                    except (json.JSONDecodeError, KeyError):
                        continue

        except Exception:
            if not self._closed:
                await asyncio.sleep(self._reconnect_ms / 1000)
                self._reconnect_ms = min(10000, int(self._reconnect_ms * 1.5))
                self._task = asyncio.create_task(self._connect())

    async def start(self):
        """Start the stream."""
        self._task = asyncio.create_task(self._connect())

    def get_depth_snapshot(self, levels: int = 10) -> Dict[str, Any]:
        """Get current depth snapshot."""
        return {
            "bids": self.bids[:levels],
            "asks": self.asks[:levels],
            "lastUpdate": self.last_update_ts,
            "bidVolume": sum(q for _, q in self.bids[:levels]),
            "askVolume": sum(q for _, q in self.asks[:levels]),
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
    """Binance WebSocket trade stream with aggressive buy/sell tracking."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.lower()
        self.last_price: Optional[float] = None
        self.last_ts: Optional[int] = None
        self._closed = False
        self._ws = None
        self._task = None
        self._reconnect_ms = 500

        # Track current candle's trades
        self.current_candle_trades: List[Dict[str, Any]] = []
        self.current_candle_start_ts: Optional[int] = None
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0

        # For aggressive classification, we need current bid/ask
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None

    def set_best_bid_ask(self, bid: Optional[float], ask: Optional[float]):
        """Set current best bid and ask for trade classification."""
        self._best_bid = bid
        self._best_ask = ask

    def reset_candle_tracking(self, candle_start_ts: int):
        """Reset trade tracking for new candle."""
        self.current_candle_start_ts = candle_start_ts
        self.current_candle_trades = []
        self.buy_volume = 0.0
        self.sell_volume = 0.0

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
                        q = to_number(msg.get("q"))

                        if p is not None:
                            self.last_price = p
                            self.last_ts = __import__("time").time_ns() // 1_000_000

                            # Track trades for current candle
                            if q is not None:
                                trade = {"price": p, "qty": q}
                                self.current_candle_trades.append(trade)

                                # Classify as aggressive buy/sell if we have bid/ask
                                if self._best_ask and p >= self._best_ask:
                                    self.buy_volume += q
                                elif self._best_bid and p <= self._best_bid:
                                    self.sell_volume += q

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

    def get_candle_volume_stats(self) -> Dict[str, Any]:
        """Get current candle's buy/sell volume stats."""
        return {
            "buyVolume": self.buy_volume,
            "sellVolume": self.sell_volume,
            "totalVolume": self.buy_volume + self.sell_volume,
            "tradeCount": len(self.current_candle_trades),
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


def start_binance_trade_stream(symbol: str = "BTCUSDT") -> BinanceTradeStream:
    """Start a Binance trade stream (non-async version for compatibility)."""
    stream = BinanceTradeStream(symbol)
    # Note: In async context, you'd call await stream.start()
    # For now, we return the stream object and start it when needed
    return stream
