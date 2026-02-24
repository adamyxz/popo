"""Chainlink data fetching."""

import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from web3 import Web3


def to_number(x: Any) -> Optional[float]:
    """Convert to number or return None."""
    try:
        n = float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


AGGREGATOR_ABI = [
    {
        "name": "latestRoundData",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"}
        ]
    },
    {
        "name": "decimals",
        "type": "function",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}]
    }
]


class ChainlinkFetcher:
    """Chainlink price fetcher with caching."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._cached_decimals: Optional[int] = None
        self._cached_result = {"price": None, "updatedAt": None, "source": "chainlink"}
        self._cached_fetched_at_ms = 0
        self._min_fetch_interval_ms = 2000
        self._rpc_timeout_ms = 1500
        self._preferred_rpc_url: Optional[str] = None

    def _get_rpc_candidates(self) -> List[str]:
        """Get list of RPC URLs to try."""
        from_list = self.config.get("polygon_rpc_urls", [])
        single = self.config.get("polygon_rpc_url")
        defaults = [
            "https://polygon-rpc.com",
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com"
        ]

        all_urls = [*from_list]
        if single:
            all_urls.append(single)
        all_urls.extend(defaults)

        # Remove duplicates and filter empty
        seen = set()
        result = []
        for url in all_urls:
            url = str(url).strip()
            if url and url not in seen:
                seen.add(url)
                result.append(url)

        return result

    def _get_ordered_rpcs(self) -> List[str]:
        """Get ordered list of RPCs with preferred first."""
        rpcs = self._get_rpc_candidates()
        pref = self._preferred_rpc_url
        if pref and pref in rpcs:
            return [pref] + [r for r in rpcs if r != pref]
        return rpcs

    async def _json_rpc_request(self, rpc_url: str, method: str, params: List[Any]) -> Any:
        """Make JSON-RPC request."""
        timeout = aiohttp.ClientTimeout(total=self._rpc_timeout_ms / 1000)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                rpc_url,
                headers={"content-type": "application/json"},
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            ) as response:
                if not response.ok:
                    raise Exception(f"rpc_http_{response.status}")

                data = await response.json()
                if data.get("error"):
                    raise Exception(f"rpc_error_{data['error']['code']}")

                return data.get("result")

    async def _fetch_decimals(self, rpc_url: str, aggregator: str) -> int:
        """Fetch decimals from aggregator."""
        # Encode function call: decimals()
        data = "0x313ce567"  # Function signature for decimals()

        result = await self._json_rpc_request(rpc_url, "eth_call", [{"to": aggregator, "data": data}, "latest"])

        if result and len(result) >= 66:
            return int(result[2:66], 16)  # Remove 0x and convert to int

        raise Exception("Failed to fetch decimals")

    async def _fetch_latest_round_data(self, rpc_url: str, aggregator: str) -> Dict[str, Any]:
        """Fetch latest round data from aggregator."""
        # Encode function call: latestRoundData()
        data = "0xfeaf968c"  # Function signature for latestRoundData()

        result = await self._json_rpc_request(rpc_url, "eth_call", [{"to": aggregator, "data": data}, "latest"])

        if not result or len(result) < 2 + 64 * 5:
            raise Exception("Invalid response from latestRoundData")

        # Decode the result (simplified, assuming standard layout)
        # Round ID (32 bytes) + Answer (32 bytes) + StartedAt (32 bytes) + UpdatedAt (32 bytes) + AnsweredInRound (32 bytes)

        # Extract answer (bytes 32-63)
        answer_hex = result[2 + 64 : 2 + 64 * 2]  # Skip round ID, get answer
        answer = int(answer_hex, 16)

        # Convert to signed integer (two's complement if needed)
        if answer >= 2**255:
            answer = answer - 2**256

        # Extract updatedAt (bytes 96-127)
        updated_at_hex = result[2 + 64 * 3 : 2 + 64 * 4]
        updated_at = int(updated_at_hex, 16)

        return {"answer": answer, "updatedAt": updated_at}

    async def fetch_chainlink_btc_usd(self) -> Dict[str, Any]:
        """Fetch Chainlink BTC/USD price."""
        aggregator = self.config.get("aggregator") or self.config.get("btc_usd_aggregator")

        rpcs = self._get_ordered_rpcs()
        if not rpcs or not aggregator:
            return {"price": None, "updatedAt": None, "source": "missing_config"}

        # Check cache
        now_ms = int(__import__("time").time() * 1000)
        if self._cached_fetched_at_ms and now_ms - self._cached_fetched_at_ms < self._min_fetch_interval_ms:
            return self._cached_result

        for rpc in rpcs:
            try:
                if self._cached_decimals is None:
                    self._cached_decimals = await self._fetch_decimals(rpc, aggregator)

                round_data = await self._fetch_latest_round_data(rpc, aggregator)
                answer = float(round_data["answer"])
                scale = 10 ** self._cached_decimals
                price = answer / scale

                self._cached_result = {
                    "price": price,
                    "updatedAt": round_data["updatedAt"] * 1000,  # Convert to milliseconds
                    "source": "chainlink"
                }
                self._cached_fetched_at_ms = now_ms
                self._preferred_rpc_url = rpc

                return self._cached_result

            except Exception as e:
                self._cached_decimals = None
                continue

        return self._cached_result


class ChainlinkPriceStream:
    """Chainlink WebSocket price stream (simplified)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_price: Optional[float] = None
        self.last_updated_at: Optional[int] = None
        self._closed = False

    def get_last(self) -> Dict[str, Any]:
        """Get last received data."""
        return {
            "price": self.last_price,
            "updatedAt": self.last_updated_at,
            "source": "chainlink_ws"
        }

    async def close(self):
        """Close the stream."""
        self._closed = True


def start_chainlink_price_stream(config: Dict[str, Any]) -> ChainlinkPriceStream:
    """Start a Chainlink price stream."""
    return ChainlinkPriceStream(config)


async def fetch_chainlink_btc_usd(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch Chainlink BTC/USD price."""
    fetcher = ChainlinkFetcher(config)
    return await fetcher.fetch_chainlink_btc_usd()
