"""Polymarket data fetching."""

import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime


def to_number(x: Any) -> Optional[float]:
    """Convert to number or return None."""
    try:
        n = float(x)
        return n if __import__("math").isfinite(n) else None
    except (TypeError, ValueError):
        return None


async def fetch_market_by_slug(session: aiohttp.ClientSession, slug: str, base_url: str = "https://gamma-api.polymarket.com") -> Optional[Dict[str, Any]]:
    """Fetch market by slug."""
    url = f"{base_url}/markets"
    params = {"slug": slug}

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, params=params, timeout=timeout) as response:
            if not response.ok:
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return None

    market = data[0] if isinstance(data, list) and data else data
    return market if market else None


async def fetch_live_events_by_series_id(
    session: aiohttp.ClientSession,
    series_id: str,
    limit: int = 20,
    base_url: str = "https://gamma-api.polymarket.com"
) -> List[Dict[str, Any]]:
    """Fetch live events by series ID."""
    url = f"{base_url}/events"
    params = {
        "series_id": series_id,
        "active": "true",
        "closed": "false",
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

    return data if isinstance(data, list) else []


def flatten_event_markets(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten markets from events."""
    out = []
    for event in (events if isinstance(events, list) else []):
        markets = event.get("markets", [])
        for market in (markets if isinstance(markets, list) else []):
            out.append(market)
    return out


def safe_time_ms(x: Any) -> Optional[int]:
    """Convert to timestamp in milliseconds."""
    if not x:
        return None
    try:
        t = int(datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp() * 1000) if isinstance(x, str) else int(x)
        return t if __import__("math").isfinite(t) else None
    except (ValueError, TypeError):
        return None


def pick_latest_live_market(markets: List[Dict[str, Any]], now_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Pick latest live market from list. Returns live market, or most recently ended market if none live."""
    if not isinstance(markets, list) or not markets:
        return None

    if now_ms is None:
        now_ms = int(__import__("time").time() * 1000)

    # Enrich markets with timestamps
    enriched = []
    for m in markets:
        end_ms = safe_time_ms(m.get("endDate"))
        start_ms = safe_time_ms(m.get("eventStartTime") or m.get("startTime") or m.get("startDate"))
        enriched.append({"m": m, "endMs": end_ms, "startMs": start_ms})

    # Filter valid end times
    enriched = [e for e in enriched if e["endMs"] is not None]

    # Find live markets (started and not ended)
    live = []
    for e in enriched:
        started = e["startMs"] is None or e["startMs"] <= now_ms
        if started and now_ms < e["endMs"]:
            live.append(e)

    live.sort(key=lambda x: x["endMs"])
    if live:
        return live[0]["m"]

    # No live markets - find most recently ended market (within last hour)
    recently_ended = [e for e in enriched if e["endMs"] <= now_ms and (now_ms - e["endMs"]) < 3600000]
    recently_ended.sort(key=lambda x: x["endMs"], reverse=True)
    if recently_ended:
        return recently_ended[0]["m"]

    # Find upcoming markets
    upcoming = [e for e in enriched if now_ms < e["endMs"]]
    upcoming.sort(key=lambda x: x["endMs"])
    return upcoming[0]["m"] if upcoming else None


async def fetch_clob_price(
    session: aiohttp.ClientSession,
    token_id: str,
    side: str,
    base_url: str = "https://clob.polymarket.com"
) -> Optional[float]:
    """Fetch CLOB price."""
    url = f"{base_url}/price"
    params = {
        "token_id": token_id,
        "side": side
    }

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, params=params, timeout=timeout) as response:
            if not response.ok:
                return None
            data = await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return None

    return to_number(data.get("price"))


async def fetch_order_book(
    session: aiohttp.ClientSession,
    token_id: str,
    base_url: str = "https://clob.polymarket.com"
) -> Optional[Dict[str, Any]]:
    """Fetch order book."""
    url = f"{base_url}/book"
    params = {"token_id": token_id}

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with session.get(url, params=params, timeout=timeout) as response:
            if not response.ok:
                return {}
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return {}


def summarize_order_book(book: Dict[str, Any], depth_levels: int = 5) -> Dict[str, Optional[float]]:
    """Summarize order book."""
    bids = book.get("bids", []) if isinstance(book, dict) else []
    asks = book.get("asks", []) if isinstance(book, dict) else []

    # Ensure lists
    if not isinstance(bids, list):
        bids = []
    if not isinstance(asks, list):
        asks = []

    # Best bid (max price)
    best_bid = None
    for level in bids:
        p = to_number(level.get("price"))
        if p is not None:
            if best_bid is None or p > best_bid:
                best_bid = p

    # Best ask (min price)
    best_ask = None
    for level in asks:
        p = to_number(level.get("price"))
        if p is not None:
            if best_ask is None or p < best_ask:
                best_ask = p

    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

    # Liquidity
    bid_liquidity = sum(
        to_number(level.get("size")) or 0
        for level in bids[:depth_levels]
    )
    ask_liquidity = sum(
        to_number(level.get("size")) or 0
        for level in asks[:depth_levels]
    )

    return {
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "spread": spread,
        "bidLiquidity": bid_liquidity if bid_liquidity > 0 else None,
        "askLiquidity": ask_liquidity if ask_liquidity > 0 else None
    }


async def fetch_polymarket_snapshot(
    session: aiohttp.ClientSession,
    polymarket_config: Dict[str, Any],
    market_slug: Optional[str] = None,
    series_id: Optional[str] = None,
    auto_select_latest: bool = True,
    base_url: str = "https://gamma-api.polymarket.com",
    clob_base_url: str = "https://clob.polymarket.com"
) -> Dict[str, Any]:
    """Fetch Polymarket snapshot including market data and order books."""

    # Resolve market
    market = None
    if market_slug:
        market = await fetch_market_by_slug(session, market_slug, base_url)
    elif auto_select_latest and series_id:
        events = await fetch_live_events_by_series_id(session, series_id, limit=25, base_url=base_url)
        markets = flatten_event_markets(events)
        market = pick_latest_live_market(markets)

    if not market:
        return {"ok": False, "reason": "market_not_found"}

    # Parse outcomes
    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        import json
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            outcomes = []

    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        import json
        try:
            outcome_prices = json.loads(outcome_prices)
        except json.JSONDecodeError:
            outcome_prices = []

    clob_token_ids = market.get("clobTokenIds", [])
    if isinstance(clob_token_ids, str):
        import json
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except json.JSONDecodeError:
            clob_token_ids = []

    # Find token IDs
    up_label = polymarket_config.get("up_outcome_label", "Up")
    down_label = polymarket_config.get("down_outcome_label", "Down")

    up_token_id = None
    down_token_id = None

    for i, outcome in enumerate(outcomes):
        label = str(outcome)
        token_id = str(clob_token_ids[i]) if i < len(clob_token_ids) else None
        if not token_id:
            continue

        if label.lower() == up_label.lower():
            up_token_id = token_id
        if label.lower() == down_label.lower():
            down_token_id = token_id

    # Get outcome prices
    up_index = next((i for i, o in enumerate(outcomes) if str(o).lower() == up_label.lower()), None)
    down_index = next((i for i, o in enumerate(outcomes) if str(o).lower() == down_label.lower()), None)

    gamma_yes = outcome_prices[up_index] if up_index is not None else None
    gamma_no = outcome_prices[down_index] if down_index is not None else None

    if not up_token_id or not down_token_id:
        return {
            "ok": False,
            "reason": "missing_token_ids",
            "market": market,
            "outcomes": outcomes,
            "clobTokenIds": clob_token_ids,
            "outcomePrices": outcome_prices
        }

    # Fetch prices and order books
    try:
        up_buy, down_buy, up_book, down_book = await asyncio.gather(
            fetch_clob_price(session, up_token_id, "buy", clob_base_url),
            fetch_clob_price(session, down_token_id, "buy", clob_base_url),
            fetch_order_book(session, up_token_id, clob_base_url),
            fetch_order_book(session, down_token_id, clob_base_url)
        )

        up_book_summary = summarize_order_book(up_book)
        down_book_summary = summarize_order_book(down_book)

    except Exception:
        # Fallback to market data
        up_buy = gamma_yes
        down_buy = gamma_no
        up_book_summary = {
            "bestBid": to_number(market.get("bestBid")),
            "bestAsk": to_number(market.get("bestAsk")),
            "spread": to_number(market.get("spread")),
            "bidLiquidity": None,
            "askLiquidity": None
        }
        down_book_summary = {
            "bestBid": None,
            "bestAsk": None,
            "spread": to_number(market.get("spread")),
            "bidLiquidity": None,
            "askLiquidity": None
        }

    return {
        "ok": True,
        "market": market,
        "tokens": {"upTokenId": up_token_id, "downTokenId": down_token_id},
        "prices": {
            "up": up_buy if up_buy is not None else gamma_yes,
            "down": down_buy if down_buy is not None else gamma_no
        },
        "orderbook": {
            "up": up_book_summary,
            "down": down_book_summary
        }
    }
