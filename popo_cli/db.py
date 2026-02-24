"""Database module for storing snapshots."""

import os
import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Database:
    """Database connection manager."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._url = os.getenv("DATABASE_URL")

    async def init(self):
        """Initialize database connection pool and create tables."""
        if not self._url:
            raise ValueError("DATABASE_URL environment variable not set")

        self.pool = await asyncpg.create_pool(self._url)

        # Create snapshots table
        await self._create_tables()

    async def _create_tables(self):
        """Create database tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id SERIAL PRIMARY KEY,
                    market_slug VARCHAR(255) NOT NULL,
                    title TEXT,
                    coin VARCHAR(50),
                    interval VARCHAR(20),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    price_to_beat NUMERIC(20, 2),
                    current_price NUMERIC(20, 2),
                    ptb_delta NUMERIC(20, 2),
                    predict_value JSONB,
                    signal VARCHAR(50),
                    signal_stats JSONB,
                    heiken_ashi VARCHAR(100),
                    rsi VARCHAR(50),
                    macd VARCHAR(100),
                    delta VARCHAR(100),
                    vwap VARCHAR(100),
                    vwap_slope VARCHAR(20),
                    liquidity NUMERIC(20, 2),
                    poly_prices JSONB,
                    binance_spot NUMERIC(20, 2),
                    signals_history JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create index on market_slug and timestamp for faster queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_market_timestamp
                ON snapshots(market_slug, timestamp DESC);
            """)

            # Add columns for existing tables
            try:
                await conn.execute("""
                    ALTER TABLE snapshots
                    ADD COLUMN IF NOT EXISTS signals_history JSONB;
                """)
            except Exception:
                # Column might already exist or table is new
                pass

            try:
                await conn.execute("""
                    ALTER TABLE snapshots
                    ADD COLUMN IF NOT EXISTS vwap_slope VARCHAR(20);
                """)
            except Exception:
                pass

    def _remove_rich_tags(self, text: Any) -> str:
        """Remove Rich formatting tags from text."""
        if text is None:
            return ""
        if not isinstance(text, str):
            return str(text)

        import re
        # Remove Rich tags like [color], [/color], [bold], [/bold], etc.
        cleaned = re.sub(r'\[/?[^\]]+\]', '', text)
        return cleaned.strip()

    def _round_numeric(self, value: Any, decimals: int = 2) -> Optional[Decimal]:
        """Round numeric value to specified decimals using Decimal for precision."""
        if value is None:
            return None
        try:
            # Convert to Decimal and round to exactly 2 decimal places
            dec_value = Decimal(str(value))
            # Create quantize string like '0.01' for 2 decimals
            quantizer = Decimal('1').scaleb(-decimals)
            return dec_value.quantize(quantizer, rounding=ROUND_HALF_UP)
        except (ValueError, TypeError):
            return None

    async def save_snapshot(self, display_data: Dict[str, Any], coin: str = "BTC", interval: str = "5m"):
        """Save a panel display data snapshot to the database.

        Args:
            display_data: The display data dictionary from panel
            coin: The coin being tracked (BTC, ETH, SOL)
            interval: The time interval (5m, 15m)
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call init() first.")

        async with self.pool.acquire() as conn:
            import re

            # Parse numeric values from strings and round them
            price_to_beat = self._round_numeric(display_data.get("priceToBeat"), 2)
            current_price = self._extract_price(display_data.get("currentPriceLine"))
            if current_price is not None:
                current_price = self._round_numeric(current_price, 2)

            # Calculate ptb_delta and round it
            ptb_delta = None
            if current_price is not None and price_to_beat is not None:
                ptb_delta = self._round_numeric(current_price - price_to_beat, 2)

            # Parse poly prices
            poly_prices = {}
            poly_header = display_data.get("polyHeader", "")
            up_match = re.search(r'UP[^\d]*(\d+)c', poly_header)
            down_match = re.search(r'DOWN[^\d]*(\d+)c', poly_header)
            if up_match:
                up_value = Decimal(up_match.group(1)) / Decimal('100')
                poly_prices["up"] = float(up_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            if down_match:
                down_value = Decimal(down_match.group(1)) / Decimal('100')
                poly_prices["down"] = float(down_value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

            # Parse predict value
            predict_value = display_data.get("predictValue", "")
            predict_json = {}
            long_match = re.search(r'LONG[^\d]*(\d+)%', predict_value)
            short_match = re.search(r'SHORT[^\d]*(\d+)%', predict_value)
            if long_match:
                predict_json["long"] = int(long_match.group(1))
            if short_match:
                predict_json["short"] = int(short_match.group(1))

            # Parse signal stats
            signal_stats = {}
            signal_stats_str = display_data.get("signalStats", "UPx0 DOWNx0")
            up_match = re.search(r'UPx(\d+)', signal_stats_str)
            down_match = re.search(r'DOWNx(\d+)', signal_stats_str)
            if up_match:
                signal_stats["up"] = int(up_match.group(1))
            if down_match:
                signal_stats["down"] = int(down_match.group(1))

            # Extract binance spot price and round it
            binance_spot = self._extract_price(display_data.get("binanceSpot", ""))
            if binance_spot is not None:
                binance_spot = self._round_numeric(binance_spot, 2)

            # Round liquidity
            liquidity = self._round_numeric(display_data.get("liquidity"), 2)

            # Clean Rich formatting tags from text fields
            heiken_ashi_clean = self._remove_rich_tags(display_data.get("heikenAshi"))

            # RSI - extract pure number without arrow
            rsi_raw = self._remove_rich_tags(display_data.get("rsi"))
            rsi_clean = rsi_raw.split()[0] if rsi_raw else ""

            macd_clean = self._remove_rich_tags(display_data.get("macd"))
            delta_clean = self._remove_rich_tags(display_data.get("delta"))

            # VWAP - extract value and slope separately
            vwap_raw = self._remove_rich_tags(display_data.get("vwap"))
            vwap_clean = vwap_raw.split("|")[0].strip() if vwap_raw else ""
            # Extract slope from VWAP string (format: "value (pct) | slope: UP")
            vwap_slope = ""
            if vwap_raw and "slope:" in vwap_raw:
                slope_part = vwap_raw.split("slope:")[1].strip()
                vwap_slope = slope_part.split()[0] if slope_part else ""

            # Round prices in signals_history
            signals_history = display_data.get("signalsHistory", [])
            cleaned_signals_history = []
            for signal in signals_history:
                up_price = signal.get("up_price")
                down_price = signal.get("down_price")

                cleaned_signal = {
                    "direction": signal.get("direction"),
                    "up_price": float(Decimal(str(up_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) if up_price is not None else None,
                    "down_price": float(Decimal(str(down_price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) if down_price is not None else None,
                    "remaining_time": signal.get("remaining_time"),
                    "timestamp": signal.get("timestamp")
                }
                cleaned_signals_history.append(cleaned_signal)

            await conn.execute("""
                INSERT INTO snapshots (
                    market_slug, title, coin, interval,
                    price_to_beat, current_price, ptb_delta,
                    predict_value, signal, signal_stats,
                    heiken_ashi, rsi, macd, delta, vwap, vwap_slope,
                    liquidity, poly_prices, binance_spot,
                    signals_history
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """,
                display_data.get("marketSlug"),
                display_data.get("title"),
                coin,
                interval,
                price_to_beat,
                current_price,
                ptb_delta,
                json.dumps(predict_json),
                display_data.get("signal"),
                json.dumps(signal_stats),
                heiken_ashi_clean,
                rsi_clean,
                macd_clean,
                delta_clean,
                vwap_clean,
                vwap_slope,
                liquidity,
                json.dumps(poly_prices),
                binance_spot,
                json.dumps(cleaned_signals_history),  # Store cleaned signals history as JSONB
            )

    def _extract_price(self, price_str: Any) -> Optional[Decimal]:
        """Extract numeric price from display string."""
        if price_str is None:
            return None
        if isinstance(price_str, (int, float, Decimal)):
            return Decimal(str(price_str))
        if isinstance(price_str, str):
            import re
            match = re.search(r'\$?([\d,]+\.?\d*)', price_str)
            if match:
                try:
                    return Decimal(match.group(1).replace(',', ''))
                except ValueError:
                    return None
        return None

    async def get_snapshots(self, market_slug: Optional[str] = None, limit: int = 10):
        """Get snapshots from database.

        Args:
            market_slug: Filter by market slug (optional)
            limit: Maximum number of snapshots to return

        Returns:
            List of snapshot dictionaries
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call init() first.")

        async with self.pool.acquire() as conn:
            if market_slug:
                query = """
                    SELECT * FROM snapshots
                    WHERE market_slug = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """
                rows = await conn.fetch(query, market_slug, limit)
            else:
                query = """
                    SELECT * FROM snapshots
                    ORDER BY timestamp DESC
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)

            return [dict(row) for row in rows]

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None


# Global database instance
_db: Optional[Database] = None


async def get_db() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
        await _db.init()
    return _db


async def close_db():
    """Close global database instance."""
    global _db
    if _db:
        await _db.close()
        _db = None
