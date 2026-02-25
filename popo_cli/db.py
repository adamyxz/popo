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

    async def get_last_snapshot_current_price(self) -> Optional[float]:
        """Get the current_price from the most recent snapshot.

        Returns:
            The current_price from the most recent snapshot, or None if no snapshots exist.
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call init() first.")

        async with self.pool.acquire() as conn:
            query = """
                SELECT current_price
                FROM snapshots
                ORDER BY timestamp DESC
                LIMIT 1
            """
            row = await conn.fetchrow(query)
            if row and row['current_price'] is not None:
                return float(row['current_price'])
            return None

    async def get_snapshots_stats(self):
        """Get comprehensive statistics about snapshots.

        Returns:
            Dictionary with statistics including:
            - total_count: Total number of snapshots
            - up_count: Count where current_price > price_to_beat
            - down_count: Count where current_price < price_to_beat
            - correct_count: Total correct predictions
            - correct_rate: Overall accuracy rate
            - up_correct_count: UP predictions that were correct
            - up_correct_rate: UP prediction accuracy rate
            - down_correct_count: DOWN predictions that were correct
            - down_correct_rate: DOWN prediction accuracy rate
            - up_entry_time_range: Optimal entry time range for UP (in seconds from start)
            - down_entry_time_range: Optimal entry time range for DOWN (in seconds from start)
        """
        if not self.pool:
            raise RuntimeError("Database not initialized. Call init() first.")

        async with self.pool.acquire() as conn:
            # Get all snapshots with relevant fields
            rows = await conn.fetch("""
                SELECT id, price_to_beat, current_price, signal_stats, signals_history, timestamp
                FROM snapshots
                ORDER BY timestamp ASC
            """)

            # Count all snapshots (including those without signals)
            total_count = len(rows)

            # Track counts with signals only
            up_with_signal = 0
            down_with_signal = 0
            up_correct_count = 0
            down_correct_count = 0

            # Store signal times for correct predictions
            up_signal_times = []  # List of seconds from snapshot timestamp
            down_signal_times = []

            for row in rows:
                # Determine direction
                if row['current_price'] and row['price_to_beat']:
                    if row['current_price'] > row['price_to_beat']:
                        direction = 'up'
                    elif row['current_price'] < row['price_to_beat']:
                        direction = 'down'
                    else:
                        direction = 'neutral'
                else:
                    continue

                # Parse signal stats to determine sentiment
                import json
                signal_stats = row['signal_stats']
                if isinstance(signal_stats, str):
                    signal_stats = json.loads(signal_stats)

                up_signals = signal_stats.get('up', 0) if signal_stats else 0
                down_signals = signal_stats.get('down', 0) if signal_stats else 0

                # Skip if no signals (up=0 and down=0 means no prediction)
                if up_signals == 0 and down_signals == 0:
                    continue

                if up_signals > down_signals:
                    sentiment = 'bullish'
                elif down_signals > up_signals:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'

                # Count directions for snapshots with signals
                if direction == 'up':
                    up_with_signal += 1
                elif direction == 'down':
                    down_with_signal += 1

                # Check if prediction was correct
                is_correct = (direction == 'up' and sentiment == 'bullish') or \
                             (direction == 'down' and sentiment == 'bearish')

                if is_correct:
                    if direction == 'up':
                        up_correct_count += 1
                        # Collect signal times for UP
                        signals_history = row['signals_history']
                        if signals_history:
                            if isinstance(signals_history, str):
                                signals_history = json.loads(signals_history)
                            snapshot_time = row['timestamp']
                            # Ensure snapshot_time is timezone-aware
                            if snapshot_time.tzinfo is None:
                                from datetime import timezone
                                snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
                            for sig in signals_history:
                                if sig.get('direction') == 'up':
                                    sig_time = sig.get('timestamp')
                                    if sig_time:
                                        from datetime import datetime
                                        if isinstance(sig_time, str):
                                            sig_dt = datetime.fromisoformat(sig_time.replace('Z', '+00:00'))
                                        else:
                                            sig_dt = sig_time
                                        # Ensure sig_dt is timezone-aware
                                        if sig_dt.tzinfo is None:
                                            from datetime import timezone
                                            sig_dt = sig_dt.replace(tzinfo=timezone.utc)
                                        # Calculate seconds from snapshot start
                                        delta = snapshot_time - sig_dt
                                        # Signals happen before snapshot, so we want how long before
                                        # Store absolute seconds (e.g., 60 seconds before = 60)
                                        up_signal_times.append(abs(int(delta.total_seconds())))

                    elif direction == 'down':
                        down_correct_count += 1
                        # Collect signal times for DOWN
                        signals_history = row['signals_history']
                        if signals_history:
                            if isinstance(signals_history, str):
                                signals_history = json.loads(signals_history)
                            snapshot_time = row['timestamp']
                            # Ensure snapshot_time is timezone-aware
                            if snapshot_time.tzinfo is None:
                                from datetime import timezone
                                snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
                            for sig in signals_history:
                                if sig.get('direction') == 'down':
                                    sig_time = sig.get('timestamp')
                                    if sig_time:
                                        from datetime import datetime
                                        if isinstance(sig_time, str):
                                            sig_dt = datetime.fromisoformat(sig_time.replace('Z', '+00:00'))
                                        else:
                                            sig_dt = sig_time
                                        # Ensure sig_dt is timezone-aware
                                        if sig_dt.tzinfo is None:
                                            from datetime import timezone
                                            sig_dt = sig_dt.replace(tzinfo=timezone.utc)
                                        delta = snapshot_time - sig_dt
                                        down_signal_times.append(abs(int(delta.total_seconds())))

            # Calculate optimal entry time ranges
            # Use interquartile range (IQR) method to find typical signal timing
            def calculate_time_range(times):
                if not times:
                    return {
                        "min": 0,
                        "max": 0,
                        "optimal_min": 0,
                        "optimal_max": 0,
                        "median": 0,
                        "sample_size": 0
                    }
                sorted_times = sorted(times)
                n = len(sorted_times)
                # Use 25th to 75th percentile as optimal range
                q25_idx = int(n * 0.25)
                q75_idx = int(n * 0.75)
                return {
                    "min": sorted_times[0],
                    "max": sorted_times[-1],
                    "optimal_min": sorted_times[q25_idx],
                    "optimal_max": sorted_times[q75_idx],
                    "median": sorted_times[n // 2],
                    "sample_size": n
                }

            up_entry_time_range = calculate_time_range(up_signal_times)
            down_entry_time_range = calculate_time_range(down_signal_times)

            # Calculate rates (only for snapshots with signals)
            with_signal_count = up_with_signal + down_with_signal
            correct_count = up_correct_count + down_correct_count
            correct_rate = (correct_count / with_signal_count * 100) if with_signal_count > 0 else 0
            up_correct_rate = (up_correct_count / up_with_signal * 100) if up_with_signal > 0 else 0
            down_correct_rate = (down_correct_count / down_with_signal * 100) if down_with_signal > 0 else 0

            return {
                "total_count": total_count,
                "up_count": up_with_signal,
                "down_count": down_with_signal,
                "correct_count": correct_count,
                "correct_rate": round(correct_rate, 2),
                "up_correct_count": up_correct_count,
                "up_correct_rate": round(up_correct_rate, 2),
                "down_correct_count": down_correct_count,
                "down_correct_rate": round(down_correct_rate, 2),
                "up_entry_time_range": up_entry_time_range,
                "down_entry_time_range": down_entry_time_range
            }

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None


# Global database instance
_db: Optional[Database] = None


async def get_db() -> Database:
    """Get or create global database instance.

    Returns a valid database connection, creating a new one if needed.
    """
    global _db
    if _db is None or _db.pool is None:
        _db = Database()
        await _db.init()
    return _db


async def close_db():
    """Close global database instance."""
    global _db
    if _db:
        await _db.close()
        _db = None
