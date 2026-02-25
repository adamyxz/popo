"""Database models and repository for trading orders."""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from decimal import Decimal, ROUND_HALF_UP
import asyncpg
from dotenv import load_dotenv

load_dotenv()


class OrderRepository:
    """Repository for order operations."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def init_tables(self):
        """Create orders table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(255) UNIQUE NOT NULL,
                    market_id VARCHAR(255),
                    token_id VARCHAR(255),
                    direction VARCHAR(10),
                    side VARCHAR(10) DEFAULT 'BUY',
                    order_type VARCHAR(10) DEFAULT 'market',
                    price FLOAT,
                    size FLOAT,
                    original_size FLOAT,
                    token_quantity FLOAT,
                    size_matched FLOAT,
                    value FLOAT,
                    time_in_force VARCHAR(10) DEFAULT 'FOK',
                    status VARCHAR(50),
                    notes TEXT,
                    raw_response TEXT,
                    entry JSONB,
                    exit JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_market_id
                ON orders(market_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_token_id
                ON orders(token_id);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_status
                ON orders(status);
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_direction
                ON orders(direction);
            """)

    async def create_order(
        self,
        order_id: str,
        market_id: str,
        token_id: str,
        direction: str,
        side: str,
        price: Optional[float] = None,
        shares: Optional[float] = None,
        amount_paid: Optional[float] = None,
        amount_in_dollars: Optional[float] = None,
        status: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new order."""
        async with self.pool.acquire() as conn:
            # For BUY: size = dollars, value = dollars
            # For SELL: size = shares, value = amount_paid
            if side == "BUY":
                size = amount_in_dollars
                value = amount_in_dollars
            else:
                size = shares
                value = amount_paid

            await conn.execute("""
                INSERT INTO orders (
                    order_id, market_id, token_id, direction, side,
                    price, size, original_size, token_quantity,
                    size_matched, value, status, raw_response, entry
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                order_id, market_id, token_id, direction, side,
                price, size, shares, shares,
                shares, value, status,
                json.dumps(raw_response) if raw_response else None,
                json.dumps(entry) if entry else None,
            )

            return {
                "order_id": order_id,
                "market_id": market_id,
                "token_id": token_id,
                "direction": direction,
                "side": side,
                "price": price,
                "shares": shares,
                "amount_paid": amount_paid,
                "amount_in_dollars": amount_in_dollars,
                "status": status,
            }

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by order_id."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM orders WHERE order_id = $1",
                order_id
            )
            if row:
                return dict(row)
            return None

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders (matched/active status)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders
                WHERE status IN ('matched', 'open', 'active', 'live', 'pending')
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in rows]

    async def get_orders_by_direction(self, direction: str) -> List[Dict[str, Any]]:
        """Get orders by direction (up/down)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders
                WHERE direction = $1
                ORDER BY created_at DESC
            """, direction)
            return [dict(row) for row in rows]

    async def update_order_status(
        self,
        order_id: str,
        status: str,
        exit_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update order status."""
        async with self.pool.acquire() as conn:
            if exit_data:
                await conn.execute("""
                    UPDATE orders
                    SET status = $1, exit = $2, updated_at = NOW()
                    WHERE order_id = $3
                """, status, json.dumps(exit_data), order_id)
            else:
                await conn.execute("""
                    UPDATE orders
                    SET status = $1, updated_at = NOW()
                    WHERE order_id = $2
                """, status, order_id)
            return True

    async def get_latest_buy_order(
        self,
        direction: str,
        market_id: str,
        minutes_ago: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent BUY order for a direction."""
        async with self.pool.acquire() as conn:
            recent_time = datetime.now(timezone.utc) - __import__('datetime').timedelta(minutes=minutes_ago)
            row = await conn.fetchrow("""
                SELECT * FROM orders
                WHERE side = 'BUY'
                AND direction = $1
                AND market_id = $2
                AND status = 'matched'
                AND created_at >= $3
                ORDER BY created_at DESC
                LIMIT 1
            """, direction, market_id, recent_time)
            if row:
                return dict(row)
            return None

    async def get_open_orders_by_market(self, market_id: str) -> List[Dict[str, Any]]:
        """Get all open orders for a specific market."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM orders
                WHERE market_id = $1
                AND status IN ('matched', 'open', 'active', 'live', 'pending')
                ORDER BY created_at DESC
            """, market_id)
            return [dict(row) for row in rows]

    async def close_orders_for_market(
        self,
        market_id: str,
        reason: str = "market_closed"
    ) -> int:
        """Close all open orders for a specific market when the market ends.

        Returns the number of orders closed.
        """
        async with self.pool.acquire() as conn:
            exit_data = {
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "market_ended": True
            }
            result = await conn.execute("""
                UPDATE orders
                SET status = 'market_closed',
                    exit = $2,
                    updated_at = NOW()
                WHERE market_id = $1
                AND status IN ('matched', 'open', 'active', 'live', 'pending')
            """, market_id, json.dumps(exit_data))

            # Parse "UPDATE n" result to get count
            count = int(result.split()[-1]) if result else 0
            return count


# Global database pool
_pool: Optional[asyncpg.Pool] = None


async def get_trading_pool() -> asyncpg.Pool:
    """Get or create database pool for trading."""
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        _pool = await asyncpg.create_pool(database_url)

        # Initialize tables
        repo = OrderRepository(_pool)
        await repo.init_tables()

    return _pool


async def close_trading_pool():
    """Close the trading database pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
