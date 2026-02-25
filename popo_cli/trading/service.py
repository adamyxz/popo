"""Polymarket trading service for order execution."""

import os
import logging
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from .models import PlaceOrderRequest, PlaceOrderResponse, CloseOrderResponse, OrderInfo
from .db import OrderRepository, get_trading_pool

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Trading wallet configuration."""
    private_key: str
    funder: Optional[str] = None
    signature_type: int = 2
    chain_id: int = 137
    clob_host: str = "https://clob.polymarket.com"
    address: Optional[str] = None


class PolymarketTradingService:
    """Service for executing Polymarket trades."""

    def __init__(self):
        self._config: Optional[TradingConfig] = None
        self._client: Any = None
        self._order_repo: Optional[OrderRepository] = None

    async def init(self):
        """Initialize the trading service from environment variables."""
        # Load config from environment
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set")
            return False

        funder = os.getenv("POLYMARKET_FUNDER", "").strip() or None
        signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))

        self._config = TradingConfig(
            private_key=private_key,
            funder=funder,
            signature_type=signature_type,
        )

        # Initialize database
        try:
            pool = await get_trading_pool()
            self._order_repo = OrderRepository(pool)
        except Exception as e:
            logger.error(f"Failed to initialize trading database: {e}")
            return False

        # Build client
        try:
            self._client = self._build_client()
            logger.info(f"Trading service initialized with signature_type={signature_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to build trading client: {e}")
            return False

    def _build_client(self, signature_type: Optional[int] = None) -> Any:
        """Build a ClobClient."""
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            raise RuntimeError(
                "py-clob-client is not installed. Install with: pip install py-clob-client"
            )

        if not self._config:
            raise RuntimeError("Trading config not initialized")

        sig_type = signature_type or self._config.signature_type

        kwargs = {
            "host": self._config.clob_host,
            "key": self._config.private_key,
            "chain_id": self._config.chain_id,
            "signature_type": sig_type,
        }
        if self._config.funder:
            kwargs["funder"] = self._config.funder

        client = ClobClient(**kwargs)

        # Derive API credentials
        try:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
        except Exception as e:
            # This is expected for some wallet types, ignore silently
            pass

        # Resolve address
        self._config.address = self._resolve_address(client)
        return client

    def _resolve_address(self, client: Any) -> Optional[str]:
        """Resolve wallet address from client."""
        candidates = [
            getattr(client, "address", None),
            getattr(getattr(client, "signer", None), "address", None),
            getattr(getattr(client, "signer", None), "addr", None),
            self._config.funder,
        ]

        for candidate in candidates:
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception:
                    continue
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def is_configured(self) -> bool:
        """Check if trading service is configured."""
        return self._client is not None and self._config is not None

    async def place_order(
        self,
        request: PlaceOrderRequest,
        neg_risk: Optional[bool] = None,
    ) -> PlaceOrderResponse:
        """Place a FOK (Fill-Or-Kill) market order."""
        if not self.is_configured():
            return PlaceOrderResponse(
                success=False,
                error="Trading service not configured. Set POLYMARKET_PRIVATE_KEY in .env"
            )

        if not self._order_repo:
            return PlaceOrderResponse(
                success=False,
                error="Order repository not initialized"
            )

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType as ClobOrderType
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError:
            return PlaceOrderResponse(
                success=False,
                error="py-clob-client not installed"
            )

        side = BUY if request.side == "BUY" else SELL
        order_type_enum = ClobOrderType.FOK

        # Calculate amount
        if side == BUY:
            amount = request.amount_in_dollars
        else:
            # SELL: check on-chain balance or recent BUY orders
            amount = await self._get_sell_amount(request.token_id, request.direction, request.market_id)

            # Check and update allowance
            await self._ensure_allowance(request.token_id)

        # Build order args
        try:
            order_args = MarketOrderArgs(
                side=side,
                token_id=request.token_id,
                order_type=order_type_enum,
                amount=amount,
            )
            if neg_risk is not None:
                # Add neg_risk if supported
                try:
                    order_args.neg_risk = neg_risk
                except (TypeError, AttributeError):
                    pass
        except Exception as e:
            return PlaceOrderResponse(
                success=False,
                error=f"Failed to build order args: {e}"
            )

        logger.info(f"Placing FOK {request.side} order: amount={amount}, token_id={request.token_id}")

        # Try to post order with signature fallback
        result = await self._post_order_with_fallback(
            order_args, order_type_enum, request
        )

        return result

    async def _get_sell_amount(self, token_id: str, direction: str, market_id: str) -> float:
        """Get amount to sell - from on-chain balance or recent BUY orders."""
        # Check on-chain balance
        try:
            allowance_info = await self.check_allowance(token_id)
            balance_raw = allowance_info.get('balance', '0')

            try:
                if isinstance(balance_raw, str):
                    balance_val = int(balance_raw) / 1e6
                else:
                    balance_val = float(balance_raw or 0) / 1e6
            except:
                balance_val = 0

            logger.info(f"On-chain balance for token {token_id}: {balance_val}")

            if balance_val > 0:
                return balance_val
        except Exception as e:
            logger.warning(f"Failed to check on-chain balance: {e}")

        # Fall back to recent BUY order
        if self._order_repo:
            recent_buy = await self._order_repo.get_latest_buy_order(direction, market_id)
            if recent_buy and recent_buy.get("token_quantity"):
                amount = recent_buy["token_quantity"]
                logger.info(f"Using recent BUY order amount: {amount}")
                return amount

        # Default fallback
        logger.warning("No balance found, using default amount of 1")
        return 1.0

    async def _ensure_allowance(self, token_id: str):
        """Ensure allowance is set for token."""
        try:
            allowance_info = await self.check_allowance(token_id)
            allowance = allowance_info.get('allowance', '0')

            try:
                allowance_val = int(allowance)
            except:
                allowance_val = 0

            INFINITE_ALLOWANCE = 10**30
            if allowance_val >= INFINITE_ALLOWANCE:
                return  # Already has infinite allowance

            # Try to update allowance
            logger.info(f"Updating allowance for token {token_id}...")
            await self.update_allowance(token_id)
        except Exception as e:
            logger.warning(f"Allowance update failed (continuing): {e}")

    async def _post_order_with_fallback(
        self,
        order_args: Any,
        order_type_enum: Any,
        request: PlaceOrderRequest,
    ) -> PlaceOrderResponse:
        """Post order with signature type fallback."""
        from py_clob_client.order_builder.constants import BUY, SELL

        async def try_post(client: Any) -> Dict[str, Any]:
            signed_order = client.create_market_order(order_args)
            result = client.post_order(signed_order, order_type_enum)
            return result if isinstance(result, dict) else {"result": result}

        def is_signature_error(exc: Exception) -> bool:
            error_str = str(exc).lower()
            return "invalid signature" in error_str or "signature verification failed" in error_str

        # Try current config
        try:
            result = await try_post(self._client)
            return await self._parse_order_result(result, request)
        except Exception as exc:
            if not is_signature_error(exc):
                return PlaceOrderResponse(
                    success=False,
                    error=f"Order failed: {exc}"
                )

        # Try fallback signature types
        for sig_type in [0, 1]:
            if sig_type == self._config.signature_type:
                continue

            try:
                logger.info(f"Trying signature_type={sig_type}...")
                next_client = self._build_client(sig_type)
                result = await try_post(next_client)

                # Success - update config
                self._config.signature_type = sig_type
                self._client = next_client
                logger.info(f"Order succeeded with signature_type={sig_type}")

                return await self._parse_order_result(result, request)
            except Exception as exc:
                if is_signature_error(exc):
                    logger.warning(f"signature_type={sig_type} failed")
                else:
                    return PlaceOrderResponse(
                        success=False,
                        error=f"Order failed with signature_type={sig_type}: {exc}"
                    )

        return PlaceOrderResponse(
            success=False,
            error="Order failed with all signature types"
        )

    async def _parse_order_result(
        self,
        result: Dict[str, Any],
        request: PlaceOrderRequest,
    ) -> PlaceOrderResponse:
        """Parse order result and save to database."""
        from py_clob_client.order_builder.constants import BUY, SELL

        # Extract order ID
        order_id = self._first_text(result, ["orderID", "orderId", "order_id", "id"])
        if not order_id:
            return PlaceOrderResponse(
                success=False,
                error="No order ID in response"
            )

        status = self._first_text(result, ["status", "state"]) or "matched"

        # Parse amounts
        making_amount_raw = self._first_text(result, ["makingAmount", "making_amount"])
        taking_amount_raw = self._first_text(result, ["takingAmount", "taking_amount"])

        making_amount = self._parse_amount(making_amount_raw)
        taking_amount = self._parse_amount(taking_amount_raw)

        # Calculate price, shares, amount_paid
        price = None
        shares = None
        amount_paid = None

        if request.side == "BUY":
            if making_amount is not None and taking_amount is not None and taking_amount > 0:
                price = making_amount / taking_amount
                shares = taking_amount
                amount_paid = making_amount
            else:
                return PlaceOrderResponse(
                    success=True,
                    order_id=order_id,
                    status=status,
                    error="Incomplete response data"
                )
        else:  # SELL
            if making_amount is not None and taking_amount is not None and making_amount > 0:
                price = taking_amount / making_amount
                shares = making_amount
                amount_paid = taking_amount
            else:
                return PlaceOrderResponse(
                    success=True,
                    order_id=order_id,
                    status=status,
                    error="Incomplete response data"
                )

        # Save to database
        await self._order_repo.create_order(
            order_id=order_id,
            market_id=request.market_id,
            token_id=request.token_id,
            direction=request.direction,
            side=request.side,
            price=price,
            shares=shares,
            amount_paid=amount_paid,
            amount_in_dollars=request.amount_in_dollars,
            status=status,
            raw_response=result,
            entry={"timestamp": datetime.now(timezone.utc).isoformat()},
        )

        return PlaceOrderResponse(
            success=True,
            order_id=order_id,
            status=status,
            price=price,
            shares=shares,
            amount_paid=amount_paid,
            amount_in_dollars=request.amount_in_dollars,
            raw=result,
        )

    def _parse_amount(self, raw: Optional[str]) -> Optional[float]:
        """Parse amount from raw string (may have 1e6 precision)."""
        if not raw:
            return None
        try:
            val = float(raw)
            return val / 1e6 if val > 10000 else val
        except (ValueError, TypeError):
            return None

    def _first_text(self, payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
        """Get first matching text value from dict."""
        for key in keys:
            if key in payload:
                value = payload.get(key)
                if value:
                    return str(value).strip()
        return None

    async def close_order(self, order_id: str) -> CloseOrderResponse:
        """Close/sell all positions for an order."""
        if not self.is_configured():
            return CloseOrderResponse(
                success=False,
                order_id=order_id,
                error="Trading service not configured"
            )

        if not self._order_repo:
            return CloseOrderResponse(
                success=False,
                order_id=order_id,
                error="Order repository not initialized"
            )

        # Get order info
        order = await self._order_repo.get_order(order_id)
        if not order:
            return CloseOrderResponse(
                success=False,
                order_id=order_id,
                error="Order not found"
            )

        # Only close BUY orders (SELL orders don't need closing)
        if order.get("side") != "BUY":
            return CloseOrderResponse(
                success=False,
                order_id=order_id,
                error="Only BUY orders can be closed"
            )

        # Create SELL order
        sell_request = PlaceOrderRequest(
            market_id=order["market_id"],
            token_id=order["token_id"],
            direction=order["direction"],
            side="SELL",
            amount_in_dollars=0,  # Ignored for SELL
        )

        result = await self.place_order(sell_request)

        if result.success:
            # Update original order status
            await self._order_repo.update_order_status(
                order_id,
                "closed",
                exit_data={"closed_at": datetime.now(timezone.utc).isoformat()}
            )

        return CloseOrderResponse(
            success=result.success,
            order_id=order_id,
            status=result.status,
            error=result.error,
            raw=result.raw,
        )

    async def close_all_orders(self) -> List[CloseOrderResponse]:
        """Close all open BUY orders."""
        if not self._order_repo:
            return []

        open_orders = await self._order_repo.get_open_orders()
        results = []

        for order in open_orders:
            if order.get("side") == "BUY":
                result = await self.close_order(order["order_id"])
                results.append(result)

        return results

    async def get_open_orders(self) -> List[OrderInfo]:
        """Get all open orders with PnL calculation."""
        if not self._order_repo:
            return []

        orders = await self._order_repo.get_open_orders()
        result = []

        for order in orders:
            # Calculate PnL
            pnl = None
            pnl_pct = None

            if order.get("side") == "BUY" and order.get("price"):
                # For BUY orders: PnL = (current_price - entry_price) * shares
                # We need current price - for now just return order info
                pass

            result.append(OrderInfo(
                order_id=order["order_id"],
                market_id=order["market_id"],
                token_id=order["token_id"],
                direction=order["direction"],
                side=order["side"],
                price=order.get("price"),
                shares=order.get("token_quantity"),
                amount_paid=order.get("value"),
                status=order.get("status"),
                created_at=order.get("created_at"),
            ))

        return result

    async def get_orders_by_direction(self, direction: str) -> List[OrderInfo]:
        """Get orders by direction (up/down)."""
        if not self._order_repo:
            return []

        orders = await self._order_repo.get_orders_by_direction(direction)
        result = []

        for order in orders:
            result.append(OrderInfo(
                order_id=order["order_id"],
                market_id=order["market_id"],
                token_id=order["token_id"],
                direction=order["direction"],
                side=order["side"],
                price=order.get("price"),
                shares=order.get("token_quantity"),
                amount_paid=order.get("value"),
                status=order.get("status"),
                created_at=order.get("created_at"),
            ))

        return result

    async def close_orders_for_market(self, market_id: str) -> Dict[str, Any]:
        """Close all open orders for a specific market when it ends.

        This updates the order status to 'market_closed' without attempting
        to sell, since trading is no longer possible for ended markets.

        Returns a dict with the count of orders closed and details.
        """
        if not self._order_repo:
            return {"success": False, "error": "Order repository not initialized", "count": 0}

        try:
            # Get the orders that will be closed for logging
            open_orders = await self._order_repo.get_open_orders_by_market(market_id)

            # Close all orders for this market
            count = await self._order_repo.close_orders_for_market(market_id, reason="market_ended")

            logger.info(f"Closed {count} order(s) for market {market_id}")

            return {
                "success": True,
                "count": count,
                "market_id": market_id,
                "orders": [
                    {
                        "order_id": o["order_id"],
                        "side": o["side"],
                        "direction": o["direction"],
                        "value": o.get("value"),
                    }
                    for o in open_orders
                ]
            }
        except Exception as e:
            logger.error(f"Failed to close orders for market {market_id}: {e}")
            return {"success": False, "error": str(e), "count": 0}

    async def check_allowance(self, token_id: str) -> Dict[str, Any]:
        """Check token balance and allowance."""
        if not self.is_configured():
            raise RuntimeError("Trading service not configured")

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(
                token_id=token_id,
                asset_type=AssetType.CONDITIONAL
            )
            balance_allowance = self._client.get_balance_allowance(params)

            return {
                "token_id": token_id,
                "balance": balance_allowance.get("balance"),
                "allowance": balance_allowance.get("allowance"),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to check allowance: {e}")

    async def update_allowance(self, token_id: str) -> Dict[str, Any]:
        """Update token allowance."""
        if not self.is_configured():
            raise RuntimeError("Trading service not configured")

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(
                token_id=token_id,
                asset_type=AssetType.CONDITIONAL
            )
            result = self._client.update_balance_allowance(params)

            return {
                "token_id": token_id,
                "success": True,
                "result": result,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to update allowance: {e}")


# Global trading service instance
_trading_service: Optional[PolymarketTradingService] = None


async def get_trading_service() -> PolymarketTradingService:
    """Get or create global trading service instance."""
    global _trading_service
    if _trading_service is None:
        _trading_service = PolymarketTradingService()
        await _trading_service.init()
    return _trading_service
