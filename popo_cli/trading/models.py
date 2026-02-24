"""Trading models for order requests and responses."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class PlaceOrderRequest:
    """Request to place a FOK (Fill-Or-Kill) order."""
    market_id: str
    token_id: str
    direction: str  # "up" or "down"
    side: str  # "BUY" or "SELL"
    amount_in_dollars: float = 10.0  # Amount in USD for BUY orders


@dataclass
class OrderInfo:
    """Information about an order."""
    order_id: str
    market_id: str
    token_id: str
    direction: str
    side: str
    price: Optional[float] = None
    shares: Optional[float] = None
    amount_paid: Optional[float] = None
    amount_in_dollars: Optional[float] = None
    status: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class PlaceOrderResponse:
    """Response from placing an order."""
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None
    price: Optional[float] = None
    shares: Optional[float] = None
    amount_paid: Optional[float] = None
    amount_in_dollars: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class CloseOrderResponse:
    """Response from closing an order."""
    success: bool
    order_id: str
    status: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
