"""Configuration for panel module."""

import os
from typing import Literal

CoinType = Literal["BTC", "ETH", "SOL"]
IntervalType = Literal["5m", "15m"]


MARKET_CONFIG = {
    "BTC": {
        "symbol": "BTCUSDT",
        "polymarket": {
            "5m": {"series_id": "10684", "series_slug": "btc-up-or-down-5m"},
            "15m": {"series_id": "10192", "series_slug": "btc-up-or-down-15m"}
        },
        "chainlink_aggregator": "0xc907E116054Ad103354f2D350FD2514433D57F6f"
    },
    "ETH": {
        "symbol": "ETHUSDT",
        "polymarket": {
            "5m": {"series_id": "10683", "series_slug": "eth-up-or-down-5m"},
            "15m": {"series_id": "10191", "series_slug": "eth-up-or-down-15m"}
        },
        "chainlink_aggregator": "0x97E7D9eC38E364e1191F62c6C9b8320735F0978d"
    },
    "SOL": {
        "symbol": "SOLUSDT",
        "polymarket": {
            "5m": {"series_id": "10686", "series_slug": "sol-up-or-down-5m"},
            "15m": {"series_id": "10423", "series_slug": "sol-up-or-down-15m"}
        },
        "chainlink_aggregator": "0x4b3EFEB0ec81b6C13E4bef17C9af19D4E091e80F"
    }
}


def get_config(coin: CoinType = "BTC", interval: IntervalType = "5m") -> dict:
    """Get configuration for a given coin and interval."""
    # Validate coin
    valid_coins = list(MARKET_CONFIG.keys())
    validated_coin = coin.upper() if coin.upper() in valid_coins else "BTC"

    # Validate interval
    valid_intervals = ["5m", "15m"]
    validated_interval = interval.lower() if interval.lower() in valid_intervals else "5m"

    market_config = MARKET_CONFIG[validated_coin]
    candle_window_minutes = 5 if validated_interval == "5m" else 15

    return {
        # Runtime config
        "coin": validated_coin,
        "interval": validated_interval,

        "symbol": market_config["symbol"],
        "binance_base_url": "https://api.binance.com",
        "gamma_base_url": "https://gamma-api.polymarket.com",
        "clob_base_url": "https://clob.polymarket.com",

        "poll_interval_ms": 150,
        "candle_window_minutes": candle_window_minutes,

        "vwap_slope_lookback_minutes": 5,
        "rsi_period": 14,
        "rsi_ma_period": 14,

        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,

        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "llm_trigger_ratio": 0.5,  # Trigger when time_left <= 50% of candle window
        },
        "binance_base_url": "https://api.binance.com",
        "gamma_base_url": "https://gamma-api.polymarket.com",
        "clob_base_url": "https://clob.polymarket.com",

        "poll_interval_ms": 150,
        "candle_window_minutes": candle_window_minutes,

        "vwap_slope_lookback_minutes": 5,
        "rsi_period": 14,
        "rsi_ma_period": 14,

        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,

        "polymarket": {
            "market_slug": os.getenv("POLYMARKET_SLUG", ""),
            "series_id": os.getenv("POLYMARKET_SERIES_ID", market_config["polymarket"][validated_interval]["series_id"]),
            "series_slug": os.getenv("POLYMARKET_SERIES_SLUG", market_config["polymarket"][validated_interval]["series_slug"]),
            "auto_select_latest": os.getenv("POLYMARKET_AUTO_SELECT_LATEST", "true").lower() == "true",
            "live_data_ws_url": os.getenv("POLYMARKET_LIVE_WS_URL", "wss://ws-live-data.polymarket.com"),
            "up_outcome_label": os.getenv("POLYMARKET_UP_LABEL", "Up"),
            "down_outcome_label": os.getenv("POLYMARKET_DOWN_LABEL", "Down")
        },

        "chainlink": {
            "polygon_rpc_urls": [s.strip() for s in os.getenv("POLYGON_RPC_URLS", "").split(",") if s.strip()],
            "polygon_rpc_url": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
            "polygon_wss_urls": [s.strip() for s in os.getenv("POLYGON_WSS_URLS", "").split(",") if s.strip()],
            "polygon_wss_url": os.getenv("POLYGON_WSS_URL", ""),
            "aggregator": market_config["chainlink_aggregator"],
            "btc_usd_aggregator": os.getenv("CHAINLINK_BTC_USD_AGGREGATOR", market_config["chainlink_aggregator"])
        }
    }
