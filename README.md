# Popo - Polymarket Crypto Assistant

A real-time console trading assistant for Polymarket **"Up or Down"** markets.

**Supported markets:**
- **BTC** (Bitcoin) - 5m / 15m
- **ETH** (Ethereum) - 5m / 15m
- **SOL** (Solana) - 5m / 15m

**Default:** BTC 5m

It combines:
- Polymarket market selection + UP/DOWN prices + liquidity
- Polymarket live WS **Chainlink CURRENT PRICE** (same feed shown on the Polymarket UI)
- Fallback to on-chain Chainlink (Polygon) via HTTP/WSS RPC
- Binance spot price for reference
- Short-term TA snapshot (Heiken Ashi, RSI, MACD, VWAP, Delta 1/3m)
- A simple live **Predict (LONG/SHORT %)** derived from the assistant's current TA scoring
- **NEW: Rich-based UI with color-coded UP/DOWN prediction boxes**

## Requirements

- Python **3.10+** (https://python.org)
- uv (Python package manager) - recommended for faster installs

## Run from terminal (step-by-step)

### 1) Clone the repository

```bash
git clone https://github.com/adamxyz/popo.git
```

Alternative (no git):

- Click the green `<> Code` button on GitHub
- Choose `Download ZIP`
- Extract the ZIP
- Open a terminal in the extracted project folder

Then open a terminal in the project folder.

### 2) Install dependencies

Using `uv` (recommended):

```bash
pip install uv
uv sync
```

Or using `pip`:

```bash
pip install -e .
```

### 3) Run the CLI

```bash
uv run popo
```

Then use the `/panel` command to start the trading panel:

```
/panel BTC 5m    # Bitcoin 5-minute panel
/panel ETH 15m   # Ethereum 15-minute panel
/panel SOL 5m    # Solana 5-minute panel
```

### 4) (Optional) Set environment variables

You can run without extra config (defaults are included), but for more stable Chainlink fallback it's recommended to set at least one Polygon RPC.

#### Windows PowerShell (current terminal session)

```powershell
$env:POLYGON_RPC_URL = "https://polygon-rpc.com"
$env:POLYGON_RPC_URLS = "https://polygon-rpc.com,https://rpc.ankr.com/polygon"
$env:POLYGON_WSS_URLS = "wss://polygon-bor-rpc.publicnode.com"
```

Optional Polymarket settings:

```powershell
$env:POLYMARKET_AUTO_SELECT_LATEST = "true"
# $env:POLYMARKET_SLUG = "btc-updown-15m-..."   # pin a specific market
```

#### Windows CMD (current terminal session)

```cmd
set POLYGON_RPC_URL=https://polygon-rpc.com
set POLYGON_RPC_URLS=https://polygon-rpc.com,https://rpc.ankr.com/polygon
set POLYGON_WSS_URLS=wss://polygon-bor-rpc.publicnode.com
```

Optional Polymarket settings:

```cmd
set POLYMARKET_AUTO_SELECT_LATEST=true
REM set POLYMARKET_SLUG=btc-updown-15m-...
```

Notes:
- These environment variables apply only to the current terminal window.
- If you want permanent env vars, set them via Windows System Environment Variables or use a `.env` loader of your choice.

## Panel UI Layout

The panel interface uses Rich for beautiful, color-coded terminal output:

### Top Section
- Market title and slug
- **PRICE TO BEAT** (prominent red background highlighting)
- Current price with delta from price to beat
- **TIME LEFT** countdown (yellow highlight)

### Upper Middle Section
Two large colored boxes showing:

| **UP Box** (Green) | **DOWN Box** (Red) |
|---|---|
| Polymarket price (e.g., 54c) | Polymarket price (e.g., 45c) |
| Prediction probability (e.g., 65%) | Prediction probability (e.g., 35%) |
| Signal count (e.g., 10x) | Signal count (e.g., 0x) |

### Lower Section - Indicators
- Signal (BUY UP / BUY DOWN / NO TRADE) with stats
- Heiken Ashi trend
- RSI with direction arrow
- MACD (bullish/bearish)
- Delta 1/3 (price change over 1min and 3min)
- VWAP (Volume Weighted Average Price) with slope
- Liquidity
- Binance Spot price

## Configuration

The coin and interval can be selected via command line arguments (see "Run" section).

This project also reads configuration from environment variables.

### Polymarket

- `POLYMARKET_AUTO_SELECT_LATEST` (default: `true`)
  - When `true`, automatically picks the latest market for the selected coin/interval.
- `POLYMARKET_SERIES_ID` (auto-selected based on coin/interval)
  - BTC 5m: `10684`
  - BTC 15m: `10192`
  - ETH 5m: `10683`
  - ETH 15m: `10191`
  - SOL 5m: `10686`
  - SOL 15m: `10423`
- `POLYMARKET_SERIES_SLUG` (auto-selected based on coin/interval)
  - Format: `{coin}-up-or-down-{interval}`
- `POLYMARKET_SLUG` (optional)
  - If set, the assistant will target a specific market slug (overrides auto-selection).
- `POLYMARKET_LIVE_WS_URL` (default: `wss://ws-live-data.polymarket.com`)

### Chainlink on Polygon (fallback)

- `CHAINLINK_BTC_USD_AGGREGATOR` (BTC only, default: `0xc907E116054Ad103354f2D350FD2514433D57F6f`)
- Other aggregators are auto-selected based on coin:
  - ETH: `0x97E7D9eC38E364e1191F62C6C9b8320735F0978d`
  - SOL: `0x4b3EFEB0ec81b6C13E4bef17C9af19D4E091e80F`

HTTP RPC:
- `POLYGON_RPC_URL` (default: `https://polygon-rpc.com`)
- `POLYGON_RPC_URLS` (optional, comma-separated)
  - Example: `https://polygon-rpc.com,https://rpc.ankr.com/polygon`

WSS RPC (optional but recommended for more real-time fallback):
- `POLYGON_WSS_URL` (optional)
- `POLYGON_WSS_URLS` (optional, comma-separated)

### Proxy support

The bot supports HTTP(S) proxies for both HTTP requests and WebSocket connections.

Supported env vars (standard):

- `HTTPS_PROXY` / `https_proxy`
- `HTTP_PROXY` / `http_proxy`
- `ALL_PROXY` / `all_proxy`

Examples:

PowerShell:

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:8080"
# or
$env:ALL_PROXY = "socks5://127.0.0.1:1080"
```

CMD:

```cmd
set HTTPS_PROXY=http://127.0.0.1:8080
REM or
set ALL_PROXY=socks5://127.0.0.1:1080
```

#### Proxy with username + password (simple guide)

1) Take your proxy host and port (example: `1.2.3.4:8080`).

2) Add your login and password in the URL:

- HTTP/HTTPS proxy:
  - `http://USERNAME:PASSWORD@HOST:PORT`
- SOCKS5 proxy:
  - `socks5://USERNAME:PASSWORD@HOST:PORT`

3) Set it in the terminal and run the bot.

PowerShell:

```powershell
$env:HTTPS_PROXY = "http://USERNAME:PASSWORD@HOST:PORT"
uv run popo
```

CMD:

```cmd
set HTTPS_PROXY=http://USERNAME:PASSWORD@HOST:PORT
uv run popo
```

Important: if your password contains special characters like `@` or `:` you must URL-encode it.

Example:

- password: `p@ss:word`
- encoded: `p%40ss%3Aword`
- proxy URL: `http://user:p%40ss%3Aword@1.2.3.4:8080`

## CLI Commands

### Available Commands

- `/hi` - Say hi
- `/panel [coin] [interval]` - Start trading panel
  - Coins: BTC, ETH, SOL
  - Intervals: 5m, 15m
  - Example: `/panel BTC 5m`
- `/help` - Show help message
- `/exit` or `/quit` - Exit the CLI

### Stopping the Panel

Press `Ctrl + C` in the terminal.

### Update to latest version

```bash
git pull
uv sync
uv run popo
```

## Notes / Troubleshooting

- If you see no Chainlink updates:
  - Polymarket WS might be temporarily unavailable. The bot falls back to Chainlink on-chain price via Polygon RPC.
  - Ensure at least one working Polygon RPC URL is configured.
- If the console displays garbled characters on Windows:
  - This is due to Windows console using GBK encoding instead of UTF-8.
  - The layout structure is correct; modern terminals (Windows Terminal, VSCode terminal) will display properly.

## Safety

This is not financial advice. Use at your own risk.

---

created by @krajekis
Refactored to Python CLI with Rich UI by @adamxyz
