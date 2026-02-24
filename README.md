# Popo - Polymarket Crypto Trading Assistant

A real-time console trading assistant for Polymarket **"Up or Down"** markets with integrated trading functionality.

**Supported markets:**
- **BTC** (Bitcoin) - 5m / 15m
- **ETH** (Ethereum) - 5m / 15m
- **SOL** (Solana) - 5m / 15m

## Features

### 📊 Real-Time Market Analysis
- Polymarket market selection + UP/DOWN prices + liquidity
- Polymarket live WebSocket Chainlink price feed
- Binance spot price for reference
- Technical indicators (Heiken Ashi, RSI, MACD, VWAP, Delta)
- AI-assisted LONG/SHORT prediction

### 💹 Integrated Trading
- **One-click trading** - Place orders directly from the panel
- **Shopping cart system** - Add amount, place order when ready
- **Real-time P&L** - Track profit/loss on open positions
- **Close all positions** - Exit all trades with one keypress
- **Market-aware filtering** - Only shows orders for current market

### 🎮 Keyboard Controls
| Key | Action |
|-----|--------|
| `←` (Left Arrow) | Add $1 to UP cart |
| `→` (Right Arrow) | Add $1 to DOWN cart |
| `Enter` | Place order |
| `Space` | Close all positions |
| `q` | Quit |

## Requirements

- Python **3.10+**
- PostgreSQL (for storing snapshots and orders)
- uv (Python package manager) - recommended

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/adamxyz/popo.git
cd popo
pip install uv
uv sync
```

### 2. Setup Database

```bash
# Create PostgreSQL database
createdb popo
```

### 3. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your configuration
# Required: DATABASE_URL
# Optional: POLYMARKET_PRIVATE_KEY (for trading)
```

### 4. Run the Panel

```bash
uv run popo
```

Then use the `/panel` command:

```
/panel BTC 5m    # Bitcoin 5-minute panel
/panel ETH 15m   # Ethereum 15-minute panel
/panel SOL 5m    # Solana 5-minute panel
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required: PostgreSQL database
DATABASE_URL=postgresql://user:password@localhost:5432/popo

# Optional: Polymarket trading
POLYMARKET_PRIVATE_KEY=your_private_key_here
POLYMARKET_FUNDER=your_funder_address  # Optional
POLYMARKET_SIGNATURE_TYPE=2  # 0, 1, or 2
```

### Getting Your Private Key

1. Open MetaMask
2. Click Account Details → Export Private Key
3. Copy the private key (64 character hex string)
4. Paste it in `.env` under `POLYMARKET_PRIVATE_KEY`

**⚠️ WARNING:** Never share your private key or commit `.env` to version control!

## Panel UI Layout

```
┌─────────────────────────────────────┐
│       Main Panel (Blue)              │
│  - Market title & slug               │
│  - PRICE TO BEAT                     │
│  - UP/DOWN prediction boxes          │
│  - Technical indicators              │
├─────────────────────────────────────┤
│     Trading Panel (Yellow)           │
│                                     │
│  ━━━ Cart ━━━                      │
│    Direction: UP                    │
│    Amount: $10                      │
│                                     │
│  ━━━ Open Positions ━━━            │
│    UP: 2 orders | P&L: +$2.50        │
│    DOWN: 0 orders | P&L: $0.00       │
│                                     │
│  ━━━ Controls ━━━                  │
│    ← UP | → DOWN | Enter: Buy       │
│    Space: Sell All | q: Quit        │
└─────────────────────────────────────┘
```

## Trading Features

### Shopping Cart System

- **Single direction** - Cart can only hold UP or DOWN, not both
- **Auto-clear on switch** - Switching directions clears the cart
- **Auto-clear on market change** - New market = empty cart
- **Immediate refresh** - Panel updates instantly on keypress

### Order Management

- Orders are stored in PostgreSQL with entry/exit snapshots
- P&L calculated in real-time based on current market prices
- Only current market orders are displayed in Open Positions
- Close all positions with one Space keypress

### Signature Auto-Fallback

The system automatically tries 3 signature types:
1. GNOSIS_SAFE (type 2)
2. POLY_PROXY (type 1)
3. EIP712 (type 0)

## Trading Workflow

### Placing an Order

1. Press `←` (left arrow) to add UP amount
2. Press `←` again to increase amount
3. Press `Enter` to place order
4. Cart clears automatically on success

### Example

```
Press ←  → Cart: UP $1
Press ←  → Cart: UP $2
Press ←  → Cart: UP $3
Press Enter → Order placed! Cart: EMPTY $0
```

### Closing Positions

- Press `Space` to close all open positions
- Each position's status is displayed
- P&L is calculated and shown

## Market Switching

When a market ends and a new one begins:
1. Cart is automatically cleared
2. Open Positions only show new market orders
3. Old market orders are filtered out
4. Message: `Market switched - cart cleared`

## Troubleshooting

### Trading service not configured

- Make sure `POLYMARKET_PRIVATE_KEY` is set in `.env`
- Verify the key is 64 hex characters (without 0x prefix)

### Database connection failed

- Verify PostgreSQL is running
- Check `DATABASE_URL` is correct
- Create database: `createdb popo`

### Keyboard not responding

- Make sure your terminal supports raw mode
- Try using a different terminal (Terminal.app, iTerm2, etc.)
- Some IDE terminals may not support keyboard input

## Safety

This is not financial advice. Use at your own risk.

- Start with small amounts to test the system
- Never share your private key
- Keep your `.env` file secure
- Monitor your positions actively

## Credits

- Original concept by @krajekis
- Refactored to Python CLI with Rich UI by @adamxyz
- Trading integration by @adamxyz

## License

See LICENSE file for details.
