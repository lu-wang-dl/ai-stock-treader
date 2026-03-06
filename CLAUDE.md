# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run via launcher (checks dependencies, starts on port 8503)
python run.py

# Or run Streamlit directly
streamlit run app.py --server.port 8503
```

Docker: `docker-compose up -d` (exposes port 8503)

## Architecture

This is an AI-powered quantitative trading platform for US stocks. It uses a **Streamlit** frontend (`app.py`), **Alpaca Markets API** for trading, and LLM models (Databricks-hosted Gemini / DeepSeek) for AI-driven decisions.

### Core Pipeline

The trading decision flow follows this chain:

1. **Data Collection** → `stock_data.py` (StockDataFetcher) fetches from yfinance/akshare/tushare with automatic fallback via `data_source_manager.py`
2. **AI Analysis** → `ai_agents.py` (StockAnalysisAgents) orchestrates multiple analyst agents (technical, fundamental, risk, macro) that call `databricks_client.py` (DatabricksClient) for LLM inference
3. **Decision Engine** → `alpaca_ai_decision.py` (AlpacaAIDecision) aggregates agent analyses into a structured BUY/SELL/HOLD proposal
4. **Hard Risk Firewall** → `hard_decision_firewall.py` (HardDecisionFirewall) validates proposals against non-bypassable rules (session validity, liquidity, spread, position limits, confidence thresholds, circuit breakers, cooldowns)
5. **State Machine** → `trade_state_machine.py` (TradeStateMachine) manages per-symbol states: WAIT → CANDIDATE → ENTERED → MANAGING → EXITED → COOLDOWN
6. **Execution** → `us_stock_trading.py` provides both `USStockTradingInterface` (real Alpaca API) and `USStockTradingSimulator` (local simulator with $100k virtual capital)
7. **Audit** → `audit_logger.py` writes JSONL audit logs to `audit_logs/` for every decision

### Strategy & Auto-Trading Layer

- `alpaca_strategy_manager.py` — manages strategy CRUD in SQLite (`alpaca_strategies.db`), orchestrates the full pipeline above, applies indicator snapshots
- `alpaca_auto_trader.py` — background thread service that periodically executes strategies (default: 5-minute interval)

### Key Data Structures

- `IndicatorSnapshot` (frozen dataclass in `indicator_snapshot.py`) — immutable snapshot of all technical indicators and boolean conditions for a symbol at a point in time
- `LLMProposal` / `FirewallResult` (in `hard_decision_firewall.py`) — structured proposal from LLM and validation result from firewall

### Configuration

- `.env` file managed by `config_manager.py` (ConfigManager singleton) — stores API keys, feature flags, and firewall thresholds
- Firewall parameters (max spread, min liquidity, position size limits, confidence thresholds, stop-loss/take-profit ranges) are all configurable via `.env`

### Trading Modes

Three modes controlled by config: Local Simulator (default, no API needed), Alpaca Paper Trading, and Live Trading.

### Trading Rules (system_prompt.txt)

- Cannot sell a stock on the same day it was bought
- If a stock loses money, cannot buy the same stock for 30 days
