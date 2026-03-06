"""
Stock Picker Agent
AI-powered agent that selects the best stocks to buy from a candidate pool.
Uses technical screening + LLM ranking to pick top N stocks.
"""

import logging
import time
from typing import Dict, List, Any, Optional

import yfinance as yf

from databricks_client import DatabricksClient
from yfinance_stock_advisor import YFinanceStockAdvisor


class StockPickerAgent:
    """
    AI agent that selects stocks to buy.

    Pipeline:
    1. Build candidate pool from sectors / screeners / custom list
    2. Run technical screening on candidates (via YFinanceStockAdvisor)
    3. Send screened candidates to LLM for intelligent ranking
    4. Return top N picks with reasoning
    """

    def __init__(self, model: str = "databricks-gemini-3-pro"):
        self.logger = logging.getLogger(__name__)
        self.advisor = YFinanceStockAdvisor(use_ai=False)
        self.llm = DatabricksClient(model=model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pick_stocks(
        self,
        num_stocks: int = 5,
        sectors: Optional[List[str]] = None,
        custom_symbols: Optional[List[str]] = None,
        min_buy_signals: int = 2,
    ) -> Dict[str, Any]:
        """
        Select the best stocks to buy.

        Args:
            num_stocks: Number of stocks to select.
            sectors: Sectors to source candidates from.
                     Defaults to all sectors if neither sectors nor custom_symbols given.
            custom_symbols: Explicit list of symbols to consider.
            min_buy_signals: Minimum technical buy signals for pre-filtering.

        Returns:
            Dict with 'picks' (list of dicts) and 'reasoning' (LLM explanation).
        """
        print(f"🔎 Stock Picker Agent: selecting top {num_stocks} stocks...")
        print("=" * 50)

        # Step 1 – gather candidates
        candidates = self._gather_candidates(sectors, custom_symbols)
        if not candidates:
            return {"picks": [], "reasoning": "No candidate symbols found."}

        print(f"📋 Candidate pool: {len(candidates)} symbols")

        # Step 2 – technical screening
        screened = self._screen_candidates(candidates, min_buy_signals)
        if not screened:
            return {"picks": [], "reasoning": "No stocks passed technical screening."}

        print(f"✅ {len(screened)} stocks passed technical screening")

        # Step 3 – LLM ranking
        picks, reasoning = self._llm_rank(screened, num_stocks)

        print(f"🏆 Selected {len(picks)} stocks")
        print("=" * 50)

        return {
            "picks": picks,
            "reasoning": reasoning,
            "candidates_total": len(candidates),
            "screened_total": len(screened),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_candidates(
        self,
        sectors: Optional[List[str]],
        custom_symbols: Optional[List[str]],
    ) -> List[str]:
        """Build candidate symbol list from sectors and/or custom list."""
        symbols: List[str] = []

        if custom_symbols:
            symbols.extend([s.strip().upper() for s in custom_symbols if s.strip()])

        if sectors:
            for sector in sectors:
                try:
                    sector_obj = yf.Sector(sector)
                    top = sector_obj.top_companies
                    buys = top[top["rating"].str.lower().isin(["buy", "strong buy"])]
                    symbols.extend(buys.index.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to get sector {sector}: {e}")

        # Default: all sectors
        if not symbols:
            all_sectors = [
                "technology", "healthcare", "financial-services",
                "consumer-cyclical", "communication-services", "industrials",
                "consumer-defensive", "energy", "basic-materials",
                "real-estate", "utilities",
            ]
            for sector in all_sectors:
                try:
                    sector_obj = yf.Sector(sector)
                    top = sector_obj.top_companies
                    buys = top[top["rating"].str.lower().isin(["buy", "strong buy"])]
                    symbols.extend(buys.index.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to get sector {sector}: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _screen_candidates(
        self, symbols: List[str], min_buy_signals: int
    ) -> List[Dict[str, Any]]:
        """Run technical screening and return passing stocks."""
        return self.advisor.screen_stocks(
            symbols, min_buy_signals=min_buy_signals, only_bullish=False
        )

    def _llm_rank(
        self, screened: List[Dict[str, Any]], num_stocks: int
    ) -> tuple:
        """Use LLM to rank screened stocks and pick the best ones."""
        # Build a concise summary table for the LLM
        summary_lines = []
        for s in screened:
            ind = s.get("indicators", {})
            cond = s.get("conditions", {})
            fund = s.get("fundamentals", {})
            summary_lines.append(
                f"- {s['symbol']} | {s['name'][:30]} | "
                f"Price: ${s['current_price']:.2f} | Chg: {s['change_pct']:+.2f}% | "
                f"Rec: {s['recommendation']} | Conf: {s['confidence']}% | "
                f"Risk: {s['risk_level']} | BuySignals: {s['buy_signal_count']}/6 | "
                f"RSI: {ind.get('rsi', 'N/A')} | MACD: {ind.get('macd', 'N/A')} | "
                f"VolRatio: {ind.get('volume_ratio', 'N/A')} | "
                f"Trend: {cond.get('trend_ok')} | "
                f"Sector: {fund.get('sector', 'N/A')} | "
                f"PE: {fund.get('pe_ratio', 'N/A')} | "
                f"MktCap: {fund.get('market_cap', 'N/A')}"
            )

        stocks_table = "\n".join(summary_lines)

        prompt = f"""You are a senior portfolio manager selecting stocks to buy.

Below are {len(screened)} US stocks that passed an initial technical screening.
Your task: pick exactly {num_stocks} stocks that form the best portfolio to buy NOW.

CANDIDATE STOCKS:
{stocks_table}

SELECTION CRITERIA (prioritize in order):
1. Strong technical momentum (high buy signal count, healthy RSI, MACD bullish)
2. Good risk/reward (lower risk level, higher confidence)
3. Sector diversification (avoid concentrating in one sector)
4. Reasonable valuation (PE ratio not excessively high)
5. Volume confirmation (volume ratio > 1.0 preferred)
6. Positive price momentum (positive change percentage)

OUTPUT FORMAT (strict JSON):
{{
    "selected_symbols": ["SYM1", "SYM2", ...],
    "rankings": [
        {{
            "rank": 1,
            "symbol": "SYM1",
            "score": 85,
            "reasons": ["reason1", "reason2"]
        }},
        ...
    ],
    "portfolio_rationale": "2-3 sentence explanation of the overall selection strategy",
    "sector_allocation": {{"sector1": N, "sector2": M}}
}}

RULES:
- Select exactly {num_stocks} stocks (or fewer if not enough good candidates)
- All selected stocks must be from the candidate list above
- Provide clear, specific reasons referencing actual data values
- Score each stock 0-100 based on overall attractiveness
- ALL text must be in ENGLISH
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior portfolio manager with 20 years of experience "
                    "in US equity markets. You make data-driven stock selections."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        print("🤖 LLM is ranking candidates...")
        response = self.llm.call_api(messages, max_tokens=4000, temperature=0.3)

        # Parse response
        picks, reasoning = self._parse_llm_response(response, screened, num_stocks)
        return picks, reasoning

    def _parse_llm_response(
        self, response: str, screened: List[Dict[str, Any]], num_stocks: int
    ) -> tuple:
        """Parse the LLM ranking response."""
        import json

        # Build lookup for screened stocks
        screened_map = {s["symbol"]: s for s in screened}

        try:
            # Extract JSON
            if "```json" in response.lower():
                start = response.lower().find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                json_str = response

            data = json.loads(json_str)
            selected = data.get("selected_symbols", [])
            rankings = data.get("rankings", [])
            rationale = data.get("portfolio_rationale", "")
            sector_alloc = data.get("sector_allocation", {})

            # Build picks list with full data
            picks = []
            ranking_map = {r["symbol"]: r for r in rankings}
            for sym in selected[:num_stocks]:
                if sym in screened_map:
                    pick = dict(screened_map[sym])
                    rank_info = ranking_map.get(sym, {})
                    pick["rank"] = rank_info.get("rank", 0)
                    pick["score"] = rank_info.get("score", 0)
                    pick["selection_reasons"] = rank_info.get("reasons", [])
                    picks.append(pick)

            reasoning = f"{rationale}\n\nSector allocation: {sector_alloc}"
            return picks, reasoning

        except Exception as e:
            self.logger.error(f"Failed to parse LLM ranking: {e}")
            # Fallback: return top N by confidence from screened list
            sorted_screened = sorted(
                screened, key=lambda x: x["confidence"], reverse=True
            )
            picks = sorted_screened[:num_stocks]
            for i, p in enumerate(picks):
                p["rank"] = i + 1
                p["score"] = p["confidence"]
                p["selection_reasons"] = ["Fallback: selected by highest confidence"]
            return picks, f"LLM parsing failed ({e}), used confidence-based fallback."
