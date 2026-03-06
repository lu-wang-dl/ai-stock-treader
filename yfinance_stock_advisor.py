"""
YFinance Stock Advisor Module
Queries stock data with yfinance and provides buy/sell suggestions based on technical analysis
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import yfinance as yf
import pandas as pd
import numpy as np
import ta

from databricks_client import DatabricksClient


@dataclass
class StockSignal:
    """Data class for stock trading signals"""
    symbol: str
    name: str
    current_price: float
    change_pct: float
    
    # Technical indicators
    ma5: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    # Signal conditions
    trend_ok: bool = False
    volume_ok: bool = False
    macd_ok: bool = False
    rsi_ok: bool = False
    bb_ok: bool = False
    breakout_ok: bool = False
    
    # Overall assessment
    buy_signal_count: int = 0
    recommendation: str = "HOLD"  # BUY, SELL, HOLD
    confidence: int = 0
    risk_level: str = "medium"  # low, medium, high
    
    # Additional info
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # Analysis notes
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class YFinanceStockAdvisor:
    """
    Stock Advisor using yfinance for data and technical analysis for suggestions.
    
    Features:
    - Fetch real-time and historical stock data
    - Calculate technical indicators (MA, RSI, MACD, Bollinger Bands)
    - Generate buy/sell/hold recommendations
    - Screen stocks based on criteria
    - AI-powered analysis (optional, using DatabricksClient)
    """
    
    def __init__(self, use_ai: bool = False, ai_model: str = "databricks-gemini-3-pro"):
        """
        Initialize the Stock Advisor
        
        Args:
            use_ai: Whether to use AI for enhanced analysis
            ai_model: AI model to use for analysis
        """
        self.logger = logging.getLogger(__name__)
        self.use_ai = use_ai
        self.ai_client = DatabricksClient(model=ai_model) if (use_ai) else None
        
        # Default screening criteria
        self.default_criteria = {
            'min_volume': 100000,  # Minimum daily volume
            'min_price': 5.0,      # Minimum stock price
            'max_price': 500.0,    # Maximum stock price
            'min_market_cap': 1e9,  # Minimum $1B market cap
        }
    
    def get_stock_data(self, symbol: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """
        Fetch stock historical data with technical indicators
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        
        # Moving Averages
        df['MA5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['MA10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA60'] = ta.trend.sma_indicator(df['Close'], window=60)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        # Volume indicators
        df['Volume_MA5'] = ta.trend.sma_indicator(df['Volume'], window=5)
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA5']
        
        # ATR for volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        return df
    
    def analyze_stock(self, symbol: str) -> Optional[StockSignal]:
        """
        Analyze a single stock and generate trading signals
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            StockSignal object with analysis results
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < 20:
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Get stock info
            info = ticker.info
            
            # Create signal object
            signal = StockSignal(
                symbol=symbol,
                name=info.get('longName', symbol),
                current_price=float(latest['Close']),
                change_pct=((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] > 0 else 0,
                ma5=float(latest['MA5']) if pd.notna(latest['MA5']) else None,
                ma20=float(latest['MA20']) if pd.notna(latest['MA20']) else None,
                ma60=float(latest['MA60']) if pd.notna(latest['MA60']) else None,
                rsi=float(latest['RSI']) if pd.notna(latest['RSI']) else None,
                macd=float(latest['MACD']) if pd.notna(latest['MACD']) else None,
                macd_signal=float(latest['MACD_signal']) if pd.notna(latest['MACD_signal']) else None,
                macd_hist=float(latest['MACD_hist']) if pd.notna(latest['MACD_hist']) else None,
                bb_upper=float(latest['BB_upper']) if pd.notna(latest['BB_upper']) else None,
                bb_middle=float(latest['BB_middle']) if pd.notna(latest['BB_middle']) else None,
                bb_lower=float(latest['BB_lower']) if pd.notna(latest['BB_lower']) else None,
                volume_ratio=float(latest['Volume_ratio']) if pd.notna(latest['Volume_ratio']) else None,
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
            )
            
            # Evaluate signal conditions
            self._evaluate_signals(signal, df)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {symbol}: {e}")
            return None
    
    def _evaluate_signals(self, signal: StockSignal, df: pd.DataFrame) -> None:
        """Evaluate trading signal conditions"""
        signals = []
        warnings = []
        buy_count = 0
        
        # 1. Trend Analysis (MA alignment)
        if signal.ma5 and signal.ma20 and signal.ma60:
            if signal.current_price > signal.ma5 > signal.ma20 > signal.ma60:
                signal.trend_ok = True
                signals.append("✅ Bullish trend: Price > MA5 > MA20 > MA60")
                buy_count += 1
            elif signal.current_price < signal.ma5 < signal.ma20 < signal.ma60:
                warnings.append("⚠️ Bearish trend: Price < MA5 < MA20 < MA60")
            else:
                signals.append("➡️ Sideways trend: MAs not aligned")
        
        # 2. Volume Analysis
        if signal.volume_ratio:
            if signal.volume_ratio > 1.2:
                signal.volume_ok = True
                signals.append(f"✅ High volume: {signal.volume_ratio:.2f}x average")
                buy_count += 1
            elif signal.volume_ratio < 0.8:
                warnings.append(f"⚠️ Low volume: {signal.volume_ratio:.2f}x average")
        
        # 3. MACD Analysis
        if signal.macd is not None and signal.macd_signal is not None:
            macd_bullish = signal.macd > signal.macd_signal
            macd_positive = signal.macd > 0
            
            # Check for golden cross (MACD crossing above signal)
            if len(df) > 1:
                prev_macd = df.iloc[-2]['MACD'] if pd.notna(df.iloc[-2]['MACD']) else 0
                prev_signal = df.iloc[-2]['MACD_signal'] if pd.notna(df.iloc[-2]['MACD_signal']) else 0
                golden_cross = (prev_macd <= prev_signal) and (signal.macd > signal.macd_signal)
                death_cross = (prev_macd >= prev_signal) and (signal.macd < signal.macd_signal)
                
                if golden_cross:
                    signals.append("✅ MACD Golden Cross detected")
                if death_cross:
                    warnings.append("⚠️ MACD Death Cross detected")
            
            if macd_bullish and macd_positive:
                signal.macd_ok = True
                signals.append("✅ MACD bullish: Above signal line and positive")
                buy_count += 1
            elif not macd_bullish:
                warnings.append("⚠️ MACD bearish: Below signal line")
        
        # 4. RSI Analysis
        if signal.rsi:
            if 50 <= signal.rsi <= 70:
                signal.rsi_ok = True
                signals.append(f"✅ RSI healthy: {signal.rsi:.1f} (bullish zone)")
                buy_count += 1
            elif signal.rsi > 70:
                warnings.append(f"⚠️ RSI overbought: {signal.rsi:.1f}")
            elif signal.rsi < 30:
                signals.append(f"📉 RSI oversold: {signal.rsi:.1f} (potential reversal)")
            else:
                signals.append(f"➡️ RSI neutral: {signal.rsi:.1f}")
        
        # 5. Bollinger Bands Analysis
        if signal.bb_upper and signal.bb_middle and signal.bb_lower:
            if signal.bb_middle < signal.current_price < signal.bb_upper:
                signal.bb_ok = True
                signals.append("✅ Price in upper BB zone (bullish)")
                buy_count += 1
            elif signal.current_price > signal.bb_upper:
                warnings.append("⚠️ Price above upper BB (overbought)")
            elif signal.current_price < signal.bb_lower:
                signals.append("📉 Price below lower BB (oversold)")
        
        # 6. Breakout Detection
        if len(df) >= 20:
            recent_high = df['High'].iloc[-20:-1].max()
            if signal.current_price > recent_high:
                signal.breakout_ok = True
                signals.append(f"✅ Breakout: Price above 20-day high ${recent_high:.2f}")
                buy_count += 1
        
        # Calculate overall recommendation
        signal.buy_signal_count = buy_count
        signal.signals = signals
        signal.warnings = warnings
        
        # Determine recommendation
        if buy_count >= 4:
            signal.recommendation = "BUY"
            signal.confidence = min(90, 50 + buy_count * 10)
            signal.risk_level = "low"
        elif buy_count >= 3:
            signal.recommendation = "BUY"
            signal.confidence = min(75, 40 + buy_count * 10)
            signal.risk_level = "medium"
        elif buy_count >= 2:
            signal.recommendation = "HOLD"
            signal.confidence = 50
            signal.risk_level = "medium"
        else:
            if len(warnings) >= 3:
                signal.recommendation = "SELL"
                signal.confidence = 60
            else:
                signal.recommendation = "HOLD"
                signal.confidence = 40
            signal.risk_level = "high"
    
    def get_stock_suggestion(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock suggestion with analysis
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock analysis and recommendation
        """
        signal = self.analyze_stock(symbol)
        
        if not signal:
            return {
                'success': False,
                'error': f'Failed to analyze {symbol}',
                'symbol': symbol
            }
        
        result = {
            'success': True,
            'symbol': signal.symbol,
            'name': signal.name,
            'current_price': signal.current_price,
            'change_pct': signal.change_pct,
            'recommendation': signal.recommendation,
            'confidence': signal.confidence,
            'risk_level': signal.risk_level,
            'buy_signal_count': signal.buy_signal_count,
            'signals': signal.signals,
            'warnings': signal.warnings,
            'indicators': {
                'ma5': signal.ma5,
                'ma20': signal.ma20,
                'ma60': signal.ma60,
                'rsi': signal.rsi,
                'macd': signal.macd,
                'macd_signal': signal.macd_signal,
                'volume_ratio': signal.volume_ratio,
                'bb_upper': signal.bb_upper,
                'bb_middle': signal.bb_middle,
                'bb_lower': signal.bb_lower,
            },
            'conditions': {
                'trend_ok': signal.trend_ok,
                'volume_ok': signal.volume_ok,
                'macd_ok': signal.macd_ok,
                'rsi_ok': signal.rsi_ok,
                'bb_ok': signal.bb_ok,
                'breakout_ok': signal.breakout_ok,
            },
            'fundamentals': {
                'sector': signal.sector,
                'industry': signal.industry,
                'market_cap': signal.market_cap,
                'pe_ratio': signal.pe_ratio,
                'dividend_yield': signal.dividend_yield,
            }
        }
        
        # Add AI analysis if enabled
        if self.use_ai and self.ai_client:
            result['ai_analysis'] = self._get_ai_analysis(signal)
        
        return result
    
    def _get_ai_analysis(self, signal: StockSignal) -> str:
        """Get AI-powered analysis for the stock"""
        try:
            stock_info = {
                'symbol': signal.symbol,
                'name': signal.name,
                'current_price': signal.current_price,
                'change_percent': signal.change_pct,
                'sector': signal.sector,
                'industry': signal.industry,
                'market_cap': signal.market_cap,
                'pe_ratio': signal.pe_ratio,
            }
            
            indicators = {
                'price': signal.current_price,
                'ma5': signal.ma5,
                'ma10': None,
                'ma20': signal.ma20,
                'ma60': signal.ma60,
                'rsi': signal.rsi,
                'macd': signal.macd,
                'macd_signal': signal.macd_signal,
                'bb_upper': signal.bb_upper,
                'bb_lower': signal.bb_lower,
                'volume_ratio': signal.volume_ratio,
            }
            
            return self.ai_client.technical_analysis(stock_info, None, indicators)
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return f"AI analysis unavailable: {str(e)}"
    
    def screen_stocks(self, symbols: List[str], 
                     min_buy_signals: int = 3,
                     only_bullish: bool = True) -> List[Dict[str, Any]]:
        """
        Screen multiple stocks and return those meeting criteria
        
        Args:
            symbols: List of stock symbols to screen
            min_buy_signals: Minimum number of buy signals required
            only_bullish: Only return stocks with BUY recommendation
            
        Returns:
            List of stock suggestions sorted by confidence
        """
        results = []
        
        for symbol in symbols:
            try:
                suggestion = self.get_stock_suggestion(symbol)
                
                if not suggestion['success']:
                    continue
                
                # Apply filters
                if suggestion['buy_signal_count'] < min_buy_signals:
                    continue
                    
                if only_bullish and suggestion['recommendation'] != 'BUY':
                    continue
                
                results.append(suggestion)
                
            except Exception as e:
                self.logger.warning(f"Failed to screen {symbol}: {e}")
                continue
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def get_top_picks(self, symbols: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N stock picks from a list of symbols
        
        Args:
            symbols: List of stock symbols to analyze
            top_n: Number of top picks to return
            
        Returns:
            Top N stock picks with BUY recommendation
        """
        screened = self.screen_stocks(symbols, min_buy_signals=3, only_bullish=True)
        return screened[:top_n]
    
    def get_sector_recommendations(self, choosen_sector: str, min_signals: int = 2) -> List[Dict[str, Any]]:
        """
        Get stock recommendations for a specific sector
        
        Args:
            sector: Sector name (e.g., 'Technology', 'Healthcare')
            
        Returns:
            List of recommended stocks in the sector
        """
        # Popular stocks by sector
        sectors = ['technology', 'healthcare', 'financial-services', 'consumer-cyclical', 'communication-services', 'industrials', 'consumer-defensive', 'energy', 'basic-materials', 'real-estate', 'utilities']
        
        sector_stocks = {}
        for sector in sectors:
            sector_comps = yf.Sector(sector)
            top_companies = sector_comps.top_companies
            suggested_companies = top_companies[top_companies['rating'].str.lower().isin(['buy', 'strong buy'])].index.tolist()
            sector_stocks[sector] = suggested_companies
        
        if choosen_sector.lower() == "all":
            import itertools
            symbols = list(itertools.chain.from_iterable(sector_stocks.values()))
        else:
            symbols = sector_stocks.get(choosen_sector, [])
        if not symbols:
            self.logger.warning(f"Unknown sector: {choosen_sector}")
            return []
        
        return self.screen_stocks(symbols, min_buy_signals=min_signals, only_bullish=False)
    
    def quick_scan(self, symbol: str) -> str:
        """
        Quick scan of a stock with summary output
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Formatted string with quick analysis
        """
        suggestion = self.get_stock_suggestion(symbol)
        
        if not suggestion['success']:
            return f"❌ Unable to analyze {symbol}: {suggestion.get('error', 'Unknown error')}"
        
        # Build summary
        rec_emoji = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '🟡'
        }
        
        risk_emoji = {
            'low': '🟢',
            'medium': '🟡',
            'high': '🔴'
        }
        
        summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 {suggestion['name']} ({suggestion['symbol']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Price: ${suggestion['current_price']:.2f} ({suggestion['change_pct']:+.2f}%)

{rec_emoji.get(suggestion['recommendation'], '⚪')} Recommendation: {suggestion['recommendation']}
📈 Confidence: {suggestion['confidence']}%
{risk_emoji.get(suggestion['risk_level'], '⚪')} Risk Level: {suggestion['risk_level'].upper()}
🎯 Buy Signals: {suggestion['buy_signal_count']}/6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 SIGNALS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for sig in suggestion['signals']:
            summary += f"  {sig}\n"
        
        if suggestion['warnings']:
            summary += "\n⚠️ WARNINGS:\n"
            for warn in suggestion['warnings']:
                summary += f"  {warn}\n"
        
        indicators = suggestion['indicators']
        ma5_str = f"${indicators['ma5']:.2f}" if indicators['ma5'] else 'N/A'
        ma20_str = f"${indicators['ma20']:.2f}" if indicators['ma20'] else 'N/A'
        ma60_str = f"${indicators['ma60']:.2f}" if indicators['ma60'] else 'N/A'
        rsi_str = f"{indicators['rsi']:.1f}" if indicators['rsi'] else 'N/A'
        macd_str = f"{indicators['macd']:.4f}" if indicators['macd'] else 'N/A'
        vol_ratio_str = f"{indicators['volume_ratio']:.2f}x" if indicators['volume_ratio'] else 'N/A'
        
        summary += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📉 KEY INDICATORS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MA5:  {ma5_str}
  MA20: {ma20_str}
  MA60: {ma60_str}
  RSI:  {rsi_str}
  MACD: {macd_str}
  Volume Ratio: {vol_ratio_str}
"""
        
        return summary
    
    def compare_stocks(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare multiple stocks side by side
        
        Args:
            symbols: List of stock symbols to compare
            
        Returns:
            DataFrame with comparison data
        """
        data = []
        
        for symbol in symbols:
            suggestion = self.get_stock_suggestion(symbol)
            
            if not suggestion['success']:
                continue
            
            data.append({
                'Symbol': symbol,
                'Name': suggestion['name'],
                'Price': suggestion['current_price'],
                'Change%': suggestion['change_pct'],
                'Recommendation': suggestion['recommendation'],
                'Confidence': suggestion['confidence'],
                'Risk': suggestion['risk_level'],
                'Buy Signals': suggestion['buy_signal_count'],
                'RSI': suggestion['indicators']['rsi'],
                'Volume Ratio': suggestion['indicators']['volume_ratio'],
            })
        
        return pd.DataFrame(data)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create advisor instance
    # advisor = YFinanceStockAdvisor(use_ai=False)

    print(yf.PREDEFINED_SCREENER_QUERIES.keys())
    response = yf.screen("aggressive_small_caps")

    # The response is a dictionary containing the results, usually in a list under a 'screen' key
    # You would then process this data, for instance, convert it to a pandas DataFrame
    import pandas as pd
    data = pd.DataFrame(response['quotes'])
    data = data[data['averageAnalystRating'].str.lower().isin(['buy', 'strong buy'])]
    print(data.columns)

    software = yf.Sector('industrials')
    top_companies = software.top_companies
    suggested_companies = top_companies[top_companies['rating'].str.lower().isin(['buy', 'strong buy'])]
    print(suggested_companies.index.tolist())
    
    # # Test single stock analysis
    # print("\n" + "="*50)
    # print("Single Stock Analysis: AAPL")
    # print("="*50)
    # print(advisor.quick_scan("AAPL"))
    
    # Test stock screening
    # print("\n" + "="*50)
    # print("Screening Tech Stocks")
    # print("="*50)
    # tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
    # picks = advisor.get_top_picks(tech_stocks, top_n=3)
    
    # for pick in picks:
    #     print(f"\n🎯 {pick['symbol']}: {pick['recommendation']} (Confidence: {pick['confidence']}%)")
    #     print(f"   Price: ${pick['current_price']:.2f} | Buy Signals: {pick['buy_signal_count']}/6")
    
    # # Test comparison
    # print("\n" + "="*50)
    # print("Stock Comparison")
    # print("="*50)
    # comparison = advisor.compare_stocks(['AAPL', 'MSFT', 'GOOGL'])
    # print(comparison.to_string(index=False))

