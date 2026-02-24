#!/usr/bin/env python3
"""
Unified Stock Analysis Script
Combines real-time price data with deep TradingAgents analysis

Features:
- Real-time and historical analysis support
- Multi-ticker batch processing
- Enhanced error handling
- Structured output formatting
- Results persistence
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import TradingAgents components
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class UnifiedAnalysis:
    """Unified interface for stock analysis with TradingAgents."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional custom configuration.
        
        Args:
            config: Custom configuration dictionary. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.ta: Optional[TradingAgentsGraph] = None
        self._setup_config()
    
    def _setup_config(self) -> None:
        """Apply default optimizations to configuration."""
        # Set defaults if not already configured
        self.config.setdefault("llm_provider", "ollama")
        self.config.setdefault("deep_think_llm", "minimax-m2.5:cloud")
        self.config.setdefault("quick_think_llm", "minimax-m2.5:cloud")
        self.config.setdefault("max_debate_rounds", 1)
        
        # Ensure data vendors are configured
        self.config.setdefault("data_vendors", {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        })
    
    def initialize(self, debug: bool = False) -> None:
        """Initialize the TradingAgents graph.
        
        Args:
            debug: Enable debug mode for tracing
        """
        print("Initializing TradingAgents graph...")
        self.ta = TradingAgentsGraph(
            debug=debug,
            config=self.config,
            callbacks=None
        )
        print("[OK] TradingAgents initialized successfully\n")
    
    def get_realtime_price(self, ticker: str) -> Optional[dict]:
        """Fetch real-time stock price using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with price data or None if failed
        """
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            previous = info.get('previousClose')
            change = info.get('regularMarketChangePercent')
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'current_price': current,
                'previous_close': previous,
                'change_pct': change,
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            print("[WARNING] yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            print(f"[WARNING] Error fetching price for {ticker}: {e}")
            return None
    
    def print_price_data(self, price_data: dict) -> None:
        """Print formatted price data.
        
        Args:
            price_data: Dictionary containing price information
        """
        if not price_data or not price_data.get('current_price'):
            print("  Price data unavailable")
            return
        
        print(f"  Company: {price_data['name']}")
        print(f"  Current Price: ${price_data['current_price']:.2f}")
        
        if price_data.get('change_pct'):
            emoji = "+" if price_data['change_pct'] > 0 else "-"
            print(f"  Change: {emoji} {price_data['change_pct']:+.2f}%")
        
        if price_data.get('market_cap'):
            mc = price_data['market_cap']
            mc_str = f"${mc/1e12:.2f}T" if mc > 1e12 else f"${mc/1e9:.2f}B"
            print(f"  Market Cap: {mc_str}")
        
        print(f"  Sector: {price_data['sector']}")
    
    def analyze(
        self,
        ticker: str,
        trade_date: Optional[str] = None,
        use_realtime: bool = True,
        save_results: bool = True
    ) -> tuple:
        """Run comprehensive analysis on a stock.
        
        Args:
            ticker: Stock ticker symbol
            trade_date: Date for analysis (YYYY-MM-DD). Uses today if None and use_realtime=True
            use_realtime: Whether to fetch live price data
            save_results: Whether to save results to file
            
        Returns:
            Tuple of (result dict, decision string)
        """
        ticker = ticker.upper().strip()
        
        # Determine analysis date
        if trade_date is None and use_realtime:
            trade_date = datetime.now().strftime("%Y-%m-%d")
        elif trade_date is None:
            trade_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Ensure TradingAgents is initialized
        if self.ta is None:
            self.initialize()
        
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker}")
        print(f"Date: {trade_date}")
        print(f"{'='*60}")
        
        # Fetch real-time price if requested
        price_data = None
        if use_realtime:
            print("\n[INFO] Fetching real-time price data...")
            price_data = self.get_realtime_price(ticker)
            self.print_price_data(price_data)
        
        # Run TradingAgents analysis
        print(f"\n[AGENT] Running TradingAgents analysis...")
        print(f"   This may take 5-15 minutes depending on LLM response time...\n")
        
        try:
            result, decision = self.ta.propagate(ticker, trade_date)
            
            # Process and display results
            self._print_analysis_results(result, decision, price_data)
            
            # Save results if requested
            if save_results:
                self._save_results(ticker, trade_date, result, decision, price_data)
            
            return result, decision
            
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {e}")
            return None, None
    
    def _print_analysis_results(
        self,
        result: dict,
        decision: str,
        price_data: Optional[dict]
    ) -> None:
        """Print formatted analysis results."""
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # Market Report
        if result.get('market_report'):
            print("\n[MARKET] MARKET REPORT:")
            print("-" * 40)
            self._print_section(result['market_report'])
        
        # Sentiment Report
        if result.get('sentiment_report'):
            print("\n[SENTIMENT] SENTIMENT REPORT:")
            print("-" * 40)
            self._print_section(result['sentiment_report'])
        
        # News Report
        if result.get('news_report'):
            print("\n[NEWS] NEWS REPORT:")
            print("-" * 40)
            self._print_section(result['news_report'])
        
        # Fundamentals Report
        if result.get('fundamentals_report'):
            print("\n[INFO] FUNDAMENTALS REPORT:")
            print("-" * 40)
            self._print_section(result['fundamentals_report'])
        
        # Investment Decision
        if result.get('trader_investment_plan'):
            print("\n[INVESTMENT] INVESTMENT DECISION:")
            print("-" * 40)
            self._print_section(result['trader_investment_plan'])
        
        # Risk Assessment
        if result.get('investment_plan'):
            print("\n[WARNING] RISK ASSESSMENT:")
            print("-" * 40)
            self._print_section(result['investment_plan'])
        
        # Final Decision
        print(f"\n{'='*60}")
        print(f"[DECISION] FINAL DECISION: {decision}")
        print(f"{'='*60}")
    
    def _print_section(self, text: str, max_length: int = 1500) -> None:
        """Print a section of text with optional truncation."""
        if text:
            # Truncate long text for display
            display_text = text[:max_length]
            if len(text) > max_length:
                display_text += "..."
            print(display_text)
    
    def _save_results(
        self,
        ticker: str,
        trade_date: str,
        result: dict,
        decision: str,
        price_data: Optional[dict]
    ) -> None:
        """Save analysis results to files."""
        # Create output directory
        output_dir = Path(f"analysis_results/{ticker}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = output_dir / f"{ticker}_analysis_{trade_date}.json"
        json_data = {
            "ticker": ticker,
            "trade_date": trade_date,
            "analysis_timestamp": timestamp,
            "decision": decision,
            "price_data": price_data,
            "reports": {
                "market_report": result.get("market_report"),
                "sentiment_report": result.get("sentiment_report"),
                "news_report": result.get("news_report"),
                "fundamentals_report": result.get("fundamentals_report"),
                "trader_investment_plan": result.get("trader_investment_plan"),
                "investment_plan": result.get("investment_plan"),
                        }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Results saved to: {json_file}")
        
        # Save human-readable summary
        txt_file = output_dir / f"{ticker}_summary_{trade_date}.txt"
        self._save_text_summary(txt_file, ticker, trade_date, result, decision, price_data)
        print(f"[SAVED] Summary saved to: {txt_file}")
        
        # Save real-time chart
        chart_file = output_dir / f"{ticker}_chart_{trade_date}.png"
        self._save_realtime_chart(chart_file, ticker, price_data, decision)
        print(f"[SAVED] Chart saved to: {chart_file}")
    
    def _save_realtime_chart(
        self,
        filepath: Path,
        ticker: str,
        price_data: Optional[dict],
        decision: str
    ) -> None:
        """Generate real-time price chart with decision zones."""
        try:
            import yfinance as yf
            import matplotlib.pyplot as plt
            from datetime import datetime, timedelta
            
            if not price_data or not price_data.get('current_price'):
                print("[WARNING] No price data for chart")
                return
            
            current_price = price_data['current_price']
            
            # Get historical data
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            hist = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                                end=end_date.strftime("%Y-%m-%d"))
            
            if hist.empty:
                print("[WARNING] No historical data for chart")
                return
            
            # Calculate support/resistance
            support = round(current_price * 0.92, 2)
            resistance = round(current_price * 1.10, 2)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot price
            ax.plot(hist.index, hist['Close'], linewidth=2, color='black', label=f'{ticker} Price')
            
            # Get min/max for zones
            min_price = hist['Close'].min() * 0.95
            max_price = hist['Close'].max() * 1.05
            
            # Color zones
            ax.fill_between(hist.index, min_price, support, 
                           alpha=0.3, color='green', label='BUY Zone')
            ax.fill_between(hist.index, support, resistance, 
                           alpha=0.2, color='yellow', label='HOLD Zone')
            ax.fill_between(hist.index, resistance, max_price, 
                           alpha=0.3, color='red', label='SELL Zone')
            
            # Lines
            ax.axhline(y=current_price, color='blue', linestyle='--', linewidth=2, 
                      label=f'Current: ${current_price:.2f}')
            ax.axhline(y=support, color='green', linestyle=':', linewidth=1.5, 
                      label=f'Support: ${support:.2f}')
            ax.axhline(y=resistance, color='red', linestyle=':', linewidth=1.5, 
                      label=f'Resistance: ${resistance:.2f}')
            
            # Title
            ax.set_title(f'{ticker} Trading Decision Chart | Real-Time Price: ${current_price:.2f} | Decision: {decision}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Info box
            textstr = f'TRADING RECOMMENDATION: {decision}\n' + '='*40 + f'\nCurrent Price:    ${current_price:.2f}\nSupport:          ${support:.2f}\nResistance:        ${resistance:.2f}\n' + '='*40 + '\nZONES:\n- BUY:  Below ${support:.2f}\n- HOLD: ${support:.2f} - ${resistance:.2f}\n- SELL: Above ${resistance:.2f}'
            
            props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props, family='monospace')
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("[WARNING] matplotlib not installed for chart")
        except Exception as e:
            print(f"[WARNING] Chart generation failed: {e}")
    
    def _save_text_summary(
        self,
        filepath: Path,
        ticker: str,
        trade_date: str,
        result: dict,
        decision: str,
        price_data: Optional[dict]
    ) -> None:
        """Save a human-readable text summary."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"STOCK ANALYSIS REPORT: {ticker}\n")
            f.write(f"Analysis Date: {trade_date}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            # Price data
            if price_data and price_data.get('current_price'):
                f.write("CURRENT MARKET DATA\n")
                f.write("-" * 40 + "\n")
                f.write(f"Price: ${price_data['current_price']:.2f}\n")
                if price_data.get('change_pct'):
                    f.write(f"Change: {price_data['change_pct']:+.2f}%\n")
                f.write(f"Company: {price_data['name']}\n")
                f.write(f"Sector: {price_data['sector']}\n\n")
            
            # Reports
            sections = [
                ("MARKET REPORT", result.get("market_report")),
                ("SENTIMENT REPORT", result.get("sentiment_report")),
                ("NEWS REPORT", result.get("news_report")),
                ("FUNDAMENTALS REPORT", result.get("fundamentals_report")),
                ("INVESTMENT DECISION", result.get("trader_investment_plan")),
                ("RISK ASSESSMENT", result.get("investment_plan")),
            ]
            
            for title, content in sections:
                if content:
                    f.write(f"\n{title}\n")
                    f.write("-" * 40 + "\n")
                    f.write(content[:5000] + "\n")
            
            f.write(f"\n{'='*60}\n")
            f.write(f"FINAL DECISION: {decision}\n")
            f.write(f"{'='*60}\n")
    
    def batch_analyze(
        self,
        tickers: list[str],
        trade_date: Optional[str] = None,
        use_realtime: bool = True
    ) -> dict:
        """Run analysis on multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            trade_date: Date for analysis
            use_realtime: Whether to fetch live price data
            
        Returns:
            Dictionary mapping ticker to (result, decision) tuple
        """
        results = {}
        
        print(f"\n[BATCH] Starting batch analysis for {len(tickers)} tickers...\n")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
            try:
                result, decision = self.analyze(
                    ticker,
                    trade_date=trade_date,
                    use_realtime=use_realtime,
                    save_results=True
                )
                results[ticker] = (result, decision)
            except Exception as e:
                print(f"[ERROR] Failed to analyze {ticker}: {e}")
                results[ticker] = (None, None)
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: dict) -> None:
        """Print summary of batch analysis results."""
        print(f"\n{'='*60}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        decisions = []
        for ticker, (result, decision) in results.items():
            if decision:
                decisions.append((ticker, decision))
                print(f"  {ticker}: {decision}")
        
        if decisions:
            buy_count = sum(1 for _, d in decisions if "BUY" in d.upper())
            sell_count = sum(1 for _, d in decisions if "SELL" in d.upper())
            hold_count = sum(1 for _, d in decisions if "HOLD" in d.upper())
            
            print(f"\n[INFO] Summary:")
            print(f"  BUY: {buy_count} | HOLD: {hold_count} | SELL: {sell_count}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Stock Analysis with TradingAgents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/unified_analysis.py NVDA
  python scripts/unified_analysis.py AMD --date 2026-02-20
  python scripts/unified_analysis.py AAPL MSFT GOOGL --batch
  python scripts/unified_analysis.py TSLA --no-realtime
        """
    )
    
    parser.add_argument(
        'tickers',
        nargs='+',
        help='Stock ticker symbol(s)'
    )
    
    parser.add_argument(
        '-d', '--date',
        type=str,
        default=None,
        help='Analysis date (YYYY-MM-DD). Defaults to today if --realtime, yesterday otherwise'
    )
    
    parser.add_argument(
        '-r', '--realtime',
        action='store_true',
        default=True,
        help='Fetch real-time price data (default: True)'
    )
    
    parser.add_argument(
        '--no-realtime',
        action='store_true',
        help='Disable real-time price fetching'
    )
    
    parser.add_argument(
        '-b', '--batch',
        action='store_true',
        help='Run batch analysis on all tickers'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with tracing'
    )
    
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai', 'anthropic', 'google'],
        help='LLM provider to use'
    )
    
    parser.add_argument(
        '--deep-think',
        type=str,
        default='minimax-m2.5:cloud',
        help='Deep thinking model'
    )
    
    parser.add_argument(
        '--quick-think',
        type=str,
        default='minimax-m2.5:cloud',
        help='Quick thinking model'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Build configuration
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["deep_think_llm"] = args.deep_think
    config["quick_think_llm"] = args.quick_think
    config["max_debate_rounds"] = 1
    
    # Create analyzer
    analyzer = UnifiedAnalysis(config=config)
    
    # Determine realtime setting
    use_realtime = args.realtime and not args.no_realtime
    
    # Run analysis
    if args.batch or len(args.tickers) > 1:
        analyzer.batch_analyze(
            args.tickers,
            trade_date=args.date,
            use_realtime=use_realtime
        )
    else:
        analyzer.analyze(
            args.tickers[0],
            trade_date=args.date,
            use_realtime=use_realtime,
            save_results=True
        )


if __name__ == "__main__":
    main()