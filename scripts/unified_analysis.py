"""
Unified Trading Analysis Script
Uses FULL TradingAgents Framework (8 AI Agents)
- Market Analyst
- Fundamentals Analyst  
- News Analyst
- Social Analyst
- Bull Researcher
- Bear Researcher
- Risk Manager
- Trader

Single command: Analysis + Chart + Report
"""
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import sys

load_dotenv()


def get_market_data(ticker):
    """Get real-time market data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'current_price': info.get('currentPrice') or info.get('regularMarketPrice') or 0,
            'previous_close': info.get('previousClose', 0),
            'change_pct': info.get('regularMarketChangePercent', 0),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'eps': info.get('trailingEps', 0),
            'beta': info.get('beta', 0),
        }
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None


def run_full_analysis(ticker, trade_date):
    """
    Run FULL TradingAgents analysis with all 8 agents:
    1. Market Analyst - Technical indicators (RSI, MACD, SMA, etc.)
    2. Fundamentals Analyst - Financial metrics (P/E, Revenue, etc.)
    3. News Analyst - News and events
    4. Social Analyst - Social media sentiment
    5. Bull Researcher - Argues for BUY
    6. Bear Researcher - Argues for SELL
    7. Risk Manager - Portfolio risk evaluation
    8. Trader - Final decision synthesis
    """
    # Full framework configuration
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["deep_think_llm"] = "minimax-m2.5:cloud"
    config["quick_think_llm"] = "minimax-m2.5:cloud"
    config["max_debate_rounds"] = 1
    
    # Configure data sources
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }
    
    # Initialize full TradingAgents graph with all analysts
    print("  Initializing TradingAgents with 8 AI agents...")
    ta = TradingAgentsGraph(
        selected_analysts=["market", "fundamentals", "news", "social"],
        debug=False, 
        config=config
    )
    
    print("  Running agent analysis (this may take 5-15 minutes)...")
    result, decision = ta.propagate(ticker, trade_date)
    
    return result, decision, ta


def create_price_chart(ticker, market_data, decision):
    """Create price chart with BUY/HOLD/SELL zones."""
    current_price = market_data['current_price']
    
    if current_price == 0:
        print("  Warning: No valid price data")
        return False
    
    # Calculate zones
    support = round(current_price * 0.92, 2)
    resistance = round(current_price * 1.10, 2)
    
    # Get historical data
    try:
        stock = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=90)
        hist = stock.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        
        if hist.empty:
            print("  No historical data available")
            return False
    except Exception as e:
        print(f"  Error fetching history: {e}")
        return False
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Price line
    ax.plot(hist.index, hist['Close'], linewidth=2, color='black', label=f'{ticker} Price')
    
    # Zones
    min_p, max_p = hist['Close'].min() * 0.95, hist['Close'].max() * 1.05
    
    ax.fill_between(hist.index, min_p, support, alpha=0.3, color='green', label='BUY Zone')
    ax.fill_between(hist.index, support, resistance, alpha=0.2, color='yellow', label='HOLD Zone')
    ax.fill_between(hist.index, resistance, max_p, alpha=0.3, color='red', label='SELL Zone')
    
    # Lines
    ax.axhline(y=current_price, color='blue', linestyle='--', linewidth=2, label=f'Current: ${current_price:.2f}')
    ax.axhline(y=support, color='green', linestyle=':', linewidth=1.5, label=f'Support: ${support:.2f}')
    ax.axhline(y=resistance, color='red', linestyle=':', linewidth=1.5, label=f'Resistance: ${resistance:.2f}')
    
    # Title
    change = market_data.get('change_pct', 0)
    change_str = f"{change:.2f}%" if change else "N/A"
    ax.set_title(f'{ticker} Trading Analysis\nPrice: ${current_price:.2f} ({change_str}) | Decision: {decision}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Info box
    textstr = f'''
TRADING RECOMMENDATION: {decision}
{'='*40}
Current Price:    ${current_price:.2f}
Change:          {change_str}
Support:          ${support:.2f}
Resistance:        ${resistance:.2f}
P/E Ratio:        {market_data.get('pe_ratio', 'N/A')}
Beta:             {market_data.get('beta', 'N/A')}
{'='*40}
ZONES:
- BUY:  Below ${support:.2f}
- HOLD: ${support:.2f} - ${resistance:.2f}
- SELL: Above ${resistance:.2f}
'''
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9),
            family='monospace')
    
    plt.tight_layout()
    
    # Save
    output = f'{ticker}_Analysis_Chart.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {output}")
    
    return True


def save_analysis_report(ticker, market_data, decision, result, trade_date):
    """Save full analysis report."""
    output = f'{ticker}_Analysis_Report_{trade_date}.txt'
    
    with open(output, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {ticker} STOCK ANALYSIS REPORT\n")
        f.write(f"  TradingAgents Framework (8 AI Agents)\n")
        f.write(f"  Date: {trade_date}\n")
        f.write(f"{'='*60}\n\n")
        
        # Market Data
        f.write("MARKET DATA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Company: {market_data.get('name', 'N/A')}\n")
        f.write(f"Sector: {market_data.get('sector', 'N/A')}\n")
        f.write(f"Current Price: ${market_data['current_price']:.2f}\n")
        f.write(f"Previous Close: ${market_data.get('previous_close', 0):.2f}\n")
        f.write(f"Change: {market_data.get('change_pct', 0):.2f}%\n")
        f.write(f"Market Cap: ${market_data.get('market_cap', 0)/1e9:.2f}B\n")
        f.write(f"P/E Ratio: {market_data.get('pe_ratio', 'N/A')}\n")
        f.write(f"EPS: ${market_data.get('eps', 0):.2f}\n")
        f.write(f"Beta: {market_data.get('beta', 'N/A')}\n\n")
        
        # Decision
        f.write(f"{'='*60}\n")
        f.write(f"  FINAL DECISION: {decision}\n")
        f.write(f"{'='*60}\n\n")
        
        # Full reports from agents
        if result:
            f.write("MARKET ANALYST REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('market_report', 'N/A')[:3000] + "\n\n")
            
            f.write("FUNDAMENTALS ANALYST REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('fundamentals_report', 'N/A')[:3000] + "\n\n")
            
            f.write("NEWS ANALYST REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('news_report', 'N/A')[:2000] + "\n\n")
            
            f.write("SENTIMENT REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('sentiment_report', 'N/A')[:2000] + "\n\n")
            
            f.write("TRADER INVESTMENT DECISION\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('trader_investment_plan', 'N/A')[:2000] + "\n\n")
            
            f.write("RISK ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            f.write(result.get('investment_plan', 'N/A')[:2000] + "\n\n")
    
    print(f"  Report saved: {output}")
    return output


def main():
    """Main unified analysis function."""
    print("\n" + "="*60)
    print("  UNIFIED TRADING ANALYSIS")
    print("  TradingAgents Framework (8 AI Agents)")
    print("="*60)
    
    # Get ticker
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("\nEnter stock ticker (e.g., AMD, NVDA, AAPL): ").upper().strip()
    
    if not ticker:
        print("Error: Please provide a ticker")
        return
    
    trade_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nAnalyzing: {ticker}")
    print(f"Date: {trade_date}\n")
    
    # Step 1: Get real-time market data
    print("[1/4] Fetching real-time market data...")
    market_data = get_market_data(ticker)
    
    if not market_data or market_data['current_price'] == 0:
        print(f"Error: Could not fetch data for {ticker}")
        return
    
    print(f"  Price: ${market_data['current_price']}")
    print(f"  Change: {market_data.get('change_pct', 0):.2f}%")
    
    # Step 2: Run full TradingAgents analysis
    print("\n[2/4] Running TradingAgents analysis...")
    result, decision, ta = run_full_analysis(ticker, trade_date)
    
    if not result:
        print("  Warning: Analysis may be incomplete")
        decision = "ANALYSIS FAILED"
    
    print(f"\n  >>> DECISION: {decision}")
    
    # Step 3: Create price chart
    print("\n[3/4] Creating price chart...")
    create_price_chart(ticker, market_data, decision)
    
    # Step 4: Save report
    print("\n[4/4] Saving analysis report...")
    save_analysis_report(ticker, market_data, decision, result, trade_date)
    
    # Summary
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)
    print(f"\n  Ticker: {ticker}")
    print(f"  Price: ${market_data['current_price']}")
    print(f"  Decision: {decision}")
    print(f"\n  Files generated:")
    print(f"    - {ticker}_Analysis_Chart.png")
    print(f"    - {ticker}_Analysis_Report_{trade_date}.txt")
    print("="*60)


if __name__ == "__main__":
    main()
