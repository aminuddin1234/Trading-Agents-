# trading-agents-setup/realtime_analysis.py
"""
Real-time Stock Analysis with Live Market Prices
Uses yfinance to fetch current market data
"""
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
import yfinance as yf
from datetime import datetime

load_dotenv()

def get_realtime_price(ticker):
    """Fetch real-time stock price."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price
        current = info.get('currentPrice') or info.get('regularMarketPrice')
        previous = info.get('previousClose')
        change = info.get('regularMarketChangePercent')
        
        # Get company info
        name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        market_cap = info.get('marketCap', 0)
        
        return {
            'ticker': ticker,
            'name': name,
            'current_price': current,
            'previous_close': previous,
            'change_pct': change,
            'sector': sector,
            'market_cap': market_cap,
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def analyze_stock(ticker):
    """Run real-time analysis on a stock."""
    
    # Get real-time price
    price_data = get_realtime_price(ticker)
    if not price_data:
        print(f"Could not fetch data for {ticker}")
        return
    
    print("\n" + "="*60)
    print(f"REAL-TIME ANALYSIS: {ticker}")
    print("="*60)
    
    if price_data['current_price']:
        print(f"Company: {price_data['name']}")
        print(f"Current Price: ${price_data['current_price']:.2f}")
        
        if price_data['change_pct']:
            print(f"Change: {price_data['change_pct']:.2f}%")
        if price_data['market_cap']:
            print(f"Market Cap: ${price_data['market_cap']/1e12:.2f}T" if price_data['market_cap'] > 1e12 else f"${price_data['market_cap']/1e9:.2f}B")
        print(f"Sector: {price_data['sector']}")
    print("="*60)
    
    # Use today's date for analysis
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Configure TradingAgents
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["deep_think_llm"] = "minimax-m2.5:cloud"
    config["quick_think_llm"] = "minimax-m2.5:cloud"
    config["max_debate_rounds"] = 1
    
    print(f"\nRunning TradingAgents analysis...")
    print(f"Analysis Date: {today}")
    print("This may take 5-15 minutes...\n")
    
    try:
        ta = TradingAgentsGraph(debug=False, config=config)
        result, decision = ta.propagate(ticker, today)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"\nFINAL DECISION: {decision}")
        
        # Save results
        output_file = f"{ticker}_Realtime_Analysis_{today}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"{ticker} Real-Time Analysis - {today}\n")
            f.write(f"Current Price: ${price_data['current_price']:.2f}\n")
            f.write(f"Decision: {decision}\n\n")
            
            # Write key reports
            if result.get('market_report'):
                f.write("MARKET REPORT:\n")
                f.write(result['market_report'][:3000])
                f.write("\n\n")
            
            if result.get('trader_investment_plan'):
                f.write("TRADER DECISION:\n")
                f.write(result['trader_investment_plan'][:2000])
        
        print(f"\nResults saved to: {output_file}")
        
        return result, decision
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        return None, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter stock ticker (e.g., AMD, NVDA, AAPL): ").upper().strip()
    
    if ticker:
        analyze_stock(ticker)
    else:
        print("Please provide a stock ticker")