"""
Simple Trading Chart - No Unicode
Fetches REAL-TIME prices from yfinance
"""
import yfinance as yf
import matplotlib.pyplot as plt
import json
import os

def get_realtime_price(ticker):
    """Fetch real-time stock price from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price (handles pre-market and after-hours)
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        previous_close = info.get('previousClose')
        
        # If no current price, use previous close
        if not current_price:
            current_price = previous_close
        
        return current_price, previous_close
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None, None

def create_simple_chart(ticker, trade_date):
    """Create simple price chart with decision zones."""
    
    # Get REAL-TIME price
    current_price, previous_close = get_realtime_price(ticker)
    
    if not current_price:
        print(f"Could not fetch real-time price for {ticker}")
        return False
    
    print(f"Real-time price for {ticker}: ${current_price}")
    
    # Calculate support and resistance based on current price
    # Support: 200 SMA (~8-10% below current)
    # Resistance: ~10% above current
    support = round(current_price * 0.92, 2)  # 8% below
    resistance = round(current_price * 1.10, 2)  # 10% above
    
    # Try to load analysis results for decision
    decision = "HOLD"
    try:
        log_file = f"eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = json.load(f)
            analysis = data.get(trade_date, {})
            final_decision = analysis.get('final_trade_decision', 'HOLD')
            if 'BUY' in final_decision.upper():
                decision = 'BUY'
            elif 'SELL' in final_decision.upper():
                decision = 'SELL'
            else:
                decision = 'HOLD'
    except:
        pass
    
    # Fetch historical data for chart
    try:
        from datetime import datetime, timedelta
        
        stock = yf.Ticker(ticker)
        # Get 90 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        hist = stock.history(start=start_date.strftime("%Y-%m-%d"), 
                            end=end_date.strftime("%Y-%m-%d"))
        
        if hist.empty:
            print("No historical data available")
            return False
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot price
    ax.plot(hist.index, hist['Close'], linewidth=2, color='black', label=f'{ticker} Price')
    
    # Fill zones based on real-time price
    min_price = hist['Close'].min() * 0.95
    max_price = hist['Close'].max() * 1.05
    
    # BUY zone (green) - below support
    ax.fill_between(hist.index, min_price, support, 
                     alpha=0.3, color='green', label='BUY Zone')
    
    # HOLD zone (yellow) - between support and resistance
    ax.fill_between(hist.index, support, resistance, 
                     alpha=0.2, color='yellow', label='HOLD Zone')
    
    # SELL zone (red) - above resistance
    ax.fill_between(hist.index, resistance, max_price, 
                     alpha=0.3, color='red', label='SELL Zone')
    
    # Lines
    ax.axhline(y=current_price, color='blue', linestyle='--', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax.axhline(y=support, color='green', linestyle=':', linewidth=1.5, 
               label=f'Support: ${support:.2f}')
    ax.axhline(y=resistance, color='red', linestyle=':', linewidth=1.5, 
               label=f'Resistance: ${resistance:.2f}')
    
    # Title with REAL-TIME price
    ax.set_title(f'{ticker} Trading Decision Chart\nReal-Time Price: ${current_price:.2f} | Decision: {decision}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box
    textstr = f'''
TRADING RECOMMENDATION: {decision}
{'='*40}
Current Price:    ${current_price:.2f}
Support:          ${support:.2f}
Resistance:        ${resistance:.2f}
Previous Close:    ${previous_close if previous_close else 'N/A':.2f}
{'='*40}
ZONES:
- BUY:  Below ${support:.2f}
- HOLD: ${support:.2f} - ${resistance:.2f}
- SELL: Above ${resistance:.2f}
'''
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output = f'{ticker}_Chart_realtime.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Chart saved: {output}")
    plt.close()
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        ticker = sys.argv[1].upper()
    else:
        ticker = input("Enter ticker (e.g., AMD, NVDA): ").upper().strip()
    
    if ticker:
        create_simple_chart(ticker, "realtime")
    else:
        print("Please provide a stock ticker")
