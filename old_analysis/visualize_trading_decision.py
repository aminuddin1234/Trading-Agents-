"""
TradingAgents Visualization Module
Creates price charts with BUY/HOLD/SELL zones based on analysis results
"""
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import json
import os
import sys


def load_analysis_results(ticker, trade_date):
    """Load analysis results from JSON log file."""
    log_file = f"eval_results/{ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json"
    
    if not os.path.exists(log_file):
        print(f"Analysis file not found: {log_file}")
        print("Please run analysis first using main.py")
        return None
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data.get(trade_date, {})


def extract_price_levels(analysis_data):
    """Extract price levels from analysis data."""
    # Default levels if extraction fails
    levels = {
        'current_price': 187.90,
        'support_200_sma': 173.17,
        'support_50_sma': 184.80,
        'resistance_high': 212.19,
        'resistance_recent': 192.51,
        'rsi': 54.88,
        'decision': 'HOLD'
    }
    
    try:
        # Try to extract from market report
        market_report = analysis_data.get('market_report', '')
        
        # Extract current price
        if 'trading at' in market_report.lower():
            import re
            price_match = re.search(r'\$([\d,]+\.\d{2})', market_report)
            if price_match:
                levels['current_price'] = float(price_match.group(1).replace(',', ''))
        
        # Extract 200 SMA
        if '200 SMA' in market_report:
            import re
            sma200_match = re.search(r'200 SMA.*?\$([\d.]+)', market_report)
            if sma200_match:
                levels['support_200_sma'] = float(sma200_match.group(1))
        
        # Extract 50 SMA
        if '50 SMA' in market_report:
            import re
            sma50_match = re.search(r'50 SMA.*?\$([\d.]+)', market_report)
            if sma50_match:
                levels['support_50_sma'] = float(sma50_match.group(1))
        
        # Extract decision
        final_decision = analysis_data.get('final_trade_decision', '')
        if 'HOLD' in final_decision.upper():
            levels['decision'] = 'HOLD'
        elif 'BUY' in final_decision.upper():
            levels['decision'] = 'BUY'
        elif 'SELL' in final_decision.upper():
            levels['decision'] = 'SELL'
            
    except Exception as e:
        print(f"Warning: Could not extract all levels: {e}")
        print("Using default values")
    
    return levels


def create_trading_chart(ticker, trade_date, output_file=None):
    """Create price chart with BUY/HOLD/SELL zones."""
    
    print(f"\nCreating trading chart for {ticker} on {trade_date}...")
    
    # Load analysis results
    analysis_data = load_analysis_results(ticker, trade_date)
    if not analysis_data:
        return False
    
    # Extract price levels
    levels = extract_price_levels(analysis_data)
    
    # Fetch historical data
    try:
        # Parse trade date
        date_obj = datetime.strptime(trade_date, "%Y-%m-%d")
        start_date = (date_obj - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = trade_date
        
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"No price data available for {ticker}")
            return False
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return False
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    current_price = levels['current_price']
    support_200 = levels['support_200_sma']
    support_50 = levels['support_50_sma']
    resistance = levels['resistance_high']
    decision = levels['decision']
    
    # Calculate zones
    buy_zone_top = support_200
    buy_zone_bottom = data['Close'].min() * 0.95
    sell_zone_bottom = resistance
    sell_zone_top = data['Close'].max() * 1.05
    
    # Plot price
    ax1.plot(data.index, data['Close'], linewidth=2.5, color='#2C3E50', label=f'{ticker} Price')
    
    # Fill decision zones
    # BUY zone (green)
    ax1.fill_between(data.index, buy_zone_bottom, buy_zone_top, 
                     alpha=0.25, color='#27AE60', label='BUY Zone')
    
    # HOLD zone (yellow)
    ax1.fill_between(data.index, buy_zone_top, sell_zone_bottom, 
                     alpha=0.2, color='#F1C40F', label='HOLD Zone')
    
    # SELL zone (red)
    ax1.fill_between(data.index, sell_zone_bottom, sell_zone_top, 
                     alpha=0.25, color='#E74C3C', label='SELL Zone')
    
    # Horizontal lines
    ax1.axhline(y=current_price, color='#3498DB', linestyle='--', linewidth=2.5, 
               label=f'Current: ${current_price:.2f}')
    ax1.axhline(y=support_200, color='#27AE60', linestyle=':', linewidth=2, 
               label=f'200 SMA: ${support_200:.2f}')
    ax1.axhline(y=support_50, color='#F39C12', linestyle=':', linewidth=2, 
               label=f'50 SMA: ${support_50:.2f}')
    ax1.axhline(y=resistance, color='#E74C3C', linestyle=':', linewidth=2, 
               label=f'Resistance: ${resistance:.2f}')
    
    # Add decision badge
    decision_colors = {
        'BUY': '#27AE60',
        'HOLD': '#F1C40F', 
        'SELL': '#E74C3C'
    }
    badge_color = decision_colors.get(decision, '#95A5A6')
    
    # Title with decision
    ax1.set_title(f'{ticker} Trading Decision Chart\nAnalysis Date: {trade_date} | Recommendation: {decision}', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # Annotations
    ax1.annotate(f'${current_price:.2f}', 
                xy=(data.index[-1], current_price),
                xytext=(15, 5), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB', alpha=0.9),
                color='white', fontweight='bold', fontsize=11)
    
    # Styling
    ax1.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([data.index[0], data.index[-1]])
    
    # Add some padding to y-axis
    price_range = data['Close'].max() - data['Close'].min()
    ax1.set_ylim([data['Close'].min() - price_range*0.1, 
                  data['Close'].max() + price_range*0.1])
    
    # Volume subplot
    colors = ['#27AE60' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
              else '#E74C3C' for i in range(len(data))]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Decision explanation text box
    explanation = f'''
╔══════════════════════════════════════╗
║   TRADING RECOMMENDATION: {decision:<8}   ║
╠══════════════════════════════════════╣
║ Current Price:    ${current_price:<8.2f}      ║
║ 200 SMA Support:  ${support_200:<8.2f}      ║
║ 50 SMA Support:   ${support_50:<8.2f}      ║
║ Resistance:       ${resistance:<8.2f}      ║
╠══════════════════════════════════════╣
║  ZONES:                              ║
║  🟢 BUY:  Below ${support_200:.0f}              ║
║  🟡 HOLD: ${support_200:.0f} - ${resistance:.0f}         ║
║  🔴 SELL: Above ${resistance:.0f}             ║
╚══════════════════════════════════════╝
'''
    
    props = dict(boxstyle='round,pad=0.8', facecolor='#ECF0F1', 
                alpha=0.95, edgecolor='#2C3E50', linewidth=2)
    ax1.text(0.02, 0.98, explanation, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace',
            color='#2C3E50')
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = f'{ticker}_Trading_Chart_{trade_date}.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ Chart saved: {output_file}")
    
    plt.show()
    return True


def main():
    """Main function - get user input and create chart."""
    print("="*60)
    print("  TradingAgents Visualization Tool")
    print("  Generate price charts with BUY/HOLD/SELL zones")
    print("="*60)
    
    # Get user input
    if len(sys.argv) >= 3:
        ticker = sys.argv[1].upper()
        trade_date = sys.argv[2]
    else:
        ticker = input("\nEnter stock ticker (e.g., NVDA, AAPL): ").upper().strip()
        trade_date = input("Enter analysis date (YYYY-MM-DD): ").strip()
    
    if not ticker or not trade_date:
        print("Error: Please provide both ticker and date")
        print("Usage: python visualize_trading_decision.py NVDA 2026-02-20")
        return
    
    # Create chart
    success = create_trading_chart(ticker, trade_date)
    
    if not success:
        print("\nFailed to create chart. Make sure:")
        print("1. Analysis has been run for this ticker/date")
        print("2. The JSON log file exists")
        print("3. Price data is available")


if __name__ == "__main__":
    main()
