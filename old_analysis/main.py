from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config for Ollama with DeepSeek R1
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"  # Use Ollama for local models
config["deep_think_llm"] = "minimax-m2.5:cloud"  # MiniMax M2.5 Cloud
config["quick_think_llm"] = "minimax-m2.5:cloud"  # Same model for quick tasks
config["max_debate_rounds"] = 1  # Debate rounds

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

# Initialize with custom config (set debug=False to avoid Unicode emoji issues in console)
ta = TradingAgentsGraph(debug=False, config=config)

# forward propagate
print("Running Trading Agents analysis for AMD on 2026-02-20...")
result, decision = ta.propagate("AMD", "2026-02-20")

print("\n" + "="*60)
print("AMD STOCK ANALYSIS - 2026-02-20")
print("="*60)

print("\nMARKET REPORT:")
print("-" * 40)
print(result.get("market_report", "N/A")[:2000] if result.get("market_report") else "N/A")

print("\nFUNDAMENTALS REPORT:")
print("-" * 40)
print(result.get("fundamentals_report", "N/A")[:2000] if result.get("fundamentals_report") else "N/A")

print("\nNEWS REPORT:")
print("-" * 40)
print(result.get("news_report", "N/A")[:2000] if result.get("news_report") else "N/A")

print("\nINVESTMENT DECISION:")
print("-" * 40)
print(result.get("trader_investment_plan", "N/A")[:2000] if result.get("trader_investment_plan") else "N/A")

print("\nRISK ASSESSMENT:")
print("-" * 40)
print(result.get("investment_plan", "N/A")[:2000] if result.get("investment_plan") else "N/A")

print("\n" + "="*60)
print(f"FINAL DECISION: {decision}")
print("="*60)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
