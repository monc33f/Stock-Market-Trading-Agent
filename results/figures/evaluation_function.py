import sys
import os
import matplotlib.pyplot as plt
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))  
results_dir = os.path.dirname(current_dir)                
project_root = os.path.dirname(results_dir)               
sys.path.append(project_root)

from src.environments.monthly_trading_env import MonthlyTradingEnv

# ==========================================
# III. EVALUATION AND BACKTEST
# ==========================================

def evaluate_and_save_plot(model, test_prices, test_features, log_dir, lookback=3, initial_balance=100000):
    """Performs backtesting, displays the plot, and saves it."""
    
    # 1. Agent Simulation
    env_test = MonthlyTradingEnv(test_prices, test_features, initial_balance=initial_balance, lookback_window=lookback)
    obs, _ = env_test.reset()
    
    rl_values = [initial_balance]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env_test.step(action)
        rl_values.append(info['portfolio_value'])
    
    # 2. Market Simulation (Buy & Hold)
    prices_sync = test_prices.iloc[lookback:]
    dates = prices_sync.index
    shares_bh = (initial_balance / test_prices.shape[1]) / prices_sync.iloc[0].values
    market_values = prices_sync.values @ shares_bh
    
    # Alignment
    rl_values = rl_values[:len(dates)]

    # 3. Plot Creation
    plt.figure(figsize=(12, 6))
    plt.plot(dates, rl_values, label="Optimized AI", color='#1f77b4', lw=2.5)
    plt.plot(dates, market_values, label="Market (Buy & Hold)", color='#7f7f7f', linestyle='--', lw=2)
    
    # Performance Zones
    plt.fill_between(dates, rl_values, market_values, where=(np.array(rl_values) >= np.array(market_values)),
                     color='green', alpha=0.1, interpolate=True)
    plt.fill_between(dates, rl_values, market_values, where=(np.array(rl_values) < np.array(market_values)),
                     color='red', alpha=0.1, interpolate=True)

    plt.title("Cumulative Performance: Agent vs Market", fontsize=14)
    plt.ylabel("Portfolio Value (€)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SAVE PLOT
    plot_path = os.path.join(log_dir, "backtest_performance.png")
    plt.savefig(plot_path)
    print(f"Performance plot saved at: {plot_path}")
    
    plt.show()
    
    print(f"AI Return: {((rl_values[-1]/initial_balance)-1)*100:.2f}%")
    print(f"Market Return: {((market_values[-1]/initial_balance)-1)*100:.2f}%")