import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

# ==========================================
#  IMPORTS FROM FOLDERS
# ==========================================

from environments.monthly_trading_env import MonthlyTradingEnv
from environments.data_fetcher import get_data, calculate_monthly_features

from algorithms.CS_PPO import cuckoo_search_ppo
from algorithms.CS_SAC import cuckoo_search_sac

from results.figures.evaluation_function import evaluate_and_save_plot

# ==========================================
#  MAIN EXECUTION
# ==========================================

def run(args):
    # --------------------------------------
    # 1. Data Loading & Processing
    # --------------------------------------
    print("--- Loading Data ---")
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    # Fetch data using function from 'environments'
    prices_raw = get_data(tickers, "15y")
    prices_m, features_m = calculate_monthly_features(prices_raw)

    # Normalization
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_m), 
        index=features_m.index, 
        columns=features_m.columns
    )

    # Train/Test Split (Last 24 months for test)
    split_idx = len(features_scaled) - 24
    train_p, test_p = prices_m.iloc[:split_idx], prices_m.iloc[split_idx:]
    train_f, test_f = features_scaled.iloc[:split_idx], features_scaled.iloc[split_idx:]

    print(f"Training months: {len(train_p)}")
    print(f"Testing months: {len(test_p)}")

    # --------------------------------------
    # 2. Hyperparameter Optimization (Cuckoo Search)
    # --------------------------------------
    print(f"\n--- Starting Optimization using Cuckoo Search ({args.algo.upper()}) ---")
    
    best_params = {}
    
    if args.algo == 'ppo':
        # Running PPO Cuckoo Search
        best_p = cuckoo_search_ppo(n_nests=5, iterations=args.iter, train_p=train_p, train_f=train_f)
        
        final_lr, final_gamma, final_n_steps_exp, final_ent = best_p
        final_n_steps = int(2**round(final_n_steps_exp))
        
        print(f"Best PPO Params: LR={final_lr:.6f}, Gamma={final_gamma:.4f}, n_steps={final_n_steps}, Ent={final_ent:.4f}")
        
        best_params = {
            "learning_rate": final_lr,
            "gamma": final_gamma,
            "n_steps": final_n_steps,
            "ent_coef": final_ent
        }

    elif args.algo == 'sac':
        # Running SAC Cuckoo Search
        best_p = cuckoo_search_sac(n_nests=5, iterations=args.iter, train_p=train_p, train_f=train_f)
        
        final_lr, final_gamma, final_tau, final_ent = best_p 
        
        print(f"Best SAC Params: LR={final_lr:.6f}, Gamma={final_gamma:.4f}, Tau={final_tau:.4f}")

        best_params = {
            "learning_rate": final_lr,
            "gamma": final_gamma,
            "tau": final_tau,
            "ent_coef": final_ent
        }

    # --------------------------------------
    # 3. Final Training
    # --------------------------------------
    print(f"\n--- Phase 2: Final Training ({args.algo.upper()}) ---")
    
    log_dir = f"./logs/final_{args.algo.upper()}/"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Environment
    env_train = MonthlyTradingEnv(train_p, train_f, lookback_window=3)
    env_train = Monitor(env_train, log_dir)

    # Initialize Model with Best Params
    if args.algo == 'ppo':
        model = PPO("MlpPolicy", env_train, verbose=1, **best_params)
    elif args.algo == 'sac':
        model = SAC("MlpPolicy", env_train, verbose=1, **best_params)

    # Train
    model.learn(total_timesteps=args.timesteps)
    model.save(f"{log_dir}/best_model_{args.algo}")

    # --------------------------------------
    # 4. Evaluation
    # --------------------------------------
    print(f"\n--- Phase 3: Evaluation ---")
    
    # Using function from 'results' folder
    evaluate_and_save_plot(
        model=model, 
        test_prices=test_p, 
        test_features=test_f, 
        log_dir=log_dir, 
        lookback=3, 
        initial_balance=100000
    )
    print("Experiment Finished. Results saved in logs folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trading Experiment")
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'sac'], help="Algorithm to use")
    parser.add_argument("--iter", type=int, default=3, help="Number of Cuckoo Search iterations")
    parser.add_argument("--timesteps", type=int, default=200, help="Timesteps for final training")
    
    args = parser.parse_args()
    run(args)