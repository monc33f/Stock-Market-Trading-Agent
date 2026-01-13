# AI Portfolio Manager 📈🤖

An autonomous **Portfolio Allocation Agent** powered by Deep Reinforcement Learning (DRL). This system implements a hybrid neuro-evolutionary architecture, using **Cuckoo Search (CS)** as a meta-optimizer to enhance the performance of **Soft Actor-Critic (SAC)** and **Proximal Policy Optimization (PPO)** agents.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stable Baselines3](https://img.shields.io/badge/SB3-SAC%20%7C%20PPO-green)
![Optimization](https://img.shields.io/badge/Metaheuristic-Cuckoo%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

## 🧠 Algorithms & Architecture

This project solves the "black box" initialization problem of DRL by using nature-inspired algorithms.

### 1. The Meta-Optimizer: Cuckoo Search (CS)
Standard RL agents often get stuck in local optima depending on their random initialization. We use **Cuckoo Search**, a meta-heuristic algorithm inspired by the brood parasitism of cuckoo birds, to explore the hyperparameter space and optimize initial network weights.
- **Global Search:** Uses Lévy flights (random walks with heavy-tailed probability) to effectively explore the solution landscape.
- **Application:** CS is "linked" to both agents, acting as a wrapper that tunes the agent before or during the training process to ensure robust convergence.

### 2. The DRL Agents (The Workers)
Once optimized by Cuckoo Search, the agents take over to learn the trading strategy:

* **SAC (Soft Actor-Critic):** * **Role:** The entropy maximizer. It learns a stochastic policy that maximizes trade returns while maintaining high randomness in its actions (entropy).
    * **Why CS?** CS helps SAC find the optimal *temperature* parameter (alpha) and network depth for the specific volatility of the target stocks.

* **PPO (Proximal Policy Optimization):**
    * **Role:** The stable stabilizer. It uses clipped objective functions to ensure safe policy updates.
    * **Why CS?** CS optimizes PPO's learning rate and clipping range to prevent the policy from collapsing during market crashes.

---

## 📊 The Strategy (Market Environment)

The agents operate within a custom Gymnasium environment (`MonthlyTradingEnv`) designed for realistic fund management.

- **Timeframe:** Monthly (Business Month End) to capture macro trends.
- **Assets:** Trades a dynamic basket of tech giants: `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `TSLA`.
- **Observation Space:** 3-month lookback window including:
  - **Log Returns:** $\ln(P_t / P_{t-1})$
  - **RSI (6-month):** Momentum health check.
  - **Trend Signal:** Spread between Short-term (3m) and Long-term (12m) moving averages.
- **Action Space:** Continuous vector + **Softmax** normalization (Portfolio Weights sum to 100%).
- **Friction:** **0.1% transaction fee** per rebalance.

---

## 📂 Project Structure

```text
├── README.md               # Documentation
├── requirements.txt        # Dependencies
├── Dockerfile              # Reproducible container
├── src/
│   ├── algorithms/
│   │   ├── CS_PPO.py      
│   │   ├── CS_SAC.py # Meta-heuristic optimization logic
│   │   ├── sac_agent.py     # SAC implementation
│   │   └── ppo_agent.py     # PPO implementation
│   ├── environments/
│   │   ├── data_fetcher.py               #yfinance datasets
│   │   └── monthly_trading_env.py        # Custom 'MonthlyTradingEnv'
│   └── run_experiment.py    # CLI entry point               
├── results/
│   ├── logs/                # Tensorboard metrics
│   └── figures/             # Backtest comparison plots
│   │   ├── evaluation_function.py              
└── monthly_trading.ipynb # Prototyping sandbox

## 🏃 How to Run the Experiment

The `src/run_experiment.py` script handles the entire pipeline: data fetching, Cuckoo Search optimization, training, and evaluation.

### 1. Prerequisites
Ensure your virtual environment is active and you are in the project root directory.

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate


2. Basic Usage (Quick Test)
Run a quick training session with default settings (3 search iterations, 200 training steps) to verify everything works.

Bash

# Train PPO Agent
python src/run_experiment.py --algo ppo

# Train SAC Agent
python src/run_experiment.py --algo sac


3. Advanced Configuration (Full Training)
Use command-line arguments to control the optimization intensity and training duration.

Syntax:

Bash

python src/run_experiment.py --algo <ALGO> --iter <ITERATIONS> --timesteps <STEPS>
Arguments: | Flag | Required | Default | Description | | :--- | :--- | :--- | :--- | | --algo | Yes | N/A | Algorithm to use (ppo or sac). | | --iter | No | 3 | Number of Cuckoo Search iterations (Nests). Higher = better hyperparameters. | | --timesteps | No | 200 | Total training steps for the final agent. Use 20000+ for real results. |

Example: Serious Training Run Train an SAC agent with extensive hyperparameter search (10 iterations) and long training (50,000 steps).

Bash

python src/run_experiment.py --algo sac --iter 10 --timesteps 50000
