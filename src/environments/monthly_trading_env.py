import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ==========================================
# II. MONTHLY TRADING ENVIRONMENT
# ==========================================

class MonthlyTradingEnv(gym.Env):
    def __init__(self, prices_df, features_df, initial_balance=100000, lookback_window=3):
        super(MonthlyTradingEnv, self).__init__()
        self.prices = prices_df.values
        self.features = features_df.values
        self.n_assets = prices_df.shape[1]
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Actions: Weights for each asset (-1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Observations: Features over the lookback window
        obs_shape = self.features.shape[1] * lookback_window
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        obs = self.features[self.current_step - self.lookback_window : self.current_step]
        return obs.flatten().astype(np.float32)

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        
        # Softmax Allocation (sum of weights = 100%)
        exp_a = np.exp(action)
        weights = exp_a / np.sum(exp_a)
        
        # Monthly transaction
        current_prices = self.prices[self.current_step]
        target_shares = (weights * self.portfolio_value) / current_prices
        
        # 0.1% fee on traded volume
        trade_volume = np.sum(np.abs(target_shares - self.shares_held) * current_prices)
        costs = trade_volume * 0.001
        
        self.shares_held = target_shares
        self.portfolio_value -= costs 
        
        # Move to the next month
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        
        # Update value with new prices
        self.portfolio_value = np.sum(self.shares_held * self.prices[self.current_step])
        reward = (self.portfolio_value / prev_portfolio_value) - 1.0
        
        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}