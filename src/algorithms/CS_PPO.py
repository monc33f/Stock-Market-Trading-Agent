import math
import numpy as np
from stable_baselines3 import PPO
from environments.monthly_trading_env import MonthlyTradingEnv

# ==========================================
# III. CUCKOO SEARCH OPTIMIZATION (PPO)
# ==========================================

def objective_function_ppo(params, train_p, train_f):
    """
    Evaluates a specific set of PPO hyperparameters.
    Returns the negative return (since we are minimizing).
    """
    # Unpack parameters: Learning Rate, Gamma, n_steps (exponent), Entropy Coefficient
    lr, gamma, n_steps_exp, ent_coef = params
    
    # Convert the exponent to an integer power of 2 (e.g., 2^5 = 32 steps)
    n_steps = int(2**round(n_steps_exp))
    
    try:
        # Initialize environment with a lookback window of 3 months
        env = MonthlyTradingEnv(train_p, train_f, lookback_window=3)
        
        # Initialize PPO model with the current hyperparameters
        model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, 
                    n_steps=n_steps, ent_coef=ent_coef, verbose=0)
        
        # Train for a short duration (100 timesteps) to get a quick estimate
        model.learn(total_timesteps=100)
        
        # Run one evaluation episode on the training data
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        
        # Return negative return for minimization (Higher return = Lower fitness value)
        return - (info['portfolio_value'] / 100000 - 1)
        
    except:
        # Return a high penalty value if the model crashes (e.g., unstable params)
        return 999

def cuckoo_search_ppo(n_nests, iterations, train_p, train_f):
    """
    Optimizes PPO hyperparameters using the Cuckoo Search algorithm.
    """
    # Initialize nests (population) with random values within bounds
    # Bounds: [Learning Rate, Gamma, n_steps_exp, Ent_Coef]
    # Lower: [1e-5, 0.8, 5 (32 steps), 0.0]
    # Upper: [1e-3, 0.99, 10 (1024 steps), 0.05]
    nests = np.random.uniform([1e-5, 0.8, 5, 0.0], [1e-3, 0.99, 10, 0.05], (n_nests, 4))
    
    # Calculate initial fitness for all nests
    fitness = np.array([objective_function_ppo(n, train_p, train_f) for n in nests])
    
    # Identify the current best solution
    best_nest = nests[np.argmin(fitness)]
    
    # Main Optimization Loop
    for i in range(iterations):
        for j in range(n_nests):
            # Calculate Levy Flight step (Mantegna's algorithm)
            sigma = (math.gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2) / 
                    (math.gamma((1 + 1.5) / 2) * 1.5 * 2**((1.5 - 1) / 2)))**(1 / 1.5)
            
            # Generate new step size
            step = 0.01 * (np.random.normal(0, sigma, 1) / abs(np.random.normal(0, 1, 1))**(1/1.5)) * (nests[j] - best_nest)
            
            # Apply step to create a new nest and clip to bounds to ensure validity
            new_nest = np.clip(nests[j] + step, [1e-5, 0.8, 5, 0.0], [1e-3, 0.99, 10, 0.05])
            
            # Evaluate the new nest
            new_fit = objective_function_ppo(new_nest, train_p, train_f)
            
            # Greedy selection: If new nest is better, replace the old one
            if new_fit < fitness[j]:
                fitness[j] = new_fit
                nests[j] = new_nest
        
        # Update the global best nest found so far
        best_nest = nests[np.argmin(fitness)]
        
        # Print progress (Converting negative fitness back to positive return percentage)
        print(f"PPO Opti | Iter {i+1} | Best Return: {-np.min(fitness)*100:.2f}%")
        
    return best_nest