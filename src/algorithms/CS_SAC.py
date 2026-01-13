import math
import numpy as np
from stable_baselines3 import SAC
from environments.monthly_trading_env import MonthlyTradingEnv
# ==========================================
# III. CUCKOO SEARCH OPTIMIZATION
# ==========================================

def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
            (math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.normal(0, sigma, 1)
    v = np.random.normal(0, 1, 1)
    step = u / abs(v)**(1 / Lambda)
    return step

def objective_function(params, train_p, train_f):
    # 1. Unpack the 4 parameters matching the new requirement
    lr, gamma, tau, ent_coef = params 
    
    # 2. Fix batch_size 
    batch_size = 256 

    env = None
    try:
        env = MonthlyTradingEnv(train_p, train_f, lookback_window=3)
        
        # 3. Pass the new parameters to SAC
        model = SAC("MlpPolicy", env, 
                    learning_rate=lr, 
                    gamma=gamma, 
                    tau=tau,           
                    ent_coef=ent_coef, 
                    batch_size=batch_size, 
                    verbose=0)
        
        model.learn(total_timesteps=100)
        
        # Evaluation logic (unchanged)
        obs, _ = env.reset()
        done = False
        final_val = 100000
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            final_val = info['portfolio_value']
            
        return - (final_val / 100000 - 1)
        
    except Exception:
        return 999
    finally:
        if env is not None:
            env.close()

def cuckoo_search_sac(n_nests, iterations, train_p, train_f):
    # ==========================================
    # DEFINING BOUNDS FOR 4 PARAMETERS
    # 0: LR       [1e-5, 1e-2]
    # 1: Gamma    [0.80, 0.999]
    # 2: Tau      [0.001, 0.1]
    # 3: Ent_Coef [0.0001, 0.5]
    # ==========================================
    lower_bound = np.array([1e-5, 0.80, 0.001, 0.0001])
    upper_bound = np.array([1e-2, 0.999, 0.1,   0.5])
    
    # Initialize nests with 4 dimensions
    nests = np.random.uniform(lower_bound, upper_bound, (n_nests, 4))
    
    fitness = np.array([objective_function(n, train_p, train_f) for n in nests])
    best_nest = nests[np.argmin(fitness)]
    
    for i in range(iterations):
        for j in range(n_nests):
            step = 0.01 * levy_flight(1.5) * (nests[j] - best_nest)
            # Clip using the new 4-dimensional bounds
            new_nest = np.clip(nests[j] + step, lower_bound, upper_bound)
            new_fit = objective_function(new_nest, train_p, train_f)
            
            if new_fit < fitness[j]:
                fitness[j] = new_fit
                nests[j] = new_nest
                
        best_nest = nests[np.argmin(fitness)]
        # Print current best
        print(f"Iter {i+1} | Best Return: {-np.min(fitness)*100:.2f}%")
        
    return best_nest