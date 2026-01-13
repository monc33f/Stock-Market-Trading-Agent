import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# --- Hyperparameters ---
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 3e-4
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # Generalized Advantage Estimation smoothing
EPS_CLIP = 0.2              # PPO Clip parameter (0.2 is standard)
K_EPOCHS = 4                # How many times to update network per batch
UPDATE_TIMESTEP = 2000      # Update policy every n timesteps
MAX_TIMESTEPS = 50000       # Total training steps
HIDDEN_DIM = 64             # Neural network hidden layer size

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Memory:
    """
    Buffer to store trajectories (experience replay).
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network.
    - Actor: Outputs action probabilities (Softmax).
    - Critic: Outputs a single value estimate (Linear).
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Shared Feature Extractor (optional, but common)
        # Here we separate them for simplicity and stability in simple envs
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        Query the actor for an action during data collection.
        """
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.actor(state)
        
        # Create a categorical distribution to sample an action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()

    def evaluate(self, state, action):
        """
        Evaluate actions for the update step (calculate new logprobs and entropy).
        """
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        
        # Old policy for PPO ratio calculation (pi_theta / pi_theta_old)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 1. Prepare Data
        # Convert list of returns to tensors
        rewards = []
        discounted_reward = 0
        
        # -- Monte Carlo Reward Calculation (Simple version) --
        # Ideally, use GAE here, but simple discounted returns work for CartPole
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards (Crucial for stability)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        # 2. PPO Update Loop (K epochs)
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Finding the Ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            # Final Loss: -min(surr1, surr2) + 0.5*MSE(val, reward) - 0.01*Entropy
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 3. Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        memory.clear()

