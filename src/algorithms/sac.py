import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque

# --- Hyperparameters ---
ENV_NAME = "Pendulum-v1"
SEED = 42
GAMMA = 0.99            # Discount factor
TAU = 0.005             # Soft update parameter for target networks
ALPHA = 0.2             # Entropy regularization coefficient (Fixed for simplicity)
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
BATCH_SIZE = 256
MEMORY_SIZE = 100000    # Replay buffer size
HIDDEN_DIM = 256
START_STEPS = 1000      # Steps to sample random actions before using policy
UPDATES_PER_STEP = 1    # How many gradient updates per env step
MAX_STEPS = 15000       # Total training steps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.tensor(np.array(state), dtype=torch.float32).to(device),
                torch.tensor(np.array(action), dtype=torch.float32).to(device),
                torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(np.array(next_state), dtype=torch.float32).to(device),
                torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# --- Networks ---

class SoftQNetwork(nn.Module):
    """
    Critic: Estimates Q(s, a). 
    SAC uses two of these (Twin Critics) to reduce overestimation bias.
    """
    def __init__(self, num_inputs, num_actions):
        super(SoftQNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear3 = nn.Linear(HIDDEN_DIM, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, HIDDEN_DIM)
        self.linear5 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear6 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class Actor(nn.Module):
    """
    Actor: Outputs Mean and Std for a Gaussian distribution.
    Uses the Reparameterization Trick to sample actions.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.layer2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        
        self.mean = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)
        
        self.max_action = max_action
        
        # Action rescaling limits (log_std clamping)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        
        mu = self.mean(x)
        log_std = self.log_std(x)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick: z ~ N(0,1), action = mu + sigma * z
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample() 
        action_tanh = torch.tanh(z)
        
        # Scale action to env limits
        action = action_tanh * self.max_action
        
        # Enforce action bound for log_prob calculation (Change of variable formula)
        # log_prob(a) = log_prob(u) - sum(log(1 - tanh(u)^2))
        log_prob = normal.log_prob(z) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

# --- SAC Agent ---

class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic (Current) and Target
        self.critic = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.max_action = max_action
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            mu, _ = self.actor(state)
            return (torch.tanh(mu) * self.max_action).detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size)

        with torch.no_grad():
            # 1. Target Policy Smoothing
            next_state_action, next_state_log_pi = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            
            # 2. Min Clipped Q-Learning (Double Q)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - ALPHA * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * GAMMA * min_qf_next_target

        # 3. Two Q-functions update
        qf1, qf2 = self.critic(state_batch, action_batch) 
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # 4. Actor Update
        pi, log_pi = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Objective: Maximize (Q - alpha * log_pi) -> Minimize (alpha * log_pi - Q)
        policy_loss = ((ALPHA * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 5. Soft Update of Target Network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

