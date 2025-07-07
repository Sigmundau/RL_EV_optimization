import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter

from environments.mnm_v1b import MnM_v1b
import settings
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# --- Rainbow DQN Components ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, obs_dim, n_actions, n_atoms=51, Vmin=-10, Vmax=10):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(Vmin, Vmax, n_atoms)
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.noisy1 = NoisyLinear(256, 128)
        self.noisy2 = NoisyLinear(128, n_actions * n_atoms)


    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = x.view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(x, dim=2)
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# --- Replay Buffer with Prioritized Experience Replay ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.2):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    def __len__(self):
        return len(self.buffer)

# --- Utilities ---
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

def get_obs(env, obs):
    fin = torch.as_tensor(obs["financial_metrics"], dtype=torch.float32).flatten()
    mac = torch.as_tensor(obs["macro_economic_data"], dtype=torch.float32).flatten()
    # Defensive: if either is empty, pad to match training obs size
    if fin.numel() + mac.numel() != 35:
        # Use zeros to pad to 35
        out = torch.cat([fin, mac])
        pad_len = 35 - out.numel()
        if pad_len > 0:
            out = torch.cat([out, torch.zeros(pad_len, dtype=out.dtype, device=out.device)])
        elif pad_len < 0:
            out = out[:35]
        return out
    return torch.cat([fin, mac])

def select_action(model, state, epsilon, n_actions, device):
    state = state.unsqueeze(0).to(device)
    log_probs = model(state)
    q_vals = (log_probs.exp() * model.support.to(device)).sum(2)
    return q_vals.argmax(1).item()


def projection_distribution(next_model, next_states, rewards, dones, gamma, n_atoms, Vmin, Vmax, device):
    batch_size = rewards.size(0)
    delta_z = float(Vmax - Vmin) / (n_atoms - 1)
    support = torch.linspace(Vmin, Vmax, n_atoms).to(device)
    with torch.no_grad():
        next_log_probs = next_model(next_states)
        next_q = (next_log_probs.exp() * support).sum(2)
        next_action = next_q.argmax(1)
        next_dist = next_log_probs[range(batch_size), next_action]
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        Tz = rewards + (1 - dones) * gamma * support
        Tz = Tz.clamp(Vmin, Vmax)
        b = (Tz - Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        proj_dist = torch.zeros_like(next_dist)
        for i in range(batch_size):
            for j in range(n_atoms):
                lj = l[i, j]
                uj = u[i, j]
                if lj == uj:
                    proj_dist[i, lj] += next_dist[i, j]
                else:
                    proj_dist[i, lj] += next_dist[i, j] * (uj - b[i, j])
                    proj_dist[i, uj] += next_dist[i, j] * (b[i, j] - lj)
    return proj_dist

# --- Main Training Script ---
def main():
    # --- Hyperparameters ---
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = 1
    batch_size = 128
    buffer_size = 100_000
    learning_rate = 1e-4
    n_atoms = 51
    Vmin = -10
    Vmax = 10
    update_target_every = 1000
    min_buffer = 1000
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 50_000
    beta_start = 0.4
    beta_frames = 100_000
    max_episodes = 1000
    T_horizon = 256
    # --- Environment Setup ---
    ticker = 'KO'
    fred_api_key = settings.fred_api_key
    terminal_growth_rate = 0.02
    credit_rating = 'A'
    simulation_horizon = 10
    csv_path = "data/KO_quarterly.csv"
    debug = False
    train_env = MnM_v1b(
        ticker=ticker,
        fred_api_key=fred_api_key,
        terminal_growth_rate=terminal_growth_rate,
        credit_rating=credit_rating,
        fixed_origin=False,
        simulation_horizon=simulation_horizon,
        csv_path=csv_path,
        fixed_path=False,
        debug=debug,
        device=device.type
    )
    test_env = MnM_v1b(
        ticker=ticker,
        fred_api_key=fred_api_key,
        terminal_growth_rate=terminal_growth_rate,
        credit_rating=credit_rating,
        fixed_origin=False,
        simulation_horizon=simulation_horizon,
        csv_path=csv_path,
        fixed_path=True,
        fixed_path_csv='data/simulated_path_10step.csv',
        debug=debug,
        device=device.type
    )
    # After resetting the environment
    obs = train_env.reset()
    fin = torch.as_tensor(obs["financial_metrics"], dtype=torch.float32).flatten()
    mac = torch.as_tensor(obs["macro_economic_data"], dtype=torch.float32).flatten()
    state = torch.cat([fin, mac])


    # Calculate obs_dim as used in your model
    obs_dim = state.numel()

    n_actions = 11 * 11  # 2D categorical action, flatten to 1D
    # --- Model and Buffer ---
    policy_net = RainbowDQN(obs_dim, n_actions, n_atoms, Vmin, Vmax).to(device)
    target_net = RainbowDQN(obs_dim, n_actions, n_atoms, Vmin, Vmax).to(device)
    try:
        policy_net = torch.compile(policy_net)
        target_net = torch.compile(target_net)
        print("Using torch.compile for model speedup.")
    except Exception:
        pass
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    buffer = PrioritizedReplayBuffer(buffer_size)
    # --- Logging ---
    log_dir = os.path.join("runs", f"mnm_rainbowdqn_v1a_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    global_step = 0
    episode_rewards = deque(maxlen=100)
    # --- Training Loop ---
    for episode in range(max_episodes):
        obs = train_env.reset()
        state = get_obs(train_env, obs).to(device)
        policy_net.reset_noise()
        total_reward = 0
        t = 0
        done = False
        while t < T_horizon and not done:
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * global_step / epsilon_decay)
            action_idx = select_action(policy_net, state, epsilon, n_actions, device)
            action_2d = [action_idx // 11, action_idx % 11]
            action_tensor = torch.as_tensor(action_2d, dtype=torch.long, device=device)
            from tensordict import TensorDict
            action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
            next_td = train_env.step(action_td)
            next_obs_td = next_td.get('next')
            reward = next_obs_td.get('reward').item()
            done = next_obs_td.get('done').item()
            next_state_td = next_obs_td.get('next')
            next_state = get_obs(train_env, next_state_td).to(device)
            buffer.push(Transition(state.cpu(), action_idx, reward, next_state.cpu(), done))
            state = next_state
            total_reward += reward
            global_step += 1
            t += 1
            if len(buffer) > min_buffer:
                beta = min(1.0, beta_start + global_step * (1.0 - beta_start) / beta_frames)
                transitions, indices, weights = buffer.sample(batch_size, beta)
                batch = Transition(*zip(*transitions))
                states = torch.stack(batch.state).to(device)
                actions = torch.tensor(batch.action, dtype=torch.long, device=device)
                rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
                next_states = torch.stack(batch.next_state).to(device)
                dones = torch.tensor(batch.done, dtype=torch.float32, device=device)
                weights = weights.to(device)
                proj_dist = projection_distribution(target_net, next_states, rewards, dones, gamma, n_atoms, Vmin, Vmax, device)
                log_probs = policy_net(states)
                log_probs_a = log_probs[range(batch_size), actions]
                loss = -(proj_dist * log_probs_a).sum(1)
                loss = (loss * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                policy_net.reset_noise()
                writer.add_scalar('Loss/Train', loss.item(), global_step)  # <-- Add this line
                if global_step % update_target_every == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        episode_rewards.append(total_reward)
        writer.add_scalar('Reward/Episodic Avg Reward', total_reward, episode)
        
        if episode % 10 == 0:

            print(f"Episode {episode}, Episode Reward: {total_reward:.2f}, Buffer: {len(buffer)}")
            # --- Evaluation on fixed-path env ---
            eval_rewards = []
            obs_eval = test_env.reset()
            # PPO-style: always extract state from 'financial_metrics' and 'macro_economic_data' keys
            state = get_obs(test_env, obs_eval).to(device)
            t = 0
            done = False
            while t < T_horizon and not done:
                with torch.no_grad():
                    action_idx = select_action(policy_net, state, 0.0, n_actions, device)
                action_2d = [action_idx // 11, action_idx % 11]
                action_tensor = torch.as_tensor(action_2d, dtype=torch.long, device=device)
                action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
                next_td = test_env.step(action_td)
                next_obs_td = next_td.get('next')
                reward = next_obs_td.get('reward').item()
                done = next_obs_td.get('done').item()
                if not done:
                    next_state_td = next_obs_td.get('next')
                    state = get_obs(test_env, next_state_td).to(device)
                eval_rewards.append(reward)
                t += 1
                if done:
                    break

            writer.add_scalar('Eval/FixedPath Avg Reward', reward, episode/10)
    writer.close()

if __name__ == "__main__":
    main()
