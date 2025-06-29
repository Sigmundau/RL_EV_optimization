import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
from collections import deque
import numpy as np
from collections import namedtuple
from environments.mnm_v1 import MnM_v1
import settings
from tensordict import TensorDict
# PPO Hyperparameters
learning_rate = 3e-4
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epochs = 4
T_horizon = 2048
mini_batch_size = 64

# Environment parameters (customize as needed)
ticker = 'KO'
fred_api_key = settings.fred_api_key  # Ensure this is set in settings.py
terminal_growth_rate = 0.02
credit_rating = 'AAA'
simulation_horizon = 10
csv_path = "data/KO_quarterly.csv"
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MnM_v1(
    ticker=ticker,
    fred_api_key=fred_api_key,
    terminal_growth_rate=terminal_growth_rate,
    credit_rating=credit_rating,
    simulation_horizon=simulation_horizon,
    csv_path=csv_path,
    debug=debug,
    device=device.type
)

# Retrieve observation and action shapes from env.reset() and action_spec
obs = env.reset()
obs_financial_dim = obs["financial_metrics"].shape[0]
obs_macro_dim = obs["macro_economic_data"].shape[0]
obs_dim = obs_financial_dim + obs_macro_dim

# Get action spec info directly from environment using .specs
action_spec = env.action_spec
if hasattr(action_spec, 'space') and hasattr(action_spec.space, 'n'):
    act_dim = action_spec.space.n
else:
    raise ValueError("Could not determine number of categories (n) from action_spec.space.")
if hasattr(action_spec, 'shape') and action_spec.shape is not None:
    action_shape = action_spec.shape[0]
else:
    raise ValueError("Could not determine action shape from action_spec.shape.")


# Multi-categorical actor-critic for 2D categorical action
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, action_shape):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_shape = action_shape
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim * action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        logits = self.actor(x).view(-1, self.action_shape, self.act_dim)
        value = self.critic(x)
        return logits, value

    def act(self, x):
        logits, _ = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        return actions.squeeze(0), log_probs.squeeze(0)

    def evaluate(self, x, actions):
        logits, value = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
        return log_probs, entropy, value

model = ActorCritic(obs_dim, act_dim, action_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'log_prob', 'done'])

def collect_trajectories(env, model, horizon):
    state = env.reset()
    memory = []
    for _ in range(horizon):
        # Flatten observation dict to vector
        state_vec = np.concatenate([state["financial_metrics"], state["macro_economic_data"]])
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
        action, log_prob = model.act(state_tensor)
        action_np = action.cpu().numpy().squeeze()
        # Prepare action as TensorDict for TorchRL
        action_tensor = torch.tensor(action_np, dtype=torch.long, device=device)
        action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
        step_td = env.step(action_td)

        # --- Robust extraction of reward, done, next_state from nested 'next' dicts ---
        td = step_td
        # descend into 'next' as far as possible
        while "next" in td and isinstance(td["next"], (dict, TensorDict)):
            td = td["next"]

        # Try to extract reward and done
        reward = td["reward"].item() if "reward" in td else (step_td["reward"].item() if "reward" in step_td else 0.0)
        done = td["done"].item() if "done" in td else (step_td["done"].item() if "done" in step_td else True)

        # Try to extract next_state
        if "financial_metrics" in td and "macro_economic_data" in td:
            next_state = {
                "financial_metrics": td["financial_metrics"].cpu().numpy().squeeze(),
                "macro_economic_data": td["macro_economic_data"].cpu().numpy().squeeze()
            }
        else:
            # If still not found, reset env (likely episode ended)
            next_state = env.reset()

        memory.append(Transition(state, action_np, reward, next_state, log_prob.item(), done))
        state = next_state
        if done:
            state = env.reset()
    return memory

def ppo_update(memory, model, optimizer):
    # Convert memory to tensors
    states_np = np.array([
        np.concatenate([t.state["financial_metrics"], t.state["macro_economic_data"]]) for t in memory
    ])
    states = torch.from_numpy(states_np).float().to(device)
    actions_np = np.array([t.action for t in memory])
    actions = torch.from_numpy(actions_np).long().to(device)
    rewards = [t.reward for t in memory]
    dones = [t.done for t in memory]
    old_log_probs = torch.tensor([t.log_prob for t in memory], dtype=torch.float32, device=device)

    # Compute returns and advantages
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, values = model(states)
        values = values.squeeze()
    advantages = returns - values

    # PPO update
    global writer, global_step
    for _ in range(K_epochs):
        for i in range(0, len(memory), mini_batch_size):
            idx = slice(i, i + mini_batch_size)
            log_probs, entropy, value = model.evaluate(states[idx], actions[idx])
            ratio = torch.exp(log_probs - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value.squeeze(), returns[idx])
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TensorBoard logging for losses
            writer.add_scalar('Loss/actor_loss', actor_loss.item(), global_step)
            writer.add_scalar('Loss/critic_loss', critic_loss.item(), global_step)
            writer.add_scalar('Loss/total_loss', loss.item(), global_step)
            global_step += 1

def main():
    # TensorBoard setup
    log_dir = os.path.join("runs", f"mnm_ppo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    global writer, global_step
    writer = SummaryWriter(log_dir)
    global_step = 0
    print(f"TensorBoard logs will be saved to: {log_dir}")
    reward_window = deque(maxlen=100)
    for episode in range(1000):
        memory = collect_trajectories(env, model, T_horizon)
        # Compute episode rewards for this batch
        episode_rewards = [t.reward for t in memory if t.done]
        if episode_rewards:
            reward_window.extend(episode_rewards)
        if episode % 10 == 0 and len(reward_window) > 0:
            avg_reward = sum(reward_window) / len(reward_window)
            print(f"Episode {episode} completed. Rolling avg reward (last 100): {avg_reward:.2f}")
            writer.add_scalar('Reward/rolling_avg_100', avg_reward, episode)
        ppo_update(memory, model, optimizer)
    writer.close()

if __name__ == "__main__":
    main()
