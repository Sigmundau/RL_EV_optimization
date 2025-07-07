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
from environments.mnm_v1b import MnM_v1b
import settings
from tensordict import TensorDict
# PPO Hyperparameters
learning_rate = 1e-4
gamma = 1
lmbda = 0.95
eps_clip = 0.2
K_epochs = 4
T_horizon = 256
mini_batch_size = 64

# Environment parameters (customize as needed)
ticker = 'KO'
fred_api_key = settings.fred_api_key  # Ensure this is set in settings.py
terminal_growth_rate = 0.02
credit_rating = 'A'
simulation_horizon = 10
csv_path = "data/KO_quarterly.csv"
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MnM_v1b(
    ticker=ticker,
    fred_api_key=fred_api_key,
    terminal_growth_rate=terminal_growth_rate,
    credit_rating=credit_rating,
    fixed_origin=False,
    simulation_horizon=simulation_horizon,
    csv_path=csv_path,
    fixed_path=False,  # Training env: random path
    debug=debug,
    device=device.type
)

# Fixed-path environment for evaluation
eval_env = MnM_v1b(
    ticker=ticker,
    fred_api_key=fred_api_key,
    terminal_growth_rate=terminal_growth_rate,
    credit_rating=credit_rating,
    fixed_origin=False,
    simulation_horizon=simulation_horizon,
    csv_path=csv_path,
    fixed_path=True,  # Evaluation env: fixed path
    fixed_path_csv='data/simulated_path_10step.csv',
    debug=debug,
    device=device.type
)


# Retrieve observation and action shapes from env.reset() and action_spec
obs = env.reset()
obs_financial_dim = len(obs["financial_metrics"])
obs_macro_dim = len(obs["macro_economic_data"])
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
    """
    Standard multi-categorical actor-critic network for PPO.
    The actor outputs logits for each action dimension; the critic outputs a state value.
    """
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
        """
        Forward pass for the actor-critic network.
        Args:
            x (torch.Tensor): Input observation tensor.
        Returns:
            logits (torch.Tensor): Action logits for each action dimension.
            value (torch.Tensor): State value estimate.
        """
        logits = self.actor(x).view(-1, self.action_shape, self.act_dim)
        value = self.critic(x)
        return logits, value

    def act(self, x):
        """
        Samples actions and computes log probabilities for the given observation.
        Args:
            x (torch.Tensor): Input observation tensor.
        Returns:
            actions (torch.Tensor): Sampled actions.
            log_probs (torch.Tensor): Log probabilities of the sampled actions.
        """
        logits, _ = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        return actions.squeeze(0), log_probs.squeeze(0)

    def evaluate(self, x, actions):
        """
        Evaluates the log probabilities, entropy, and value for given states and actions.
        Args:
            x (torch.Tensor): Input observation tensor.
            actions (torch.Tensor): Actions to evaluate.
        Returns:
            log_probs (torch.Tensor): Log probabilities of the actions.
            entropy (torch.Tensor): Entropy of the action distributions.
            value (torch.Tensor): State value estimate.
        """
        logits, value = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
        return log_probs, entropy, value

class ImprovedActorCritic(nn.Module):
    """
    Improved multi-categorical actor-critic network with a shared feature extractor,
    separate actor and critic heads, and dropout for regularization.
    """
    def __init__(self, obs_dim, act_dim, action_shape):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_shape = action_shape
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        # Separate actor and critic heads
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim * action_shape)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Forward pass for the improved actor-critic network.
        Args:
            x (torch.Tensor): Input observation tensor.
        Returns:
            logits (torch.Tensor): Action logits for each action dimension.
            value (torch.Tensor): State value estimate.
        """
        shared_features = self.shared(x)
        logits = self.actor(shared_features).view(-1, self.action_shape, self.act_dim)
        value = self.critic(shared_features)
        return logits, value

    def act(self, x):
        """
        Samples actions and computes log probabilities for the given observation.
        Args:
            x (torch.Tensor): Input observation tensor.
        Returns:
            actions (torch.Tensor): Sampled actions.
            log_probs (torch.Tensor): Log probabilities of the sampled actions.
        """
        logits, _ = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        return actions.squeeze(0), log_probs.squeeze(0)

    def evaluate(self, x, actions):
        """
        Evaluates the log probabilities, entropy, and value for given states and actions.
        Args:
            x (torch.Tensor): Input observation tensor.
            actions (torch.Tensor): Actions to evaluate.
        Returns:
            log_probs (torch.Tensor): Log probabilities of the actions.
            entropy (torch.Tensor): Entropy of the action distributions.
            value (torch.Tensor): State value estimate.
        """
        logits, value = self.forward(x)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
        return log_probs, entropy, value

# model = ActorCritic(obs_dim, act_dim, action_shape).to(device)
model = ImprovedActorCritic(obs_dim, act_dim, action_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'log_prob', 'done'])

def collect_trajectories(env, model, horizon):
    """
    Collects a batch of trajectories (state, action, reward, etc.) from the environment using the current policy.

    Args:
        env: The environment to interact with.
        model: The policy/value network.
        horizon: Number of steps to collect.

    Returns:
        memory: List of Transition namedtuples.
        knockout_count: Number of knockout events in the batch.
    """
    state = env.reset()
    memory = []
    knockout_count = 0
    # Determine expected macro_economic_data length from initial obs
    expected_macro_dim = len(state["macro_economic_data"])
    for _ in range(horizon):
        # Convert observation dict to torch tensor on GPU, always flatten to 1D
        state_fin = torch.as_tensor(state["financial_metrics"], dtype=torch.float32, device=device).flatten()
        state_mac_raw = state["macro_economic_data"]
        state_mac = torch.as_tensor(state_mac_raw, dtype=torch.float32, device=device).flatten()
        # Pad macro_economic_data if needed
        if state_mac.shape[0] < expected_macro_dim:
            pad_len = expected_macro_dim - state_mac.shape[0]
            state_mac = torch.cat([state_mac, torch.zeros(pad_len, dtype=state_mac.dtype, device=state_mac.device)])
        elif state_mac.shape[0] > expected_macro_dim:
            state_mac = state_mac[:expected_macro_dim]
        state_tensor = torch.cat([state_fin, state_mac]).unsqueeze(0)  # shape: (1, obs_dim)
        action, log_prob = model.act(state_tensor)
        action_np = action.detach().cpu().numpy().squeeze()
        # Prepare action as TensorDict for TorchRL
        action_tensor = torch.as_tensor(action_np, dtype=torch.long, device=device)
        action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
        step_td = env.step(action_td)

        # --- Robust extraction of reward, done, next_state, knockout from nested 'next' dicts ---
        td = step_td
        reward = td.get('next').get('reward').item()
        done = td.get('next').get('done').item()
        knockout = td.get('next').get('knockout').item() if 'knockout' in td.get('next').keys() else False
        if knockout:
            knockout_count += 1
        next_state = td.get('next').get('next')
        memory.append(Transition(state, action_np, reward, next_state, log_prob.item(), done))
        state = next_state
        if done:
            state = env.reset()
    return memory, knockout_count

def compute_gae(rewards, values, dones, gamma=0.99, lmbda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) for a batch of rewards and values.

    Args:
        rewards: Tensor of rewards.
        values: Tensor of value estimates.
        dones: Tensor of done flags.
        gamma: Discount factor.
        lmbda: GAE lambda parameter.

    Returns:
        Tensor of advantage estimates.
    """
    """Compute GAE advantages"""
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[i]
            nextvalues = 0
        else:
            nextnonterminal = 1.0 - dones[i+1]
            nextvalues = values[i+1]
        delta = rewards[i] + gamma * nextvalues * nextnonterminal - values[i]
        gae = delta + gamma * lmbda * nextnonterminal * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32, device=values.device)

def ppo_update(memory, model, optimizer):
    """
    Performs a PPO update step using the collected batch of transitions.

    Args:
        memory: List of Transition namedtuples.
        model: The policy/value network.
        optimizer: The optimizer for updating model parameters.
    """
    # Convert memory to tensors (all on GPU)
    # Determine expected macro_economic_data length from first transition
    expected_macro_dim = len(memory[0].state["macro_economic_data"]) if hasattr(memory[0].state["macro_economic_data"], '__len__') else 1
    def to_1d_np(arr):
        arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
        arr = np.atleast_1d(arr)
        return arr
    states_np = []
    for t in memory:
        fin = to_1d_np(t.state["financial_metrics"])
        mac_raw = t.state["macro_economic_data"]
        mac = to_1d_np(mac_raw)
        # Pad macro if needed
        if mac.shape[0] < expected_macro_dim:
            mac = np.concatenate([mac, np.zeros(expected_macro_dim - mac.shape[0], dtype=mac.dtype)])
        elif mac.shape[0] > expected_macro_dim:
            mac = mac[:expected_macro_dim]
        states_np.append(np.concatenate([fin, mac]))
    states_np = np.array(states_np)
    states = torch.from_numpy(states_np).float().to(device)
    actions_np = np.array([t.action for t in memory])
    actions = torch.from_numpy(actions_np).long().to(device)
    rewards = torch.tensor([t.reward for t in memory], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in memory], dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([t.log_prob for t in memory], dtype=torch.float32, device=device)

    # Compute returns and advantages
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards.cpu().numpy()), reversed(dones.cpu().numpy())):
        if done:
            G = 0
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, values = model(states)
        values = values.squeeze()
    advantages = compute_gae(rewards, values, dones, gamma, lmbda)
    returns = advantages + values

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
            if writer is not None:
                writer.add_scalar('Loss/actor_loss', actor_loss.item(), global_step)
                writer.add_scalar('Loss/critic_loss', critic_loss.item(), global_step)
                writer.add_scalar('Loss/total_loss', loss.item(), global_step)
            global_step += 1


def main(log_to_tensorboard=True):
    """
    Main training loop for PPO agent.
    Trains on the random-path environment and periodically evaluates on the fixed-path environment.

    Args:
        log_to_tensorboard (bool): If True, logs metrics to TensorBoard.
    """
    # TensorBoard setup
    if log_to_tensorboard:
        log_dir = os.path.join("runs", f"mnm_ppo_v1b_float_10step_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        global writer, global_step
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    else:
        writer = None
    global_step = 0
    test_step = 0
    reward_window = deque(maxlen=100)
    # --- Training loop ---
    for episode in range(1000):
        memory, knockout_count = collect_trajectories(env, model, T_horizon)
        episode_reward = sum(t.reward for t in memory) / len(memory)
        reward_window.append(episode_reward)
        knockout_pct = knockout_count / len(memory) if len(memory) > 0 else 0.0
        writer.add_scalar('Reward/Episodic Avg Reward', episode_reward, episode)
        writer.add_scalar('Env/Knockout Percentage', knockout_pct, episode)
        writer.flush()
        ppo_update(memory, model, optimizer)
        if episode % 10 == 0 and len(reward_window) > 0:
            print(f"Episode {episode} completed. Episodic reward: {episode_reward:.2f} | Knockout %: {knockout_pct:.2%}")
            # --- Evaluation on fixed-path environment ---
            eval_fixed_path(model, eval_env, simulation_horizon, device, test_step,writer)
            test_step += 1
    writer.close()

def eval_fixed_path(model, eval_env, simulation_horizon, device, test_step, writer=None):
    """
    Run 1 full episode on the fixed-path environment and log results to TensorBoard if writer is provided.

    Args:
        model: The policy/value network.
        eval_env: The fixed-path evaluation environment.
        simulation_horizon: Number of steps to run in the episode.
        device: Torch device.
        test_step: Current evaluation step (for TensorBoard logging).
        writer: TensorBoard SummaryWriter (optional).

    Returns:
        None
    """
    """
    Run 1 full episode on the fixed-path environment and log results to TensorBoard if writer is provided.
    """
    eval_rewards = []
    eval_knockout_count = 0
    state = eval_env.reset()
    expected_macro_dim = len(state["macro_economic_data"])
    for step in range(simulation_horizon):
        state_fin = torch.as_tensor(state["financial_metrics"], dtype=torch.float32, device=device).flatten()
        state_mac_raw = state["macro_economic_data"]
        state_mac = torch.as_tensor(state_mac_raw, dtype=torch.float32, device=device).flatten()
        if state_mac.shape[0] < expected_macro_dim:
            pad_len = expected_macro_dim - state_mac.shape[0]
            state_mac = torch.cat([state_mac, torch.zeros(pad_len, dtype=state_mac.dtype, device=state_mac.device)])
        elif state_mac.shape[0] > expected_macro_dim:
            state_mac = state_mac[:expected_macro_dim]
        state_tensor = torch.cat([state_fin, state_mac]).unsqueeze(0)
        action, log_prob = model.act(state_tensor)
        action_np = action.detach().cpu().numpy().squeeze()
        action_tensor = torch.as_tensor(action_np, dtype=torch.long, device=device)
        action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
        step_td = eval_env.step(action_td)
        td = step_td
        reward = td.get('next').get('reward').item()
        done = td.get('next').get('done').item()
        knockout = td.get('next').get('knockout').item() if 'knockout' in td.get('next').keys() else False
        if knockout:
            eval_knockout_count += 1
        eval_rewards.append(reward)
        state = td.get('next').get('next')
        if done:
            break
    avg_eval_reward = sum(eval_rewards) / len(eval_rewards) if len(eval_rewards) > 0 else 0.0
    eval_knockout_pct = eval_knockout_count / len(eval_rewards) if len(eval_rewards) > 0 else 0.0
    if writer is not None:
        writer.add_scalar('Eval/FixedPath Avg Reward', avg_eval_reward, test_step)
        writer.add_scalar('Eval/FixedPath Knockout Percentage', eval_knockout_pct, test_step)
        writer.flush()
    return



if __name__ == "__main__":
    # Set log_to_tensorboard=False to disable logging
    main(log_to_tensorboard=True)
