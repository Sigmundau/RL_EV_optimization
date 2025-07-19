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
from environments.mnm_v1c import MnM_v1c
import settings
from tensordict import TensorDict
import torch.nn.init as init

# PPO Hyperparameters
learning_rate = 1e-4
gamma = 1
lmbda = 0.95
eps_clip = 0.2
K_epochs = 4
T_horizon = 256
mini_batch_size = 64  # For LSTM, this is the sequence length
max_grad_norm = 0.5
adam_epsilon = 1e-5
reward_scale = 1.0
lstm_hidden_size = 128

# Running mean/std for state and reward normalization
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4
    def update(self, x):
        x = np.asarray(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

state_rms = None
reward_rms = RunningMeanStd(())

# Environment parameters
ticker = 'KO'
fred_api_key = settings.fred_api_key
terminal_growth_rate = 0.02
credit_rating = 'A'
simulation_horizon = 10
csv_path = "data/KO_quarterly.csv"
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_dir = os.path.join("runs", "2013_v1d_float")
env = MnM_v1c(
    ticker=ticker,
    fred_api_key=fred_api_key,
    terminal_growth_rate=terminal_growth_rate,
    credit_rating=credit_rating,
    fixed_origin=False,
    simulation_earliest_origin = datetime.datetime(2007, 6, 30),
    # fixed_origin=True,
    # simulation_earliest_origin = datetime.datetime(2013, 6, 30),
    simulation_horizon=simulation_horizon,
    csv_path=csv_path,
    fixed_path=False,  # Training env: random path
    debug=debug,
    device=device.type
)

# Fixed-path environment for evaluation
eval_env = MnM_v1c(
    ticker=ticker,
    fred_api_key=fred_api_key,
    terminal_growth_rate=terminal_growth_rate,
    credit_rating=credit_rating,
    fixed_origin=False,
    simulation_horizon=simulation_horizon,
    csv_path=csv_path,
    fixed_path=True,  # Evaluation env: fixed path
    fixed_path_csv='data/20131231_real_path.csv',
    debug=debug,
    device=device.type
)

obs = env.reset()
obs_financial_dim = len(obs["financial_metrics"])
obs_macro_dim = len(obs["macro_economic_data"])
obs_dim = obs_financial_dim + obs_macro_dim

action_spec = env.action_spec
if hasattr(action_spec, 'space') and hasattr(action_spec.space, 'n'):
    act_dim = action_spec.space.n
else:
    raise ValueError("Could not determine number of categories (n) from action_spec.space.")
if hasattr(action_spec, 'shape') and action_spec.shape is not None:
    action_shape = action_spec.shape[0]
else:
    raise ValueError("Could not determine action shape from action_spec.shape.")

def orthogonal_init(module):
    if isinstance(module, (nn.Linear, nn.LSTM)):
        if isinstance(module, nn.Linear):
            init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    init.orthogonal_(param)

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_dim, act_dim, action_shape, lstm_hidden_size):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_shape = action_shape
        self.lstm_hidden_size = lstm_hidden_size

        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(256, lstm_hidden_size, batch_first=False)
        self.actor_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim * action_shape)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, x, hidden):
        # x shape: (seq_len, batch_size, obs_dim)
        features = self.shared_fc(x)
        lstm_out, hidden = self.lstm(features, hidden)
        logits = self.actor_head(lstm_out).view(lstm_out.shape[0], lstm_out.shape[1], self.action_shape, self.act_dim)
        value = self.critic_head(lstm_out)
        return logits, value, hidden

    def act(self, x, hidden):
        # x shape: (1, 1, obs_dim)
        logits, _, hidden = self.forward(x, hidden)
        logits = logits.squeeze(0)
        dists = [torch.distributions.Categorical(logits=logits[:, i, :]) for i in range(self.action_shape)]
        actions = torch.stack([dist.sample() for dist in dists], dim=1)
        log_probs = torch.stack([dist.log_prob(actions[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        return actions.squeeze(0), log_probs.squeeze(0), hidden

    def evaluate(self, x, actions, hidden):
        # x shape: (seq_len, batch_size, obs_dim)
        # actions shape: (seq_len, batch_size, action_shape)
        logits, value, _ = self.forward(x, hidden)
        logits_flat = logits.view(-1, self.action_shape, self.act_dim)
        actions_flat = actions.view(-1, self.action_shape)
        dists = [torch.distributions.Categorical(logits=logits_flat[:, i, :]) for i in range(self.action_shape)]
        log_probs = torch.stack([dist.log_prob(actions_flat[:, i]) for i, dist in enumerate(dists)], dim=1).sum(dim=1)
        entropy = torch.stack([dist.entropy() for dist in dists], dim=1).sum(dim=1)
        log_probs = log_probs.view(x.shape[0], x.shape[1])
        entropy = entropy.view(x.shape[0], x.shape[1])
        value = value.squeeze(-1)
        return log_probs, entropy, value

model = ActorCriticLSTM(obs_dim, act_dim, action_shape, lstm_hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=adam_epsilon)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / 1000)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'log_prob', 'done', 'hidden'])

def collect_trajectories(env, model, horizon):
    state = env.reset()
    memory = []
    knockout_count = 0
    expected_macro_dim = len(state["macro_economic_data"])
    hidden = (torch.zeros(1, 1, model.lstm_hidden_size, device=device),
              torch.zeros(1, 1, model.lstm_hidden_size, device=device))
    for _ in range(horizon):
        state_fin = torch.as_tensor(state["financial_metrics"], dtype=torch.float32, device=device).flatten()
        state_mac_raw = state["macro_economic_data"]
        state_mac = torch.as_tensor(state_mac_raw, dtype=torch.float32, device=device).flatten()
        if state_mac.shape[0] < expected_macro_dim:
            pad_len = expected_macro_dim - state_mac.shape[0]
            state_mac = torch.cat([state_mac, torch.zeros(pad_len, dtype=state_mac.dtype, device=state_mac.device)])
        elif state_mac.shape[0] > expected_macro_dim:
            state_mac = state_mac[:expected_macro_dim]
        state_np = torch.cat([state_fin, state_mac]).cpu().numpy()
        global state_rms
        if state_rms is not None:
            state_rms.update([state_np])
            state_np_norm = state_rms.normalize(state_np)
        else:
            state_np_norm = state_np
        state_tensor = torch.as_tensor(state_np_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        current_hidden = hidden
        action, log_prob, hidden = model.act(state_tensor, hidden)
        action_np = action.detach().cpu().numpy().squeeze()
        action_tensor = torch.as_tensor(action_np, dtype=torch.long, device=device)
        action_td = TensorDict({"action": action_tensor}, batch_size=[], device=device)
        step_td = env.step(action_td)
        td = step_td
        reward = td.get('next').get('reward').item()
        done = td.get('next').get('done').item()
        knockout = td.get('next').get('knockout').item() if 'knockout' in td.get('next').keys() else False
        if knockout:
            knockout_count += 1
        next_state = td.get('next').get('next')
        memory.append(Transition(state, action_np, reward, next_state, log_prob.item(), done, current_hidden))
        state = next_state
        if done:
            state = env.reset()
            hidden = (torch.zeros(1, 1, model.lstm_hidden_size, device=device),
                      torch.zeros(1, 1, model.lstm_hidden_size, device=device))
    return memory, knockout_count

def compute_gae(rewards, values, dones, gamma=0.99, lmbda=0.95):
    advantages = []
    gae = 0
    next_values = torch.cat([values[1:], torch.tensor([0.0], device=values.device)])
    for i in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_values[i] * nextnonterminal - values[i]
        gae = delta + gamma * lmbda * nextnonterminal * gae
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=values.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

def ppo_update(memory, model, optimizer):
    expected_macro_dim = len(memory[0].state["macro_economic_data"]) if hasattr(memory[0].state["macro_economic_data"], '__len__') else 1
    def to_1d_np(arr):
        arr = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else arr
        return np.atleast_1d(arr)
    def process_state(state):
        fin = to_1d_np(state["financial_metrics"])
        mac_raw = state["macro_economic_data"]
        mac = to_1d_np(mac_raw)
        if mac.shape[0] < expected_macro_dim:
            mac = np.concatenate([mac, np.zeros(expected_macro_dim - mac.shape[0], dtype=mac.dtype)])
        elif mac.shape[0] > expected_macro_dim:
            mac = mac[:expected_macro_dim]
        state_vec = np.concatenate([fin, mac])
        global state_rms
        if state_rms is not None:
            state_vec = state_rms.normalize(state_vec)
        return state_vec
    rewards_np = np.array([t.reward for t in memory], dtype=np.float32)
    global reward_rms
    reward_rms.update(rewards_np)
    rewards_np = reward_rms.normalize(rewards_np) * reward_scale
    states_np = np.array([process_state(t.state) for t in memory])
    states = torch.from_numpy(states_np).float().to(device)
    with torch.no_grad():
        values = []
        start_idx = 0
        for i in range(len(memory)):
            if memory[i].done or i == len(memory) - 1:
                end_idx = i + 1
                sub_traj_states = states[start_idx:end_idx].unsqueeze(1)
                initial_hidden = memory[start_idx].hidden
                _, sub_traj_values, _ = model.forward(sub_traj_states, initial_hidden)
                values.extend(sub_traj_values.squeeze().tolist())
                start_idx = end_idx
    values = torch.tensor(values, dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in memory], dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    advantages = compute_gae(rewards, values, dones, gamma, lmbda)
    returns = advantages + values
    old_log_probs = torch.tensor([t.log_prob for t in memory], dtype=torch.float32, device=device)
    actions_np = np.array([t.action for t in memory])
    actions = torch.from_numpy(actions_np).long().to(device)
    global writer, global_step
    for _ in range(K_epochs):
        for i in range(0, len(memory), mini_batch_size):
            end = i + mini_batch_size
            if end > len(memory): continue
            idx = slice(i, end)
            h0, c0 = memory[i].hidden
            initial_hidden = (h0.detach(), c0.detach())
            states_seq = states[idx].unsqueeze(1)
            actions_seq = actions[idx].unsqueeze(1)
            log_probs, entropy, value = model.evaluate(states_seq, actions_seq, initial_hidden)
            log_probs, entropy, value = log_probs.squeeze(1), entropy.squeeze(1), value.squeeze(1)
            ratio = torch.exp(log_probs - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(value, returns[idx])
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if writer is not None:
                writer.add_scalar('Loss/actor_loss', actor_loss.item(), global_step)
                writer.add_scalar('Loss/critic_loss', critic_loss.item(), global_step)
                writer.add_scalar('Loss/total_loss', loss.item(), global_step)
            global_step += 1
    lr_scheduler.step()

def main(log_to_tensorboard=True):
    if log_to_tensorboard:
        # log_dir = os.path.join("runs", f"mnm_ppo_v1e_lstm_float_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        global writer, global_step
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    else:
        writer = None
    global_step = 0
    test_step = 0
    reward_window = deque(maxlen=100)
    global state_rms
    state_rms = RunningMeanStd(obs_dim)
    for episode in range(1000):
        memory, knockout_count = collect_trajectories(env, model, T_horizon)
        episode_reward = sum(t.reward for t in memory) / len(memory)
        reward_window.append(episode_reward)
        knockout_pct = knockout_count / len(memory) if len(memory) > 0 else 0.0
        if writer is not None:
            writer.add_scalar('Reward/Episodic Avg Reward', episode_reward, episode)
            writer.add_scalar('Env/Knockout Percentage', knockout_pct, episode)
            writer.flush()
        ppo_update(memory, model, optimizer)
        if episode % 10 == 0 and len(reward_window) > 0:
            print(f"Episode {episode} completed. Episodic reward: {episode_reward:.2f} | Knockout %: {knockout_pct:.2%}")
            eval_fixed_path(model, eval_env, simulation_horizon, device, test_step, writer)
            test_step += 1
    if writer is not None:
        writer.close()

def eval_fixed_path(model, eval_env, simulation_horizon, device, test_step, writer=None):
    eval_rewards = []
    eval_knockout_count = 0
    state = eval_env.reset()
    expected_macro_dim = len(state["macro_economic_data"])
    hidden = (torch.zeros(1, 1, model.lstm_hidden_size, device=device),
              torch.zeros(1, 1, model.lstm_hidden_size, device=device))
    for step in range(simulation_horizon):
        state_fin = torch.as_tensor(state["financial_metrics"], dtype=torch.float32, device=device).flatten()
        state_mac_raw = state["macro_economic_data"]
        state_mac = torch.as_tensor(state_mac_raw, dtype=torch.float32, device=device).flatten()
        if state_mac.shape[0] < expected_macro_dim:
            pad_len = expected_macro_dim - state_mac.shape[0]
            state_mac = torch.cat([state_mac, torch.zeros(pad_len, dtype=state_mac.dtype, device=state_mac.device)])
        elif state_mac.shape[0] > expected_macro_dim:
            state_mac = state_mac[:expected_macro_dim]
        state_np = torch.cat([state_fin, state_mac]).cpu().numpy()
        state_np_norm = state_rms.normalize(state_np) if state_rms is not None else state_np
        state_tensor = torch.as_tensor(state_np_norm, dtype=torch.float32, device=device).view(1, 1, -1)
        action, _, hidden = model.act(state_tensor, hidden)
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
    main(log_to_tensorboard=True)