import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.envs import GymEnv, TransformedEnv, ObservationNorm, default_info_dict_reader
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import TensorDictModule
from environments.naive_debt_control_env import EnvParams

import gymnasium as gym

# Step 0: Define the environment
gym.register(
    id="NaiveDebtControl-v0",
    entry_point="environments.naive_debt_control_env:NaiveDebtControlEnv",
    max_episode_steps=EnvParams().ep_len,
)

def make_env(device):
    env = GymEnv("NaiveDebtControl-v0",
                 categorical_action_encoding=True,
                 device=device)
    reader = default_info_dict_reader(["ev", "ebit"])
    env = env.set_info_dict_reader(info_dict_reader=reader)
    env = TransformedEnv(env, ObservationNorm(in_keys=["observation"]))
    env.transform.init_stats(num_iter=1000)
    return env

# Step 1: Define Policy and Value Network
class SharedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class PolicyHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(x)

class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(x)

# Step 2: Set up environment and network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env(device=device)
state_dim = env.observation_space.shape[0]  # 2 (leverage, EBIT)
action_dim = env.action_space.n  # 3 (–5 pp, 0, +5 pp)
hidden_dim = 64

shared_net = SharedNetwork(state_dim, hidden_dim).to(device)
policy_head = PolicyHead(hidden_dim, action_dim).to(device)
value_head = ValueHead(hidden_dim).to(device)

policy_module = TensorDictModule(
    module=nn.Sequential(shared_net, policy_head),
    in_keys=["observation"],
    out_keys=["logits"],
)
value_module = TensorDictModule(
    module=nn.Sequential(shared_net, value_head),
    in_keys=["observation"],
    out_keys=["state_value"],
)

policy = ProbabilisticActor(
    module=policy_module,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True
)
value_net = ValueOperator(
    module=value_module,
    in_keys=["observation"]
)

# Step 3; Set up PPO components
optimizer = optim.Adam(
    list(shared_net.parameters()) + list(policy_head.parameters()) + list(value_head.parameters()),
    lr=3e-4
)
loss_module = ClipPPOLoss(
    actor=policy,
    critic=value_net,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
)

collector = SyncDataCollector(
    create_env_fn=lambda: make_env(device=device),
    policy=policy,
    frames_per_batch=2048,
    total_frames=204800,
    split_trajs=False,
    device=device,
)

# Step 4: Training loop
print(f"Starting training with {device}...")
for i, batch in enumerate(collector):
    # Compute PPO loss
    loss_vals = loss_module(batch)
    loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

    # Backprop and update
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(shared_net.parameters()) + list(policy_head.parameters()) + list(value_head.parameters()),
        max_norm=1.0
    )
    optimizer.step()

    # Log progress
    if (i + 1) % 10 == 0:
        avg_reward = batch["next", "reward"].mean().item()
        avg_ev = batch["ev"].mean().item() if "ev" in batch else 0.0
        avg_ebit = batch["ebit"].mean().item() if "ebit" in batch else 0.0
        print(f"Batch {i + 1}, Average Reward: {avg_reward:.2f}, Average EV: {avg_ev:.2f}, Average EBIT: {avg_ebit:.2f}")

collector.shutdown()
print("Training completed!")