import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from environments.mnm_mvp_torch import MnM_MVP_torchrl
import datetime
import settings
from torchrl.envs.utils import check_env_specs
from tensordict import TensorDict
# --- Hyperparameters ---
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
PPO_EPSILON = 0.2 # Clip parameter
PPO_EPOCHS = 10 # Number of optimization epochs per rollout
PPO_BATCH_SIZE = 64
ROLLOUT_STEPS = 2048 # Number of steps to collect per rollout
MAX_TRAINING_STEPS = 1_000_000
ENTROPY_COEF = 0.01

class ActorCritic(nn.Module):
    def __init__(self, financial_dim, macro_dim, action_dims,debug :bool = False):
        """
        Actor-Critic Network for the MnM_MVP environment.
        Processes two separate inputs (financial and macro) and outputs
        action probabilities (actor) and a state value (critic).

        Args:
            financial_dim (int): Dimension of the financial metrics observation.
            macro_dim (int): Dimension of the macroeconomic data observation.
            action_dims (list[int]): List of numbers of discrete actions for each dimension.
        """
        super(ActorCritic, self).__init__()
        
        self.debug = debug
        # Shared layers for feature extraction
        self.shared_financial_layers = nn.Sequential(
            nn.Linear(financial_dim, 64),
            nn.Tanh(),
        )
        self.shared_macro_layers = nn.Sequential(
            nn.Linear(macro_dim, 64),
            nn.Tanh(),
        )
        
        # Combined feature layer
        self.combined_layers = nn.Sequential(
            nn.Linear(128, 128), # 64 from financial + 64 from macro
            nn.Tanh()
        )

        # Actor Head - outputs action probabilities for each action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(128, dim) for dim in action_dims
        ])

        # Critic Head - outputs a single value for the state
        self.critic_head = nn.Linear(128, 1)

    def forward(self, financial_obs, macro_obs):
        """
        Forward pass through the network.
        """
        if self.debug:
            print(f"Inside forward - financial_obs shape: {financial_obs.shape}")
            print(f"Inside forward - macro_obs shape: {macro_obs.shape}")

        # Process financial and macro observations through their respective shared layers
        fin_features = self.shared_financial_layers(financial_obs)
        mac_features = self.shared_macro_layers(macro_obs)
        
        # Concatenate features from both observation streams
        combined_features = torch.cat([fin_features, mac_features], dim=-1)
        x = self.combined_layers(combined_features)

        # Critic value
        value = self.critic_head(x)

        # Actor action probabilities
        action_logits = [head(x) for head in self.actor_heads]
        
        return action_logits, value

    def act(self, obs, deterministic=False):
        """
        Select an action based on the current policy.
        """
        financial_np_obs = obs["financial_metrics"]
        macro_np_obs = obs["macro_economic_data"]

        if self.debug: # Assuming you have a debug flag or can add one
            print(f"Shape of financial_np_obs from env: {financial_np_obs.shape}")
            print(f"Shape of macro_np_obs from env: {macro_np_obs.shape}")

        if financial_np_obs.ndim == 2 and financial_np_obs.shape[1] == 1:
            financial_np_obs = financial_np_obs.squeeze(1)

        if isinstance(financial_np_obs, torch.Tensor):
            financial_obs = financial_np_obs.detach().clone().float().unsqueeze(0)
        else:
            financial_obs = torch.from_numpy(financial_np_obs).float().unsqueeze(0)

        if isinstance(macro_np_obs, torch.Tensor):
            macro_obs = macro_np_obs.detach().clone().float().unsqueeze(0)
        else:
            macro_obs = torch.from_numpy(macro_np_obs).float().unsqueeze(0)

        if self.debug:
            print(f"Shape of financial_obs (after unsqueeze): {financial_obs.shape}")
            print(f"Shape of macro_obs (after unsqueeze): {macro_obs.shape}")

        action_logits, value = self.forward(financial_obs, macro_obs)

        dists = [Categorical(logits=logits) for logits in action_logits]
        
        if deterministic:
            actions = [dist.probs.argmax(dim=-1) for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]

        log_probs_list = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        log_probs = torch.stack(log_probs_list, dim=0).sum(dim=0)

        # Squeeze the action tensors to remove the batch dimension for the environment
        actions_tensor = torch.stack(actions).squeeze()
        if actions_tensor.ndim == 0:
            actions_tensor = actions_tensor.unsqueeze(0)
        
        action_tensordict = TensorDict({"action": actions_tensor}, batch_size=[])
        
        return action_tensordict, log_probs.detach(), value.detach()


def main():
    # --- Initialization ---
    # env = MnM_MVP(
    #     ticker='KO',
    #     fred_api_key=settings.fred_api_key,
    #     simulation_horizon=12, # 12 quarters = 3 years
    #     simulation_earliest_origin=datetime.datetime(2010, 1, 1),
    #     csv_path="data/KO_quarterly.csv",
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     debug=False
    # )
    env = MnM_MVP_torchrl(
        ticker='KO',
        fred_api_key=settings.fred_api_key,
        simulation_horizon=4, # 12 quarters = 3 years
        simulation_earliest_origin=datetime.datetime(2010, 1, 1),
        csv_path="data/KO_quarterly.csv",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        debug=False
    )

    financial_dim = env.observation_spec["financial_metrics"].shape[0]
    macro_dim = env.observation_spec["macro_economic_data"].shape[0]
    action_dims = [env.action_spec.space.n] * env.action_spec.shape[0]

    policy = ActorCritic(financial_dim, macro_dim, action_dims,debug=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    obs = env.reset()
    total_steps = 0
    
    while total_steps < MAX_TRAINING_STEPS:
        
        # --- Rollout Phase ---
        rollout_buffer = []
        episode_rewards = []
        
        for _ in range(ROLLOUT_STEPS):
            total_steps += 1
            
            action_tensordict, log_prob, value = policy.act(obs)
            step_result = env.step(action_tensordict)

            # Extract values from the TensorDict
            next_obs = {
                "financial_metrics": step_result["next"]["next"]["financial_metrics"].cpu().numpy(),
                "macro_economic_data": step_result["next"]["next"]["macro_economic_data"].cpu().numpy()
            }
            reward = step_result["next"]["reward"].item()
            terminated = step_result["next"]["done"].item()
            truncated = False  # Your environment doesn't use truncation
            info = {}

            rollout_buffer.append((obs, action_tensordict, reward, log_prob, value, terminated))
            
            obs = next_obs
            
            if terminated or truncated:
                episode_rewards.append(info.get("final_reward", reward))
                obs = env.reset()

        if episode_rewards:
            print(f"Step: {total_steps}, Avg Reward: {np.mean(episode_rewards):.2f}")

        # --- GAE and Returns Calculation ---
        # Fix the GAE and Returns Calculation section
        _, _, last_value = policy.act(obs)
        returns_list = []
        advantages = []
        gae = 0

        # Initialize the last return value
        last_return = last_value.clone().detach()

        for step in reversed(rollout_buffer):
            _, _, reward, _, value, done = step
            td_error = reward + GAMMA * last_return * (1 - done) - value
            gae = td_error + GAMMA * GAE_LAMBDA * (1 - done) * gae
            
            # Calculate return for this step
            current_return = reward + GAMMA * last_return * (1 - done)
            
            # Store the return and advantage
            returns_list.insert(0, current_return)
            advantages.insert(0, gae)
            
            # Update for next iteration
            last_return = current_return
            last_value = value

        # Convert to tensors
        advantages = torch.stack(advantages).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.stack(returns_list).detach()

        # --- PPO Update Phase ---
        
        # Extract data from buffer
        old_obs_fin = torch.tensor(np.array([s[0]["financial_metrics"] for s in rollout_buffer]), dtype=torch.float32)
        old_obs_mac = torch.tensor(np.array([s[0]["macro_economic_data"] for s in rollout_buffer]), dtype=torch.float32)
        old_actions = torch.stack([s[1]["action"] for s in rollout_buffer])
        old_log_probs = torch.cat([s[3] for s in rollout_buffer]).detach()
        
        for _ in range(PPO_EPOCHS):
            # Create batches
            indices = np.random.permutation(len(rollout_buffer))
            for i in range(0, len(rollout_buffer), PPO_BATCH_SIZE):
                batch_indices = indices[i:i+PPO_BATCH_SIZE]
                
                # Get new log_probs, values, and entropy
                action_logits, values = policy.forward(old_obs_fin[batch_indices], old_obs_mac[batch_indices])
                values = values.squeeze()
                
                dists = [Categorical(logits=logits) for logits in action_logits]
                
                # Sum log probs across action dimensions
                log_probs = torch.stack([dist.log_prob(act) for dist, act in zip(dists, old_actions[batch_indices].T)]).sum(0)
                entropy = torch.stack([dist.entropy() for dist in dists]).sum(0).mean()

                # Calculate Ratio for PPO objective
                ratio = torch.exp(log_probs - old_log_probs[batch_indices])
                
                # Policy Loss (PPO-Clip)
                surr1 = ratio * advantages[batch_indices]
                surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = nn.functional.mse_loss(values, returns[batch_indices].squeeze())
                
                # Total Loss
                loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
                
                # Gradient step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

if __name__ == '__main__':
    main()