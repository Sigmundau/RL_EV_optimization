import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from collections import deque
import datetime
from environments.mnm_v1c import MnM_v1c
import settings
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

# Environment parameters (reuse from main code)
ticker = 'KO'
fred_api_key = settings.fred_api_key
terminal_growth_rate = 0.02
credit_rating = 'A'
simulation_horizon = 10
csv_path = "data/KO_quarterly.csv"
debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

obs = eval_env.reset()
obs_financial_dim = len(obs["financial_metrics"])
obs_macro_dim = len(obs["macro_economic_data"])
obs_dim = obs_financial_dim + obs_macro_dim

action_spec = eval_env.action_spec
if hasattr(action_spec, 'space') and hasattr(action_spec.space, 'n'):
    act_dim = action_spec.space.n
else:
    raise ValueError("Could not determine number of categories (n) from action_spec.space.")
if hasattr(action_spec, 'shape') and action_spec.shape is not None:
    action_shape = action_spec.shape[0]
else:
    raise ValueError("Could not determine action shape from action_spec.shape.")

# Dummy state_rms for normalization (no normalization for random agent)
state_rms = None

def eval_fixed_path_random(eval_env, simulation_horizon, device, writer=None, test_step=0):
    eval_rewards = []
    eval_knockout_count = 0
    state = eval_env.reset()
    expected_macro_dim = len(state["macro_economic_data"])
    for step in range(simulation_horizon):
        # Random action
        action_np = np.random.randint(0, act_dim, size=(action_shape,))
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
    print(f"Random Agent - Avg Reward: {avg_eval_reward:.2f} | Knockout %: {eval_knockout_pct:.2%}")
    if writer is not None:
        writer.add_scalar('Eval/FixedPath Avg Reward', avg_eval_reward, test_step)
        writer.add_scalar('Eval/FixedPath Knockout Percentage', eval_knockout_pct, test_step)
        writer.flush()

if __name__ == "__main__":
    log_dir = os.path.join("runs", f"2013_random")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    for i in range(100):  # Run multiple evaluations
        print(f"Evaluation {i + 1}/100")
        eval_fixed_path_random(eval_env, simulation_horizon, device, writer=writer, test_step=i)