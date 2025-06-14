"""
Debt-Control Phase-0 MVP environment
-----------------------------------
• Quarterly time-step
• State  : [L, E]  (leverage %, EBIT)
• Actions: {-5 pp, 0, +5 pp} leverage move
• Reward : ΔEV – λ·default_penalty – φ·recap_cost
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass


# --------------------------------------------------
# Environment parameters
# --------------------------------------------------
@dataclass
class EnvParams:
    mu_E: float   = 100.0   # mean quarterly EBIT
    sigma_E: float = 15.0   # EBIT volatility
    m_u: float    = 8.0     # un-levered EV/EBIT multiple
    tax_rate: float = 0.25  # statutory tax rate
    # Hard-coded convenience constants
    l_max: float  = 0.80    # leverage ceiling
    ep_len: int   = 40      # 10 years @ quarterly


# --------------------------------------------------
# Gym environment
# --------------------------------------------------
class NaiveDebtControlEnv(gym.Env):
    """
    A naive implementation of a debt-control environment for reinforcement learning.

    This environment models quarterly debt control decisions based on leverage and EBIT (Earnings Before Interest and Taxes).
    The agent can take discrete actions to adjust leverage and observe the resulting changes in enterprise value (EV).

    Key Features:
    - **State**: `[L, E]` where `L` is leverage (percentage) and `E` is EBIT.
    - **Actions**: Discrete choices to adjust leverage by {-5 pp, 0, +5 pp}.
    - **Reward**: Change in enterprise value (`ΔEV`).
    - **Termination**: Episode ends if EBIT becomes negative or the maximum episode length is reached.

    Parameters:
    - `p` (EnvParams): Environment parameters including EBIT mean/volatility, leverage ceiling, and episode length.
    - `seed` (int | None): Random seed for reproducibility.

    Attributes:
    - `action_space` (gymnasium.spaces.Discrete): Discrete action space for leverage adjustments.
    - `observation_space` (gymnasium.spaces.Box): Continuous observation space for leverage and EBIT.
    - `np_random` (numpy.random.Generator): Random number generator for sampling EBIT values.
    - `L` (float): Current leverage percentage.
    - `E` (float): Current EBIT value.
    - `EV_prev` (float): Previous enterprise value.
    - `t` (int): Current time step in the episode.

    Methods:
    - `reset()`: Resets the environment to its initial state.
    - `step(action: int)`: Applies the selected action, updates the state, and computes the reward.
    - `_internal_reset()`: Helper method to initialize internal state variables.
    - `_get_obs()`: Helper method to retrieve the current observation.

    Example:
    ```python
    env = NaiveDebtControlEnv()
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    ```
    """

    metadata = {"render_modes": []}

    def __init__(self, p: EnvParams = EnvParams(), seed: int | None = None):
        super().__init__()
        self.p = p
        self.action_space = spaces.Discrete(3)  # –5 pp, 0, +5 pp
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([p.l_max, np.inf], dtype=np.float32)
        )
        self.np_random = np.random.default_rng(seed)
        self._internal_reset()

    # -------------- Gym API --------------
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._internal_reset()
        return self._get_obs(), {}

    def step(self, action: int):
        """
        Executes a single step in the environment based on the selected action.

        Args:
            action (int): The action to take, representing leverage adjustment:
                          - 0: No change in leverage
                          - 1: Increase leverage by +5 percentage points
                          - 2: Decrease leverage by -5 percentage points

        Returns:
            tuple: A tuple containing the following elements:
                - obs (np.ndarray): The next state `[L, E]` where `L` is leverage and `E` is EBIT.
                - reward (float): The reward for the step, calculated as the change in enterprise value (`ΔEV`).
                - terminated (bool): Whether the episode has ended due to EBIT < 0 or reaching the maximum episode length.
                - truncated (bool): Always `False` (no truncation logic implemented).
                - info (dict): Additional information including:
                    - "ev": Current enterprise value (`EV_t`).
                    - "ebit": Current EBIT value (`E`).

        Raises:
            ValueError: If the action is not valid (not in the range [0, 2]).
        """
        # 1. apply leverage move (–5, 0, +5 pp)
        delta_L = (-0.05, 0.0, 0.05)[action]
        self.L = np.clip(self.L + delta_L, 0.0, self.p.l_max)

        # 2. draw next-quarter EBIT (iid Normal)
        self.E = self.np_random.normal(self.p.mu_E, self.p.sigma_E)

        # 3. compute new enterprise value
        #    Debt uses previous EV -> simple, avoids implicit equation
        D_t = self.L * self.EV_prev
        EV_t = self.p.m_u * self.E + self.p.tax_rate * D_t
        delta_EV = EV_t - self.EV_prev
        self.EV_prev = EV_t

        # 4. reward is just ΔEV
        reward = delta_EV

        # 5. termination: EBIT < 0 or horizon reached
        self.t += 1
        terminated = (self.E < 0.0) or (self.t >= self.p.ep_len)

        return self._get_obs(), reward, terminated, False, {"ev": EV_t, "ebit": self.E}

    # -------------- helpers --------------
    def _internal_reset(self):
        self.L = 0.40  # start leverage 40 %
        self.E = self.p.mu_E  # start EBIT at mean
        self.EV_prev = self.p.m_u * self.E  # no debt in first EV term
        self.t = 0

    def _get_obs(self):
        return np.array([self.L, self.E], dtype=np.float32)
