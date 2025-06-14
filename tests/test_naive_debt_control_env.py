import unittest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from unittest.mock import patch, MagicMock
from environments.naive_debt_control_env import NaiveDebtControlEnv, EnvParams


class TestNaiveDebtControlEnv(unittest.TestCase):
    def setUp(self):
        """Set up the environment with fixed parameters and seed."""
        self.params = EnvParams(
            mu_E=100.0,
            sigma_E=15.0,
            m_u=8.0,
            tax_rate=0.25,
            l_max=0.80,
            ep_len=40
        )
        self.env = NaiveDebtControlEnv(p=self.params)

    def test_initialization(self):
        """Test environment initialization."""
        self.assertIsInstance(self.env, gym.Env)
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 3)
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        np.testing.assert_array_almost_equal(self.env.observation_space.low, [0.0, -np.inf])
        np.testing.assert_array_almost_equal(self.env.observation_space.high, [0.80, np.inf])
        self.assertEqual(self.env.p, self.params)
        self.assertIsInstance(self.env.np_random, np.random.Generator)

    def test_reset(self):
        """Test environment reset functionality."""
        obs, info = self.env.reset()
        expected_obs = np.array([0.40, 100.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(obs, expected_obs, decimal=6)
        self.assertEqual(self.env.L, 0.40)
        self.assertEqual(self.env.E, 100.0)
        self.assertEqual(self.env.EV_prev, 800.0)  # m_u * E = 8 * 100
        self.assertEqual(self.env.t, 0)
        self.assertEqual(info, {})

    def test_step_no_action(self):
        """Test step with no leverage change (action=0)."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = 105.0
        self.env.np_random = mock_generator
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(1)

        # Expected calculations
        expected_L = 0.40
        expected_E = 105.0
        expected_EV_prev = 800.0  # From reset: m_u * E = 8 * 100
        expected_D_t = expected_L * expected_EV_prev  # 0.40 * 800 = 320
        expected_EV_t = self.params.m_u * expected_E + self.params.tax_rate * expected_D_t  # 8 * 105 + 0.25 * 320 = 840 + 80 = 920
        expected_delta_EV = expected_EV_t - expected_EV_prev  # 920 - 800 = 120

        np.testing.assert_array_almost_equal(obs, [expected_L, expected_E], decimal=6)
        self.assertAlmostEqual(reward, expected_delta_EV, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["ev"], expected_EV_t, places=6)
        self.assertAlmostEqual(info["ebit"], expected_E, places=6)
        self.assertEqual(self.env.t, 1)


    def test_step_increase_leverage(self):
        """Test step with +5 pp leverage increase (action=1)."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = 105.0
        self.env.np_random = mock_generator
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(2)

        # Expected calculations
        expected_L = 0.45  # 0.40 + 0.05
        expected_E = 105.0
        expected_EV_prev = 800.0
        expected_D_t = expected_L * expected_EV_prev  # 0.45 * 800 = 360
        expected_EV_t = self.params.m_u * expected_E + self.params.tax_rate * expected_D_t  # 8 * 105 + 0.25 * 360 = 840 + 90 = 930
        expected_delta_EV = expected_EV_t - expected_EV_prev  # 930 - 800 = 130

        np.testing.assert_array_almost_equal(obs, [expected_L, expected_E], decimal=6)
        self.assertAlmostEqual(reward, expected_delta_EV, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["ev"], expected_EV_t, places=6)
        self.assertAlmostEqual(info["ebit"], expected_E, places=6)

    def test_step_decrease_leverage(self):
        """Test step with -5 pp leverage decrease (action=2)."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = 105.0
        self.env.np_random = mock_generator
        self.env.reset()

        obs, reward, terminated, truncated, info = self.env.step(0)

        # Expected calculations
        expected_L = 0.35  # 0.40 - 0.05
        expected_E = 105.0
        expected_EV_prev = 800.0
        expected_D_t = expected_L * expected_EV_prev  # 0.35 * 800 = 280
        expected_EV_t = self.params.m_u * expected_E + self.params.tax_rate * expected_D_t  # 8 * 105 + 0.25 * 280 = 840 + 70 = 910
        expected_delta_EV = expected_EV_t - expected_EV_prev  # 910 - 800 = 110

        np.testing.assert_array_almost_equal(obs, [expected_L, expected_E], decimal=6)
        self.assertAlmostEqual(reward, expected_delta_EV, places=6)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["ev"], expected_EV_t, places=6)
        self.assertAlmostEqual(info["ebit"], expected_E, places=6)

    def test_leverage_boundaries(self):
        """Test leverage stays within [0, l_max]."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = 105.0
        self.env.np_random = mock_generator
        self.env.reset()
        self.env.L = 0.0  # Set leverage to lower bound
        obs, _, _, _, _ = self.env.step(0)  # Try to decrease leverage
        self.assertAlmostEqual(obs[0], 0.0, places=6)  # Should stay at 0.0

        self.env.reset()
        self.env.np_random = mock_generator  # Reuse mock
        self.env.L = 0.80  # Set leverage to upper bound
        obs, _, _, _, _ = self.env.step(2)  # Try to increase leverage
        self.assertAlmostEqual(obs[0], 0.80, places=6)  # Should stay at 0.80

    def test_termination_negative_ebit(self):
        """Test termination when EBIT becomes negative."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = -0.1
        self.env.np_random = mock_generator
        self.env.reset()
        _, _, terminated, truncated, _ = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_termination_max_steps(self):
        """Test termination when maximum episode length is reached."""
        mock_generator = MagicMock()
        mock_generator.normal.return_value = 105
        self.env.np_random = mock_generator
        self.env.reset()
        self.env.t = 39
        _, _, terminated, truncated, _ = self.env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_invalid_action(self):
        """Test that invalid actions raise an IndexError."""
        self.env.reset()
        with self.assertRaises(IndexError):
            self.env.step(3)  # Action outside [0, 2]


if __name__ == '__main__':
    unittest.main()