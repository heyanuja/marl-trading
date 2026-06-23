import unittest

import numpy as np

from src.environment.trading_env import PriceProcess, TradingEnv


class TestTradingEnv(unittest.TestCase):
    def test_reset_shapes(self):
        env = TradingEnv(num_agents=2, num_resources=3, max_steps=5)
        obs, info = env.reset(seed=0)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.isfinite(obs).all())
        self.assertIn("prices", info)
        self.assertIn("portfolio_values", info)

    def test_step_shapes(self):
        env = TradingEnv(num_agents=2, num_resources=3, max_steps=5)
        obs, _ = env.reset(seed=0)

        actions = np.zeros((2, 3), dtype=np.float32)
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        self.assertEqual(next_obs.shape, obs.shape)
        self.assertEqual(rewards.shape, (2,))
        self.assertFalse(terminated)
        self.assertIsInstance(truncated, bool)
        self.assertIn("execution_prices", info)
        self.assertIn("executed_units", info)

        self.assertTrue((env.inventories >= 0).all())
        self.assertTrue((env.cash >= 0).all())
        self.assertTrue(((env.prices >= env.min_price) & (env.prices <= env.max_price)).all())

    def test_single_agent_action_shape(self):
        env = TradingEnv(num_agents=1, num_resources=3, max_steps=3)
        env.reset(seed=0)

        action = np.zeros(3, dtype=np.float32)
        _, rewards, _, _, _ = env.step(action)
        self.assertEqual(rewards.shape, (1,))

    def test_prices_static_with_zero_vol_and_no_impact(self):
        env = TradingEnv(
            num_agents=1,
            num_resources=2,
            max_steps=3,
            permanent_impact=0.0,
            price_process=PriceProcess(drift=0.0, volatility=0.0, return_autocorr=0.0, corr=0.0),
        )
        env.reset(seed=0)
        p0 = env.prices.copy()
        env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(np.allclose(env.prices, p0))

    def test_buy_pressure_moves_price_up_without_exogenous_noise(self):
        env = TradingEnv(
            num_agents=1,
            num_resources=1,
            max_steps=2,
            permanent_impact=0.2,
            temporary_impact=0.0,
            spread=0.0,
            price_process=PriceProcess(drift=0.0, volatility=0.0, return_autocorr=0.0, corr=0.0),
        )
        env.reset(seed=0)
        p0 = float(env.prices[0])
        env.step(np.array([1.0], dtype=np.float32))
        self.assertGreater(float(env.prices[0]), p0)


if __name__ == "__main__":
    unittest.main()

