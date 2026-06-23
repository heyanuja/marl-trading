import unittest

import numpy as np

from src.environment.trading_env import TradingEnv
from src.evaluation.execution import execution_costs_from_info


class TestExecutionMetrics(unittest.TestCase):
    def test_execution_costs_shapes_and_signs(self):
        env = TradingEnv(
            num_agents=2,
            num_resources=3,
            max_steps=3,
            temporary_impact=0.02,
            spread=0.01,
            transaction_cost=1e-3,
        )
        obs, _ = env.reset(seed=0)

        action = np.ones((2, 3), dtype=np.float32)
        _, _, _, _, info = env.step(action)

        costs = execution_costs_from_info(info)
        self.assertEqual(costs["fees"].shape, (2,))
        self.assertEqual(costs["slippage"].shape, (2,))
        self.assertEqual(costs["notional"].shape, (2,))
        self.assertEqual(costs["turnover"].shape, (2,))

        self.assertTrue(np.isfinite(costs["fees"]).all())
        self.assertTrue(np.isfinite(costs["slippage"]).all())
        self.assertTrue(np.isfinite(costs["notional"]).all())
        self.assertTrue(np.isfinite(costs["turnover"]).all())

        self.assertTrue((costs["fees"] >= 0).all())
        self.assertTrue((costs["notional"] >= 0).all())
        self.assertTrue((costs["turnover"] >= 0).all())
        self.assertTrue((costs["slippage"] >= -1e-6).all())


if __name__ == "__main__":
    unittest.main()

