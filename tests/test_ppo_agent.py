import subprocess
import sys
import textwrap
import unittest

from src.utils.probe import numpy_torch_import_works


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )


_TORCH_OK = numpy_torch_import_works(trials=1)


@unittest.skipUnless(_TORCH_OK, "torch runtime unavailable in this environment")
class TestPPOAgent(unittest.TestCase):
    def test_select_action_shapes(self):
        code = textwrap.dedent(
            """
            import numpy as np
            from src.environment.trading_env import TradingEnv
            from src.agents.ppo_agent import PPOAgent

            env = TradingEnv(num_agents=2, num_resources=3, max_steps=5)
            agent = PPOAgent(
                input_dim=env.single_observation_space.shape[0],
                action_dim=env.single_action_space.shape[0],
            )
            obs, _ = env.reset(seed=0)
            actions, log_probs, values = agent.select_action(obs)
            assert actions.shape == (env.num_agents, env.num_resources)
            assert log_probs.shape == (env.num_agents,)
            assert values.shape == (env.num_agents,)
            assert np.isfinite(actions).all()
            assert ((actions >= -1.0) & (actions <= 1.0)).all()
            print("ok")
            """
        )
        res = _run_python(code)
        self.assertEqual(
            res.returncode,
            0,
            msg=f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}",
        )

    def test_update_smoke(self):
        code = textwrap.dedent(
            """
            import numpy as np
            from src.environment.trading_env import TradingEnv
            from src.agents.ppo_agent import PPOAgent, PPOConfig, compute_gae

            env = TradingEnv(num_agents=2, num_resources=2, max_steps=8)
            agent = PPOAgent(
                input_dim=env.single_observation_space.shape[0],
                action_dim=env.single_action_space.shape[0],
                config=PPOConfig(ppo_epochs=1, minibatch_size=16),
            )

            obs, _ = env.reset(seed=0)
            states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
            for _ in range(6):
                act, log_p, val = agent.select_action(obs)
                next_obs, rew, terminated, truncated, _ = env.step(act)
                states.append(obs)
                actions.append(np.asarray(act, dtype=np.float32))
                log_probs.append(np.asarray(log_p, dtype=np.float32))
                values.append(np.asarray(val, dtype=np.float32))
                rewards.append(np.asarray(rew, dtype=np.float32))
                dones.append(np.full(env.num_agents, float(bool(terminated)), dtype=np.float32))
                obs = next_obs
                if terminated or truncated:
                    break

            states_arr = np.asarray(states, dtype=np.float32)
            actions_arr = np.asarray(actions, dtype=np.float32)
            log_probs_arr = np.asarray(log_probs, dtype=np.float32)
            values_arr = np.asarray(values, dtype=np.float32)
            rewards_arr = np.asarray(rewards, dtype=np.float32)
            dones_arr = np.asarray(dones, dtype=np.float32)

            last_values = agent.value(obs).astype(np.float32)
            adv, rets = compute_gae(
                rewards_arr,
                values_arr,
                dones_arr,
                last_values=last_values,
                gamma=agent.config.gamma,
                gae_lambda=agent.config.gae_lambda,
            )

            t_steps, num_agents, obs_dim = states_arr.shape
            act_dim = actions_arr.shape[-1]
            batch = {
                "states": states_arr.reshape(t_steps * num_agents, obs_dim),
                "actions": actions_arr.reshape(t_steps * num_agents, act_dim),
                "old_log_probs": log_probs_arr.reshape(t_steps * num_agents),
                "advantages": adv.reshape(t_steps * num_agents),
                "returns": rets.reshape(t_steps * num_agents),
            }
            metrics = agent.update(batch)
            assert "loss" in metrics
            assert np.isfinite(metrics["loss"])
            print("ok")
            """
        )
        res = _run_python(code)
        self.assertEqual(
            res.returncode,
            0,
            msg=f"stdout:\n{res.stdout}\n\nstderr:\n{res.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
