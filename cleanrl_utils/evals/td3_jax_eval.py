import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from typing import Callable

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import numpy as np

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    seed=1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    max_action = float(envs.single_action_space.high[0])
    obs, _ = envs.reset()

    Actor, QNetwork = Model
    action_scale = np.array((envs.action_space.high - envs.action_space.low) / 2.0)
    action_bias = np.array((envs.action_space.high + envs.action_space.low) / 2.0)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=action_scale,
        action_bias=action_bias,
    )
    qf = QNetwork()
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    actor_params = actor.init(actor_key, obs)
    qf1_params = qf.init(qf1_key, obs, envs.action_space.sample())
    qf2_params = qf.init(qf2_key, obs, envs.action_space.sample())
    with open(model_path, "rb") as f:
        (actor_params, qf1_params, qf2_params) = flax.serialization.from_bytes(
            (actor_params, qf1_params, qf2_params), f.read()
        )
    # note: qf1_params and qf2_params are not used in this script
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    episodic_returns = []
    rewards_per_episode = []
    fill_levels_per_episode = []

    os.makedirs(f"saved_rewards/{run_name}", exist_ok=True)
    while len(episodic_returns) < eval_episodes:
        actions = actor.apply(actor_params, obs)
        actions = np.array(
            [
                (
                    jax.device_get(actions)[0]
                    + np.random.normal(0, max_action * exploration_noise, size=envs.single_action_space.shape)
                ).clip(envs.single_action_space.low, envs.single_action_space.high)
            ]
        )

        # CHANGED: original version used final_info to detect done which was removed in gymnasium 1.0.0
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        rewards_per_episode.append(rewards[0]) # currently works for only one environment in the SyncVectorEnv
        fill_levels_per_episode.append(infos['current_fill_level'][0]) # currently works for only one environment in the SyncVectorEnv

        dones = terminated | truncated

        # Support both single and multiple environments
        if isinstance(infos, dict):  # Single environment
            if (terminated or truncated) and "episode" in infos:
                np.savez(f"saved_rewards/{run_name}/episodes_{len(episodic_returns)}.npz", rewards_per_episode)
                np.savez(f"saved_rewards/{run_name}/episodes_{len(episodic_returns)}_filllevels.npz", fill_levels_per_episode)
                print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}")
                episodic_returns.append(infos["episode"]["r"])
                rewards_per_episode = []
                fill_levels_per_episode = []
        else:  # Vectorized environment
            for i, done in enumerate(dones):
                if done and "episode" in infos[i]:
                    np.savez(f"saved_rewards/{run_name}/episodes_{len(episodic_returns)}.npz", rewards_per_episode)
                    np.savez(f"saved_rewards/{run_name}/episodes_{len(episodic_returns)}_filllevels.npz", fill_levels_per_episode)
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={infos[i]['episode']['r']}")
                    episodic_returns.append(infos[i]['episode']['r'])
                    rewards_per_episode = []
                    fill_levels_per_episode = []

        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from cleanrl.td3_continuous_action_jax import Actor, QNetwork, make_env

    run_nr = 1750411979
    run_name= f"PouringEnv-v0__td3_continuous_action_jax__42__{run_nr}"
    model_path = os.path.join("/home/carola/masterthesis/cleanrl/cleanrl/runs", run_name, "td3_continuous_action_jax.cleanrl_model")      
    evaluate(
        model_path,
        make_env,
        "PouringEnv-v0",
        eval_episodes=10,
        run_name=f"{run_nr}_eval",
        Model=(Actor, QNetwork),
        exploration_noise=0.1,
    )