import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from typing import Callable

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import numpy as np
import pandas as pd

from flax.core import freeze, unfreeze

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    signal_noise: float = 0.1,
    min_signal_noise: float = 0.05,
    max_signal_noise: float = 0.2,
    seed=1,
    env_kwargs: dict = None,
    video_folder: str = "videos",
    rewards_folder: str = "saved_rewards",
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, env_kwargs, video_folder, video_capture_trigger=(lambda ep_id: True))])
    max_action = float(envs.single_action_space.high[0])
    obs, _ = envs.reset()

    #np.savez(f"{rewards_folder}/{run_name}/reset_init_particle_positions.npz", infos["particle_positions"])

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

    """def print_shapes(tree):
        for k, v in tree.items():
            if isinstance(v, dict) or hasattr(v, "keys"):
                print(f"{k}:")
                print_shapes(v)
            else:
                print(f"{k}: {v.shape}")

    print_shapes(unfreeze(actor_params))"""

    episodic_returns = []
    rewards_per_episode = []
    fill_levels_per_episode = []
    jug_rotations_per_episode = []
    cup_fill_rates_per_episode = []
    jug_flow_rates_per_episode = []
    actions_per_episode = []

    records = []

    min_exploration_noise = 0.02
    max_exploration_noise = exploration_noise * 1.2

    os.makedirs(f"{rewards_folder}/{run_name}", exist_ok=True)
    while len(episodic_returns) < eval_episodes:
        actions_det = actor.apply(actor_params, obs)
        actions_det = jax.device_get(actions_det)#[0]

        expl_noise = np.random.normal(0, max_action * exploration_noise, size=envs.single_action_space.shape)

        # signal-dependent noise
        noise_scale = (
            min_signal_noise
            + signal_noise * (np.abs(actions_det) / max_action)
        )

        # Clip noise scale for stability
        noise_scale = np.clip(
            noise_scale,
            min_signal_noise,
            max_signal_noise,
        )

        execution_noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=actions_det.shape,
        )

        actions = actions_det + expl_noise + execution_noise

        actions = actions.clip(
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        """# Signal-DEPENDENT noise after warmup
        noise_scale = (
            min_exploration_noise
            + exploration_noise * np.abs(actions)#np.sqrt(np.abs(actions_det))
        )

        # Clip noise scale for stability
        noise_scale = np.clip(
            noise_scale,
            min_exploration_noise,
            max_exploration_noise,
        )

        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale,
            size=actions.shape,
        )

        actions = actions + noise

        actions = np.array(
            [
                actions.clip(
                    envs.single_action_space.low,
                    envs.single_action_space.high)
            ]
        )"""
        """actions = np.array(
            [
                (
                    jax.device_get(actions)[0]
                    + np.random.normal(0, max_action * exploration_noise, size=envs.single_action_space.shape)
                ).clip(envs.single_action_space.low, envs.single_action_space.high)
            ]
        )"""

        # CHANGED: original version used final_info to detect done which was removed in gymnasium 1.0.0
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        rewards_per_episode.append(rewards[0]) # currently works for only one environment in the SyncVectorEnv
        fill_levels_per_episode.append(infos['current_fill_level'][0]) # currently works for only one environment in the SyncVectorEnv
        jug_rotations_per_episode.append(infos['jug_rotation'][0]) # currently works for only one environment in the SyncVectorEnv
        cup_fill_rates_per_episode.append(infos['cup_fill_rate'][0]) # currently works for only one environment in the SyncVectorEnv
        jug_flow_rates_per_episode.append(infos['jug_flow_rate'][0])
        actions_per_episode.append(actions[0])

        """if infos["step_id"][0] == 2:
            np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_particles_first_step.npz", infos["particle_positions"][0])"""

        data_trial = {key : value[0] for key, value in infos.items() if ("episode" not in key) and (key[0] != "_")}
        data_trial["reward"] = rewards[0]
        data_trial["action"] = actions[0][0]
        data_trial["policy_action"] = actions_det[0][0]
        data_trial["signal_noise"] = execution_noise[0][0]
        data_trial["exploration_noise"] = expl_noise[0]
        
        records.append(data_trial.copy())

        dones = terminated | truncated

        # Support both single and multiple environments
        if isinstance(infos, dict):  # Single environment
            if (terminated or truncated) and "episode" in infos:
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}.npz", rewards_per_episode)
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_filllevels.npz", fill_levels_per_episode)
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_jug_rotations.npz", jug_rotations_per_episode)
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_cup_fill_rates.npz", cup_fill_rates_per_episode)
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_jug_flow_rates.npz", jug_flow_rates_per_episode)
                np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_actions.npz", actions_per_episode)

                df = pd.DataFrame(records)
                df.to_csv(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}.csv", index=False)
                
                print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}")
                episodic_returns.append(infos["episode"]["r"])
                rewards_per_episode = []
                fill_levels_per_episode = []
                jug_rotations_per_episode = []
                cup_fill_rates_per_episode = []
                jug_flow_rates_per_episode = []
                actions_per_episode = []
                records = []
        else:  # Vectorized environment
            for i, done in enumerate(dones):
                if done and "episode" in infos[i]:
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}.npz", rewards_per_episode)
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_filllevels.npz", fill_levels_per_episode)
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_jug_rotations.npz", jug_rotations_per_episode)
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_cup_fill_rates.npz", cup_fill_rates_per_episode)
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_jug_flow_rates.npz", jug_flow_rates_per_episode)
                    np.savez(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}_actions.npz", actions_per_episode)

                    df = pd.DataFrame(records)
                    df.to_csv(f"{rewards_folder}/{run_name}/episodes_{len(episodic_returns)}.csv", index=False)

                    print(f"eval_episode={len(episodic_returns)}, episodic_return={infos[i]['episode']['r']}")
                    episodic_returns.append(infos[i]['episode']['r'])
                    rewards_per_episode = []
                    fill_levels_per_episode = []
                    jug_rotations_per_episode = []
                    cup_fill_rates_per_episode = []
                    jug_flow_rates_per_episode = []
                    actions_per_episode = []
                    records = []

        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from cleanrl.td3_continuous_action_jax import Actor, QNetwork, make_env
    import json

    run_nr = "1769605845_c84da4"
    run_name= f"PouringEnvIsaac-v0__td3_continuous_action_jax__42__{run_nr}"#f"PouringEnv-v0__td3_continuous_action_jax__42__{run_nr}"

    #run_name = "PouringEnvIsaacVisual-v0__td3_continuous_action_jax__42__1761689808_403a80"
    #model_path = os.path.join("/home/carola/masterthesis/cleanrl/cleanrl/outputs/runs", run_name, "td3_continuous_action_jax_42.cleanrl_model")    
    model_path = os.path.join("/home/carola/masterthesis/cleanrl/cleanrl/outputs/runs", run_name, "td3_continuous_action_jax.cleanrl_model")    
    
    with open(os.path.join("/home/carola/masterthesis/cleanrl/cleanrl/outputs/runs", run_name, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)

    relevant_keys = [
        "target_level_wgt",
        "pt_cup_wgt",
        "pt_flow_wgt",
        "pt_spill_wgt",
        "action_cost",
        "jug_resting_wgt",
        "jug_velocity_wgt",
        "distance_wgt",
        "fovea_radius",
        "time_penalty",
        ]

    reward_weights = {k: hyperparams[k] for k in relevant_keys if k in hyperparams}

    env_kwargs = {
        #"gnn_model_path": '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl',
        "data_path": '/shared_data/Pouring_mpc_1D_1902/',
        "reward_weights": reward_weights
        }
    
    video_folder = os.path.abspath(f"/home/carola/masterthesis/cleanrl/cleanrl/outputs/videos/")
    rewards_folder = os.path.abspath(f"/home/carola/masterthesis/cleanrl/cleanrl/outputs/saved_rewards/")
      
    evaluate(
        model_path,
        make_env,
        hyperparams["env_id"],
        eval_episodes=1,
        run_name=f"{run_name}-eval_3",
        Model=(Actor, QNetwork),
        exploration_noise=0,#hyperparams["exploration_noise"],
        signal_noise=hyperparams["signal_noise"],
        min_signal_noise=hyperparams["min_signal_noise"],
        max_signal_noise=hyperparams["max_signal_noise"],
        env_kwargs=env_kwargs,
        video_folder=video_folder,
        rewards_folder=rewards_folder
    )