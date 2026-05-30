import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import flax
import gymnasium as gym
from typing import Callable
import flax.linen as nn

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: tuple,  # (Actor, QNetwork)
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
    # Fallback to empty dict if None to safely use .get()
    if env_kwargs is None:
        env_kwargs = {}

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name, env_kwargs, video_folder, video_capture_trigger=(lambda ep_id: True))])
    max_action = float(envs.single_action_space.high[0])
    obs, _ = envs.reset()

    gnn_model_path = '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl'
    data_path = env_kwargs.get("data_path")
    print(gnn_model_path)
    print(data_path)
    gnn = GNNEncoder(gnn_model_path=gnn_model_path, data_path=data_path)

    @jax.jit
    def batched_gnn_forward(jug_hist_batch, pt_hist_batch):
        batch_size = jug_hist_batch.shape[0]
        
        if batch_size <= 128:
            return jax.vmap(gnn._apply_gnn_processing_step)(jug_hist_batch, pt_hist_batch)
        
        chunk_size = 128
        num_chunks = batch_size // chunk_size
        
        jug_chunks = jug_hist_batch.reshape((num_chunks, chunk_size) + jug_hist_batch.shape[1:])
        pt_chunks = pt_hist_batch.reshape((num_chunks, chunk_size) + pt_hist_batch.shape[1:])
        
        chunk_vmap = jax.vmap(gnn._apply_gnn_processing_step)
        
        def scan_step(carry, chunk_args):
            j_chunk, p_chunk = chunk_args
            return None, chunk_vmap(j_chunk, p_chunk)
            
        _, out_chunks = jax.lax.scan(scan_step, None, (jug_chunks, pt_chunks))
        return out_chunks.reshape((batch_size, *out_chunks.shape[2:]))

    
    obs_jug_init, obs_jug_buf_init, obs_pt_buf_init, obs_init_target_level = unpack_obs(jnp.array(obs))
    init_gnn_latents, obs_jug_init = jax.lax.stop_gradient(batched_gnn_forward(obs_jug_buf_init, obs_pt_buf_init))
    
    print("liq: ", init_gnn_latents.shape)
    print("jug: ", obs_jug_init.shape)
    print("target: ", obs_init_target_level)
    Actor, QNetwork = Model
    action_scale = jnp.array((envs.action_space.high - envs.action_space.low) / 2.0)
    action_bias = jnp.array((envs.action_space.high + envs.action_space.low) / 2.0)
    
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=action_scale,
        action_bias=action_bias,
    )
    qf = QNetwork()
    
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    
    
    actor_params = actor.init(actor_key, obs_jug_init, init_gnn_latents, obs_init_target_level)
    qf1_params = qf.init(qf1_key, obs_jug_init, init_gnn_latents, obs_init_target_level, envs.action_space.sample())
    qf2_params = qf.init(qf2_key, obs_jug_init, init_gnn_latents, obs_init_target_level, envs.action_space.sample())

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
    jug_rotations_per_episode = []
    cup_fill_rates_per_episode = []
    jug_flow_rates_per_episode = []
    actions_per_episode = []

    records = []

    min_exploration_noise = 0.02
    max_exploration_noise = exploration_noise * 1.2

    os.makedirs(f"{rewards_folder}/{run_name}", exist_ok=True)
    
    while len(episodic_returns) < eval_episodes:
        # Unpack and compute latents for the current step ---
        obs_jug, obs_jug_buf, obs_pt_buf, obs_target_level = unpack_obs(jnp.array(obs))
        gnn_latents, obs_jug = jax.lax.stop_gradient(batched_gnn_forward(obs_jug_buf, obs_pt_buf))
        
        # Pass both the jug state and latents into the actor
        actions_det = actor.apply(actor_params, obs_jug, gnn_latents, obs_target_level)
        actions_det = jax.device_get(actions_det)
        # ------------------------------------------------------------

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

        # Sample half-normal (always positive)
        epsilon = np.abs(
            np.random.normal(loc=0.0, scale=noise_scale, size=actions_det.shape)
        )

        # Add noise in direction of action sign (increases amplitude only)
        actions_exec = actions_det + np.sign(actions_det) * epsilon

        # Execution noise
        execution_noise = actions_exec - actions_det
        actions = actions_det + expl_noise + execution_noise

        actions = actions.clip(
            envs.single_action_space.low,
            envs.single_action_space.high,
        )

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        rewards_per_episode.append(rewards[0])
        fill_levels_per_episode.append(infos['current_fill_level'][0])
        jug_rotations_per_episode.append(infos['jug_rotation'][0])
        cup_fill_rates_per_episode.append(infos['cup_fill_rate'][0])
        jug_flow_rates_per_episode.append(infos['jug_flow_rate'][0])
        actions_per_episode.append(actions[0])

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
        
    envs.close()
    return episodic_returns


if __name__ == "__main__":
    from cleanrl.td3_continuous_action_jax import Actor, QNetwork, make_env, unpack_obs, GNNEncoder
    import json

    #fill_targets = [x / 10 for x in range(2, 10)] # 0.2 to 0.9 in 0.1 increments
    fill_targets = [0.25 + 0.1 * i for i in range(7)]

    with open("/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/test.txt", "r") as f:#test_diff_action_costs.txt
        runs_ids = [line.strip().split("__")[-1] for line in f]

    for current_fill_target in fill_targets:
        for run_nr in runs_ids:
        #run_nr = "1773629276_bc85d2"
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
            reward_weights["fill_target_level"] = current_fill_target 

            env_kwargs = {
                #"gnn_model_path": '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl',
                "data_path": '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/Pouring_mpc_1D_1902/',
                "reward_weights": reward_weights
                }
            
            video_folder = os.path.abspath(f"/home/carola/masterthesis/cleanrl/cleanrl/outputs/videos/")
            rewards_folder = os.path.abspath(f"/home/carola/masterthesis/cleanrl/cleanrl/outputs/saved_rewards/")
            
            evaluate(
                model_path,
                make_env,
                hyperparams["env_id"],
                eval_episodes=10,
                run_name=f"{run_name}-eval_2_{current_fill_target}",
                Model=(Actor, QNetwork),
                exploration_noise=0,#hyperparams["exploration_noise"],
                signal_noise=hyperparams["signal_noise"],
                min_signal_noise=hyperparams["min_signal_noise"],
                max_signal_noise=hyperparams["max_signal_noise"],
                env_kwargs=env_kwargs,
                video_folder=video_folder,
                rewards_folder=rewards_folder
            )