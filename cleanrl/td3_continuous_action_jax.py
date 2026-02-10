# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import random
import time
from dataclasses import dataclass
import sys

"""# Add the parent directory (root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))"""

import gymnasium as gym
import learning_to_simulate_pouring.register_env  # this triggers the registration


import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import json


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    output_dir: str = "outputs/"
    """path for the output directory"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    initial_exploration_noise: float = 0.2
    min_exploration_noise: float= 0.05
    max_exploration_noise: float  = 0.2
    exploration_warmup_steps: float = 5000

    signal_noise: float = 0.1
    min_signal_noise: float = 0.05
    max_signal_noise: float = 0.2

    # Enviroment specific arguments
    gnn_model_path: str = '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl'
    """the path to the GNN model checkpoint"""
    data_path: str = '/shared_data/Pouring_mpc_1D_1902/'
    """the path to the dataset for the GNN model"""
    target_particles_path: str = (
        "/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/particle_states/saved_particles_final_state.npz"
    )
    """path to the target particles for the environment (required for chamfer loss)"""

    target_level_wgt: float = 1.0
    pt_cup_wgt: float = 5.0
    pt_flow_wgt: float = -30.0
    pt_spill_wgt: float = -2.0
    action_cost: float = -0.01
    jug_resting_wgt: float = -0.000001
    jug_velocity_wgt: float = 0.0
    distance_wgt: float = 0.0
    fovea_radius: int = 30
    time_penalty: float = -0.001
    """weights for the different components of the reward function"""


def make_env(env_id, seed, idx, capture_video, run_name, env_kwargs=None, video_folder="videos", video_capture_trigger=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
            env.reset(seed=seed + idx)
            if video_capture_trigger is None:
                env = gym.wrappers.RecordVideo(env, os.path.join(video_folder, run_name))
            else:
                env = gym.wrappers.RecordVideo(
                    env,
                    os.path.join(video_folder, run_name),
                    episode_trigger=video_capture_trigger,
                )
        else:
            env = gym.make(env_id, **env_kwargs)
            env.reset(seed=seed + idx)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
"""# networks for reduced observations
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray  # shape: (action_dim,)
    action_bias: jnp.ndarray   # shape: (action_dim,)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Dense(256)(x)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)  # constrain to [-1, 1]

        # Rescale to [action_low, action_high]
        return x * self.action_scale + self.action_bias
    
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x"""


# networks for full state
# JugEncoder and ParticleEncoder are shared between actor and critic.
class JugEncoder(nn.Module):
    @nn.compact
    def __call__(self, jug_obs):
        # jug_obs shape: (batch, 19)
        x = nn.Dense(64)(jug_obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        return x  # (batch, 64)

class ParticleEncoder(nn.Module):
    @nn.compact
    def __call__(self, particles):  # particles shape: (batch, 1048, 128)
        x = nn.Dense(64)(particles)       # (batch, 1048, 64)
        x = nn.relu(x)
        x = nn.Dense(64)(x)               # (batch, 1048, 64)
        x = nn.relu(x)
        x = jnp.mean(x, axis=1)           # mean pool over particles → (batch, 64)
        return x
    
# Image + Gaze Encoder
class ImageGazeEncoder(nn.Module):
    gaze_dim: int = 2

    @nn.compact
    def __call__(self, img, gaze):
        """
        img: (batch, H*W), flattened grayscale image
        gaze: (batch, 2), normalized gaze position
        """
        batch_size, flat_dim = img.shape
        x = img.reshape((batch_size, 64, 64, 1))

        # Efficient CNN
        x = nn.Conv(16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64)

        # Combine with gaze
        x = jnp.concatenate([x, gaze], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return x  # (batch, 128)

# Image Encoder (NO GAZE!!!!)
class ImageEncoder(nn.Module):

    @nn.compact
    def __call__(self, img):
        """
        img: (batch, H*W), flattened grayscale image
        gaze: (batch, 2), normalized gaze position
        """
        batch_size, flat_dim = img.shape
        x = img.reshape((batch_size, 64, 64, 1))

        # Efficient CNN
        x = nn.Conv(16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (batch, 64)

        # Combine with gaze
        x = jnp.concatenate([x], axis=-1)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return x  # (batch, 128)
"""
# with jug and gaze action together
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, flat_obs):
        # Split flat observation
        jug_obs = flat_obs[:, :18]
        #particle_flat = flat_obs[:, 18:]
        #particle_flat = flat_obs[:, 18:18+1047*9]# TODO: add back in for liquid data
        #particles = particle_flat.reshape((flat_obs.shape[0], 1048, 128))
        #particles = particle_flat.reshape((flat_obs.shape[0], 1047, 9))

        #img_flat_start = 18 + 1047*9 # TODO: add back in for liquid data
        img_flat_start = 18 
        img_flat_end = -2  # last 2 values are gaze
        img_flat = flat_obs[:, img_flat_start:img_flat_end]
        gaze = flat_obs[:, -2:]

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        #liquid_emb = ParticleEncoder()(particles) # TODO: add back in for liquid data
        img_emb = ImageGazeEncoder()(img_flat, gaze)

        # Combine and pass through actor MLP
        #x = jnp.concatenate([jug_emb, liquid_emb, img_emb], axis=-1) # TODO: add back in for liquid data
        x = jnp.concatenate([jug_emb, img_emb], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x * self.action_scale + self.action_bias
"""
"""
# for visual processing (with gaze)
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, flat_obs):
        jug_action_dim = self.action_dim - 2
        gaze_action_dim = 2  # last two dimensions for gaze

        # Split observation
        jug_obs = flat_obs[:, :18]
        #particle_flat = flat_obs[:, 18:18+1047*9]# TODO: add back in for liquid data
        #particles = particle_flat.reshape((flat_obs.shape[0], 1047, 9))# TODO: add back in for liquid data

        #img_flat_start = 18 + 1047*9 # TODO: add back in for liquid data
        img_flat_start = 18 
        img_flat_end = -2
        img_flat = flat_obs[:, img_flat_start:img_flat_end]
        gaze = flat_obs[:, -2:]

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        #liquid_emb = ParticleEncoder()(particles)# TODO: add back in for liquid data
        img_emb = ImageGazeEncoder()(img_flat, gaze)

        # Shared representation
        #trunk = jnp.concatenate([jug_emb, liquid_emb, img_emb], axis=-1)# TODO: add back in for liquid data
        trunk = jnp.concatenate([jug_emb, img_emb], axis=-1)

        # Jug action head
        x_jug = nn.Dense(256)(trunk)
        x_jug = nn.relu(x_jug)
        x_jug = nn.Dense(256)(x_jug)
        x_jug = nn.relu(x_jug)
        jug_action = nn.tanh(nn.Dense(jug_action_dim)(x_jug))

        # Gaze action head (image-focused)
        x_gaze = nn.Dense(128)(img_emb)  # rely more directly on image
        x_gaze = nn.relu(x_gaze)
        gaze_action = nn.tanh(nn.Dense(gaze_action_dim)(x_gaze))

        # Concatenate final actions
        action = jnp.concatenate([jug_action, gaze_action], axis=-1)
        return action * self.action_scale + self.action_bias

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, flat_obs, action):
        # Split flat observation
        jug_obs = flat_obs[:, :18]
        #particle_flat = flat_obs[:, 18:]

        #img_flat_start = 18 + 1047*9 # TODO add back in 
        img_flat_start = 18 
        img_flat_end = -2
        img_flat = flat_obs[:, img_flat_start:img_flat_end]
        gaze = flat_obs[:, -2:]

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        #liquid_emb = ParticleEncoder()(particles) # TODO add back in
        img_emb = ImageGazeEncoder()(img_flat, gaze)

        # Combine with action
        #x = jnp.concatenate([jug_emb, liquid_emb, img_emb, action], axis=-1)
        x = jnp.concatenate([jug_emb, img_emb, action], axis=-1) # TODO add back in
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
"""


# for visual processing (without gaze)
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, flat_obs):
        jug_action_dim = self.action_dim
        #gaze_action_dim = 2  # last two dimensions for gaze

        # Split observation
        jug_obs = flat_obs[:, :18]
        #particle_flat = flat_obs[:, 18:18+1047*9]# TODO: add back in for liquid data
        #particles = particle_flat.reshape((flat_obs.shape[0], 1047, 9))# TODO: add back in for liquid data

        #img_flat_start = 18 + 1047*9 # TODO: add back in for liquid data
        img_flat_start = 18 
        img_flat = flat_obs[:, img_flat_start:]

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        #liquid_emb = ParticleEncoder()(particles)# TODO: add back in for liquid data
        img_emb = ImageEncoder()(img_flat)

        # Shared representation
        #trunk = jnp.concatenate([jug_emb, liquid_emb, img_emb], axis=-1)# TODO: add back in for liquid data
        trunk = jnp.concatenate([jug_emb, img_emb], axis=-1)
        #trunk = jnp.concatenate([jug_emb], axis=-1)

        # Jug action head
        x_jug = nn.Dense(256)(trunk)
        x_jug = nn.relu(x_jug)
        x_jug = nn.Dense(256)(x_jug)
        x_jug = nn.relu(x_jug)
        jug_action = nn.tanh(nn.Dense(jug_action_dim)(x_jug))

        # Concatenate final actions
        action = jnp.concatenate([jug_action], axis=-1)
        return action * self.action_scale + self.action_bias

class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, flat_obs, action):
        # Split flat observation
        jug_obs = flat_obs[:, :18]
        #particle_flat = flat_obs[:, 18:]

        #img_flat_start = 18 + 1047*9 # TODO add back in 
        img_flat_start = 18 
        img_flat = flat_obs[:, img_flat_start:]

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        #liquid_emb = ParticleEncoder()(particles) # TODO add back in
        img_emb = ImageEncoder()(img_flat)

        # Combine with action
        #x = jnp.concatenate([jug_emb, liquid_emb, img_emb, action], axis=-1)
        x = jnp.concatenate([jug_emb, img_emb, action], axis=-1) # TODO add back in
        #x = jnp.concatenate([jug_emb, action], axis=-1) # TODO add back in
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
"""

# without visual processing
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, flat_obs):
        # Split flat observation
        jug_obs = flat_obs[:, :19]
        particle_flat = flat_obs[:, 19:]
        #particles = particle_flat.reshape((flat_obs.shape[0], 1048, 128))
        particles = particle_flat.reshape((flat_obs.shape[0], 1047, 9))

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        liquid_emb = ParticleEncoder()(particles)

        # Combine and pass through actor MLP
        x = jnp.concatenate([jug_emb, liquid_emb], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        return x * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, flat_obs, action):
        # Split flat observation
        jug_obs = flat_obs[:, :19]
        particle_flat = flat_obs[:, 19:]
        #particles = particle_flat.reshape((flat_obs.shape[0], 1048, 128))
        particles = particle_flat.reshape((flat_obs.shape[0], 1047, 9))

        # Encode
        jug_emb = JugEncoder()(jug_obs)
        liquid_emb = ParticleEncoder()(particles)

        # Combine with action
        x = jnp.concatenate([jug_emb, liquid_emb, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
"""

class TrainState(TrainState):
    target_params: flax.core.FrozenDict


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)

    import uuid
    suffix = uuid.uuid4().hex[:6]
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}_{suffix}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    runs_folder = os.path.abspath(f"{args.output_dir}/runs/{run_name}")
    video_folder = os.path.abspath(f"{args.output_dir}/videos")

    writer = SummaryWriter(runs_folder)
    print(f"TensorBoard logs will be saved to: {runs_folder}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    json.dump(vars(args), open(os.path.join(runs_folder, "hyperparameters.json"), "w"), indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)

    reward_weights = {
        "target_level_wgt": args.target_level_wgt,
        "pt_cup_wgt": args.pt_cup_wgt,
        "pt_flow_wgt": args.pt_flow_wgt,
        "pt_spill_wgt": args.pt_spill_wgt,
        "action_cost": args.action_cost,
        "jug_resting_wgt": args.jug_resting_wgt,
        "jug_velocity_wgt": args.jug_velocity_wgt,
        "distance_wgt": args.distance_wgt,
        "fovea_radius": args.fovea_radius,
        "time_penalty": args.time_penalty,
    }

    env_kwargs = {
        "data_path": args.data_path,
        "target_particles_path": args.target_particles_path,
        "reward_weights": reward_weights,
        }

    if "Isaac" not in args.env_id:
        env_kwargs["gnn_model_path"] = args.gnn_model_path

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, env_kwargs, video_folder=video_folder)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key = jax.random.split(key, 2)
        
        # with signal independent noise
        clipped_noise = (
            jnp.clip(
                (jax.random.normal(noise_key, actions.shape) * args.policy_noise),
                -args.noise_clip,
                args.noise_clip,
            )
            * actor.action_scale
        ) # this is signal-independent noise (independent of action magnitude)
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_observations) + clipped_noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        """
        # with action-dependent noise
        action = actor.apply(actor_state.target_params, next_observations)

        # variance matching
        abs_action = jnp.abs(action)
        scale = abs_action / (jnp.mean(abs_action) + 1e-6)

        noise = (
            jax.random.normal(noise_key, action.shape)
            * args.policy_noise
            * scale
        )
    
        # different scaling attempt
        #scale = jnp.maximum(1.0, jnp.abs(action))
        #noise = jax.random.normal(noise_key, action.shape) * args.policy_noise * scale

        noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    
        next_state_actions = jnp.clip(
            action + noise,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )
        """

        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
        qf2_next_target = qf.apply(qf2_state.target_params, next_observations, next_state_actions).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (rewards + (1 - terminations) * args.gamma * (min_qf_next_target)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        (qf2_loss_value, qf2_a_values), grads2 = jax.value_and_grad(mse_loss, has_aux=True)(qf2_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, args.tau)
        )
        return actor_state, (qf1_state, qf2_state), actor_loss_value

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        start_time_step = time.time()
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            
            # deterministic action from actor
            # signal-independent exploration noise 
            actions_det = actor.apply(actor_state.params, obs)
            actions_det = np.array(jax.device_get(actions_det))

            expl_noise = np.random.normal(0, max_action * args.exploration_noise, size=envs.single_action_space.shape)
            """actions = np.array(
                [
                    (
                        jax.device_get(actions)[0]
                        + np.random.normal(0, max_action * args.exploration_noise, size=envs.single_action_space.shape)
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )"""
            # signal-dependent noise
            noise_scale = (
                args.min_signal_noise
                + args.signal_noise * (np.abs(actions_det) / max_action)
            )

            # Clip noise scale for stability
            noise_scale = np.clip(
                noise_scale,
                args.min_signal_noise,
                args.max_signal_noise,
            )

            execution_noise = np.random.normal(
                loc=0.0,
                scale=noise_scale,
                size=actions_det.shape,
            )
            """if global_step < args.learning_starts + args.exploration_warmup_steps:
                # Signal-INDEPENDENT noise during warmup
                noise = np.random.normal(
                    loc=0.0,
                    scale=args.initial_exploration_noise,
                    size=actions_det.shape,
                )
            else:
                # Signal-DEPENDENT noise after warmup
                noise_scale = (
                    args.min_exploration_noise
                    + args.exploration_noise * np.abs(actions_det)#np.sqrt(np.abs(actions_det))
                )

                # Clip noise scale for stability
                noise_scale = np.clip(
                    noise_scale,
                    args.min_exploration_noise,
                    args.max_exploration_noise,
                )

                noise = np.random.normal(
                    loc=0.0,
                    scale=noise_scale,
                    size=actions_det.shape,
                )"""

            actions = actions_det + expl_noise + execution_noise

            actions = actions.clip(
                envs.single_action_space.low,
                envs.single_action_space.high,
            )

            writer.add_scalar("charts/total_noise_actions", expl_noise + execution_noise, global_step)
            writer.add_scalar("charts/exploration_noise", expl_noise, global_step)
            writer.add_scalar("charts/signal_noise", execution_noise, global_step)

        # CHANGED: original version used final_info to detect done which was removed in gymnasium 1.0.0
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Handle both single and multiple environments
        is_vectorized = isinstance(infos, (list, tuple))  # True if infos is a list

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if is_vectorized:
            for i, (terminated, truncated) in enumerate(zip(terminations, truncations)):
                if (terminated or truncated) and "episode" in infos[i]:
                    print(f"global_step={global_step}, episodic_return={infos[i]['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", infos[i]['episode']['r'], global_step)
                    writer.add_scalar("charts/episodic_length", infos[i]['episode']['l'], global_step)
                    writer.add_scalar("charts/fill_level", infos[i]['current_fill_level'], global_step)
                    break # only log the first finished episode for consistency
        else:
            if (terminations or truncations) and "episode" in infos:
                print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                writer.add_scalar("charts/fill_level", infos['current_fill_level'], global_step)

        # TRY NOT TO MODIFY: save data to replay buffer
        real_next_obs = next_obs.copy()
        if is_vectorized:
            for idx, trunc in enumerate(truncations):
                if trunc and "terminal_observation" in infos[idx]:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
        else:
            if truncations and "terminal_observation" in infos:
                real_next_obs[0] = infos["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print("step time: ", time.time() - start_time_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                writer.add_scalar("charts/step_time", time.time() - start_time_step, global_step)
    envs.close()# moved up here to make sure that there is no conflict with make_env in eval
    if args.save_model:
        model_path = f"{runs_folder}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        actor_state.params,
                        qf1_state.params,
                        qf2_state.params,
                    ]
                )
            )
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.td3_jax_eval import evaluate

        eval_rewards_folder = os.path.abspath(f"{args.output_dir}/saved_rewards")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=20,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            exploration_noise=args.exploration_noise,
            signal_noise=args.signal_noise,
            min_signal_noise=args.min_signal_noise,
            max_signal_noise=args.max_signal_noise,
            env_kwargs=env_kwargs,
            video_folder=video_folder,
            rewards_folder=eval_rewards_folder)
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "TD3", f"{runs_folder}", f"{video_folder}/{run_name}-eval",)

    writer.close()
