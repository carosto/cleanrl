# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_action_jaxpy
import functools
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import pickle
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

from learning_to_simulate_pouring import learned_simulator, model_utils, pouring_env
import haiku as hk
import jax.tree_util as tree

import open3d as o3d

from scipy.spatial.transform import Rotation as R


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

def unpack_obs(flat_obs):
    # Adjust sizes according to your environment
    BASE_SIZE = 19
    LIQ_CUR_SIZE = 1047 * 9
    JUG_BUF_SIZE = 36
    PT_BUF_SIZE  = 6 * 1047 * 3
    
    jug_obs = flat_obs[:, :BASE_SIZE]
    
    start = BASE_SIZE + LIQ_CUR_SIZE
    end = start + JUG_BUF_SIZE
    jug_buffer = flat_obs[:, start:end].reshape(-1, 6, 6)
    
    start = end
    end = start + PT_BUF_SIZE
    pt_buffer = flat_obs[:, start:end].reshape(-1, 6, 1047, 3)
    
    return jug_obs, jug_buffer, pt_buffer

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

# Image Encoder for multiple consecutive frames  
class ImageEncoderMultiple(nn.Module):
    num_frames: int = 4

    @nn.compact
    def __call__(self, img):
        """
        img: (batch, num_frames * H * W)
        """

        batch_size, flat_dim = img.shape
        x = img.reshape((batch_size, 64, 64, self.num_frames))  # ← key change

        # CNN
        x = nn.Conv(32, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        x = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        return x

class ImageEncoderMultiple3D(nn.Module):
    num_frames: int = 4

    @nn.compact
    def __call__(self, img):
        """
        img: (batch, num_frames * H * W)
        """

        batch_size, flat_dim = img.shape
        H = 64
        W = 64

        # Reshape to (batch, time, height, width, channels)
        x = img.reshape((batch_size, self.num_frames, H, W, 1))

        # 3D CNN layers (time × space)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 5, 5),
            strides=(1, 2, 2),
            padding="SAME"
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3, 3, 3),
            strides=(1, 2, 2),
            padding="SAME"
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=128,
            kernel_size=(3, 3, 3),
            strides=(1, 2, 2),
            padding="SAME"
        )(x)
        x = nn.relu(x)

        # Global average pooling over time + space
        x = jnp.mean(x, axis=(1, 2, 3))  # (batch, features)

        x = nn.Dense(128)(x)
        x = nn.relu(x)

        return x

class GNNEncoder():
    INPUT_SEQUENCE_LENGTH : int = 6
    def __init__(self, gnn_model_path, data_path):
        self.gnn_model_path = gnn_model_path
        self.data_path = data_path

        self.max_time = 1

        self.metadata_model = self._read_metadata(self.data_path)
        self.collision_mesh_info_list = self.metadata_model["collision_mesh"]
        self.mesh_pt_type_list = [
            z[1] for z in self.collision_mesh_info_list
        ]  # mesh pt type for handling in v_o

        self.connectivity_radius = self.metadata_model["default_connectivity_radius"]
        self.max_n_liq_node_per_graph = int(
            self.metadata_model["max_n_liq_node"]
        )  #  can be read from position as well.. ignore for now
        self.max_edges_l_per_graph = int(self.metadata_model["max_n_edge_l"])
        self.max_edges_m_per_graph = int(self.metadata_model["max_n_edge_m"])

        self.max_nodes_edges_info = [
            len(self.collision_mesh_info_list),
            self.max_n_liq_node_per_graph,
            self.max_edges_l_per_graph,
            self.max_edges_m_per_graph,
        ]

        self.jug_name = self.metadata_model["collision_mesh"][0][0]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.jug_path = os.path.join(BASE_DIR, "ObjectFiles", self.jug_name)
        # self.jug_path = f'./ObjectFiles/{self.jug_name}'

        self.jug_vertices = self._read_mesh_vertices(self.jug_path)
        self.config_dict = {
            "jug_vertices_count": len(self.jug_vertices),
        }

         # set up stuff from pouring_env
        self.particle_types = pouring_env.get_particle_types(self.data_path)
        self.initial_features = pouring_env.build_initial_features(
            self.data_path, self.mesh_pt_type_list
        )

        # saved trajectories for the run (0 -> max timesteps)
        self.state_liq_pos_full_traj = None
        self.state_mesh_node_pos_full_traj = None
        self.state_mesh_pose_full_traj = None

        self._load_model()

        # initialize features for model input (liquid position, object positions,....)
        self.state_liq_pos_full_traj = jnp.zeros(
            (self.max_time, self.max_n_liq_node_per_graph, self.INPUT_SEQUENCE_LENGTH, 3)
        )
        self.state_mesh_node_pos_full_traj = jnp.zeros(
            (self.max_time, self.max_edges_m_per_graph, self.INPUT_SEQUENCE_LENGTH, 3)
        )
        self.state_mesh_pose_full_traj = jnp.zeros(
            (self.max_time, 3, self.INPUT_SEQUENCE_LENGTH, 6)
        )  # 3 objects

        self.state_liq_pos_full_traj = self.state_liq_pos_full_traj.at[0].set(
            self.initial_features["liq_position"]
        )
        self.state_mesh_node_pos_full_traj = self.state_mesh_node_pos_full_traj.at[
            0
        ].set(self.initial_features["mesh_position"][0])
        self.state_mesh_pose_full_traj = self.state_mesh_pose_full_traj.at[0].set(
            self.initial_features["mesh_pose"][0]
        )
        
        self._init_input_features()
    
    def _load_model(self):
        # load the model for the first time
        self.model, self.network_params = self._load_gnn_model(
            self.gnn_model_path,
            self.connectivity_radius,
            self.collision_mesh_info_list,
            self.max_nodes_edges_info,
        )
        self._fast_apply = jax.jit(lambda p, s, g: self.model.apply(p, s, g))
        self._fast_model_step = jax.jit(
            lambda j, p: self._apply_gnn_processing_step(j, p)
        )
        print("Model loaded successfully.")

    def _init_input_features(self):
        # requires 6 consecutive positions for liquid and jug (current + 5 past timesteps)
        state_liq_pos_cur = self.state_liq_pos_full_traj[0]
        state_mesh_node_pos_cur = self.state_mesh_node_pos_full_traj[0]
        state_mesh_pose_cur = self.state_mesh_pose_full_traj[0]

        # put the current state to the model and the encoded + processed liquid values
        input_features = {
            "liq_position": state_liq_pos_cur,
            "mesh_position": state_mesh_node_pos_cur[jnp.newaxis],
            "mesh_pose": state_mesh_pose_cur[jnp.newaxis],
            "particle_type": self.particle_types,
            "particle_type_obj": jnp.array(self.mesh_pt_type_list),
        }
        self.input_graph = pouring_env.build_graph(
            input_features, self.max_n_liq_node_per_graph, self.max_edges_l_per_graph
        )

    def _apply_gnn_processing_step(self, past_jug_poses, past_particle_positions):
        """
        Apply the GNN model processing step to the input graph.
        Args:
            updated_jug_pose: Updated jug pose.
            input_graph: Input graph for the model.
        Returns:
            input_graph: Modified input graph after processing.
            latent_liquid_representation: Latent representation of liquid particles.
        """
        """with open("old_input_graph.pkl", "wb") as f:
            pickle.dump(self.input_graph.nodes, f)"""
        input_graph_local = tree.tree_map(lambda x: x.copy(), self.input_graph)
        prev_liq_position = input_graph_local.nodes["liq_position"]
        prev_mesh_position = input_graph_local.nodes["mesh_position"]
        prev_mesh_pose = input_graph_local.nodes["mesh_pose"]

        # fill up history with the transformed jug positions, oldest to newest
        past_jug_poses = jnp.asarray(past_jug_poses).reshape(-1, 6)
        jug_nodes_all = jax.vmap(
                            lambda pose: pouring_env.transform_mesh_to_local_coordinates(
                                self.jug_vertices,
                                pose[:3],
                                ref_orientation=pose[3:],
                            )
                        )(past_jug_poses)
        
        next_mesh_position = prev_mesh_position[0]   # (N, L)
        next_mesh_pose = prev_mesh_pose[0]           # (P, L)

        next_mesh_position = next_mesh_position.at[
                : self.config_dict["jug_vertices_count"], :
            ].set(jug_nodes_all.T)
        
        next_mesh_pose = next_mesh_pose.at[0, :].set(past_jug_poses)

        next_mesh_position_padded = prev_mesh_position.at[0].set(next_mesh_position)
        next_mesh_pose_padded = prev_mesh_pose.at[0].set(next_mesh_pose)

        input_graph_local.nodes["mesh_position"] = next_mesh_position_padded
        input_graph_local.nodes["mesh_pose"] = next_mesh_pose_padded

        # do the updating for the liquid positions (1047x3 for each timestep -> what about that last particle?
        # past_particle_positions = (batch, time, n_particles, xyz)
        """# past_particle_positions = (batch, time, n_particles, xyz)
        past_particle_positions = jnp.stack(past_particle_positions, axis=0)

        total_nodes = jnp.sum(self.input_graph.n_node[:-1])

        # Flatten batch + particle → nodes
        # New shape → (nodes, time, xyz)
        past_particle_positions = past_particle_positions.reshape(
            -1,
            past_particle_positions.shape[1],
            past_particle_positions.shape[3]
        )

        # Ensure sequence dimension ordering is (nodes, time, xyz)
        past_particle_positions = jnp.transpose(past_particle_positions, (0, 1, 2))

        # Copy previous tensor
        next_pos_seq = prev_liq_position

        # Assign only valid nodes
        next_pos_seq = next_pos_seq.at[:total_nodes, :, :].set(
            past_particle_positions[:total_nodes, :, :]
        )

        # Optional strict masking
        node_padding_mask = jnp.arange(prev_liq_position.shape[0]) < total_nodes

        next_pos_seq = jnp.where(
            node_padding_mask[:, None, None],
            next_pos_seq,
            prev_liq_position
        )"""
        # past_particle_positions = (batch, time, n_particles, xyz) 
        past_particle_positions = jnp.stack(past_particle_positions, axis=0) 

        total_nodes = jnp.sum(input_graph_local.n_node[:-1]) 
        node_padding_mask = jnp.arange(prev_liq_position.shape[0]) < total_nodes 

        # transpose to match (nodes, time, 3) # from (L, N, 3) → (N, L, 3) 
        past_particle_positions = jnp.transpose(past_particle_positions, (1, 0, 2)) 

        # start from previous tensor (keeps padding untouched initially) 
        next_pos_seq = prev_liq_position # fill only valid (non-padded) nodes

        mask = jnp.arange(past_particle_positions.shape[0]) < total_nodes

        past_particles_masked = jnp.where(
            mask[:, None, None],
            past_particle_positions,
            jnp.zeros_like(past_particle_positions)
        )

        next_pos_seq = jax.lax.dynamic_update_slice(
            next_pos_seq,
            past_particles_masked,
            (0, 0, 0)
        )

        # alternatively using mask (if you prefer strict masking) 
        next_pos_seq = jnp.where( 
            node_padding_mask[:, None, None], next_pos_seq, prev_liq_position
        )

        # update graph
        input_graph_local.nodes["liq_position"] = next_pos_seq

        """with open("new_input_graph.pkl", "wb") as f:
            pickle.dump(self.input_graph.nodes, f)"""

        (_, dict_latent_graphs), _ = self._fast_apply(
            self.network_params["params"], self.network_params["state"], input_graph_local
        )

        latent_liquid_representation = dict_latent_graphs[
            "latent_graph_before_decoding"
        ].nodes[
            "v_l"
        ]  # latent representation of liquid particles after processing step of model
        del input_graph_local
        return latent_liquid_representation

    def _load_gnn_model(
        self,
        model_path,
        connectivity_radius,
        collision_mesh_info_list,
        max_nodes_edges_info,
    ):
        """
        Load the GNN model for the environment.
        Returns:
            model: Loaded GNN model.
        """
        graph_network_kwargs = dict(
            include_sent_messages_in_node_update=False,
            latent_size=128,
            mlp_hidden_size=128,
            mlp_num_hidden_layers=2,
            num_message_passing_steps=10,
            node_types=["v_l", "v_m", "v_o"],
            edge_types=["e_l", "e_mo", "e_om", "e_ol"],
            use_layer_norm=True,
        )
        model_kwargs = {"graph_network_kwargs": graph_network_kwargs}
        flatten_kwargs = {"apply_normalization": True}

        flatten_fn = functools.partial(model_utils.flatten_features, **flatten_kwargs)
        haiku_model = functools.partial(
            learned_simulator.LearnedSimulator,
            connectivity_radius=connectivity_radius,
            collision_mesh_info_lists=[
                collision_mesh_info_list,
            ],
            max_nodes_edges_info=max_nodes_edges_info,
            flatten_features_fn=flatten_fn,
            **model_kwargs,
        )

        model = hk.without_apply_rng(
            hk.transform_with_state(lambda x: haiku_model()(x))
        )
        network_params = self._load_network_params(model_path)["network"]
        print(f"Loading gnn model at path: {model_path}")

        return model, network_params

    def _load_network_params(self, model_path):
        """
        Load the network parameters from a file.
        Args:
            model_path: Path to the model file.
        Returns:
            network_params: Loaded network parameters.
        """
        # taken from MPC file (-> see load_model)
        with open(model_path, "rb") as f:
            numpy_params = pickle.load(f)
        return jax.tree_util.tree_map(lambda x: jnp.array(x), numpy_params)

    def _read_metadata(self, data_path):
        with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
            return json.loads(fp.read())

    def _read_mesh_vertices(self, mesh_path):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        scale_factor = 10  # isaac is 10x reality
        return jnp.array(vertices * scale_factor)

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
"""
# visual processing with multiple frames
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, flat_obs):

        # Expect 4 stacked frames
        img_flat = flat_obs   # no jug anymore

        # Encode stacked frames
        img_emb = ImageEncoderMultiple(num_frames=4)(img_flat)

        trunk = img_emb

        x = nn.Dense(256)(trunk)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        action = nn.tanh(nn.Dense(self.action_dim)(x))

        return action * self.action_scale + self.action_bias

class QNetwork(nn.Module):

    @nn.compact
    def __call__(self, flat_obs, action):

        # Entire observation is stacked frames now
        img_flat = flat_obs

        # Encode frames
        img_emb = ImageEncoderMultiple(num_frames=4)(img_flat)

        # Combine with action
        x = jnp.concatenate([img_emb, action], axis=-1)

        # Q MLP
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        x = nn.Dense(256)(x)
        x = nn.relu(x)

        q = nn.Dense(1)(x)
        return q

"""
# without visual processing
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, jug_obs, gnn_latents):
        jug_emb = JugEncoder()(jug_obs)
        liquid_emb = ParticleEncoder()(gnn_latents) # gnn_latents shape: (batch, 1048, 128)

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
    def __call__(self, jug_obs, gnn_latents, action):
        jug_emb = JugEncoder()(jug_obs)
        liquid_emb = ParticleEncoder()(gnn_latents)

        x = jnp.concatenate([jug_emb, liquid_emb, action], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


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
        "clear_cache_bool" : False,
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

    """gnn_model_path = '/home/carola/masterthesis/pouring_env/learning_to_simulate_pouring/models/sdf_fullpose_lessPt_2412/model_checkpoint_globalstep_1770053.pkl'
    data_path = "/home/carola/masterthesis/Pouring_mpc_1D_1902"#'/shared_data/Pouring_mpc_1D_1902/'

    gnn = GNNEncoder(gnn_model_path=gnn_model_path, data_path=data_path)"""

    # Before entering the training loop, initialize the GNN
    gnn = GNNEncoder(gnn_model_path=args.gnn_model_path, data_path=args.data_path)
    base_graph = gnn.input_graph
    gnn_params = gnn.network_params["params"]
    gnn_state = gnn.network_params["state"]

    # --- Chunking-Version von batched_gnn_forward ---
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
        
        # lax.scan ist wesentlich schneller als lax.map für solche Operationen!
        def scan_step(carry, chunk_args):
            j_chunk, p_chunk = chunk_args
            return None, chunk_vmap(j_chunk, p_chunk)
            
        _, out_chunks = jax.lax.scan(scan_step, None, (jug_chunks, pt_chunks))
        
        return out_chunks.reshape((batch_size, *out_chunks.shape[2:]))

    # --- NEU: Initiale Beobachtung entpacken und GNN Latents berechnen ---
    obs_jug_init, obs_jug_buf_init, obs_pt_buf_init = unpack_obs(jnp.array(obs))
    init_gnn_latents = jax.lax.stop_gradient(batched_gnn_forward(obs_jug_buf_init, obs_pt_buf_init))
    # ---------------------------------------------------------------------

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        # BEIDE initialen Variablen übergeben:
        params=actor.init(actor_key, obs_jug_init, init_gnn_latents),
        target_params=actor.init(actor_key, obs_jug_init, init_gnn_latents),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    
    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        # Auch das QNetwork braucht jetzt beide Variablen plus die Action:
        params=qf.init(qf1_key, obs_jug_init, init_gnn_latents, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs_jug_init, init_gnn_latents, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs_jug_init, init_gnn_latents, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs_jug_init, init_gnn_latents, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        obs_jug: jnp.ndarray,           # <-- Neu
        gnn_latents: jnp.ndarray,       # <-- Neu
        actions: np.ndarray,
        next_jug: jnp.ndarray,          # <-- Neu
        next_gnn_latents: jnp.ndarray,  # <-- Neu
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, noise_key = jax.random.split(key, 2)
        
        # HIER KEIN GNN MEHR AUFRUFEN! Einfach direkt die übergebenen Variablen nutzen:
        clipped_noise = jnp.clip((jax.random.normal(noise_key, actions.shape) * args.policy_noise), -args.noise_clip, args.noise_clip) * actor.action_scale
        next_state_actions = jnp.clip(
            actor.apply(actor_state.target_params, next_jug, next_gnn_latents) + clipped_noise,
            envs.single_action_space.low, envs.single_action_space.high
        )

        qf1_next_target = qf.apply(qf1_state.target_params, next_jug, next_gnn_latents, next_state_actions).reshape(-1)
        qf2_next_target = qf.apply(qf2_state.target_params, next_jug, next_gnn_latents, next_state_actions).reshape(-1)
        min_qf_next_target = jnp.minimum(qf1_next_target, qf2_next_target)
        next_q_value = (rewards + (1 - terminations) * args.gamma * (min_qf_next_target)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, obs_jug, gnn_latents, actions).squeeze()
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
        obs_jug: jnp.ndarray,     # <-- Neu
        gnn_latents: jnp.ndarray, # <-- Neu
    ):
        # HIER EBENFALLS KEIN GNN MEHR!
        def actor_loss(params):
            actions = actor.apply(params, obs_jug, gnn_latents)
            return -qf.apply(qf1_state.params, obs_jug, gnn_latents, actions).mean()

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
            obs_jug, obs_jug_buf, obs_pt_buf = unpack_obs(jnp.array(obs))
            gnn_latents = jax.lax.stop_gradient(batched_gnn_forward(obs_jug_buf, obs_pt_buf))
            actions_det = actor.apply(actor_state.params, obs_jug, gnn_latents)
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

            """
            # original Normal noise (over- and undershoot)
            execution_noise = np.random.normal(
                loc=0.0,
                scale=noise_scale,
                size=actions_det.shape,
            )"""

            # Sample half-normal (always positive)
            epsilon = np.abs(
                np.random.normal(loc=0.0, scale=noise_scale, size=actions_det.shape)
            )

            # Add noise in direction of action sign (increases amplitude only)
            actions_exec = actions_det + np.sign(actions_det) * epsilon

            # Execution noise
            execution_noise = actions_exec - actions_det
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
            # --- Dein aktuelles Sampling ---
            data = rb.sample(args.batch_size)
            obs_batch = jax.device_put(data.observations.numpy())
            next_obs_batch = jax.device_put(data.next_observations.numpy())
            actions_batch = jax.device_put(data.actions.numpy())
            rewards_batch = jax.device_put(data.rewards.flatten().numpy())
            dones_batch = jax.device_put(data.dones.flatten().numpy())

            # --- NEU: Beobachtungen entpacken und GNN anwenden ---
            # 1. Aktuelle Beobachtungen entpacken und Latents berechnen
            obs_jug, obs_jug_buf, obs_pt_buf = unpack_obs(obs_batch)
            gnn_latents = jax.lax.stop_gradient(batched_gnn_forward(obs_jug_buf, obs_pt_buf))
            
            # 2. Nächste Beobachtungen entpacken und Latents berechnen
            next_jug, next_jug_buf, next_pt_buf = unpack_obs(next_obs_batch)
            next_gnn_latents = jax.lax.stop_gradient(batched_gnn_forward(next_jug_buf, next_pt_buf))
            # -----------------------------------------------------

            # --- 3. Den Critic updaten ---
            (qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value), (qf1_a_values, qf2_a_values), key = update_critic(
                actor_state, 
                qf1_state, 
                qf2_state, 
                obs_jug,             # Nutzt den isolierten Krug-Zustand
                gnn_latents,         # Nutzt die vorberechneten GNN-Latents
                actions_batch, 
                next_jug,            # Nächster Krug-Zustand
                next_gnn_latents,    # Nächste GNN-Latents
                rewards_batch, 
                dones_batch, 
                key
            )

            # --- 4. Den Actor updaten (verzögert) ---
            if global_step % args.policy_frequency == 0:
                actor_state, (qf1_state, qf2_state), actor_loss_value = update_actor(
                    actor_state, 
                    qf1_state, 
                    qf2_state, 
                    obs_jug,         # Wir recyclen die Krug-Daten...
                    gnn_latents      # ...und die GNN-Latents! (Das spart massiv Zeit)
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
