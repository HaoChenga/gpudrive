"""Base Gym Environment that interfaces with the GPU Drive simulator."""

from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
import torch
import copy
import gpudrive
import imageio
from itertools import product

from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.base_env import GPUDriveGymEnv
from pygpudrive.env import constants


class GPUDriveTorchEnv(GPUDriveGymEnv):
    """Torch Gym Environment that interfaces with the GPU Drive simulator."""

    def __init__(
        self,
        config,
        scene_config,
        max_cont_agents,
        device="cuda",
        action_type="discrete",
        render_config: RenderConfig = RenderConfig(),
    ):
        # Initialization of environment configurations
        self.config = config
        self.num_worlds = scene_config.num_scenes
        self.max_cont_agents = max_cont_agents
        self.device = device
        self.render_config = render_config

        # Environment parameter setup
        self.params = self._setup_environment_parameters()

        # Initialize simulator with parameters
        self.sim = self._initialize_simulator(self.params, scene_config)
        # Controlled agents setup
        self.cont_agent_mask = self.get_controlled_agents_mask()
        self.max_agent_count = self.cont_agent_mask.shape[1]
        self.num_valid_controlled_agents_across_worlds = (
            self.cont_agent_mask.sum().item()
        )

        # Setup action and observation spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.get_obs()[0].shape[-1],)
        )
        self._setup_action_space(action_type)
        self.info_dim = 5  # Number of info features
        self.episode_len = self.config.episode_len
        # Rendering setup
        self.visualizer = self._setup_rendering()


    def reset(self):
        """Reset the worlds and return the initial observations."""
        self.sim.reset(list(range(self.num_worlds)))
        return self.get_obs()

    def get_dones(self):
        return self.sim.done_tensor().to_torch().squeeze(dim=2).to(torch.float)

    def get_infos(self):
        return (
            self.sim.info_tensor()
            .to_torch()
            .squeeze(dim=2)
            .to(torch.float)
            .to(self.device)
        )

    def get_rewards(
        self, collision_weight=0, goal_achieved_weight=1.0, off_road_weight=0
    ):
        """Obtain the rewards for the current step.
        By default, the reward is a weighted combination of the following components:
        - collision
        - goal_achieved
        - off_road

        The importance of each component is determined by the weights.
        """
        if self.config.reward_type == "sparse_on_goal_achieved":
            return self.sim.reward_tensor().to_torch().squeeze(dim=2)

        elif self.config.reward_type == "weighted_combination":
            # Return the weighted combination of the reward components
            info_tensor = self.sim.info_tensor().to_torch()
            off_road = info_tensor[:, :, 0].to(torch.float)

            # True if the vehicle collided with another road object
            # (i.e. a cyclist or pedestrian)
            collided = info_tensor[:, :, 1:3].to(torch.float).sum(axis=2)
            goal_achieved = info_tensor[:, :, 3].to(torch.float)

            weighted_rewards = (
                collision_weight * collided
                + goal_achieved_weight * goal_achieved
                + off_road_weight * off_road
            )

            return weighted_rewards

    def step_dynamics(self, actions):
        if actions is not None:
            self._apply_actions(actions)
        self.sim.step()

    def _apply_actions(self, actions):
        """Apply the actions to the simulator."""

        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
            or self.config.dynamics_model == "delta_local"
        ):
            if actions.dim() == 2:  # (num_worlds, max_agent_count)
                # Map action indices to action values if indices are provided
                actions = (
                    torch.nan_to_num(actions, nan=0).long().to(self.device)
                )
                action_value_tensor = self.action_keys_tensor[actions]

            elif actions.dim() == 3:
                if actions.shape[2] == 1:
                    actions = actions.squeeze(dim=2).to(self.device)
                    action_value_tensor = self.action_keys_tensor[actions]
                elif (
                    actions.shape[2] == 3
                ):  # Assuming we are given the actual action values 
                    # (acceleration, steering, heading)
                    action_value_tensor = actions.to(self.device)
            else:
                raise ValueError(f"Invalid action shape: {actions.shape}")

        else:
            action_value_tensor = actions.to(self.device)

        # Feed the action values to gpudrive
        self._copy_actions_to_simulator(action_value_tensor)

    def _copy_actions_to_simulator(self, actions):
        """Copy the provived actions to the simulator."""
        if (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            # Action space: (acceleration, steering, heading)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "delta_local":
            # Action space: (dx, dy, dyaw)
            self.sim.action_tensor().to_torch()[:, :, :3].copy_(actions)
        elif self.config.dynamics_model == "state":
            # Following the StateAction struct in types.hpp
            # Need to provide: (x, y, z, yaw, velocity x, vel y, vel z, ang_vel_x, ang_vel_y, ang_vel_z)
            self.sim.action_tensor().to_torch()[:, :, :10].copy_(actions)
        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

    def _set_discrete_action_space(self) -> None:
        """Configure the discrete action space based on dynamics model."""
        products = None

        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            products = product(self.dx, self.dy, self.dyaw)
        elif (
            self.config.dynamics_model == "classic"
            or self.config.dynamics_model == "bicycle"
        ):
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            products = product(
                self.accel_actions, self.steer_actions, self.head_actions
            )
        elif self.config.dynamics_model == "state":
            self.x = self.config.x.to(self.device)
            self.y = self.config.y.to(self.device)
            self.yaw = self.config.yaw.to(self.device)
            self.vx = self.config.vx.to(self.device)
            self.vy = self.config.vy.to(self.device)

        else:
            raise ValueError(
                f"Invalid dynamics model: {self.config.dynamics_model}"
            )

        # Create a mapping from action indices to action values
        self.action_key_to_values = {}
        self.values_to_action_key = {}
        if products is not None:
            for action_idx, (action_1, action_2, action_3) in enumerate(
                products
            ):
                self.action_key_to_values[action_idx] = [
                    action_1.item(),
                    action_2.item(),
                    action_3.item(),
                ]
                self.values_to_action_key[
                    round(action_1.item(), 3),
                    round(action_2.item(), 3),
                    round(action_3.item(), 3),
                ] = action_idx

            self.action_keys_tensor = torch.tensor(
                [
                    self.action_key_to_values[key]
                    for key in sorted(self.action_key_to_values.keys())
                ]
            ).to(self.device)

            return Discrete(n=int(len(self.action_key_to_values)))
        else:
            return Discrete(n=1)

    def _set_continuous_action_space(self) -> None:
        """Configure the continuous action space."""
        if self.config.dynamics_model == "delta_local":
            self.dx = self.config.dx.to(self.device)
            self.dy = self.config.dy.to(self.device)
            self.dyaw = self.config.dyaw.to(self.device)
            action_1 = self.dx.clone().cpu().numpy()
            action_2 = self.dy.clone().cpu().numpy()
            action_3 = self.dyaw.clone().cpu().numpy()
        else:
            self.steer_actions = self.config.steer_actions.to(self.device)
            self.accel_actions = self.config.accel_actions.to(self.device)
            self.head_actions = torch.tensor([0], device=self.device)
            action_1 = self.steer_actions.clone().cpu().numpy()
            action_2 = self.accel_actions.clone().cpu().numpy()
            action_3 = self.head_actions.clone().cpu().numpy()

        action_space = Tuple(
            (
                Box(action_1.min(), action_1.max(), shape=(1,)),
                Box(action_2.min(), action_2.max(), shape=(1,)),
                Box(action_3.min(), action_3.max(), shape=(1,)),
            )
        )
        return action_space

    def get_obs(self):
        """Get observation: Combine different types of environment information into a single tensor.

        Returns:
            torch.Tensor: (num_worlds, max_agent_count, num_features)
        """

        # EGO STATE [modify to (speed, length, width, rel_goal_x, rel_goal_y, global_goal_x, global_goal_y, collision, global_start_x, global_start_y)]
        if self.config.ego_state:
            ego_states_unprocessed = (
                self.sim.self_observation_tensor().to_torch()
            )

            if self.config.norm_obs:
                ego_states = self.normalize_ego_state(ego_states_unprocessed)
            else:
                ego_states = ego_states_unprocessed
        else:
            ego_states = torch.Tensor().to(self.device)


        ego_states_full = ego_states.clone()
        obs_filter_ids = [0, 1, 2, 3, 4, 7] # [speed, length, width, rel_goal_x, rel_goal_y, collision] 
        ego_states = ego_states[:,:,obs_filter_ids]

        #map_obs=self.sim.map_observation_tensor().to_torch()[0,:5,0:2]
        # PARTNER OBSERVATIONS
        if self.config.partner_obs:
            partner_observations = (
                self.sim.partner_observations_tensor().to_torch()
            )
            if self.config.norm_obs:  # Normalize observations and then flatten
                partner_observations = self.normalize_and_flatten_partner_obs(
                    partner_observations
                )
            else:  # Flatten along the last two dimensions
                partner_observations = partner_observations.flatten(
                    start_dim=2
                )
        else:
            partner_observations = torch.Tensor().to(self.device)

        # ROAD MAP OBSERVATIONS
        if self.config.road_map_obs:

            road_map_observations_unprocessed = (
                self.sim.agent_roadmap_tensor().to_torch()
            )

            if self.config.norm_obs:
                road_map_observations = self.normalize_and_flatten_map_obs(
                    road_map_observations_unprocessed
                )
            else:
                road_map_observations = (
                    road_map_observations_unprocessed.flatten(start_dim=2)
                )
        else:
            road_map_observations = torch.Tensor().to(self.device)

        # Combine the observations
        obs_filtered = torch.cat(
            (
                ego_states, #6
                partner_observations,
                road_map_observations,#2600
            ),
            dim=-1,
        )

        return obs_filtered, ego_states_full

    def get_controlled_agents_mask(self):
        """Get the control mask."""
        return (self.sim.controlled_state_tensor().to_torch() == 1).squeeze(
            axis=2
        )

    def normalize_ego_state(self, state):
        """Normalize ego state features."""

        # Speed, vehicle length, vehicle width
        state[:, :, 0] /= constants.MAX_SPEED
        state[:, :, 1] /= constants.MAX_VEH_LEN
        state[:, :, 2] /= constants.MAX_VEH_WIDTH

        # Relative goal coordinates
        state[:, :, 3] = self.normalize_tensor(
            state[:, :, 3],
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )
        state[:, :, 4] = self.normalize_tensor(
            state[:, :, 4],
            # do the same
            constants.MIN_REL_GOAL_COORD,
            constants.MAX_REL_GOAL_COORD,
        )

        if state.shape[-1] > 5:
            # Global goal coordinates
            state[:, :, 5] = self.normalize_tensor(
                state[:, :, 5],
                constants.MIN_GLOBAL_COORD,
                constants.MAX_GLOBAL_COORD,
            )
            state[:, :, 6] = self.normalize_tensor(
                state[:, :, 6],
                constants.MIN_GLOBAL_COORD,
                constants.MAX_GLOBAL_COORD,
            )

            state[:, :, 8] = self.normalize_tensor(
                state[:, :, 8],
                constants.MIN_GLOBAL_COORD,
                constants.MAX_GLOBAL_COORD,
            )
            state[:, :, 9] = self.normalize_tensor(
                state[:, :, 9],
                constants.MIN_GLOBAL_COORD,
                constants.MAX_GLOBAL_COORD,
            )
        # Uncommment this to exclude the collision state
        # (1 if vehicle is in collision, 1 otherwise)
        # state = state[:, :, :5]

        return state

    def get_expert_actions(self, debug_world_idx=None, debug_veh_idx=None):
        """Get expert actions for the full trajectories across worlds."""

        expert_traj = self.sim.expert_trajectory_tensor().to_torch()
        # Global positions
        positions = expert_traj[:, :, : 2 * self.episode_len].view(
            self.num_worlds, self.max_agent_count, self.episode_len, -1
        )

        # Global velocity
        velocity = expert_traj[
            :, :, 2 * self.episode_len : 4 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        headings = expert_traj[
            :, :, 4 * self.episode_len : 5 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        inferred_expert_actions = expert_traj[
            :, :, 6 * self.episode_len : 16 * self.episode_len
        ].view(self.num_worlds, self.max_agent_count, self.episode_len, -1)

        if self.config.dynamics_model == "delta_local":
            inferred_expert_actions = inferred_expert_actions[..., :3]
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
            )
            inferred_expert_actions[..., 1] = torch.clamp(
                inferred_expert_actions[..., 1], -6, 6
            )
            inferred_expert_actions[..., 2] = torch.clamp(
                inferred_expert_actions[..., 2], -3.14, 3.14
            )
        elif self.config.dynamics_model == "state":
            # Extract (x, y, yaw, velocity x, velocity y)
            inferred_expert_actions = torch.cat(
                (
                    positions,  # xy
                    torch.ones((*positions.shape[:-1], 1), device=self.device),
                    headings,  # float (yaw)
                    velocity,  # xy velocity
                    torch.zeros((*positions.shape[:-1], 4), device=self.device)
                ),
                dim=-1,
            )

        else:  # classic or bicycle
            inferred_expert_actions = inferred_expert_actions[..., :3]
            inferred_expert_actions[..., 0] = torch.clamp(
                inferred_expert_actions[..., 0], -6, 6
            )
            inferred_expert_actions[..., 1] = torch.clamp(
                inferred_expert_actions[..., 1], -0.3, 0.3
            )
        velo2speed = None
        debug_positions = None
        if debug_world_idx is not None and debug_veh_idx is not None:
            velo2speed = (
                torch.norm(velocity[debug_world_idx, debug_veh_idx], dim=-1)
                / constants.MAX_SPEED
            )
            positions[..., 0] = self.normalize_tensor(
                positions[..., 0],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
            )
            positions[..., 1] = self.normalize_tensor(
                positions[..., 1],
                constants.MIN_REL_GOAL_COORD,
                constants.MAX_REL_GOAL_COORD,
            )
            debug_positions = positions[debug_world_idx, debug_veh_idx]

        return inferred_expert_actions, velo2speed, debug_positions

    def normalize_and_flatten_partner_obs(self, obs):
        """Normalize partner state features.
        Args:
            obs: torch.Tensor of shape (num_worlds, kMaxAgentCount, kMaxAgentCount - 1, num_features)
        """

        # TODO: Fix (there should not be nans in the obs)
        obs = torch.nan_to_num(obs, nan=0)

        # Speed
        obs[:, :, :, 0] /= constants.MAX_SPEED

        # Relative position
        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )
        obs[:, :, :, 2] = self.normalize_tensor(
            obs[:, :, :, 2],
            constants.MIN_REL_AGENT_POS,
            constants.MAX_REL_AGENT_POS,
        )

        # Orientation (heading)
        obs[:, :, :, 3] /= constants.MAX_ORIENTATION_RAD

        # Vehicle length and width
        obs[:, :, :, 4] /= constants.MAX_VEH_LEN
        obs[:, :, :, 5] /= constants.MAX_VEH_WIDTH

        # One-hot encode the type of the other visible objects
        one_hot_encoded_object_types = self.one_hot_encode_object_type(
            obs[:, :, :, 6]
        )
        # Concat the one-hot encoding with the rest of the features
        obs = torch.concat(
            (obs[:, :, :, :6], one_hot_encoded_object_types), dim=-1
        )

        return obs.flatten(start_dim=2)

    def one_hot_encode_roadpoints(self, roadmap_type_tensor):

        # Set garbage object types to zero
        road_types = torch.where(
            (roadmap_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (roadmap_type_tensor > self.ROAD_MAP_OBJECT_TYPES),
            0.0,
            roadmap_type_tensor,
        ).int()

        return torch.nn.functional.one_hot(
            road_types.long(),
            num_classes=self.ROAD_MAP_OBJECT_TYPES,
        )

    def one_hot_encode_object_type(self, object_type_tensor):
        """One-hot encode the object type."""

        VEHICLE = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Vehicle]
        PEDESTRIAN = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Pedestrian]
        CYCLIST = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType.Cyclist]
        PADDING = self.ENTITY_TYPE_TO_INT[gpudrive.EntityType._None]

        # Set garbage object elements to zero
        object_types = torch.where(
            (object_type_tensor < self.MIN_OBJ_ENTITY_ENUM)
            | (object_type_tensor > self.MAX_OBJ_ENTITY_ENUM),
            0.0,
            object_type_tensor,
        ).int()

        one_hot_object_type = torch.nn.functional.one_hot(
            torch.where(
                condition=(object_types == VEHICLE)
                | (object_types == PEDESTRIAN)
                | (object_types == CYCLIST)
                | object_types
                == PADDING,
                input=object_types,
                other=0,
            ).long(),
            num_classes=self.ROAD_OBJECT_TYPES,
        )
        return one_hot_object_type

    def normalize_and_flatten_map_obs(self, obs):
        """Normalize map observation features."""

        # Road point coordinates
        obs[:, :, :, 0] = self.normalize_tensor(
            obs[:, :, :, 0],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        obs[:, :, :, 1] = self.normalize_tensor(
            obs[:, :, :, 1],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        # Road line segment length
        obs[:, :, :, 2] /= constants.MAX_ROAD_LINE_SEGMENT_LEN

        # Road scale (width and height)
        obs[:, :, :, 3] /= constants.MAX_ROAD_SCALE
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, :, 5] /= constants.MAX_ORIENTATION_RAD

        # Road types: one-hot encode them
        one_hot_road_types = self.one_hot_encode_roadpoints(obs[:, :, :, 6])


        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.cat((obs[:, :, :, :6], one_hot_road_types), dim=-1)

        return obs.flatten(start_dim=2)
    

    def normalize_and_flatten_global_map_objects(self, obs):
        """Normalize global map objects."""
        # Road point coordinates
        obs[:, :,  0] = self.normalize_tensor(
            obs[:, :,  0],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        obs[:, :,  1] = self.normalize_tensor(
            obs[:, :,  1],
            constants.MIN_RG_COORD,
            constants.MAX_RG_COORD,
        )

        # Road line segment length
        obs[:, :, 2] /= constants.MAX_ROAD_LINE_SEGMENT_LEN

        # Road scale (width and height)
        obs[:, :, 3] /= constants.MAX_ROAD_SCALE
        # obs[:, :, :, 4] seems already scaled

        # Road point orientation
        obs[:, :, 5] /= constants.MAX_ORIENTATION_RAD

        # Road types: one-hot encode them
        one_hot_road_types = self.one_hot_encode_roadpoints(obs[:, :, 6])

        # Concatenate the one-hot encoding with the rest of the features
        obs = torch.cat((obs[:, :,  :6], one_hot_road_types), dim=-1)

        return obs.flatten(start_dim=1)


if __name__ == "__main__":

    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 10

    env_config = EnvConfig(dynamics_model="state")
    render_config = RenderConfig()
    scene_config = SceneConfig("data/", NUM_WORLDS)

    # MAKE ENV
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Number of agents to control
        device="cpu",
        render_config=render_config,
    )

    expert_actions, _, _ = env.get_expert_actions()

    # RUN
    obs = env.reset()
    frames = []

    for i in range(TOTAL_STEPS):
        print(f"Step: {i}")

        # Take a random actions
        rand_action = torch.Tensor(
            [
                [
                    env.action_space.sample()
                    for _ in range(
                        env_config.max_num_agents_in_scene * NUM_WORLDS
                    )
                ]
            ]
        ).reshape(NUM_WORLDS, env_config.max_num_agents_in_scene)

        # Step the environment
        env.step_dynamics(rand_action)

        frames.append(env.render())

        obs = env.get_obs()
        reward = env.get_rewards()
        done = env.get_dones()

    # import imageio
    imageio.mimsave("world1.gif", np.array(frames))

    env.close()
