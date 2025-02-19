import torch
import imageio
import numpy as np
from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
    SelectionDiscipline,
    RenderMode,
)
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.agents.random_actor import RandomActor
from pygpudrive.agents.policy_actor import PolicyActor
from pygpudrive.agents.core import merge_actions
import pickle
import os

if __name__ == "__main__":

    # Constants
    DATASET_EPISODE_LENGTH = 91
    EPISODE_LENGTH = 90
    MAX_CONTROLLED_AGENTS = 128 # Number of agents to control per scene
    NUM_WORLDS = 1000
    DEVICE = "cuda"
    
    TRAINED_POLICY_PATH = "models/learned_sb3_policy.zip"
    #VIDEO_PATH = f"save_data/world_num{NUM_WORLDS}_episode_length_{EPISODE_LENGTH}_agent_{MAX_CONTROLLED_AGENTS}/videos"
    
    FPS = 23
    traj_list = []
    action_list = []
    obs_list = []
    cont_agent_mask_list = []
    for i in range(1,36):
        DATA_PATH = f"dataset/dataset/formatted_json_v2_no_tl_train_split_1000/batch_{i}"
        DATA_SAVE_DIR = f"save_data/world_num{NUM_WORLDS}_episode_length_{EPISODE_LENGTH}_agent_{MAX_CONTROLLED_AGENTS}_has_map_objects/batch_{i}"
        # Configs
        env_config = EnvConfig()
        scene_config = SceneConfig(
            path=DATA_PATH,
            num_scenes=NUM_WORLDS,
            discipline=SelectionDiscipline.PAD_N,
        )
        render_config = RenderConfig(
            draw_obj_idx=True,
            color_scheme='light',
        )

        # Make environment
        env = GPUDriveTorchEnv(
            config=env_config,
            scene_config=scene_config,
            render_config=render_config,
            max_cont_agents=MAX_CONTROLLED_AGENTS,  # Maximum number of agents to control per scene
            device=DEVICE,
        )

        # Create sim agent
        obj_idx = torch.arange(env_config.max_num_agents_in_scene)

        rand_actor = RandomActor(
            env=env,
            is_controlled_func=(obj_idx <2), #(obj_idx == 0) | (obj_idx == 1),
            valid_agent_mask=env.cont_agent_mask,
            device=DEVICE,
        )

        policy_actor = PolicyActor(
            is_controlled_func=(obj_idx ==1),
            valid_agent_mask=env.cont_agent_mask,
            saved_model_path=TRAINED_POLICY_PATH,
            device=DEVICE,
        )

        obs_ori, obs_global = env.reset() #obs: [num_world, agent_num, 3876]
        frames_dict = {f"scene_{idx}": [] for idx in range(NUM_WORLDS)}

        expert_traj = env.sim.expert_trajectory_tensor().to_torch()

        #[num_world, agent_num, 91,2]
        expert_traj = expert_traj[:, :, : 2 *  DATASET_EPISODE_LENGTH].view(
                NUM_WORLDS, MAX_CONTROLLED_AGENTS, DATASET_EPISODE_LENGTH, -1
            )
        

        expert_action, _, _ = env.get_expert_actions() #[num_world, agent_num, 91,3]

        obs_of_episodes = [] # [episodes_num ,num_world, agent_num, 10]
        obs_ori = []

        # STEP THROUGH ENVIRONMENT
        for time_step in range(EPISODE_LENGTH):
            print(f"Step {time_step}/{EPISODE_LENGTH}")

            # SELECT ACTIONS
            #rand_actions = rand_actor.select_action()
            #policy_actions = policy_actor.select_action(obs)
            # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
            # actions = merge_actions(
            #     actor_actions_dict={
            #         "pi_rand": rand_actions,
            #         "pi_rl": rand_actions,
            #     },
            #     actor_ids_dict={
            #         "pi_rand": rand_actor.actor_ids,
            #         "pi_rl": rand_actor.actor_ids,
            #     },
            #     reference_action_tensor=env.cont_agent_mask,
            #     device=DEVICE,
            # )

            # STEP
            obs_of_episodes.append(obs_global[:,:,:10].clone())
            obs_ori.append(obs_global.clone())
            
            env.step_dynamics(actions=None)

            # GET NEXT OBS
            obs = env.get_obs() #obs: self_obs = [num_world, agent_num, :10] 



            # RENDER
            # for world_idx in range(NUM_WORLDS):
            #     frame = env.render(
            #         world_render_idx=world_idx,
            #         color_objects_by_actor={
            #             "rand": rand_actor.actor_ids[world_idx].tolist(),
            #             "policy": policy_actor.actor_ids[world_idx].tolist(),
            #         },
            #     )
            #     frames_dict[f"scene_{world_idx}"].append(frame)

        obs_ori = torch.stack(obs_ori, dim=0).permute(1,2,0,3) #[num_world, agent_num, episodes_num, 3876]
        obs_of_episodes = torch.stack(obs_of_episodes, dim=0).permute(1,2,0,3) #[num_world, agent_num, episodes_num, 10]
        traj_list.append(expert_traj.clone())
        action_list.append(expert_action.clone())
        obs_list.append(obs_of_episodes.clone())    
        cont_agent_mask_list.append(env.cont_agent_mask.clone())
        pkl_save_path = os.path.join(DATA_SAVE_DIR, 'pkl')

        os.makedirs(pkl_save_path, exist_ok=True)
        with open(os.path.join(pkl_save_path, 'expert_traj.pkl'), 'wb') as f:
            pickle.dump(expert_traj, f)
        with open(os.path.join(pkl_save_path, 'expert_action.pkl'), 'wb') as f:
            pickle.dump(expert_action, f)

        with open(os.path.join(pkl_save_path, 'obs_of_episodes.pkl'), 'wb') as f:
            pickle.dump(obs_of_episodes, f)

        with open(os.path.join(pkl_save_path, 'cont_agent_mask.pkl'), 'wb') as f:
            pickle.dump(env.cont_agent_mask, f)

        with open(os.path.join(pkl_save_path, 'obs_ori.pkl'), 'wb') as f:
            pickle.dump(obs_ori, f)
        # # # # # # # #
        # Done. Save videos
        # video_save_dir = os.path.join(DATA_SAVE_DIR, 'videos')
        # os.makedirs(video_save_dir, exist_ok=True)
        # for scene_name, frames_list in frames_dict.items():
        #     frames_arr = np.array(frames_list)
        #     save_path = os.path.join(video_save_dir, f'{scene_name}.gif')
        #     imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)

        print(f"all file are saved to {DATA_SAVE_DIR}")