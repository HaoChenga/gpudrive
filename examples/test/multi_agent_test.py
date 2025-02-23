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
from multi_llm.config_parse.config_parse import data_generation_parser_args
import time


if __name__ == "__main__":

    args = data_generation_parser_args()
    # Constants
    DATASET_EPISODE_LENGTH = 91
    EPISODE_LENGTH = 91
    MAX_CONTROLLED_AGENTS = 128 # Number of agents to control per scene
    NUM_WORLDS = 1000
    DEVICE = "cuda"
    
    TRAINED_POLICY_PATH = "gpudrive/models/learned_sb3_policy.zip"

    BATCH_NUM=args.batch_num
    #VIDEO_PATH = f"save_data/world_num{NUM_WORLDS}_episode_length_{EPISODE_LENGTH}_agent_{MAX_CONTROLLED_AGENTS}/videos"
    DATA_PATH = f"dataset/dataset/formatted_json_v2_no_tl_train_split_1000/batch_{BATCH_NUM}"
    DATA_SAVE_DIR = f"save_data/world_num_{NUM_WORLDS}_episode_length_{EPISODE_LENGTH}_agent_{MAX_CONTROLLED_AGENTS}_actor_{args.actor}_no_normlized_from_exper_batch/batch_{BATCH_NUM}"
    #scenes_path = f"dataset/output/gpudrive_ori_data_34000/test_scenes.txt"
    scenes_path = None
    FPS = 23
    start_time = time.time()

    if scenes_path is not None:
        with open (scenes_path, 'r') as f:
            scenes = f.readlines()
            scenes = [scene.strip() for scene in scenes]
            NUM_WORLDS = len(scenes)
    else:
        scenes = None

    # Configs
    env_config = EnvConfig()

    if not args.normlize:
        env_config.norm_obs = False
        print("\n>>>>>obs is not normlized")

    scene_config = SceneConfig(
        path=DATA_PATH,
        num_scenes=NUM_WORLDS,
        discipline=SelectionDiscipline.PAD_N,
        #special_scence=scenes,
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

    # rand_actor = RandomActor(
    #     env=env,
    #     is_controlled_func=(obj_idx <2), #(obj_idx == 0) | (obj_idx == 1),
    #     valid_agent_mask=env.cont_agent_mask,
    #     device=DEVICE,
    # )

    policy_actor = PolicyActor(
        is_controlled_func=(obj_idx >=0),
        valid_agent_mask=env.cont_agent_mask,
        saved_model_path=TRAINED_POLICY_PATH,
        device=DEVICE,
    )

    obs, obs_gloabl = env.reset() #obs: [num_world, agent_num, 3876]
    #obs_full = obs.clone()

    frames_dict = {f"scene_{idx}": [] for idx in range(NUM_WORLDS)}

    expert_traj_full = env.sim.expert_trajectory_tensor().to_torch()


    global_map_objects = env.sim.map_observation_tensor().to_torch()
    if args.normlize:
        global_map_objects = env.normalize_and_flatten_global_map_objects(global_map_objects)

    #[num_world, agent_num, 91,2]
    expert_traj = expert_traj_full[:, :, : 2 *  DATASET_EPISODE_LENGTH].view(
            NUM_WORLDS, MAX_CONTROLLED_AGENTS, DATASET_EPISODE_LENGTH, -1
        )
    
    expert_velocity = expert_traj_full[
            :, :, 2 * DATASET_EPISODE_LENGTH : 4 * DATASET_EPISODE_LENGTH
        ].view(NUM_WORLDS, MAX_CONTROLLED_AGENTS, DATASET_EPISODE_LENGTH, -1)
    
    expert_action, _, _ = env.get_expert_actions() #[num_world, agent_num, 91,3]

    expert_traj = torch.cat([expert_traj, expert_velocity], dim=-1) #[num_world, agent_num, 91,4]
    #breakpoint()
    #obs_of_episodes = [] # [episodes_num ,num_world, agent_num, 10]
    actions_of_episodes = [] # [episodes_num ,num_world, agent_num]
    obs_global = []
    map_objects = []
    

    # STEP THROUGH ENVIRONMENT
    for time_step in range(EPISODE_LENGTH):
        print(f"Step {time_step}/{EPISODE_LENGTH}")

        # SELECT ACTIONS
        #rand_actions = rand_actor.select_action()
        
        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        if args.actor !='expert':
            policy_actions = policy_actor.select_action(obs)
            actions = merge_actions(
                actor_actions_dict={
                    #"pi_rand": rand_actions,
                    "pi_rl": policy_actions,
                },
                actor_ids_dict={
                    #"pi_rand": rand_actor.actor_ids,
                    "pi_rl": policy_actor.actor_ids,
                },
                reference_action_tensor=env.cont_agent_mask,
                device=DEVICE,
            )
        else:
            actions = None
        # STEP
        #obs_of_episodes.append(obs.clone().cpu().numpy())
        #map_objects.append(obs[:,:,-2600:].clone().cpu().numpy())
        obs_global.append(obs_gloabl.clone().cpu().numpy())
        if args.actor !='expert':
            actions_of_episodes.append(actions.clone())
            env.step_dynamics(actions)
        else:
            env.step_dynamics(actions)
        
        #del obs, obs_gloabl
        # GET NEXT OBS
        obs, obs_gloabl = env.get_obs() #obs: self_obs = [num_world, agent_num, :10] 


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
    obs_global = np.stack(obs_global, axis=0).transpose(1,2,0,3) #[num_world, agent_num, episodes_num, 10]
    #map_objects = np.stack(map_objects, axis=0).transpose(1,2,0,3) #[num_world, agent_num, episodes_num, 2600]
    #obs_of_episodes = np.stack(obs_of_episodes, axis=0).transpose(1,2,0,3) #[num_world, agent_num, episodes_num, 10]

    if args.actor !='expert':
        actions_of_episodes = torch.stack(actions_of_episodes, dim=0).permute(1,2,0) #[num_world, agent_num, episodes_num, 3]

    pkl_save_path = os.path.join(DATA_SAVE_DIR, 'pkl')
    os.makedirs(pkl_save_path, exist_ok=True)
    # with open(os.path.join(pkl_save_path, 'expert_traj.pkl'), 'wb') as f:
    #     pickle.dump(expert_traj, f)
    # with open(os.path.join(pkl_save_path, 'expert_action.pkl'), 'wb') as f:
    #     pickle.dump(expert_action, f)
    if args.actor !='expert':

        with open(os.path.join(pkl_save_path, 'actions_of_episodes.pkl'), 'wb') as f:
            pickle.dump(actions_of_episodes, f)
    
    # with open(os.path.join(pkl_save_path, 'obs_full.pkl'), 'wb') as f:
    #     pickle.dump(obs_of_episodes, f)
    with open(os.path.join(pkl_save_path, 'traj.pkl'), 'wb') as f:
        pickle.dump(expert_traj, f)


    with open(os.path.join(pkl_save_path, 'global_map_objects.pkl'), 'wb') as f:
        pickle.dump(global_map_objects, f)

    with open(os.path.join(pkl_save_path, 'obs_global.pkl'), 'wb') as f:
        pickle.dump(torch.from_numpy(obs_global), f)

    with open(os.path.join(pkl_save_path, 'cont_agent_mask.pkl'), 'wb') as f:
        pickle.dump(env.cont_agent_mask, f)

    # with open(os.path.join(pkl_save_path, 'cont_agent_mask.pkl'), 'wb') as f:
    #     pickle.dump(env.cont_agent_mask, f)
    # # # # # # # #
    # Done. Save videos
    # video_save_dir = os.path.join(DATA_SAVE_DIR, 'videos')
    # os.makedirs(video_save_dir, exist_ok=True)
    # for scene_name, frames_list in frames_dict.items():
    #     frames_arr = np.array(frames_list)
    #     save_path = os.path.join(video_save_dir, f'{scene_name}.gif')
    #     imageio.mimwrite(save_path, frames_arr, fps=FPS, loop=0)
    print(f"all file are saved to {DATA_SAVE_DIR}")
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f} seconds")