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
    if args.scene_path is None:
        DATA_PATH = None
    else:
        DATA_PATH = f"dataset/dataset/formatted_json_v2_no_tl_train_split_1000/batch_{BATCH_NUM}"
    #DATA_SAVE_DIR = f"save_data/world_num_{NUM_WORLDS}_episode_length_{EPISODE_LENGTH}_agent_{MAX_CONTROLLED_AGENTS}_actor_{args.actor}_for_test"
    DATA_SAVE_DIR = args.output_dir
    #scenes_path = f"dataset/output/gpudrive_ori_data_34000/reserve_world_num_362_agent_num_5_interact_coeff_3.0/coll_coeff_1.4_prob_planning_per_frame/eval/baseline/test_mask/test_scenes.txt"
    scenes_path = args.scene_path

    FPS = 23
    start_time = time.time()

    with open (scenes_path, 'r') as f:
        scenes = f.readlines()
        scenes = [scene.strip() for scene in scenes]
        NUM_WORLDS = len(scenes)
    # Configs
    env_config = EnvConfig()
    scene_config = SceneConfig(
        path=DATA_PATH,
        num_scenes=NUM_WORLDS,
        discipline=SelectionDiscipline.PAD_N,
        special_scene=scenes,
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

    #obs, obs_gloabl = env.reset() #obs: [num_world, agent_num, 3876]
    #obs_full = obs.clone()

    frames_dict = {f"scene_{idx}": [] for idx in range(NUM_WORLDS)}

    expert_traj = env.sim.expert_trajectory_tensor().to_torch()

    #[num_world, agent_num, 91,2]
    expert_traj = expert_traj[:, :, : 2 *  DATASET_EPISODE_LENGTH].view(
            NUM_WORLDS, MAX_CONTROLLED_AGENTS, DATASET_EPISODE_LENGTH, -1
        )
    

    expert_action, _, _ = env.get_expert_actions() #[num_world, agent_num, 91,3]

    actions_of_epe_list = [] 
    obs_global_of_epe_list = []


    # init_done=env.sim.done_tensor().to_torch().clone()
    # init_info = env.sim.info_tensor().to_torch().clone()
    # init_reward = env.sim.reward_tensor().to_torch().clone()
    # init_action = env.sim.action_tensor().to_torch().clone()
    # init_self_obs = env.sim.self_observation_tensor().to_torch().clone()
    # init_partner_obs = env.sim.partner_observations_tensor().to_torch().clone()
    # inti_agent_roadmap = env.sim.agent_roadmap_tensor().to_torch().clone()
    # init_cont_state = env.sim.controlled_state_tensor().to_torch().clone()
    
    # STEP THROUGH ENVIRONMENT
    
    for time_step in range(EPISODE_LENGTH):
        print(f"Step {time_step}/{EPISODE_LENGTH}")

        # SELECT ACTIONS
        #rand_actions = rand_actor.select_action()
        
        # MERGE ACTIONS FROM DIFFERENT SIM AGENTS
        #obs_of_episodes_pred_step = [] # [episodes_num ,num_world, agent_num, 10]

        #initialize the env

        obs, obs_gloabl = env.reset()
        for i in range(time_step):
            env.step_dynamics(None)
        obs, obs_gloabl = env.get_obs()


        actions_pred_setp_list = [] 
        obs_global_pred_setp_list = []
        obs_global_pred_setp_list.append(obs_gloabl) # include the obs_global of current frame and next 6 frames

        for pred_step in range(6):

            if(pred_step+1+time_step>=EPISODE_LENGTH):
                obs_global_pred_setp_list.append(obs_gloabl)
                actions_pred_setp_list.append(actions)
            else:
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

                env.step_dynamics(actions)
                # GET NEXT OBS
                obs, obs_gloabl = env.get_obs() #obs: self_obs = [num_world, agent_num, :10] 
                obs_global_pred_setp_list.append(obs_gloabl)
                actions_pred_setp_list.append(actions)

        obs_global_pred_setp = torch.stack(obs_global_pred_setp_list, dim=0)
        actions_pred_setp = torch.stack(actions_pred_setp_list, dim=0)

        actions_of_epe_list.append(actions_pred_setp)
        obs_global_of_epe_list.append(obs_global_pred_setp)

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
    actions_of_epe = torch.stack(actions_of_epe_list, dim=0).permute(2,0,1,3) 
    obs_global_of_epe = torch.stack(obs_global_of_epe_list, dim=0).permute(2,3,0,1,4) 

    pkl_save_path = os.path.join(DATA_SAVE_DIR, 'pkl')
    os.makedirs(pkl_save_path, exist_ok=True)
    # with open(os.path.join(pkl_save_path, 'expert_traj.pkl'), 'wb') as f:
    #     pickle.dump(expert_traj, f)
    # with open(os.path.join(pkl_save_path, 'expert_action.pkl'), 'wb') as f:
    #     pickle.dump(expert_action, f)
    
    with open(os.path.join(pkl_save_path, 'obs_of_action_bl_test.pkl'), 'wb') as f:
        pickle.dump(actions_of_epe, f)

    with open(os.path.join(pkl_save_path, 'obs_global_bl_test.pkl'), 'wb') as f:
        pickle.dump(obs_global_of_epe, f)

        

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