from multi_llm.tools.process_gpudrive_data import generate_path, load_pkl
import yaml
import os
import itertools
import torch
import pickle

def select_test_scence(scenes_path, reserve_mask_dir, test_mask_path):


    reserve_agent_mask_all_path, reserve_agent_mask_num_over_thrd_path, reserve_mask_path = generate_path(reserve_mask_dir, ['reserve_agent_mask_all.pkl', 'reserve_agent_mask_num_over_thrd.pkl', 'reserve_mask.pkl'])

    print("\n>>>>start loading mask data...\n")
    reserve_agent_mask_all, reserve_agent_mask_num_over_thrd, reserve_mask = load_pkl([reserve_agent_mask_all_path, reserve_agent_mask_num_over_thrd_path, reserve_mask_path])
    print(">>>>finish loading mask data and align the test data to the LLM evaluation.\n")

    #load a txt file
    with open (scenes_path, 'r') as f:
        scenes = f.readlines()
        scenes = [scene.strip() for scene in scenes]
        ids= [25, 18, 17, 34, 24, 30, 10, 8, 26, 23, 3, 29, 32, 19, 5, 20, 14, 13, 16, 21, 28, 1, 6, 9, 33, 31, 11, 4, 7, 2, 22, 27, 15, 12]
        ids = [x - 1 for x in ids]
        #scenes_align = []
        scenes_align = [scenes[i*1000:i*1000+1000] for i in ids]
        scenes_align = list(itertools.chain(*scenes_align))


    
    

    with open(test_mask_path, 'r') as f:
        test_mask = yaml.load(f, Loader=yaml.FullLoader)
        train_count = test_mask['train_world_num']
        test_world_small_num = test_mask['test_world_small_num']



    #scenes_over_thrd = scenes[reserve_agent_mask_num_over_thrd]
    #reserve_scenes = scenes_over_thrd[reserve_mask]
    scenes_over_thrd = [v for v, m in zip(scenes_align, reserve_agent_mask_num_over_thrd) if m]
    reserve_scenes = [v for v, m in zip(scenes_over_thrd, reserve_mask) if m]

    reserve_agent_mask_list = []
    for i in range(reserve_agent_mask_all.shape[0]):
        if reserve_agent_mask_num_over_thrd[i]: 
            reserve_agent_mask_list.append(reserve_agent_mask_all[i])
    reserve_agent_mask = torch.stack(reserve_agent_mask_list, dim=0)
    reserve_agent_mask = reserve_agent_mask[reserve_mask]

    breakpoint()
    test_scenes = reserve_scenes[train_count:]
    test_small_scenes = test_scenes[:test_world_small_num]   

    reserve_agent_mask_test = reserve_agent_mask[train_count:]
    reserve_agent_mask_test_small = reserve_agent_mask_test[:test_world_small_num]


    #breakpoint()
    save_scence_path = os.path.join(os.path.dirname(scenes_path), 'test_scenes.txt')
    with open(save_scence_path, "w") as f:
        for scene_path in test_small_scenes:
            f.write(scene_path + "\n")

    save_reserve_agent_mask_path = os.path.join(os.path.dirname(scenes_path), 'reserve_agent_mask_for_test.pkl')
    with open(save_reserve_agent_mask_path, 'wb') as f:
        pickle.dump(reserve_agent_mask_test_small, f)

    print(f"\nsaved scene paths to {save_scence_path}\n")
    print(f"\nsaved scene paths to {save_reserve_agent_mask_path}\n")

def compare_test_with_LLM(llm_obs_dir, compared_obs_dir, reserve_agent_mask_test_dir, test_mask_path):
    CONT_NUM = 5
    obs_path_llm = generate_path(llm_obs_dir,  'obs.pkl')
    obs_global_path_bl = generate_path(compared_obs_dir, 'obs_global.pkl')

    reserve_agent_mask_test_path= generate_path(reserve_agent_mask_test_dir, 'reserve_agent_mask_for_test.pkl')

    with open(test_mask_path, 'r') as f:
        test_mask = yaml.load(f, Loader=yaml.FullLoader)
        train_count = test_mask['train_world_num']
        test_world_small_num = test_mask['test_world_small_num']


    

    obs_llm = load_pkl(obs_path_llm)
    # reserve_agent_mask_all, reserve_agent_mask_num_over_thrd = load_pkl([reserve_agent_mask_all_path, reserve_agent_mask_num_over_thrd_path])
    reserve_agent_mask_test = load_pkl(reserve_agent_mask_test_path)

    reserve_global_obs_bl_list = []


    


    obs_llm_test = obs_llm[train_count:]
    obs_llm_test_small = obs_llm_test[:test_world_small_num]

    obs_global_bl = load_pkl(obs_global_path_bl)
    
    obs_global_bl_list = []
    for i in range(obs_global_bl.shape[0]):
        obs_global_bl_world = obs_global_bl[i, reserve_agent_mask_test[i], :, :]
        obs_global_bl_world = obs_global_bl_world[:CONT_NUM]
        obs_global_bl_list.append(obs_global_bl_world)

    obs_global_bl = torch.stack(obs_global_bl_list, dim=0)
    breakpoint()
    #flag_count=0
    # for i in range(reserve_agent_mask_all.shape[0]):
    #     if reserve_agent_mask_num_over_thrd[i]: 
    #         selected_obs_global_bl = obs_global_bl[flag_count, reserve_agent_mask_all[i], :, :]
    #         selected_obs_global_bl = selected_obs_global_bl[:CONT_NUM]
    #         flag_count += 1


            #reserve_global_obs_bl_list.append(selected_obs_global_bl)
    #reserve_global_obs_bl = torch.stack(reserve_global_obs_bl_list, dim=0) #[reserve_num, CONT_NUM, episode_len, 2]
    


    

if __name__ == '__main__':
    # scenes_path = 'dataset/output/gpudrive_ori_data_34000/scene_paths.txt'
    # reserve_mask_dir = 'dataset/output/gpudrive_ori_data_34000/reserve_world_num_362_agent_num_5_interact_coeff_3.0'
    # test_mask_path = 'dataset/output/gpudrive_ori_data_34000/reserve_world_num_362_agent_num_5_interact_coeff_3.0/coll_coeff_1.4_prob_planning_per_frame/prompt/config.yaml'
    # select_test_scence(scenes_path, reserve_mask_dir, test_mask_path)

    #compare the test data with LLM data
    llm_obs_dir = 'dataset/output/gpudrive_ori_data_34000/reserve_world_num_362_agent_num_5_interact_coeff_3.0'
    compared_obs_path = 'save_data/world_num_1000_episode_length_91_agent_128_actor_expert_for_test/pkl'
    test_mask_path = 'dataset/output/gpudrive_ori_data_34000/reserve_world_num_362_agent_num_5_interact_coeff_3.0/coll_coeff_1.4_prob_planning_per_frame/prompt/config.yaml'
    reserve_agent_mask_test_dir = 'dataset/output/gpudrive_ori_data_34000'
    compare_test_with_LLM(llm_obs_dir, compared_obs_path, reserve_agent_mask_test_dir, test_mask_path)