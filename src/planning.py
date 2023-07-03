import hydra
import json
import random
import os

from tqdm import tqdm

from wah.wah_task_planner import WahTaskPlanner

import pdb

@hydra.main(version_base=None, config_path="../conf", config_name="config_wah")
def planning(cfg):
    ### Load dataset
    with open(cfg.dataset.wah_testset, "r") as json_file:
        test_set = json.load(json_file)
    with open(cfg.dataset.obj_dict_sim2nl, 'r') as file:
        obj_dict_sim2nl = json.load(file)
    with open(cfg.dataset.obj_dict_nl2sim, 'r') as file:
        obj_dict_nl2sim = json.load(file)
        
    ### Load task planner
    task_planner = WahTaskPlanner(cfg)
    
    ### Planning
    planning_results = {}
    for i, task_d in tqdm(enumerate(test_set[:10]), total=len(test_set[:10])):    
    # for i, task_d in tqdm(enumerate(test_set), total=len(test_set)):
        sim_obj_list = list(set([node['class_name'] for node in task_d['init_graph']['nodes']]))
        nl_obj_list = [obj_dict_sim2nl[sim_obj] for sim_obj in sim_obj_list]
        nl_act_list = ["walk", "grab", "open", "close", "switch on"]
        
        ### Reset planner
        task_planner.reset(nl_act_list, nl_obj_list)
        
        ### Task planning
        random.seed(0)
        nl_instruction = random.choice(task_d['nl_instructions'])
        step_seq, skill_set_size_seq = task_planner.plan(nl_instruction)
        
        ### Results
        planning_results[task_d['task_id']] = step_seq
        
    ### Save results
    result_dir = os.path.join("results", f"{cfg.experiment.exp_name}")
    os.makedirs(result_dir, exist_ok=True)
    if cfg.planner.fast_mode:
        fast_mode = "fast"
    else:
        fast_mode = "original"
    model_name = cfg.planner.model_name.split('/')[1]
    with open(os.path.join(result_dir, f"{model_name}_{fast_mode}.json"), "w") as json_file:
        json.dump(planning_results, json_file)

if __name__ == "__main__":
    planning()