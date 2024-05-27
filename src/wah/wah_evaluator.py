import json 
import logging
import random
import time

from evaluator import Evaluator
from wah.wah_task_planner import WahTaskPlanner
from wah.wah_environment import WahEnv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from wah.wah_utils import check_goal_condition

import pdb

log = logging.getLogger(__name__)
debug_executor = 0  # disable LLM planner


class WahEvaluator(Evaluator):
    def __init__(self, cfg):
        self.cfg = cfg
    
    def evaluate(self):
        cfg = self.cfg
        with open(cfg.dataset.wah_testset, "r") as json_file:
            test_set = json.load(json_file)
            
        log.info(OmegaConf.to_yaml(cfg))

        ### llm planner
        if not debug_executor:
            planner = WahTaskPlanner(cfg)
        else:
            planner = None
        
        ### virtualhome envoronment
        env = WahEnv(cfg)
        
        ### evaluation
        start = time.time()
        save_path = cfg.out_dir
        results = self.evaluate_main(test_set, planner, env, save_path)
        
        ### print results
        log.info(results)
        num_tasks = len(results)
        num_goal_success, sum_subgoal_success_rate = 0, 0
        for result in results:
            num_goal_success += result['goal_success']
            sum_subgoal_success_rate += result['subgoal_success_rate']
        log.info(f'success rate: {num_goal_success / num_tasks * 100:.2f} %')
        log.info(f'average subgoal success rate: {sum_subgoal_success_rate / num_tasks * 100:.2f} %')
        log.info(f'took {(time.time() - start) / 60:.1f} mins')
            
    def evaluate_main(self, test_set, task_planner, env, save_path):
        cfg = self.cfg
        with open(cfg.dataset.obj_dict_sim2nl, 'r') as file:
            obj_dict_sim2nl = json.load(file)
        with open(cfg.dataset.obj_dict_nl2sim, 'r') as file:
            obj_dict_nl2sim = json.load(file)
        nl_act_list = cfg.planner.nl_act_list
        results = []
        for i, task_d in tqdm(enumerate(test_set), total=len(test_set)):
            try:
                sim_obj_list = list(set([node['class_name'] for node in task_d['init_graph']['nodes']]))
                nl_obj_list = [obj_dict_sim2nl[sim_obj] for sim_obj in sim_obj_list]
                
                ### Select NL instruction
                random.seed(0)
                nl_instruction = random.choice(task_d['nl_instructions'])
                log.info(nl_instruction)

                ### Reset planner
                task_planner.reset(nl_act_list, nl_obj_list, task_d, nl_instruction)
                ### Reset environment
                env.reset(task_d)
                
                ### Task planning and simulation
                result = self.evaluate_task(task_planner, env, nl_instruction, task_d, log_prompt=False)
                results.append(result)
        
            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))
            
        return results
            
    def evaluate_task(self, task_planner, env, nl_instruction, task_d, log_prompt=False):
        task_goal = task_d['task_goal']
        
        done, success = False, False

        # mode selection
        if self.cfg.planner.model_name.endswith('gpt-3.5-turbo') or 'gpt-4' in self.cfg.planner.model_name:
            # plan whole sequences with chat style api
            step_by_step_mode = False
        else:
            step_by_step_mode = True
        
        if step_by_step_mode:
            prev_steps = []
            prev_action_msg = []
            skill_set_size_seq = []
            while not done:
                ### Task planning
                skill_set_size_seq.append(len(task_planner.skill_set))
                step, prompt = task_planner.plan_step_by_step(nl_instruction, prev_steps, prev_action_msg)
                if step is None:
                    log.info("\tmax step reached")
                    break
                
                if log_prompt:
                    log.info(prompt)
                # log.info(f'{len(prev_steps) + 1}. {step}')
                prev_steps.append(step)
                
                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break
                
                ### TODO: Update skill set
                task_planner.update_skill_set(step, task_planner.nl_obj_list)
                
                ### Simualtion
                possible, feedback = env.step(step, step_form='nl', instance=False)
                log.info(f'{len(prev_steps)}. {step} ({possible})')
                ### TODO: NL feedback ######################################
                prev_action_msg.append('')
                ### TODO: Visualization Save ################################
        else:
            
            steps, prompt = task_planner.plan_whole(nl_instruction)
            prev_steps = steps

            if log_prompt:
                log.info(prompt)

            for si, step in enumerate(steps):
                log.info(f'{si + 1}. {step}')

                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    break

                # execute
                step_to_execute = step
                

                task_planner.update_skill_set(step, task_planner.nl_obj_list)
                
                ### Simualtion
                possible, feedback = env.step(step, step_form='nl', instance=False)

        ### Check goal
        graph = env.get_graph()
        ### TODO: check_goal_condition refactorying
        to_obj_id, final_state = check_goal_condition(graph, task_goal)      
        total_num_subgoal_satisfied, total_num_subgoal = 0, 0
        goal_success = True
        for subgoal, subgoal_result in final_state.items():
            num_subgoal_satisfied, num_subgoal = subgoal_result
            goal_success = goal_success and (num_subgoal_satisfied>=num_subgoal)
            
            total_num_subgoal_satisfied += min(num_subgoal_satisfied, num_subgoal) 
            total_num_subgoal += num_subgoal
            
        subgoal_succes_rate = total_num_subgoal_satisfied / total_num_subgoal
        ### Record results
        log_entry = {'trial': task_d['task_id'],
                    'goal_instr': nl_instruction,
                    'inferred_steps': prev_steps,
                    'goal_success': goal_success,
                    'subgoal_success_rate': subgoal_succes_rate}
        return log_entry