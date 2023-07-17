import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from alfred.data.preprocess import Dataset
from src.alfred.alfred_task_planner import AlfredTaskPlanner
from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import dotdict, load_task_json
from tqdm import tqdm

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluator import Evaluator


splits = 'alfred/data/splits/oct21.json'
debug_executor = 0  # disable LLM planner
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)


class AlfredEvaluator(Evaluator):
    def __init__(self, hparams):
        self.cfg = hparams

    def evaluate(self):
        cfg = self.cfg

        log.info(OmegaConf.to_yaml(cfg))
        global splits

        random.seed(cfg.planner.random_seed)

        # llm planner
        if not debug_executor:
            planner = AlfredTaskPlanner(cfg)
        else:
            planner = None
        planner.reset()  # init skill set

        # prepare
        args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}

        with open(splits) as f:
            splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in splits.items()})

        # preprocessing
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None  # todo
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        # load tasks
        assert cfg.alfred.eval_set in splits.keys()
        files = splits[cfg.alfred.eval_set]

        if cfg.alfred.eval_portion_in_percent < 100:
            random.seed(1)  # fix seed for reproducibility
            n_sample = int(len(files) * cfg.alfred.eval_portion_in_percent / 100)
            files = random.sample(files, n_sample)
            random.seed(cfg.planner.random_seed)

        # random.shuffle(files)
        # files = [files[30]]  # debugging
        # files = files[24:30]  # debugging

        if False:  # debug mode
            for file in files:
                if 'T20190907_001129_214240' in file['task'] and file['repeat_idx'] == 1:
                    new_files = [file]
                    break
            files = new_files

        # run
        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        results = self.evaluate_main(files, args_dict, planner, x_display, save_path)

        # print results
        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
        log.info(f'success rate: {n_success / n * 100:.2f} %')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        model_args = dotdict(args_dict)
        env = ThorConnector(x_display=x_display)

        # save prompt
        with open(os.path.join(save_path, 'prompt.txt'), 'w') as f:
            f.write(planner.prompt)

        # run
        for i, task in enumerate(tqdm(tasks)):
            try:
                log.info(task)
                traj_data = load_task_json(task)
                r_idx = task['repeat_idx']
                log.info(f"Evaluating ({i+1}/{len(tasks)}): {traj_data['root']}")
                result = self.evaluate_task(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0))
                results.append(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))

        return results
    
    def evaluate_task(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False):
        # setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize
        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        # print goal instr
        instruction_text = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        log.info("Task: %s" % instruction_text)

        # inference steps from instruction
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        prev_steps = []
        prev_action_msg = []
        while not done:
            # find next step
            step, prompt = planner.plan_step_by_step(instruction_text, prev_steps, prev_action_msg)
            if step is None:
                log.info("\tmax step reached")
                break

            if log_prompt:
                log.info(prompt)
            log.info(f'{len(prev_steps) + 1}. {step}')
            prev_steps.append(step)

            if step in ['done', 'done.', 'done.\n']:
                done = True
                prev_action_msg.append('')
                break

            # execute
            step_to_execute = step
            action_ret = env.llm_skill_interact(step_to_execute)
            imgs.append(env.write_step_on_img(t+1, step_to_execute))
            prev_action_msg.append(action_ret['message'])

            if not action_ret['success']:
                print(action_ret['message'])

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            # print("Goal Reached")
            success = True
            # exit()

        # record results
        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'success': success}

        # save
        self.save_result(log_entry, imgs, save_path)

        return log_entry

    def save_result(self, result_dict, imgs, base_path='results'):
        if result_dict:
            filename = f"{result_dict['trial']}_{result_dict['repeat_idx']}"

            # write json
            with open(os.path.join(base_path, filename + '.json'), "w") as outfile:
                json.dump(result_dict, outfile)
        else:
            filename = "images"

        # write img
        widths, heights = zip(*(i.size for i in imgs))
        left, top, right, bottom = font.getbbox('hello')
        line_height = bottom - top
        total_width = widths[0] * 5
        total_height = math.ceil(len(imgs) / 5) * heights[0] + line_height * 1.2
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        # draw text
        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text_color = (0, 0, 0)  # black
            # text_color = (100, 255, 100) if result_dict['success'] else (255, 100, 100)
            lines = textwrap.wrap(text, width=100)
            draw = ImageDraw.Draw(new_im)
            y_offset = 10
            for line in lines:
                draw.text((10, y_offset), line, font=font, fill=text_color)
                y_offset += line_height
            y_offset += line_height
        else:
            y_offset = 0

        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]

        success_str = 'success' if result_dict['success'] else 'fail'
        new_im.save(os.path.join(base_path, f"{filename}_{success_str}.png"))
