from task_planner import TaskPlanner
from wah.wah_utils import find_indefinite_article
import json

import pdb


class WahTaskPlanner(TaskPlanner):
    def init_prompt(self, prefix, example_file_path, num_examples="all", splitter=""):
        with open(example_file_path, "r") as json_file:
            examples = json.load(json_file)
        
        if num_examples == "all":
            init_prompt = f"{prefix}{splitter}"
            for example in examples:
                example_template = f"Human: {example['task_nl_des']}\nRobot:"
                for step_id, step in enumerate(example['steps']):
                    example_template += f" {step_id+1}. {step}"
                    if step_id == len(example['steps'])-1:
                        example_template += ".\n"
                    else:
                        example_template += ","
                init_prompt += f"{example_template}{splitter}"
        else:
            num_examples = int(num_examples)
            examples = examples[:5*num_examples]
            init_prompt = f"{prefix}{splitter}"
            for example in examples:
                example_template = f"Human: {example['task_nl_des']}\nRobot:"
                for step_id, step in enumerate(example['steps']):
                    example_template += f" {step_id+1}. {step}"
                    if step_id == len(example['steps'])-1:
                        example_template += ".\n"
                    else:
                        example_template += ","
                init_prompt += f"{example_template}{splitter}"
        return init_prompt
    
    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)
    
    def init_skill_set(self, nl_act_list, nl_obj_list):
        skill_set = ["done", "done.", "done.\n"]
        nl_all_objs = ['alcohol', 'amplifier', 'apple', 'balance ball', 'bananas', 'bar soap', 'bathroom', 'bathroom cabinet', 'bathroom counter', 'bathtub', 'bed', 'bedroom', 'bell pepper', 'bench', 'board game', 'book', 'bookshelf', 'bottled water', 'box', 'slice of bread', 'bucket', 'cabinet', 'candle', 'candy bar', 'carrot', 'ceiling', 'ceiling fan', 'ceilinglamp', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'closet', 'closet drawer', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee maker', 'coffee pot', 'coffee table', 'computer', 'condiment bottle', 'condiment shaker', 'cooking pot', 'cpu screen', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'curtains', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'deodorant', 'desk', 'bowl', 'dishwasher', 'dishwashing liquid', 'door', 'doorjamb', 'face cream', 'faucet', 'floor', 'folder', 'fridge', 'frying pan', 'game', 'garbage can', 'glasses', 'globe', 'guitar', 'hair product', 'hanger', 'juice', 'keyboard', 'kitchen', 'kitchen cabinet', 'kitchen counter', 'kitchen counter drawer', 'kitchen table', 'knife block', 'light switch', 'lime', 'living room', 'longboard', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'milkshake', 'minced meat', 'mouse', 'mouse mat', 'mug', 'nightstand', 'notes', 'orchid', 'oven tray', 'painkillers', 'pancake', 'paper', 'paper tray', 'peach', 'pear', 'perfume', 'photo frame', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'power socket', 'printer', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmod', 'shelf', 'sink', 'slippers', 'sofa', 'speaker', 'ball', 'stall', 'standing mirror', 'stove', 'stove fan', 'sundae', 'table lamp', 'teddy bear', 'toaster', 'toilet', 'toiletpaper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'tv', 'tv stand', 'vase', 'wall', 'wall lamp', 'wall phone', 'wall picture frame', 'wall shelf', 'washing machine', 'washing sponge', 'water glass', 'whipped cream', 'window', 'wine', 'wine glass']
        nl_grab_objs = ['alcohol', 'apple', 'bananas', 'bar soap', 'bell pepper', 'board game', 'book', 'box', 'slice of bread', 'bucket', 'candle', 'candy bar', 'carrot', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee pot', 'condiment bottle', 'condiment shaker', 'cooking pot', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'bowl', 'dishwashing liquid', 'face cream', 'folder', 'frying pan', 'glasses', 'globe', 'hair product', 'hanger', 'juice', 'keyboard', 'lime', 'lotion bottle', 'magazine', 'milk', 'milkshake', 'minced meat', 'mouse', 'mug', 'notes', 'oven tray', 'pancake', 'paper', 'pear', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmod', 'slippers', 'ball', 'sundae', 'teddy bear', 'toiletpaper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'wall phone', 'wall picture frame', 'washing sponge', 'water glass', 'whipped cream', 'wine', 'wine glass']
        nl_open_objs = ['bathroom cabinet', 'book', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'coffee pot', 'cooking pot', 'curtains', 'desk', 'dishwasher', 'door', 'folder', 'fridge', 'garbage can', 'hair product', 'kitchen cabinet', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'nightstand', 'printer', 'radio', 'stove', 'toilet', 'toothpaste', 'washing machine', 'window']
        nl_switch_objs = ['candle', 'cell phone', 'clock', 'coffee maker', 'computer', 'dishwasher', 'faucet', 'fridge', 'light switch', 'microwave oven', 'printer', 'radio', 'remote control', 'stove', 'toaster', 'tv', 'wall phone', 'washing machine']
        nl_recepticel_objs = ['cabinet', 'kitchen cabinet', 'stove', 'garbage can', 'sink', 'washing machine', 'bathroom cabinet', 'microwave oven', 'coffee maker', 'box', 'closet', 'nightstand', 'dishwasher', 'toilet', 'bookshelf', 'fridge']
        # for act_nl in nl_act_list:
        #     if act_nl == "walk":
        #         act_nl = "walk to"
        #         for obj_nl in nl_obj_list:
        #             if obj_nl in nl_all_objs:
        #                 skill_set.append(f"{act_nl} {obj_nl}")
        #     elif act_nl == "grab":
        #         for obj_nl in nl_obj_list:
        #             if obj_nl in nl_grab_objs:
        #                 skill_set.append(f"{act_nl} {obj_nl}")
        #     elif act_nl in ["open", "close"]:
        #         for obj_nl in nl_obj_list:
        #             if obj_nl in nl_open_objs:
        #                 skill_set.append(f"{act_nl} {obj_nl}")
        #     elif act_nl in ["switch on"]:
        #         for obj_nl in nl_obj_list:
        #             if obj_nl in nl_switch_objs:
        #                 skill_set.append(f"{act_nl} {obj_nl}")
        #     else:
        #         NotImplementedError
        for act_nl in nl_act_list:
            if act_nl == "find":
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_grab_objs:
                        article = find_indefinite_article(obj_nl)
                        skill_set.append(f"{act_nl} {article} {obj_nl}")
            elif act_nl == "go to":
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_recepticel_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl == "grab":
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_grab_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl in ["open", "close"]:
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_open_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl in ["switch on"]:
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_switch_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            else:
                NotImplementedError
        return skill_set    

    def update_skill_set(self, previous_step, nl_obj_list):
        ### Case 1. previous_step: grab something -> add put grabbed_obj ?? obj
        nl_putin_objs = ['bathroom cabinet', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'dishwasher', 'folder', 'fridge', 'frying pan', 'garbage can', 'kitchen cabinet', 'microwave oven', 'nightstand', 'printer', 'sink', 'stove', 'toaster', 'toilet', 'washing machine']
        nl_putback_objs = ['bathroom cabinet', 'bathroom counter', 'bed', 'bench', 'board game', 'bookshelf', 'cabinet', 'chair', 'coffee table', 'cutting board', 'desk', 'floor', 'frying pan', 'kitchen cabinet', 'kitchen counter', 'kitchen table', 'mouse mat', 'nightstand', 'oven tray', 'plate', 'radio', 'rug', 'sofa', 'stove', 'towel rack']
        
        skill_set = self.skill_set
        
        if "grab " in previous_step:
            grabbed_obj = previous_step.split('grab the ')[1]
            for obj_nl in nl_obj_list:
                if obj_nl in nl_putin_objs:
                    skill_set.append(f'put the {grabbed_obj} in the {obj_nl}')
                if obj_nl in nl_putback_objs:
                    skill_set.append(f'put the {grabbed_obj} on the {obj_nl}')
        ### Case 2. previous_step: put something ?? obj -> remove put grabbed_obj ?? obj
        elif "put " in previous_step:
            if " on " in previous_step:
                # grabbed_obj = previous_step.split("put ")[1].split(" on ")[0]                
                grabbed_obj = previous_step.split("put the ")[1].split(" on the ")[0]
            elif " in " in previous_step:
                # grabbed_obj = previous_step.split("put ")[1].split(" in ")[0]
                grabbed_obj = previous_step.split("put the ")[1].split(" in the ")[0]
            skill_set = [skill for skill in skill_set if not f"put the {grabbed_obj}" in skill]
            # pdb.set_trace()
        self.skill_set = list(set(skill_set))