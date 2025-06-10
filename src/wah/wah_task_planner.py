from task_planner import TaskPlanner
from wah.wah_utils import find_indefinite_article, divide_total_into_keys
import json
import random
import torch

import pdb


class WahTaskPlanner(TaskPlanner):
    def init_prompt(self, cfg):
        ### Initialize with same number of examples for task type
        prefix = cfg.prompt.prefix
        example_file_path = cfg.prompt.example_file_path
        num_examples = cfg.prompt.num_examples
        splitter = cfg.prompt.splitter
        seed = cfg.prompt.seed
        select_method = cfg.prompt.select_method # Initialize the prompt with same number of examples for task type
        
        ### Load examples
        with open(example_file_path, 'r') as file:
            prompt_ex = json.load(file)

        if select_method == "uniform":
            task_name2task_d = self.get_task_name2task_d(prompt_ex)
            selected_examples = self.select_examples_uniform(task_name2task_d, num_examples, seed)
            prompt = self.make_prompt(prefix, splitter, selected_examples, seed)
            
        elif select_method == "same_task":
            task_name2task_d = self.get_task_name2task_d(prompt_ex)
            selected_examples_dict = {task_name: self.select_examples_same_task(task_name2task_d, num_examples, seed, task_name) for task_name in ['prepare_snack', 'prepare_food', 'setup_table', 'put_dishwasher', 'put_fridge']}
            self.prompt_dict = {task_name: self.make_prompt(prefix, splitter, selected_examples_dict[task_name], seed) for task_name in ['prepare_snack', 'prepare_food', 'setup_table', 'put_dishwasher', 'put_fridge']}
            prompt = None
        elif select_method == "topk":
            from sentence_transformers import SentenceTransformer
            self.sentence_emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.inst2_task_d_w_emb = self.get_inst2task_d_w_emb(prompt_ex)
            prompt = None
        else:
            raise NotImplementedError()        

        return prompt
    
    def get_task_name2task_d(self, prompt_ex):
        task_name2task_d ={
            'prepare_snack': [], 
            'prepare_food': [], 
            'setup_table': [],
            'put_dishwasher': [],
            'put_fridge': []
            }
        for ex_d in prompt_ex:
            task_name2task_d[ex_d['task_name']].append(ex_d)
        return task_name2task_d
    
    def get_inst2task_d_w_emb(self, prompt_ex):
        inst2task_d = {}
        for ex in prompt_ex:
            nl_instructions = ex['nl_instructions']
            for nl_instruction in nl_instructions:
                inst2task_d[nl_instruction] = ex
        
        from sentence_transformers import SentenceTransformer
        sentences = list(inst2task_d.keys())
        model = self.sentence_emb_model
        embeddings = model.encode(sentences, convert_to_tensor=True)
        for i, sentence in enumerate(sentences):
            embedding = embeddings[i]
            inst2task_d[sentence]['embedding'] = embedding.cpu()
        return inst2task_d


    def select_examples_uniform(self, task_name2task_d, num_examples, seed):
        task_names = list(task_name2task_d.keys())
        task_per_num = divide_total_into_keys(task_names, num_examples)
        
        random.seed(seed)
        selected_examples = []
        if num_examples == 0:
            pass
        else:
            for task_name in task_names:
                num_task_ex = task_per_num[task_name]
                selected_examples += random.sample(task_name2task_d[task_name], num_task_ex)
        return selected_examples
    
    def select_examples_same_task(self, task_name2task_d, num_examples, seed, query_task_name):
        random.seed(seed)
        same_task_dataset = task_name2task_d[query_task_name]
        selected_examples = random.sample(same_task_dataset, num_examples)
        return selected_examples
    
    def select_examples_topk(self, nl_instruction):
        inst_emb = self.sentence_emb_model.encode(nl_instruction, convert_to_tensor=True).cpu()
        num_exs = self.cfg.prompt.num_examples
        similarities = {nl_inst: torch.dot(inst_emb, self.inst2_task_d_w_emb[nl_inst]['embedding']).item() for nl_inst in list(self.inst2_task_d_w_emb.keys())}
        sorted_keys = sorted(similarities, key=lambda k: similarities[k], reverse=True)
        return sorted_keys[:num_exs]
                      
    def make_prompt(self, prefix, splitter, selected_examples, seed):
        random.seed(seed)
        random.shuffle(selected_examples)
        prompt = f"{prefix}{splitter}"
        
        for task_d in selected_examples:
            instruction = random.choice(task_d['nl_instructions'])        
            prompt_for_example = f"Human: {instruction}\nRobot:"

            for step_id, step in enumerate(task_d['task_plan']):
                prompt_for_example += f" {step_id+1}. {step},"
                
            prompt_for_example += f" {step_id + 2}. done.\n"
            prompt += f"{prompt_for_example}{splitter}"
        
        return prompt
    
    def reset(self, nl_act_list, nl_obj_list, task_d, nl_instruction):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)
        # self.prompt = self.init_prompt(self.cfg, task_d)
        if self.cfg.prompt.select_method == "uniform":
            pass
        elif self.cfg.prompt.select_method == "same_task":
            query_task_name = task_d['task_name']
            self.prompt = self.prompt_dict[query_task_name]
        elif self.cfg.prompt.select_method == "topk":
            topk_instructions = self.select_examples_topk(nl_instruction)
            selected_examples = [self.inst2_task_d_w_emb[instruction] for instruction in topk_instructions]
            self.prompt = self.make_prompt(self.cfg.prompt.prefix, self.cfg.prompt.splitter, selected_examples, self.cfg.prompt.seed)
        else:
            raise NotImplementedError()    
 
    def init_skill_set(self, nl_act_list, nl_obj_list):
        skill_set = ["done", "done.", "done.\n"]
        nl_all_objs = ['alcohol', 'amplifier', 'apple', 'balance ball', 'bananas', 'bar soap', 'bathroom', 'bathroom cabinet', 'bathroom counter', 'bathtub', 'bed', 'bedroom', 'bell pepper', 'bench', 'board game', 'book', 'bookshelf', 'bottled water', 'box', 'slice of bread', 'bucket', 'cabinet', 'candle', 'candy bar', 'carrot', 'ceiling', 'ceiling fan', 'ceilinglamp', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'closet', 'closet drawer', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee maker', 'coffee pot', 'coffee table', 'computer', 'condiment bottle', 'condiment shaker', 'cooking pot', 'cpu screen', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'curtains', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'deodorant', 'desk', 'bowl', 'dishwasher', 'dishwashing liquid', 'door', 'doorjamb', 'face cream', 'faucet', 'floor', 'folder', 'fridge', 'frying pan', 'game', 'garbage can', 'glasses', 'globe', 'guitar', 'hair product', 'hanger', 'juice', 'keyboard', 'kitchen', 'kitchen cabinet', 'kitchen counter', 'kitchen counter drawer', 'kitchen table', 'knife block', 'light switch', 'lime', 'living room', 'longboard', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'milkshake', 'minced meat', 'mouse', 'mouse mat', 'mug', 'nightstand', 'notes', 'orchid', 'oven tray', 'painkillers', 'pancake', 'paper', 'paper tray', 'peach', 'pear', 'perfume', 'photo frame', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'power socket', 'printer', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmod', 'shelf', 'sink', 'slippers', 'sofa', 'speaker', 'ball', 'stall', 'standing mirror', 'stove', 'stove fan', 'sundae', 'table lamp', 'teddy bear', 'toaster', 'toilet', 'toiletpaper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'tv', 'tv stand', 'vase', 'wall', 'wall lamp', 'wall phone', 'wall picture frame', 'wall shelf', 'washing machine', 'washing sponge', 'water glass', 'whipped cream', 'window', 'wine', 'wine glass']
        nl_pick_objs = ['alcohol', 'apple', 'bananas', 'bar soap', 'bell pepper', 'board game', 'book', 'box', 'slice of bread', 'bucket', 'candle', 'candy bar', 'carrot', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee pot', 'condiment bottle', 'condiment shaker', 'cooking pot', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'bowl', 'dishwashing liquid', 'face cream', 'folder', 'frying pan', 'glasses', 'globe', 'hair product', 'hanger', 'juice', 'keyboard', 'lime', 'lotion bottle', 'magazine', 'milk', 'milkshake', 'minced meat', 'mouse', 'mug', 'notes', 'oven tray', 'pancake', 'paper', 'pear', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmod', 'slippers', 'ball', 'sundae', 'teddy bear', 'toiletpaper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'wall phone', 'wall picture frame', 'washing sponge', 'water glass', 'whipped cream', 'wine', 'wine glass']
        nl_open_objs = ['bathroom cabinet', 'book', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'coffee pot', 'cooking pot', 'curtains', 'desk', 'dishwasher', 'door', 'folder', 'fridge', 'garbage can', 'hair product', 'kitchen cabinet', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'nightstand', 'printer', 'radio', 'stove', 'toilet', 'toothpaste', 'washing machine', 'window']
        nl_switch_objs = ['candle', 'cell phone', 'clock', 'coffee maker', 'computer', 'dishwasher', 'faucet', 'fridge', 'light switch', 'microwave oven', 'printer', 'radio', 'remote control', 'stove', 'toaster', 'tv', 'wall phone', 'washing machine']
        nl_putin_objs = ['bathroom cabinet', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'dishwasher', 'folder', 'fridge', 'frying pan', 'garbage can', 'kitchen cabinet', 'microwave oven', 'nightstand', 'printer', 'sink', 'stove', 'toaster', 'toilet', 'washing machine']
        nl_putback_objs = ['bathroom cabinet', 'bathroom counter', 'bed', 'bench', 'board game', 'bookshelf', 'cabinet', 'chair', 'coffee table', 'cutting board', 'desk', 'floor', 'frying pan', 'kitchen cabinet', 'kitchen counter', 'kitchen table', 'mouse mat', 'nightstand', 'oven tray', 'plate', 'radio', 'rug', 'sofa', 'stove', 'towel rack']
        # nl_receptacle_objs = ['cabinet', 'kitchen cabinet', 'stove', 'garbage can', 'sink', 'washing machine', 'bathroom cabinet', 'microwave oven', 'coffee maker', 'box', 'closet', 'nightstand', 'dishwasher', 'toilet', 'bookshelf', 'fridge']
        nl_find_objs = list(set(nl_pick_objs + nl_open_objs + nl_switch_objs + nl_putin_objs + nl_putback_objs))

        for act_nl in nl_act_list:
            if act_nl == "find":
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_find_objs:
                        article = find_indefinite_article(obj_nl)
                        skill_set.append(f"{act_nl} {article} {obj_nl}")
            elif act_nl == "pick up":
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_pick_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl in ["open", "close"]:
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_open_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl in ["switch on"]:
                for obj_nl in nl_obj_list:
                    if obj_nl in nl_switch_objs:
                        skill_set.append(f"{act_nl} the {obj_nl}")
            elif act_nl == "put down":
                for obj_nl in nl_obj_list:
                    skill_set.append(f"put down the {obj_nl}")
            else:
                NotImplementedError
        return skill_set    

    def update_skill_set(self, previous_step, nl_obj_list):
        if self.cfg.planner.dynamic_skill_set:
            ### Case 1. previous_step: grab something -> add put grabbed_obj ?? obj
            nl_putin_objs = ['bathroom cabinet', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'dishwasher', 'folder', 'fridge', 'frying pan', 'garbage can', 'kitchen cabinet', 'microwave oven', 'nightstand', 'printer', 'sink', 'stove', 'toaster', 'toilet', 'washing machine']
            nl_putback_objs = ['bathroom cabinet', 'bathroom counter', 'bed', 'bench', 'board game', 'bookshelf', 'cabinet', 'chair', 'coffee table', 'cutting board', 'desk', 'floor', 'frying pan', 'kitchen cabinet', 'kitchen counter', 'kitchen table', 'mouse mat', 'nightstand', 'oven tray', 'plate', 'radio', 'rug', 'sofa', 'stove', 'towel rack']
            
            skill_set = self.skill_set
            
            if "pick up " in previous_step:
                grabbed_obj = previous_step.split('pick up the ')[1]
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
        else:
            pass