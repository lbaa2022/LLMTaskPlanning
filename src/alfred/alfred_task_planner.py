import json
import random
from collections import defaultdict
import logging

from src.task_planner import TaskPlanner
from src.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

log = logging.getLogger(__name__)

class AlfredTaskPlanner(TaskPlanner):
    alfred_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                   'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl',
                   'Toilet', 'DiningTable', 'Spoon', 'TissueBox', 'Shelf', 'Apple', 'TennisRacket', 'SoapBar',
                   'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'CoffeeTable', 'Spatula', 'Plate', 'Bed',
                   'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave', 'CounterTop',
                   'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Cabinet',
                   'Desk', 'Newspaper', 'Drawer', 'Sofa', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
                   'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
                   'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
                   'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Sink']
    alfred_pick_obj = ['KeyChain', 'Potato', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
                       'SprayBottle', 'BaseballBat', 'Bread', 'CellPhone', 'Book', 'Lettuce', 'CreditCard', 'Mug',
                       'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
                       'Apple', 'TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
                       'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
                       'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato',
                       'ButterKnife', 'WateringCan', 'Vase']
    alfred_open_obj = ['Safe', 'Laptop', 'Fridge', 'Box', 'Microwave', 'Cabinet', 'Drawer']
    alfred_slice_obj = ['Potato', 'Lettuce', 'Tomato', 'Apple', 'Bread']
    alfred_toggle_obj = ['Microwave', 'DeskLamp', 'FloorLamp', 'Faucet']
    alfred_recep = ['ArmChair', 'Safe', 'Cart', 'Ottoman', 'Pot', 'CoffeeMachine', 'Desk', 'Cabinet', 'Pan',
                    'Drawer', 'Sofa', 'Mug', 'StoveBurner', 'SideTable', 'Toilet', 'Bowl', 'Box', 'DiningTable',
                    'Shelf', 'ToiletPaperHanger', 'CoffeeTable', 'Cup', 'Plate', 'Bathtub', 'Bed', 'Dresser',
                    'Fridge', 'Microwave', 'CounterTop', 'Sink', 'GarbageCan']

    def init_prompt(self, cfg):
        if cfg.planner.use_predefined_prompt:
            # use predefined prompt
            return self.load_prompt(cfg)

        prefix = cfg.prompt.prefix
        splitter = cfg.prompt.splitter

        # load examples
        examples = defaultdict(list)
        with open(cfg.prompt.example_file_path, 'r') as f:
            examples_json = json.load(f)
            for e in examples_json:
                examples[e['task type']].append(e)

        # example sampling
        examples_selected = []
        # task_types = ['pick_and_place_simple', 'look_at_obj_in_light', 'pick_two_obj_and_place',
        #               'pick_cool_then_place_in_recep',
        #               'pick_clean_then_place_in_recep', 'pick_and_place_with_movable_recep']
        task_types = ['pick_and_place_simple', 'look_at_obj_in_light', 
                      'pick_and_place_with_movable_recep', 'pick_cool_then_place_in_recep', 
                      'pick_heat_then_place_in_recep', 'pick_clean_then_place_in_recep']
        num_examples_per_task_max = 10
        num_examples_per_task = int(cfg.prompt.num_examples / len(task_types))
        assert num_examples_per_task < num_examples_per_task_max
        for k in task_types:
            assert k in examples.keys()
            candidates = random.sample(examples[k], num_examples_per_task_max)  # sample a fixed random set for reproducibility
            examples_selected.extend(candidates[:num_examples_per_task])

        # make prompt string
        sentence_ending = '\n'
        prompt = f"{prefix}{splitter}"
        for e in examples_selected:
            task_desc = e["task description"].strip()
            if task_desc[-1].isalnum():
                task_desc += '.'
            task_desc = task_desc.capitalize()
            prompt += f'Human: {task_desc}' + sentence_ending
            prompt += 'Robot: '
            last_i = 0
            for i, step in enumerate(e['NL steps']):
                prompt += f'{i+1}. {step}, '
                last_i = i+1
            prompt += f'{last_i+1}. done.' + sentence_ending

        return prompt

    def load_prompt(self, cfg):
        with open('resource/predefined_prompt.txt') as f:
            prompt = f.read()
        return prompt

    def init_skill_set(self):
        # convert ithor names to natural words
        alfred_objs = [ithor_name_to_natural_word(w) for w in self.alfred_objs]
        alfred_pick_obj = [ithor_name_to_natural_word(w) for w in self.alfred_pick_obj]
        alfred_open_obj = [ithor_name_to_natural_word(w) for w in self.alfred_open_obj]
        alfred_slice_obj = [ithor_name_to_natural_word(w) for w in self.alfred_slice_obj]
        alfred_toggle_obj = [ithor_name_to_natural_word(w) for w in self.alfred_toggle_obj]
        alfred_recep = [ithor_name_to_natural_word(w) for w in self.alfred_recep]

        # make skill sets
        skills = ['done']

        # find
        for o in set(alfred_pick_obj + alfred_slice_obj + alfred_open_obj + alfred_toggle_obj + alfred_recep):
            article = find_indefinite_article(o)
            skills.append(f'find {article} {o}')

        # pick / put
        for o in alfred_pick_obj:
            skills.append(f'pick up the {o}')
            skills.append(f'put down the {o}')

        # open / close
        for o in alfred_open_obj:
            skills.append(f'open the {o}')
            skills.append(f'close the {o}')

        # slice
        for o in alfred_slice_obj:
            skills.append(f'slice the {o}')

        # turn on / off
        for o in alfred_toggle_obj:
            skills.append(f'turn on the {o}')
            skills.append(f'turn off the {o}')

        # add a leading whitespace
        skills = [' ' + c for c in skills]

        log.info(f'# of skills: {len(skills)}')
        log.info(skills)

        return skills
