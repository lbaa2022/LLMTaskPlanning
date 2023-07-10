"""
Extract samples from the ALFRED train set for language model fine tunine
You should run 'evaluate.py' for alfred at least once to get proprocessed alfred data (see preprocess_splits function).
"""

import json
import pprint
from tqdm import tqdm

import sys
sys.path.insert(0, '.')

from src.alfred.utils import load_task_json

#TODO: alfred_examples_for_prompt.json


splits = 'alfred/data/splits/oct21.json'
with open(splits) as f:
    splits = json.load(f)
    pprint.pprint({k: len(v) for k, v in splits.items()})

train_set = splits['train']

print(len(train_set))
print(train_set[:2])

for task in tqdm(train_set):
    print(task)
    # traj_data = load_task_json(task)
    # r_idx = task['repeat_idx']
    # print(f"{task['task']}, {r_idx}")
    # print(traj_data)
    # print(traj_data.keys())
    print(task.keys())

    # for e in examples_selected:
    #         task_desc = e["task description"].strip()
    #         if task_desc[-1].isalnum():
    #             task_desc += '.'
    #         task_desc = task_desc.capitalize()
    #         prompt += f'Human: {task_desc}' + sentence_ending
    #         prompt += 'Robot: '
    #         last_i = 0
    #         for i, step in enumerate(e['NL steps']):
    #             prompt += f'{i+1}. {step}, '
    #             last_i = i+1
    #         prompt += f'{last_i+1}. done.' + sentence_ending

    break
