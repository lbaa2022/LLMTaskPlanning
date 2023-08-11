import random
from collections import defaultdict
from pathlib import Path
import json

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from src.alfred.alfred_task_planner import AlfredTaskPlanner

data_path = 'alfred/data/json_2.1.0'


def list_actions_objs():
    splits = ['valid_seen', 'valid_unseen', 'train']

    action_set = set()
    object_set = set()
    receptacle_set = set()
    action_obj = defaultdict(set)
    action_receptacle = defaultdict(set)

    for split in splits:
        base_path = Path(data_path) / split

        for path in base_path.glob("**/traj_data.json"):
            print(path)
            with open(path) as f:
                data = json.load(f)
            steps = data['plan']['low_actions']
            for s in steps:
                action = s['api_action']['action']
                action_set.add(action)
                if 'objectId' in s['api_action']:
                    obj_name = s['api_action']['objectId'].split('|')[0]
                    object_set.add(obj_name)
                    action_obj[action].add(obj_name)
                if 'receptacleObjectId' in s['api_action']:
                    recep_name = s['api_action']['receptacleObjectId'].split('|')[0]
                    receptacle_set.add(recep_name)
                    action_receptacle[action].add(recep_name)

    print('actions', len(action_set), action_set)
    print('objections', len(object_set), object_set)
    print('receptacles', len(receptacle_set), receptacle_set)
    all_things = object_set.union(receptacle_set)
    print('obj+recep', len(all_things), all_things)
    print('action-obj', action_obj)
    print('action-recep', action_receptacle)


def load_tasks(split='train'):
    base_path = Path(data_path) / split
    tasks = defaultdict(list)
    for path in sorted(base_path.glob("**/traj_data.json")):
        with open(path) as f:
            data = json.load(f)
        # print(data)
        task_type = data['task_type']
        tasks[task_type].append(data)

    return tasks


def convert_action_to_nl_skill(action, next_action, args):
    """ not accurate. need to manually update converted skills """
    from src.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

    steps = []

    args_converted = []  # e.g., desklamp -> DeskLamp
    alfred_obj_list = [x.lower() for x in AlfredTaskPlanner.alfred_objs]
    for a in args:
        if len(a) == 0:
            continue
        elif 'basin' in a:
            a = a.replace('basin', '')
        elif a == 'handtowelholder':
            a = 'handtowel'
        assert a in alfred_obj_list, f'unknown object {a}'
        idx = alfred_obj_list.index(a)
        args_converted.append(AlfredTaskPlanner.alfred_objs[idx])

    args_converted = [ithor_name_to_natural_word(e) for e in args_converted]
    if len(args_converted) > 0:
        o = args_converted[0]
    elif action != 'NoOp':
        raise ValueError(f'no objects in args')

    if action == 'GotoLocation':
        if next_action == 'OpenObject' or next_action == 'PickupObject' or next_action == 'SliceObject':
            # we avoid using receptacle names if the next step is open/pickup
            # we use the object name that a robot interact with instead of receptacle names
            pass
        else:
            steps.append(f'find {find_indefinite_article(o)} {o}')
    elif action == 'OpenObject':
        steps.append(f'find {find_indefinite_article(o)} {o}')
        steps.append(f'open the {o}')
    elif action == 'CloseObject':
        steps.append(f'close the {o}')
    elif action == 'PutObject':
        steps.append(f'put down the {o}')
    elif action == 'PickupObject':
        steps.append(f'find {find_indefinite_article(o)} {o}')
        steps.append(f'pick up the {o}')
    elif action == 'CleanObject':
        steps.append(f'put down the {o}')
        steps.append(f'turn on the faucet')
        steps.append(f'turn off the faucet')
        steps.append(f'pick up the {o}')
    elif action == 'HeatObject':
        # note: incorrect if the robot uses a stove
        steps.append(f'open the microwave')
        steps.append(f'put down the {o}')
        steps.append(f'close the microwave')
        steps.append(f'turn on the microwave')
        steps.append(f'open the microwave')
        steps.append(f'pick up the {o}')
    elif action == 'CoolObject':
        steps.append(f'open the fridge')
        steps.append(f'put down the {o}')
        steps.append(f'close the fridge')
        steps.append(f'open the fridge')
        steps.append(f'pick up the {o}')
    elif action == 'ToggleObject':
        steps.append(f'turn on the {o}')
    elif action == 'SliceObject':
        steps.append(f'find {find_indefinite_article(o)} {o}')
        steps.append(f'slice the {o}')
    elif action == 'NoOp':
        pass
    else:
        assert False, f'unexpected action {action}, {args}'

    return steps


def convert_low_level_action_to_nl_skill(action, args, cur_obj):
    from src.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

    steps = []
    ret_obj = None

    def obj_id_to_nl(s):
        return ithor_name_to_natural_word(s.split('|')[0])

    if action == 'OpenObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o:
            steps.append(f'find {find_indefinite_article(o)} {o}')
        ret_obj = o
        steps.append(f'open the {o}')
    elif action == 'CloseObject':
        o = obj_id_to_nl(args['objectId'])
        ret_obj = o
        steps.append(f'close the {o}')
    elif action == 'PutObject':
        o_recep = obj_id_to_nl(args['receptacleObjectId'])
        if cur_obj != o_recep:
            steps.append(f'find {find_indefinite_article(o_recep)} {o_recep}')
        ret_obj = o_recep
        o = obj_id_to_nl(args['objectId'])
        steps.append(f'put down the {o}')
    elif action == 'PickupObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o:
            steps.append(f'find {find_indefinite_article(o)} {o}')
        ret_obj = o
        steps.append(f'pick up the {o}')
    elif action == 'ToggleObjectOn':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o:
            steps.append(f'find {find_indefinite_article(o)} {o}')
        ret_obj = o
        steps.append(f'turn on the {o}')
    elif action == 'ToggleObjectOff':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o:
            steps.append(f'find {find_indefinite_article(o)} {o}')
        ret_obj = o
        steps.append(f'turn off the {o}')
    elif action == 'SliceObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o:
            steps.append(f'find {find_indefinite_article(o)} {o}')
        ret_obj = o
        steps.append(f'slice the {o}')
    else:
        pass

    return steps, ret_obj

def export_train_examples(export=True, export_text_samples=True):
    tasks = load_tasks('train')
    selected_samples = []

    for key in sorted(tasks.keys()):
        print('---------------------------------------------')
        print(key)
        print('N=', len(tasks[key]))
        print('---------------------------------------------')
        print()

        samples = tasks[key]  # all samples
        # samples = random.sample(tasks[key], 10)  # sampling

        for e in samples:
            print('Task id:', e['task_id'])

            # exclude pick_two_obj_and_place type
            if e['task_type'] == 'pick_two_obj_and_place':
                continue

            NL_steps = []
            print('PDDL high-level actions:')
            skip_this_sample = False
            # for s_i, s in enumerate(e['plan']['high_pddl']):
            #     action = s['discrete_action']['action']
            #     next_action = None
            #     if s_i < len(e['plan']['high_pddl']) - 1:
            #         next_action = e['plan']['high_pddl'][s_i + 1]['discrete_action']['action']
            #     args = s['discrete_action']['args']
            #     print('  ', action, args)
            #
            #     try:
            #         nl_skill = convert_action_to_nl_skill(action, next_action, args)
            #     except ValueError:
            #         skip_this_sample = True
            #         break
            #
            #     if nl_skill:
            #         NL_steps.extend(nl_skill)

            cur_obj = None
            for s_i, s in enumerate(e['plan']['low_actions']):
                action = s['api_action']['action']
                next_action = None
                if s_i < len(e['plan']['low_actions']) - 1:
                    next_action = e['plan']['low_actions'][s_i + 1]['api_action']['action']
                args = s['api_action']
                print('  ', action, args)

                try:
                    nl_skill, cur_obj = convert_low_level_action_to_nl_skill(action, args, cur_obj)
                except ValueError:
                    skip_this_sample = True
                    break

                if nl_skill:
                    NL_steps.extend(nl_skill)

            if skip_this_sample or len(NL_steps) <= 0:
                continue

            n_annotations = len(e['turk_annotations']['anns'])
            for ann_i in range(n_annotations):
                print('Task desc:', e['turk_annotations']['anns'][ann_i]['task_desc'])
                print('Step descriptions:')
                for s in e['turk_annotations']['anns'][ann_i]['high_descs']:
                    print('  ', s)

                print('Converted NL skills')
                for s in NL_steps:
                    print('  ', s)
                print(NL_steps)
                selected_samples.append({
                    'task type': key,
                    'task id': e['task_id'],
                    'task description': e['turk_annotations']['anns'][ann_i]['task_desc'].strip(),
                    'step description': e['turk_annotations']['anns'][ann_i]['high_descs'],
                    'NL steps': NL_steps,
                    # 'PDDL steps': e['plan']['high_pddl']
                    })
                print()

    if export:
        with open('resource/alfred_examples_for_prompt.json', 'w') as fp:
            json.dump(selected_samples, fp, indent=4)

    if export_text_samples:
        with open(f'resource/alfred_train_text_samples.txt', 'w') as fp:
            for sample in selected_samples:
                # fp.write(f'TaskId: {sample["task id"]}\n')
                fp.write(f'Human: {sample["task description"]}\n')
                fp.write(f'Robot: ')
                for i, step in enumerate(sample['NL steps']):
                    fp.write(f"{i+1}. {step}, ")
                fp.write(f"{i+2}. done.\n")


if __name__ == '__main__':
    # list_actions_objs()
    export_train_examples(True, True)
