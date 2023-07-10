import random
from collections import defaultdict
from pathlib import Path
import json

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
    for path in base_path.glob("**/traj_data.json"):
        with open(path) as f:
            data = json.load(f)
        # print(data)
        task_type = data['task_type']
        tasks[task_type].append(data)

    return tasks


def convert_action_to_nl_skill(action, args):
    """ not accurate. need to manually update converted skills """
    from src.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

    steps = []
    args_converted = [ithor_name_to_natural_word(e) for e in args]

    if action == 'GotoLocation':
        o = args_converted[0]
        steps.append(f'find {find_indefinite_article(o)} {o}')
    elif action == 'OpenObject':
        steps.append(f'open the {args_converted[0]}')
    elif action == 'CloseObject':
        steps.append(f'close the {args_converted[0]}')
    elif action == 'PutObject':
        steps.append(f'put down the {args_converted[0]}')
    elif action == 'PickupObject':
        steps.append(f'pick up the {args_converted[0]}')
    elif action == 'CleanObject':
        steps.append(f'put down the {args_converted[0]}')
        steps.append(f'turn on the faucet')
        steps.append(f'turn off the faucet')
        steps.append(f'pick up the {args_converted[0]}')
    elif action == 'HeatObject':
        # note: incorrect if the robot uses a stove
        steps.append(f'open the microwave')
        steps.append(f'put down the {args_converted[0]}')
        steps.append(f'close the microwave')
        steps.append(f'turn on the microwave')
        steps.append(f'open the microwave')
        steps.append(f'pick up the {args_converted[0]}')
    elif action == 'CoolObject':
        steps.append(f'open the fridge')
        steps.append(f'put down the {args_converted[0]}')
        steps.append(f'close the fridge')
        steps.append(f'open the fridge')
        steps.append(f'pick up the {args_converted[0]}')
    elif action == 'ToggleObject':
        steps.append(f'turn on the {args_converted[0]}')
    elif action == 'SliceObject':
        steps.append(f'slice the {args_converted[0]}')
        steps.append(f'pick up the {args_converted[0]}')
    elif action == 'NoOp':
        pass
    else:
        assert False, f'unexpected action {action}, {args}'

    return steps


def export_train_examples(export=True, export_text_samples=True):
    tasks = load_tasks('train')
    selected_samples = []

    for key in tasks.keys():
        print('---------------------------------------------')
        print(key)
        print('N=', len(tasks[key]))
        print('---------------------------------------------')
        print()

        samples = tasks[key]  # all samples
        # samples = random.sample(tasks[key], 10)  # sampling

        for e in samples:
            print('Task id:', e['task_id'])
            print('Task desc:', e['turk_annotations']['anns'][0]['task_desc'])
            print('Step descriptions:')
            for s in e['turk_annotations']['anns'][0]['high_descs']:
                print('  ', s)

            NL_steps = []
            print('PDDL high-level actions:')
            for s in e['plan']['high_pddl']:
                action = s['discrete_action']['action']
                args = s['discrete_action']['args']
                print('  ', action, args)
                NL_steps.extend(convert_action_to_nl_skill(action, args))

            print('Converted NL skills')
            for s in NL_steps:
                print('  ', s)
            print(NL_steps)
            selected_samples.append({
                'task type': key,
                'task id': e['task_id'],
                'task description': e['turk_annotations']['anns'][0]['task_desc'],
                'step description': e['turk_annotations']['anns'][0]['high_descs'],
                'NL steps': NL_steps,
                'PDDL steps': e['plan']['high_pddl']
                })
            print()

    if export:
        with open('resource/alfred_examples_for_prompt.json', 'w') as fp:
            json.dump(selected_samples, fp, indent=4)

    if export_text_samples:
        with open(f'resource/alfred_train_text_samples.txt', 'w') as fp:
            for sample in selected_samples:
                fp.write(f'Human: {sample["task description"]}\n')
                fp.write(f'Robot: ')
                for i, step in enumerate(sample['NL steps']):
                    fp.write(f"{i+1}. {step}, ")
                fp.write(f"{i+2}. done.\n")


if __name__ == '__main__':
    # list_actions_objs()
    export_train_examples(True, True)
