import os, json, re
import string
import subprocess


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_task_json(task, split='valid_seen'):
    '''
    load task json from disk, trying preprocessed first then raw
    '''
    base_path = 'alfred/data/json_2.1.0'
    task_path = task['task']

    # Try preprocessed file first (at top level)
    pp_path = os.path.join(base_path, task_path, 'pp', 'ann_%d.json' % task['repeat_idx'])
    if os.path.exists(pp_path):
        with open(pp_path) as f:
            return json.load(f)

    # Try raw traj_data.json under split folder
    raw_path = os.path.join(base_path, split, task_path, 'traj_data.json')
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            data = json.load(f)
        # Add root field needed by evaluator
        data['root'] = os.path.join(base_path, split, task_path)
        data['repeat_idx'] = task['repeat_idx']
        return data

    # Last fallback: try at top level without split
    fallback_path = os.path.join(base_path, task_path, 'traj_data.json')
    if os.path.exists(fallback_path):
        with open(fallback_path) as f:
            data = json.load(f)
        data['root'] = os.path.join(base_path, task_path)
        data['repeat_idx'] = task['repeat_idx']
        return data

    raise FileNotFoundError(f"Could not find task data at: {pp_path}, {raw_path}, or {fallback_path}")


def print_gpu_usage(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


def ithor_name_to_natural_word(w):
    # e.g., RemoteController -> remote controller
    if w == 'CD':
        return w
    else:
        return re.sub(r"(\w)([A-Z])", r"\1 \2", w).lower()


def natural_word_to_ithor_name(w):
    # e.g., floor lamp -> FloorLamp
    if w == 'CD':
        return w
    else:
        return ''.join([string.capwords(x) for x in w.split()])


def find_indefinite_article(w):
    # simple rule, not always correct
    w = w.lower()
    if w[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    else:
        return 'a'
