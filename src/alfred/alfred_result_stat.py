import os
import json
import numpy as np
from collections import defaultdict
import argparse


def get_json_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            files.append(file)
    return files


def get_json_data(path, file):
    with open(path + '/' + file) as f:
        data = json.load(f)
    return data


def main(path):
    files = get_json_files(path)
    data = []
    for file in files:
        data.append(get_json_data(path, file))
    data_by_type = defaultdict(list)
    for i in range(len(data)):
        task_type = data[i]['type']
        data_by_type[task_type].append(data[i])

    # calculate stats
    for key in data_by_type:
        print(key)
        n = len(data_by_type[key])
        print(f'number of trials: {n}')
        n_success = np.sum([d["success"] for d in data_by_type[key]])
        print(f'success rate: {np.mean([d["success"] for d in data_by_type[key]])*100:.2f} ({n_success}/{n})')
        print('')

    # total stat
    print('total success rate: {:.2f}'.format(np.mean([d['success'] for d in data]) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=str)
    p = parser.parse_args()

    main(p.result_path)
