import time

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
sys.path.insert(0, 'src')
sys.path.insert(0, './alfred')

from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import load_task_json


def main():
    task = {'task': 'pick_clean_then_place_in_recep-SoapBar-None-Toilet-410/trial_T20190906_201106_979461', 'repeat_idx': 1}
    traj_data = load_task_json(task)
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    env = ThorConnector(x_display='1')
    env.reset('FloorPlan%d' % scene_num)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data['scene']['init_action']))

    instructions = [
        'find a soap bar',
        'pick up the soap bar',
        'find a toilet',
        'put down the soap bar',
        'find a soap bar',
        # 'find a sink',
        # 'find a sink',
        # 'open the fridge',
        # 'put down the plate',
        # 'put down the plate',
        # 'put down the plate',
    ]

    imgs = []
    for i, instruction in enumerate(instructions):
        ret_dict = env.llm_skill_interact(instruction)
        if not ret_dict['success']:
            print(ret_dict['message'])
        imgs.append(env.write_step_on_img(i, instruction))

    # save_result(None, imgs, 'results_test')


def minimal_test():
    import ai2thor.controller

    controller = ai2thor.controller.Controller()
    controller.start()

    # controller.reset('FloorPlan28')
    # controller.step(dict(action='Initialize', gridSize=0.25))

    for i in range(10):
        controller.step(dict(action='RotateRight'))
        print('rotate')
        time.sleep(0.1)


if __name__ == '__main__':
    # minimal_test()
    main()
