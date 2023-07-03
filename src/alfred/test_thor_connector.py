import time

from src.alfred.eval_alfred import save_result
from src.alfred.thor_connector import ThorConnector
from src.alfred.utils import load_task_json


def main():
    task = {'task': 'pick_and_place_with_movable_recep-DishSponge-Pot-SinkBasin-1/trial_T20190908_103955_680867', 'repeat_idx': 1}
    traj_data = load_task_json(task)
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    env = ThorConnector(x_display='1')
    env.reset('FloorPlan%d' % scene_num)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    instructions = [
        'find a pot',
        'pick up the pot'
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
