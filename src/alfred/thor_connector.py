import os, math, re
import textwrap

import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

from env.thor_env import ThorEnv
from gen import constants
from gen.utils.game_util import get_objects_with_name_and_prop
from alfred.utils import natural_word_to_ithor_name


log = logging.getLogger(__name__)

class ThorConnector(ThorEnv):
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        super().__init__(x_display, player_screen_height, player_screen_width, quality, build_path)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None

    def get_reachable_positions(self):
        free_positions = super().step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def write_step_on_img(self, idx, description):
        img = Image.fromarray(self.last_event.frame)
        text = str(idx) + ':' + description
        lines = textwrap.wrap(text, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        return img

    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def llm_skill_interact(self, instruction: str):
        print(instruction)
        # todo: parsing 없애자. action type 이 주어진다고 가정?
        if instruction.startswith("find "):
            obj_name = instruction.replace('find a ', '').replace('find an ', '')
            self.cur_receptacle = obj_name
            ret = self.nav_obj(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("pick up "):
            obj_name = instruction.replace('pick up the ', '')
            ret = self.pick(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("put down "):
            # m = re.match(r'put down (.+) on (.+)', instruction)
            # obj = m.group(1).replace('the ', '')
            # receptacle = m.group(2).replace('the ', '')
            m = re.match(r'put down (.+)', instruction)
            obj = m.group(1).replace('the ', '')
            receptacle = self.cur_receptacle
            ret = self.put(natural_word_to_ithor_name(receptacle))
        elif instruction.startswith("open "):
            obj_name = instruction.replace('open the ', '')
            ret = self.open(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("close "):
            obj_name = instruction.replace('close the ', '')
            ret = self.close(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("turn on "):
            obj_name = instruction.replace('turn on the ', '')
            ret = self.toggleon(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("turn off "):
            obj_name = instruction.replace('turn off the ', '')
            ret = self.toggleoff(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("slice "):
            obj_name = instruction.replace('slice the ', '')
            ret = self.slice(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("drop"):
            ret = self.drop()
        else:
            assert False, 'instruction not supported'

        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"action failed: {self.last_event.metadata['errorMessage']}")

        ret_dict = {
            'action': instruction,
            'success': len(ret) <= 0,
            'message': ret
        }

        return ret_dict

    def nav_obj(self, target_obj: str):
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        ret_msg = ''
        print(f'{action_name} ({target_obj})')

        # get the object location
        obj_idx = -1
        min_dist = 1e+8
        for i, obj in enumerate(objects):
            obj_type = obj['objectId'].split('|')[0].lower()
            if obj_type.casefold() == target_obj.casefold():
                if obj["distance"] < min_dist:  # choose the closest one
                    obj_idx = i
                    pick_penalty = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    if obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            openable = get_objects_with_name_and_prop(p, 'openable', self.last_event.metadata)
                            if len(openable) > 0:
                                pick_penalty = 10000
                                break
                    min_dist = obj["distance"] + pick_penalty

        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}'
            print(ret_msg)
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            n_attempts = 10
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']

            for i in range(n_attempts):
                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], i + 1)

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                super().step(dict(action="TeleportFull",
                                  x=closest_loc[0], y=self.agent_height, z=closest_loc[2],
                                  rotation=rot_angle, horizon=-hor_angle))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}")
                    print('try again')
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}'

        return ret_msg

    def get_obj_id_from_name(self, obj_name, only_pickupable=False, only_toggleable=False, get_inherited=False):
        obj_id = None
        obj_data = None
        min_distance = 1e+8
        for obj in self.last_event.metadata['objects']:
            if (only_pickupable is False or obj['pickupable']) and \
                    (only_toggleable is False or obj['toggleable']) and \
                    obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and \
                    (get_inherited is False or len(obj['objectId'].split('|')) == 5):
                if obj["distance"] < min_distance:
                    obj_id = obj["objectId"]
                    min_distance = obj["distance"]
                    obj_data = obj

        return obj_id, obj_data

    def pick(self, obj_name):
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_pickupable=True)
        ret_msg = ''
        print(f'pick {obj_id}')

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to pick up'
        elif obj_data['visible'] is False and len(obj_data['parentReceptacles']) > 0:
            recep_name = obj_data["parentReceptacles"][0].split('|')[0]
            ret_msg = f'{obj_name} is not visible because it is in {recep_name}'
        else:
            super().step(dict(
                action="PickupObject",
                objectId=obj_id,
                forceAction=False  # todo: need forceaction?
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"PickupObject action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f'Cannot pick up {obj_name}'

        return ret_msg

    def put(self, receptacle_name):
        # assume the agent always put the object currently holding
        ret_msg = ''
        holding_obj_id = None
        all_objects = self.last_event.metadata['objects']
        for o in all_objects:
            if o['isPickedUp']:
                holding_obj_id = o['objectId']
                break

        if not holding_obj_id:
            ret_msg = f'Robot is not holding any object'
            log.warning(ret_msg)
            return ret_msg

        for i in range(2):
            if i == 0:
                recep_id, _ = self.get_obj_id_from_name(receptacle_name)
            else:
                recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True)

            if not recep_id:
                ret_msg = f'Cannot find {receptacle_name}'
                log.warning(ret_msg)
                break

            print(f'put {holding_obj_id} on {recep_id}')

            # this somehow make putobject success in some cases (for instance, put the book on the sofa).
            # todo: investigate this
            super().step(dict(
                action="RotateHand",
                x=40
            ))

            super().step(dict(
                action="PutObject",
                objectId=holding_obj_id,
                receptacleObjectId=recep_id,
                forceAction=True
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"PutObject action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f'Putting the object on {receptacle_name} failed'
                print('try again')
            else:
                ret_msg = ''
                break

        return ret_msg

    def drop(self):
        ret_msg = ''
        super().step(dict(
            action="DropHandObject",
            forceAction=True
        ))

        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"DropHandObject action failed: {self.last_event.metadata['errorMessage']}")
            ret_msg = f"Drop action failed"
        else:
            ret_msg = ''

        return ret_msg

    def open(self, obj_name):
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to open'
        else:
            for i in range(2):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(f"OpenObject action failed: {self.last_event.metadata['errorMessage']}")
                    ret_msg = f"Open action failed"
                    print('move backward and try again')
                    super().step(dict(action="MoveBack"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def close(self, obj_name):
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to close'
        else:
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"CloseObject action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f"Close action failed"

        return ret_msg

    def toggleon(self, obj_name):
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn on'
        else:
            super().step(dict(
                action="ToggleObjectOn",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"ToggleObjectOn action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f"Turn on action failed"

        return ret_msg

    def toggleoff(self, obj_name):
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn off'
        else:
            super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"ToggleObjectOff action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f"Turn off action failed"

        return ret_msg

    def slice(self, obj_name):
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to slice'
        else:
            super().step(dict(
                action="SliceObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"SliceObject action failed: {self.last_event.metadata['errorMessage']}")
                ret_msg = f"Slice action failed"

        return ret_msg
