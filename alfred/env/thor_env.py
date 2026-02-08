import cv2
import copy
import gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from env.tasks import get_task
from ai2thor.controller import Controller
import gen.utils.image_util as image_util
from gen.utils import game_util
from gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):

        # AI2-THOR 4.x compatible initialization
        import platform
        init_kwargs = {
            'quality': quality,
            'height': player_screen_height,
            'width': player_screen_width,
            'fieldOfView': 60,
        }

        # Only add x_display on Linux (not needed on macOS)
        if platform.system() == 'Linux' and x_display:
            init_kwargs['x_display'] = x_display

        # Use local build if specified and exists
        if build_path:
            import os
            if os.path.exists(build_path):
                init_kwargs['local_executable_path'] = build_path

        super().__init__(**init_kwargs)
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intemediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        print("ThorEnv started.")

    def _convert_teleport_action(self, action):
        '''
        Convert old-style TeleportFull action to AI2-THOR 4.x format
        '''
        new_action = {'action': 'TeleportFull'}

        # Handle rotation - convert single value to Vector3
        if 'rotation' in action:
            rot = action['rotation']
            if isinstance(rot, (int, float)):
                new_action['rotation'] = {'x': 0, 'y': rot, 'z': 0}
            else:
                new_action['rotation'] = rot

        # Copy position coordinates
        for key in ['x', 'y', 'z', 'horizon']:
            if key in action:
                new_action[key] = action[key]

        # Add standing parameter (required in 4.x)
        new_action['standing'] = True

        # Copy forceAction if present
        if 'forceAction' in action:
            new_action['forceAction'] = action['forceAction']

        return new_action

    def _set_objects_dirty(self):
        '''
        Set all dirtyable objects to dirty state.
        Replaces SetStateOfAllObjects which was removed in AI2-THOR 5.x
        '''
        for obj in self.last_event.metadata['objects']:
            if obj.get('dirtyable', False):
                super().step(dict(action='DirtyObject', objectId=obj['objectId'], forceAction=True))

    def _empty_fillable_objects(self):
        '''
        Empty all fillable objects that contain liquid.
        Note: EmptyLiquidObject action was removed in AI2-THOR 5.x, so this is a no-op.
        Objects start empty by default when scene is reset.
        '''
        pass  # AI2-THOR 5.x doesn't have EmptyLiquidObject action

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num

        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibilityDistance=visibility_distance,
            makeAgentsVisible=False,
        ))

        # reset task if specified
        if hasattr(self, 'task') and self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibilityDistance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        if len(object_toggles) > 0:
            super().step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        # Set dirty states for individual objects (AI2-THOR 5.x compatible)
        # Note: EmptyLiquidObject action was removed in AI2-THOR 5.x
        if dirty_and_empty:
            for obj in self.last_event.metadata['objects']:
                if obj.get('dirtyable', False):
                    super().step(dict(action='DirtyObject', objectId=obj['objectId'], forceAction=True))

        # Map old object names to current scene object names (AI2-THOR 5.x compatibility)
        # The ALFRED dataset was created with AI2-THOR 2.x which has different object IDs
        mapped_object_poses = self._map_object_poses_to_scene(object_poses)
        if mapped_object_poses:
            super().step((dict(action='SetObjectPoses', objectPoses=mapped_object_poses)))

    def _map_object_poses_to_scene(self, object_poses):
        '''
        Map object names from ALFRED dataset (AI2-THOR 2.x) to current scene (AI2-THOR 5.x).
        Objects are matched by type (e.g., 'Plate' from 'Plate_abc123').

        IMPORTANT: AI2-THOR 5.x SetObjectPoses removes objects not in the list,
        so we must include ALL scene objects to preserve furniture, etc.
        '''
        # Build mapping of object types to available scene object names
        scene_objects_by_type = {}
        scene_objects_by_name = {}  # For preserving unmapped objects
        for obj in self.last_event.metadata['objects']:
            obj_name = obj['name']
            obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
            if obj_type not in scene_objects_by_type:
                scene_objects_by_type[obj_type] = []
            scene_objects_by_type[obj_type].append(obj_name)
            scene_objects_by_name[obj_name] = obj

        # Create mapping from old unique names to new names
        old_to_new_name = {}
        type_usage_count = {}
        mapped_scene_objects = set()  # Track which scene objects get mapped

        # First pass: identify unique old names and map them
        unique_old_names = []
        seen = set()
        for pose in object_poses:
            old_name = pose.get('objectName', '')
            if old_name and old_name not in seen:
                seen.add(old_name)
                unique_old_names.append(old_name)

        # Map each unique old name to a scene object of the same type
        for old_name in unique_old_names:
            obj_type = old_name.split('_')[0] if '_' in old_name else old_name

            if obj_type in scene_objects_by_type:
                count = type_usage_count.get(obj_type, 0)
                available = scene_objects_by_type[obj_type]
                if count < len(available):
                    new_name = available[count]
                    old_to_new_name[old_name] = new_name
                    mapped_scene_objects.add(new_name)
                    type_usage_count[obj_type] = count + 1

        # Second pass: create mapped poses for objects from the trajectory
        mapped_poses = []
        for pose in object_poses:
            old_name = pose.get('objectName', '')
            if old_name in old_to_new_name:
                mapped_pose = pose.copy()
                mapped_pose['objectName'] = old_to_new_name[old_name]
                mapped_poses.append(mapped_pose)

        # Third pass: Add unmapped scene objects with their current positions
        # This is critical for AI2-THOR 5.x which removes objects not in the poses list
        # Include all objects EXCEPT structural elements that can't be repositioned
        STRUCTURAL_OBJECTS = {
            'Floor', 'Wall', 'Ceiling', 'Window', 'Door', 'Doorway',
            'LightSwitch', 'Doorframe', 'ShowerDoor', 'ShowerGlass',
            'Room', 'StandardWallSize', 'Void'
        }
        for obj_name, obj in scene_objects_by_name.items():
            if obj_name not in mapped_scene_objects:
                # Get object type from name (e.g., 'Desk' from 'Desk_abc123')
                obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name
                # Skip structural elements that can't be repositioned
                if obj_type not in STRUCTURAL_OBJECTS:
                    preserve_pose = {
                        'objectName': obj_name,
                        'position': obj['position'],
                        'rotation': obj['rotation']
                    }
                    mapped_poses.append(preserve_pose)

        return mapped_poses

    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def step(self, action=None, smooth_nav=False, **kwargs):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        # Handle string actions or keyword-only calls (used during initialization by ai2thor Controller)
        if action is None or isinstance(action, str):
            return super().step(action=action, **kwargs)

        # Convert old TeleportFull format to AI2-THOR 4.x format
        if isinstance(action, dict) and action.get('action') == 'TeleportFull':
            action = self._convert_teleport_action(action)

        if smooth_nav:
            if "MoveAhead" in action['action']:
                self.smooth_move_ahead(action)
            elif "Rotate" in action['action']:
                self.smooth_rotate(action)
            elif "Look" in action['action']:
                self.smooth_look(action)
            else:
                super().step(action)
        else:
            if "LookUp" in action['action']:
                self.look_angle(-constants.AGENT_HORIZON_ADJ)
            elif "LookDown" in action['action']:
                self.look_angle(constants.AGENT_HORIZON_ADJ)
            else:
                super().step(action)

        event = self.update_states(action)
        self.check_post_conditions(action)
        return event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                faucet_obj = game_util.get_object(action['objectId'], event.metadata)
                if faucet_obj is not None:
                    sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', faucet_obj, event.metadata)
                    if sink_basin is not None:
                        cleaned_object_ids = sink_basin.get('receptacleObjectIds')
                        self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                # AI2-THOR 5.x: TeleportFull doesn't accept render parameters
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': {'x': 0, 'y': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3), 'z': 0},
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'standing': True,
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': {'x': 0, 'y': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3), 'z': 0},
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'standing': True,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                # AI2-THOR 5.x: TeleportFull doesn't accept render parameters
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': {'x': 0, 'y': rotation, 'z': 0},
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'standing': True,
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': {'x': 0, 'y': rotation, 'z': 0},
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'standing': True,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        # AI2-THOR 5.x: TeleportFull doesn't accept render parameters
        teleport_action = {
            'action': 'TeleportFull',
            'rotation': {'x': 0, 'y': rotation, 'z': 0},
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'standing': True,
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        # AI2-THOR 5.x: TeleportFull doesn't accept render parameters
        teleport_action = {
            'action': 'TeleportFull',
            'rotation': {'x': 0, 'y': np.round(end_rotation, 3), 'z': 0},
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'standing': True,
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            # AI2-THOR 5.x: PutObject takes objectId (receptacle) not receptacleObjectId
            action = dict(action="PutObject",
                          objectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            # Look up the faucet object dict from the object ID
            faucet_obj = game_util.get_object(object_id, event.metadata)
            if faucet_obj is not None:
                sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", faucet_obj, event.metadata)
                if sink_basin_obj is not None and sink_basin_obj.get('receptacleObjectIds'):
                    for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                        in_sink_obj = game_util.get_object(in_sink_obj_id, event.metadata)
                        if in_sink_obj and in_sink_obj.get('dirtyable') and in_sink_obj.get('isDirty'):
                            event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)
