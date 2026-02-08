"""
Game utilities for ALFRED.
"""
import math


def sample_templated_task_desc_from_traj_data(traj_data):
    """
    Generate a templated task description from trajectory data.
    Falls back to the first annotation if no template available.
    """
    task_type = traj_data.get('task_type', '')

    if 'turk_annotations' in traj_data and 'anns' in traj_data['turk_annotations']:
        anns = traj_data['turk_annotations']['anns']
        if anns and len(anns) > 0:
            return anns[0].get('task_desc', task_type)

    return task_type


def get_object(object_id, metadata):
    """Get object by objectId from metadata."""
    if metadata is None:
        return None
    objects = metadata.get('objects', [])
    for obj in objects:
        if obj['objectId'] == object_id:
            return obj
    return None


def get_objects_with_name_and_prop(object_type, prop, metadata):
    """Get objects matching type and property."""
    if metadata is None:
        return []
    objects = metadata.get('objects', [])
    matching = []
    for obj in objects:
        if object_type in obj.get('objectType', ''):
            if prop in obj and obj[prop]:
                matching.append(obj)
    return matching


def get_objects_of_type(object_type, metadata):
    """Get all objects of a given type."""
    if metadata is None:
        return []
    objects = metadata.get('objects', [])
    matching = []
    for obj in objects:
        if object_type.lower() in obj.get('objectType', '').lower():
            matching.append(obj)
    return matching


def get_obj_of_type_closest_to_obj(object_type, ref_object, metadata):
    """Get object of type closest to reference object."""
    candidates = get_objects_of_type(object_type, metadata)
    if not candidates or ref_object is None:
        return None

    ref_pos = ref_object.get('position', {})
    ref_x = ref_pos.get('x', 0)
    ref_z = ref_pos.get('z', 0)

    closest = None
    min_dist = float('inf')

    for obj in candidates:
        pos = obj.get('position', {})
        x = pos.get('x', 0)
        z = pos.get('z', 0)
        dist = math.sqrt((x - ref_x) ** 2 + (z - ref_z) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = obj

    return closest
