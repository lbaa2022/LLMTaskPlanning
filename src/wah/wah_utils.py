import copy
import random
import math
import pdb
import json

##### Some functions are forked from wahtch_and_help github (https://github.com/xavierpuigf/watch_and_help)
def separate_new_ids_graph(graph, max_id):
    new_graph = copy.deepcopy(graph)
    for node in new_graph['nodes']:
        if node['id'] > max_id:
            node['id'] = node['id'] - max_id + 1000
    for edge in new_graph['edges']:
        if edge['from_id'] > max_id:
            edge['from_id'] = edge['from_id'] - max_id + 1000
        if edge['to_id'] > max_id:
            edge['to_id'] = edge['to_id'] - max_id + 1000
    return new_graph


##### 
def step_nl2sim(step_nl, obj_dict_nl2sim, cur_recep):
    if "put down" in step_nl:
        nl_putin_objs = ['bathroom cabinet', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee maker', 'dishwasher', 'folder', 'fridge', 'frying pan', 'garbage can', 'kitchen cabinet', 'microwave oven', 'nightstand', 'printer', 'sink', 'stove', 'toaster', 'toilet', 'washing machine']
        nl_putback_objs = ['bathroom cabinet', 'bathroom counter', 'bed', 'bench', 'board game', 'bookshelf', 'cabinet', 'chair', 'coffee table', 'cutting board', 'desk', 'floor', 'frying pan', 'kitchen cabinet', 'kitchen counter', 'kitchen table', 'mouse mat', 'nightstand', 'oven tray', 'plate', 'radio', 'rug', 'sofa', 'stove', 'towel rack']
        obj1_name = step_nl.split("put down the ")[1]
        if cur_recep == None:
            obj2_name = obj1_name
        else:
            obj2_name = cur_recep
        obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
        if cur_recep in nl_putin_objs:
            script = f"<char0> [putin] <{obj1_sim}> (1) <{obj2_sim}> (1)"
        else:
            script = f"<char0> [putback] <{obj1_sim}> (1) <{obj2_sim}> (1)"
    # if "put " in step_nl and " in " in step_nl:
    #     obj1_name, obj2_name = step_nl.replace('put the ', '').split(' in the ')
    #     obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
    #     script = f"<char0> [putin] <{obj1_sim}> (1) <{obj2_sim}> (1)"
    # elif "put " in step_nl and " on " in step_nl:
    #     obj1_name, obj2_name = step_nl.replace('put the ', '').split(' on the ')
    #     obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
    #     script = f"<char0> [putback] <{obj1_sim}> (1) <{obj2_sim}> (1)"
    else:
        if "find " in step_nl:
            action = "walk"
            obj1_w_article = step_nl.split("find ")[1]
            if "a " in obj1_w_article:
                obj1_name = obj1_w_article.split("a ")[1]
            elif "an " in obj1_w_article:
                obj1_name = obj1_w_article.split("an ")[1]
            else:
                raise NotImplementedError
        elif "go to" in step_nl:
            action = "walk"
            obj1_name = step_nl.split("go to the ")[1]
        elif "pick up " in step_nl:
            action = "grab"
            obj1_name = step_nl.split("pick up the ")[1]
        elif "open " in step_nl:
            action = "open"
            obj1_name = step_nl.split("open the ")[1]
        elif "close " in step_nl:
            action = "close"
            obj1_name = step_nl.split("close the ")[1]
        elif "switch on " in step_nl:
            action = "switchon"
            obj1_name = step_nl.split("switch on the ")[1]
        else:
            raise NotImplementedError
        obj1_sim = obj_dict_nl2sim[obj1_name]
        script = f"<char0> [{action}] <{obj1_sim}> (1)"
    return script


# def step_nl2sim(step_nl, obj_dict_nl2sim):
#     if "put " in step_nl and " in " in step_nl:
#         obj1_name, obj2_name = step_nl.replace('put ', '').split(' in ')
#         obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
#         script = f"<char0> [putin] <{obj1_sim}> (1) <{obj2_sim}> (1)"
#     elif "put " in step_nl and " on " in step_nl:
#         obj1_name, obj2_name = step_nl.replace('put ', '').split(' on ')
#         obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
#         script = f"<char0> [putback] <{obj1_sim}> (1) <{obj2_sim}> (1)"
#     else:
#         if "walk to " in step_nl:
#             action = "walk"
#             obj1_name = step_nl.split("walk to ")[1]
#         elif "grab " in step_nl:
#             action = "grab"
#             obj1_name = step_nl.split("grab ")[1]
#         elif "open " in step_nl:
#             action = "open"
#             obj1_name = step_nl.split("open ")[1]
#         elif "close " in step_nl:
#             action = "close"
#             obj1_name = step_nl.split("close ")[1]
#         elif "switch on " in step_nl:
#             action = "switchon"
#             obj1_name = step_nl.split("switch on ")[1]
#         elif "switch off " in step_nl:
#             action = "switchoff"
#             obj1_name = step_nl.split("switch off ")[1]
#         else:
#             raise NotImplementedError
#         obj1_sim = obj_dict_nl2sim[obj1_name]
#         script = f"<char0> [{action}] <{obj1_sim}> (1)"
#     return script


def split_step_sim(step_sim, with_ids=False):
    step_elements = step_sim.split(' ')
    act = step_elements[1].replace('[','').replace(']','')
    obj1_name = step_elements[2].replace('<','').replace('>','')
    if with_ids == False:
        if len(step_elements)==4:
            return act, obj1_name
        elif len(step_elements)==6:
            obj2_name = step_elements[4].replace('<','').replace('>','')
            return act, obj1_name, obj2_name
        else:
            raise NotImplementedError
    else:
        if len(step_elements)==4:
            return act, obj1_name, int(step_elements[3].replace('(','').replace(')',''))
        elif len(step_elements)==6:
            obj2_name = step_elements[4].replace('<','').replace('>','')
            return act, obj1_name, int(step_elements[3].replace('(','').replace(')','')), obj2_name, int(step_elements[5].replace('(','').replace(')',''))
        else:
            raise NotImplementedError

        
# def step_sim2nl(step_sim, obj_dict_sim2nl):
#     elements = split_step_sim(step_sim)
#     if len(elements) == 2:
#         act, obj1_name = elements[0], elements[1]
#         if act == "walk":
#             act = "walk to"
#         elif act == "switchon":
#             act = "switch on"
#         elif act == "switchoff":
#             act = "switch off"
#         step_nl = f"{act} {obj_dict_sim2nl[obj1_name]}"
#     elif len(elements) == 3:
#         act, obj1_name, obj2_name = elements[0], elements[1], elements[2]
#         if act == "putin":
#             step_nl = f"put {obj_dict_sim2nl[obj1_name]} in {obj_dict_sim2nl[obj2_name]}"
#         elif act == "putback":
#             step_nl = f"put {obj_dict_sim2nl[obj1_name]} on {obj_dict_sim2nl[obj2_name]}"
#     else:
#         raise NotImplementedError
#     return step_nl
    
def change_step_sim_obj_ids(step_sim, obj_ids):
    if len(obj_ids) == 1:
        return f"{step_sim.split('(')[0]}({obj_ids[0]})"
    elif len(obj_ids) == 2:
        return f"{step_sim.split('(')[0]}({obj_ids[0]}){step_sim.split(')')[1].split('(')[0]}({obj_ids[1]})"
    else:
        raise NotImplementedError
    
def find_indefinite_article(w):
    # simple rule, not always correct
    w = w.lower()
    if w[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    else:
        return 'a'



def get_ids_by_class_name(graph, class_name):
    return [node['id'] for node in graph['nodes'] if node['class_name'] == class_name]

def euclidean_dist(pos1, pos2):
    distance = 0
    for i in range(len(pos1)):
        distance += (pos1[i]-pos2[i])**2
    return math.sqrt(distance)

def select_obj_id(graph, obj_ids, method="distance"):
    id2nodes = {node['id']: node for node in graph['nodes']}
    if method == "distance":
        agent_ids = get_ids_by_class_name(graph, "character")
        if len(agent_ids) == 1:
            agent_id = agent_ids[0]
            agent_position = id2nodes[agent_id]['obj_transform']['position']
        else:
            raise NotImplementedError
        min_dist = 1e3
        for obj_id in obj_ids:
            if euclidean_dist(agent_position, id2nodes[obj_id]['obj_transform']['position']) < min_dist:
                selected_obj_id = obj_id
    else:
        raise NotImplementedError
    return selected_obj_id



###
def get_related_edges_by_id(graph, id):
    edges_from_id = [edge for edge in graph['edges'] if edge['from_id']==id]
    edges_to_id = [edge for edge in graph['edges'] if edge['to_id']==id]
    return edges_from_id, edges_to_id

def get_location_info_by_id(graph, node_id):
    id2node = {node['id']: node for node in graph['nodes']}
    edges_from_id, _ = get_related_edges_by_id(graph, node_id)
    room_node_ids, in_recepticle_ids, on_recepticle_ids = [], [], []
    for related_edge in edges_from_id:
        to_obj_id = related_edge['to_id']
        if related_edge['relation_type'] == 'INSIDE':
            if id2node[to_obj_id]['category'] == 'Rooms':
                room_node_ids.append(to_obj_id)
            else:
                in_recepticle_ids.append(to_obj_id)
        elif related_edge['relation_type'] == 'ON':
            on_recepticle_ids.append(to_obj_id)
    obj_location = {'room_node_ids': room_node_ids,
                    'in_recepticle_ids': in_recepticle_ids,
                    'on_recepticle_ids': on_recepticle_ids}
    return obj_location

def check_node_is_state(node, state):
    return state in node['states']

def check_node_is_close_to_agent(graph, agent_id, obj_id):
    edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, obj_id)
    return agent_id in [edge['to_id'] for edge in edges_from_obj]

def check_in_recep_is_open(graph, obj_id):
    id2node = {node['id']: node for node in graph['nodes']}
    obj_in_recep_ids = get_location_info_by_id(graph, obj_id)['in_recepticle_ids']
    if len(obj_in_recep_ids) == 0:
        return True, None
    elif len(obj_in_recep_ids) == 1:
        if not check_node_is_state(id2node[obj_in_recep_ids[0]], 'CLOSED'):
            return True, None
        else:
            return False, obj_in_recep_ids[0] 
    else:
        raise NotImplementedError

def check_free_hand(graph, agent_id):
    edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, agent_id)
    grabbed_objs = [edge_from_obj['to_id'] for edge_from_obj in edges_from_obj if 'HOLDS' in edge_from_obj['relation_type']]
    return len(grabbed_objs) < 2

def check_holding_obj(graph, agent_id, obj_id):
    edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, agent_id)
    x =  [edge_from_obj['relation_type'] for edge_from_obj in edges_from_obj if edge_from_obj['to_id']==obj_id]
    return 'HOLDS_RH' in x or 'HOLDS_LH' in x

def check_goal_condition(graph, task_goal):
    # task_goal keys -> 'inside_X_Y' 'on_X_Y' 'turnOn_X'
    id2node = {node['id']: node for node in graph['nodes']}
    task_goal_first_key = next(iter(task_goal))
    to_obj_name = task_goal_first_key.split('_')[-1]
    to_obj_ids = get_ids_by_class_name(graph, to_obj_name)
    
    final_state_candi = {}
    for to_obj_id in to_obj_ids:
        final_state = {}
        for goal_key, goal_n in task_goal.items():
            if 'turnOn' in goal_key:
                relation, _ = goal_key.split('_')
                states = id2node[to_obj_id]['states']
                if 'ON' in states:
                    final_state[goal_key] = (1, goal_n)
                else:
                    final_state[goal_key] = (0, goal_n)
            elif 'on' in goal_key or 'inside' in goal_key:
                relation, from_obj_name, _ = goal_key.split('_')
                _, edges_to_id = get_related_edges_by_id(graph, to_obj_id)
                count_satisfied = 0
                for edge in edges_to_id:
                    if id2node[edge['from_id']]['class_name'] == from_obj_name and relation == edge['relation_type'].lower():
                        count_satisfied += 1
                final_state[goal_key] = (count_satisfied, goal_n)
        final_state_candi[to_obj_id] = final_state
    
    
    scores = {to_obj_id: score_accomplish(final_state) for to_obj_id, final_state in final_state_candi.items()}
    max_score = max(scores.values())
    max_key = [key for key, value in scores.items() if value == max_score]
    # pdb.set_trace()
    return max_key[0], final_state_candi[max_key[0]]

def score_accomplish(final_state):
    # final_state example: {'on_juice_coffeetable': (0, 1), 'on_wine_coffeetable': (0, 1), 'on_pudding_coffeetable': (0, 1)}
    score = 0
    for k, v in final_state.items():
        score += min(v[0], v[1])/v[1]
    return score


###
def divide_total_into_keys(keys, total):
    if not keys or total <= 0:
        return {}

    num_keys = len(keys)
    value_per_key = total // num_keys
    remainder = total % num_keys

    result_dict = {}
    for i, key in enumerate(keys):
        value = value_per_key + 1 if i < remainder else value_per_key
        result_dict[key] = value

    return result_dict

##############################################################################################################################



















# ##### 1. Related to graph #####

# def get_obj_names_from_graph(graph):
#     return list(set([node['class_name'] for node in graph['nodes']]))

# def get_related_edges_by_id(graph, id):
#     edges_from_id = [edge for edge in graph['edges'] if edge['from_id']==id]
#     edges_to_id = [edge for edge in graph['edges'] if edge['to_id']==id]
#     return edges_from_id, edges_to_id

# def get_location_info_by_id(graph, node_id):
#     id2node = {node['id']: node for node in graph['nodes']}
#     edges_from_id, _ = get_related_edges_by_id(graph, node_id)
#     room_node_ids, in_recepticle_ids, on_recepticle_ids = [], [], []
#     for related_edge in edges_from_id:
#         to_obj_id = related_edge['to_id']
#         if related_edge['relation_type'] == 'INSIDE':
#             if id2node[to_obj_id]['category'] == 'Rooms':
#                 room_node_ids.append(to_obj_id)
#             else:
#                 in_recepticle_ids.append(to_obj_id)
#         elif related_edge['relation_type'] == 'ON':
#             on_recepticle_ids.append(to_obj_id)
#     obj_location = {'room_node_ids': room_node_ids,
#                     'in_recepticle_ids': in_recepticle_ids,
#                     'on_recepticle_ids': on_recepticle_ids}
#     return obj_location

# def get_ids_by_class_name(graph, class_name):
#     return [node['id'] for node in graph['nodes'] if node['class_name'] == class_name]

# def euclidean_dist(pos1, pos2):
#     distance = 0
#     for i in range(len(pos1)):
#         distance += (pos1[i]-pos2[i])**2
#     return math.sqrt(distance)

# def select_obj_id(graph, obj_ids, method="distance"):
#     id2nodes = {node['id']: node for node in graph['nodes']}
#     if method == "distance":
#         agent_ids = get_ids_by_class_name(graph, "character")
#         if len(agent_ids) == 1:
#             agent_id = agent_ids[0]
#             agent_position = id2nodes[agent_id]['obj_transform']['position']
#         else:
#             raise NotImplementedError
#         min_dist = 1e3
#         for obj_id in obj_ids:
#             if euclidean_dist(agent_position, id2nodes[obj_id]['obj_transform']['position']) < min_dist:
#                 selected_obj_id = obj_id
#     else:
#         raise NotImplementedError

#     return selected_obj_id

# def check_node_has_prop(node, prop):
#     return prop in node['properties']

# def check_node_is_state(node, state):
#     return state in node['states']

# def check_node_is_close_to_agent(graph, agent_id, obj_id):
#     edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, obj_id)
#     return agent_id in [edge['to_id'] for edge in edges_from_obj]

# def check_in_recep_is_open(graph, obj_id):
#     id2node = {node['id']: node for node in graph['nodes']}
#     obj_in_recep_ids = get_location_info_by_id(graph, obj_id)['in_recepticle_ids']
#     if len(obj_in_recep_ids) == 0:
#         return True, None
#     elif len(obj_in_recep_ids) == 1:
#         if not check_node_is_state(id2node[obj_in_recep_ids[0]], 'CLOSED'):
#             return True, None
#         else:
#             return False, obj_in_recep_ids[0] 
#     else:
#         raise NotImplementedError

# def check_free_hand(graph, agent_id):
#     edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, agent_id)
#     grabbed_objs = [edge_from_obj['to_id'] for edge_from_obj in edges_from_obj if 'HOLDS' in edge_from_obj['relation_type']]
#     return len(grabbed_objs) < 2

# def check_holding_obj(graph, agent_id, obj_id):
#     edges_from_obj, edges_to_obj = get_related_edges_by_id(graph, agent_id)
#     x =  [edge_from_obj['relation_type'] for edge_from_obj in edges_from_obj if edge_from_obj['to_id']==obj_id]
#     return 'HOLDS_RH' in x or 'HOLDS_LH' in x
    
# def get_close_obj_name_list(graph):
#     id2node = {node['id']: node for node in graph['nodes']}
#     agent_ids = get_ids_by_class_name(graph, 'character')
#     if len(agent_ids) == 1:
#         agent_id = agent_ids[0]
#     else:
#         NotImplementedError
#     edges_from_id, edges_to_id = get_related_edges_by_id(graph, agent_id)
#     close_obj_id_list = [edge['to_id'] for edge in edges_from_id if edge['relation_type'] == 'CLOSE'] + [edge['from_id'] for edge in edges_to_id if edge['relation_type'] == 'CLOSE']
#     close_obj_id_list = list(set(close_obj_id_list))
#     # close_obj_name_list = [id2node[obj_id]['class_name'] for obj_id in close_obj_id_list]
#     close_obj_name_list = list(set([id2node[obj_id]['class_name'] for obj_id in close_obj_id_list]))
#     return close_obj_name_list

# def get_grab_obj_name_list(graph):
#     id2node = {node['id']: node for node in graph['nodes']}
#     agent_ids = get_ids_by_class_name(graph, 'character')
#     if len(agent_ids) == 1:
#         agent_id = agent_ids[0]
#     else:
#         NotImplementedError
#     edges_from_id, edges_to_id = get_related_edges_by_id(graph, agent_id)
#     grab_obj_id_list = [edge['to_id'] for edge in edges_from_id if edge['relation_type'] in ['HOLDS_RH', 'HOLDS_LH']]
#     grab_obj_id_list = list(set(grab_obj_id_list))
#     # close_obj_name_list = [id2node[obj_id]['class_name'] for obj_id in close_obj_id_list]
#     grab_obj_name_list = list(set([id2node[obj_id]['class_name'] for obj_id in grab_obj_id_list]))
#     return grab_obj_name_list


# # def extract_sub_keys_from_graph_nodes(graph, keys):
# #     new_graph = {}
# #     new_graph['edges'] = graph['edges']
# #     new_nodes = []
# #     for node in graph['nodes']:
# #         new_node = {}
# #         for key in keys:
# #             new_node[key] = node[key]
# #         new_nodes.append(new_node)
# #     new_graph['nodes'] = new_nodes
# #     return new_graph



# ##### 2. Check goal condition #####
# def check_goal_condition(graph, task_goal):
#     # task_goal keys -> 'inside_X_Y' 'on_X_Y' 'turnOn_X'
#     id2node = {node['id']: node for node in graph['nodes']}
#     task_goal_first_key = next(iter(task_goal))
#     to_obj_name = task_goal_first_key.split('_')[-1]
#     to_obj_ids = get_ids_by_class_name(graph, to_obj_name)
    
#     final_state_candi = {}
#     for to_obj_id in to_obj_ids:
#         final_state = {}
#         for goal_key, goal_n in task_goal.items():
#             if 'turnOn' in goal_key:
#                 relation, _ = goal_key.split('_')
#                 states = id2node[to_obj_id]['states']
#                 if 'ON' in states:
#                     final_state[goal_key] = (1, goal_n)
#                 else:
#                     final_state[goal_key] = (0, goal_n)
#             elif 'on' in goal_key or 'inside' in goal_key:
#                 relation, from_obj_name, _ = goal_key.split('_')
#                 _, edges_to_id = get_related_edges_by_id(graph, to_obj_id)
#                 count_satisfied = 0
#                 for edge in edges_to_id:
#                     if id2node[edge['from_id']]['class_name'] == from_obj_name and relation == edge['relation_type'].lower():
#                         count_satisfied += 1
#                 final_state[goal_key] = (count_satisfied, goal_n)
#         final_state_candi[to_obj_id] = final_state
    
    
#     scores = {to_obj_id: score_accomplish(final_state) for to_obj_id, final_state in final_state_candi.items()}
#     max_score = max(scores.values())
#     max_key = [key for key, value in scores.items() if value == max_score]
#     # pdb.set_trace()
#     return max_key[0], final_state_candi[max_key[0]]

# def score_accomplish(final_state):
#     # final_state example: {'on_juice_coffeetable': (0, 1), 'on_wine_coffeetable': (0, 1), 'on_pudding_coffeetable': (0, 1)}
#     score = 0
#     for k, v in final_state.items():
#         score += min(v[0], v[1])/v[1]
#     return score

# ##### 3. Environment utils from WAH repository #####
# def separate_new_ids_graph(graph, max_id):
#     new_graph = copy.deepcopy(graph)
#     for node in new_graph['nodes']:
#         if node['id'] > max_id:
#             node['id'] = node['id'] - max_id + 1000
#     for edge in new_graph['edges']:
#         if edge['from_id'] > max_id:
#             edge['from_id'] = edge['from_id'] - max_id + 1000
#         if edge['to_id'] > max_id:
#             edge['to_id'] = edge['to_id'] - max_id + 1000
#     return new_graph

# def inside_not_trans(graph):
#     id2node = {node['id']: node for node in graph['nodes']}
#     parents = {}
#     grabbed_objs = []
#     for edge in graph['edges']:
#         if edge['relation_type'] == 'INSIDE':

#             if edge['from_id'] not in parents:
#                 parents[edge['from_id']] = [edge['to_id']]
#             else:
#                 parents[edge['from_id']] += [edge['to_id']]

#         elif edge['relation_type'].startswith('HOLDS'):
#             grabbed_objs.append(edge['to_id'])

#     edges = []
#     for edge in graph['edges']:
#         if edge['relation_type'] == 'INSIDE' and id2node[edge['to_id']]['category'] == 'Rooms':
#             if len(parents[edge['from_id']]) == 1:
#                 edges.append(edge)           
#         else:
#             edges.append(edge)
#     graph['edges'] = edges
#     parent_for_node = {}
#     char_close = {1: [], 2: []}
#     for char_id in range(1, 3):
#         for edge in graph['edges']:
#             if edge['relation_type'] == 'CLOSE':
#                 if edge['from_id'] == char_id and edge['to_id'] not in char_close[char_id]:
#                     char_close[char_id].append(edge['to_id'])
#                 elif edge['to_id'] == char_id and edge['from_id'] not in char_close[char_id]:
#                     char_close[char_id].append(edge['from_id'])
#     ## Check that each node has at most one parent
#     objects_to_check = []
#     for edge in graph['edges']:
#         if edge['relation_type'] == 'INSIDE':
#             if edge['from_id'] in parent_for_node and not id2node[edge['from_id']]['class_name'].startswith('closet'):
#                 print('{} has > 1 parent'.format(edge['from_id']))
#                 pdb.set_trace()
#                 raise Exception
#             parent_for_node[edge['from_id']] = edge['to_id']
#             # add close edge between objects in a container and the character
#             if id2node[edge['to_id']]['class_name'] in ['fridge', 'kitchencabinet', 'cabinet', 'microwave',
#                                                         'dishwasher', 'stove']:
#                 objects_to_check.append(edge['from_id'])
#                 for char_id in range(1, 3):
#                     if edge['to_id'] in char_close[char_id] and edge['from_id'] not in char_close[char_id]:
#                         graph['edges'].append({
#                             'from_id': edge['from_id'],
#                             'relation_type': 'CLOSE',
#                             'to_id': char_id
#                         })
#                         graph['edges'].append({
#                             'from_id': char_id,
#                             'relation_type': 'CLOSE',
#                             'to_id': edge['from_id']
#                         })
#     ## Check that all nodes except rooms have one parent
#     nodes_not_rooms = [node['id'] for node in graph['nodes'] if node['category'] not in ['Rooms', 'Doors']]
#     nodes_without_parent = list(set(nodes_not_rooms) - set(parent_for_node.keys()))
#     nodes_without_parent = [node for node in nodes_without_parent if node not in grabbed_objs]
#     graph['edges'] = [edge for edge in graph['edges'] if not (edge['from_id'] in objects_to_check and edge['relation_type'] == 'ON')]
#     if len(nodes_without_parent) > 0:
#         for nd in nodes_without_parent:
#             print(id2node[nd])
#         pdb.set_trace()
#         raise Exception
#     return graph


# ##### 4. Related to sim nl transformation ####
# def split_step_sim(step_sim, with_ids=False):
#     step_elements = step_sim.split(' ')
#     act = step_elements[1].replace('[','').replace(']','')
#     obj1_name = step_elements[2].replace('<','').replace('>','')
#     if with_ids == False:
#         if len(step_elements)==4:
#             return act, obj1_name
#         elif len(step_elements)==6:
#             obj2_name = step_elements[4].replace('<','').replace('>','')
#             return act, obj1_name, obj2_name
#     else:
#         if len(step_elements)==4:
#             return act, obj1_name, int(step_elements[3].replace('(','').replace(')',''))
#         elif len(step_elements)==6:
#             obj2_name = step_elements[4].replace('<','').replace('>','')
#             return act, obj1_name, int(step_elements[3].replace('(','').replace(')','')), obj2_name, int(step_elements[5].replace('(','').replace(')',''))
        
# def step_sim2nl(step_sim, obj_dict_sim2nl):
#     elements = split_step_sim(step_sim)
#     if len(elements) == 2:
#         act, obj1_name = elements[0], elements[1]
#         if act == "walk":
#             act = "walk to"
#         elif act == "switchon":
#             act = "switch on"
#         elif act == "switchoff":
#             act = "switch off"
#         step_nl = f"{act} {obj_dict_sim2nl[obj1_name]}"
#     elif len(elements) == 3:
#         act, obj1_name, obj2_name = elements[0], elements[1], elements[2]
#         if act == "putin":
#             step_nl = f"put {obj_dict_sim2nl[obj1_name]} in {obj_dict_sim2nl[obj2_name]}"
#         elif act == "putback":
#             step_nl = f"put {obj_dict_sim2nl[obj1_name]} on {obj_dict_sim2nl[obj2_name]}"
#     else:
#         pdb.set_trace()
#     return step_nl
    
# def step_nl2sim(step_nl, obj_dict_nl2sim):
#     if "put " in step_nl and " in " in step_nl:
#         obj1_name, obj2_name = step_nl.replace('put ', '').split(' in ')
#         obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
#         script = f"<char0> [putin] <{obj1_sim}> (1) <{obj2_sim}> (1)"
#     elif "put " in step_nl and " on " in step_nl:
#         obj1_name, obj2_name = step_nl.replace('put ', '').split(' on ')
#         obj1_sim, obj2_sim = obj_dict_nl2sim[obj1_name], obj_dict_nl2sim[obj2_name]
#         script = f"<char0> [putback] <{obj1_sim}> (1) <{obj2_sim}> (1)"
#     else:
#         if "walk to " in step_nl:
#             action = "walk"
#             obj1_name = step_nl.split("walk to ")[1]
#         elif "grab " in step_nl:
#             action = "grab"
#             obj1_name = step_nl.split("grab ")[1]
#         elif "open " in step_nl:
#             action = "open"
#             obj1_name = step_nl.split("open ")[1]
#         elif "close " in step_nl:
#             action = "close"
#             obj1_name = step_nl.split("close ")[1]
#         elif "switch on " in step_nl:
#             action = "switchon"
#             obj1_name = step_nl.split("switch on ")[1]
#         elif "switch off " in step_nl:
#             action = "switchoff"
#             obj1_name = step_nl.split("switch off ")[1]
#         else:
#             raise NotImplementedError
#         obj1_sim = obj_dict_nl2sim[obj1_name]
#         script = f"<char0> [{action}] <{obj1_sim}> (1)"
#     return script

# def change_step_sim_obj_ids(step_sim, obj_ids):
#     if len(obj_ids) == 1:
#         return f"{step_sim.split('(')[0]}({obj_ids[0]})"
#     elif len(obj_ids) == 2:
#         return f"{step_sim.split('(')[0]}({obj_ids[0]}){step_sim.split(')')[1].split('(')[0]}({obj_ids[1]})"
#     else:
#         raise NotImplementedError
    
    
# ##### 5. Extract Complete Plan #####
# def extract_complete_plan(new_data):
#     task_name = new_data['task_name']
    
#     complete_plan = []
#     if task_name in ['prepare_snack', 'prepare_food', 'setup_table']:
#         for subgoal_k, subgoal_info in new_data['subgoals'].items():
#             subplan_info = new_data['subplans'][subgoal_k]
#             subgoal_n = subgoal_info['goal_n']
#             ### N 개 뽑기
#             subplan = random.sample(subplan_info, subgoal_n)
#             for plan in subplan:
#                 complete_plan += plan
#     elif task_name in ['put_dishwasher']:
#         for subgoal_k, subgoal_info in new_data['subgoals'].items():
#             subplan_info = new_data['subplans'][subgoal_k]
#             subgoal_n = subgoal_info['goal_n']
#             if subgoal_info['relation'] == 'turnOn':
#                 last_plan = subplan_info[0]
#             else:
#                 subplan = random.sample(subplan_info, subgoal_n)
#                 for plan in subplan:
#                     complete_plan += plan
#         first_plan = [f"<char0> [walk] <{subgoal_info['to_obj_name']}> ({subgoal_info['to_obj_id']})",
#                       f"<char0> [open] <{subgoal_info['to_obj_name']}> ({subgoal_info['to_obj_id']})"]
#         complete_plan = first_plan + complete_plan + last_plan
#     elif task_name in ['put_fridge']:
#         for subgoal_k, subgoal_info in new_data['subgoals'].items():
#             subplan_info = new_data['subplans'][subgoal_k]
#             subgoal_n = subgoal_info['goal_n']
#             subplan = random.sample(subplan_info, subgoal_n)
#             for plan in subplan:
#                 complete_plan += plan
#         first_plan = [f"<char0> [walk] <{subgoal_info['to_obj_name']}> ({subgoal_info['to_obj_id']})",
#                       f"<char0> [open] <{subgoal_info['to_obj_name']}> ({subgoal_info['to_obj_id']})"]
#         last_plan =  [f"<char0> [close] <{subgoal_info['to_obj_name']}> ({subgoal_info['to_obj_id']})"]
#         complete_plan = first_plan + complete_plan + last_plan
#     else:
#         print("NOT DEFINED TASK ERROR")
#         pdb.set_trace()
#     return complete_plan

# ##### 6. Related to Survey #####
# # from_obj_names = ['cupcake', 'coffeepot', 'cutleryfork', 'juice', 'wineglass', 'pudding', 'apple', 'pancake', 'waterglass', 'poundcake', 'plate', 'wine']
# def combine_words(word_list):
#     if len(word_list) == 1:
#         combine_words = word_list[0]
#     elif len(word_list) == 2:
#         combine_words = " and ".join(word_list)
#     else:
#         combine_words = ", ".join(word_list[:-1]) + ", and " + word_list[-1]
#     # pdb.set_trace()
#     return combine_words

# plural_dict = {'cupcake': "cupcakes", 
#                'coffeepot': "coffee pots", 
#                'cutleryfork': "cutlery forks", 
#                'juice': "juices", 
#                'wineglass': "wine glasses", 
#                'pudding': "puddings", 
#                'apple': "apples", 
#                'pancake': "pancakes", 
#                'waterglass': "water glasses", 
#                'poundcake': "pound cakes", 
#                'plate': "plates", 
#                'wine': "wines"}
# number_dict = {1: "one",
#                2: "two",
#                3: "three",
#                4: "four",
#                5: "five",
#                6: "six",
#                7: "seven"}

# def task_goal2nl(subgoals, task_name, obj_dict_sim2nl):
#     from_obj_nls = []
#     to_obj_nl = []
#     for subgoal_key, subgoal_info in subgoals.items():
#         elements = subgoal_key.split('_')
#         if len(elements) == 3:
#             from_obj_name = subgoal_info['from_obj_name']
#             to_obj_name = subgoal_info['to_obj_name']
#             goal_n = subgoal_info['goal_n']
#             if goal_n == 1:
#                 from_obj_nl = f"{goal_n} {obj_dict_sim2nl[from_obj_name]}"
#             elif goal_n > 1:
#                 from_obj_nl = f"{goal_n} {plural_dict[from_obj_name]}"
#             from_obj_nls.append(from_obj_nl)
#             to_obj_nl.append(f"the {obj_dict_sim2nl[to_obj_name]}")
#         elif len(elements) == 2:
#             to_obj_name = subgoal_info['to_obj_name']
#             to_obj_nl.append(f"the {obj_dict_sim2nl[to_obj_name]}")
#     from_obj_nls = list(set(from_obj_nls))
#     to_obj_nl = list(set(to_obj_nl))
#     if len(to_obj_nl)>1:
#         pdb.set_trace()
    
#     is_from_objs_single = (len(from_obj_nls)==1) and (from_obj_nls[0].split(' ')[0]=='1')
#     if task_name in ['prepare_snack', 'prepare_food', 'setup_table']:
#         if is_from_objs_single:
#             nl_task_goal = f"{combine_words(from_obj_nls)} is on {to_obj_nl[0]}"
#         else:
#             nl_task_goal = f"{combine_words(from_obj_nls)} are on {to_obj_nl[0]}"
#     elif task_name in ['put_fridge']:
#         if is_from_objs_single:
#             nl_task_goal = f"{combine_words(from_obj_nls)} is in {to_obj_nl[0]}"
#         else:
#             nl_task_goal = f"{combine_words(from_obj_nls)} are in {to_obj_nl[0]}"
#     elif task_name in ['put_dishwasher']:
#         if is_from_objs_single:
#             nl_task_goal = f"{combine_words(from_obj_nls)} is in {to_obj_nl[0]}, and {to_obj_nl[0]} is turned on"
#         else:
#             nl_task_goal = f"{combine_words(from_obj_nls)} are in {to_obj_nl[0]}, and {to_obj_nl[0]} is turned on"
#     else:
#         pdb.set_trace()
#     return nl_task_goal

# def extract_location_tuple(location_info, id2node, obj_dict_sim2nl):
#     # pdb.set_trace()
#     room_names = [f"the {obj_dict_sim2nl[id2node[id]['class_name']]}" for id in location_info['room_node_ids']]
#     in_recep_names = [f"the {obj_dict_sim2nl[id2node[id]['class_name']]}" for id in location_info['in_recepticle_ids']]
#     on_recep_names = [f"the {obj_dict_sim2nl[id2node[id]['class_name']]}" for id in location_info['on_recepticle_ids']]
#     if not len(room_names) == 1:
#         pdb.set_trace()
#     # pdb.set_trace()
#     return room_names, in_recep_names, on_recep_names

# def location_tuple2nl(location_tuple, obj_nl):
#     location_nl = f"{obj_nl} is in {location_tuple[0][0]}"
#     num_in_recep = len(location_tuple[1])
#     num_on_recep = len(location_tuple[2])
#     if num_in_recep==0 and num_on_recep==0:
#         location_nl += "."
#     elif num_in_recep==0 and num_on_recep>0:
#         location_nl += f", specially on {combine_words(location_tuple[2])}."
#     elif num_in_recep>0 and num_on_recep==0:
#         location_nl += f", specially in {combine_words(location_tuple[1])}."
#     elif num_in_recep>0 and num_on_recep>0:
#         location_nl += f", specially in {combine_words(location_tuple[1])} and on {combine_words(location_tuple[2])}."
#     else:
#         pdb.set_trace()
#     return location_nl    
    
# def combine_location_sentences(location_sentence_list):
#     combined_sentence = ""
#     for id, location_sentence in enumerate(location_sentence_list):
#         if id == 0:
#             combined_sentence += "The first " + location_sentence
#         elif id == 1:
#             combined_sentence += " The second " + location_sentence
#         elif id == 2:
#             combined_sentence += " The third " + location_sentence
#         else:
#             combined_sentence += f" The {id+1}th " + location_sentence
#     return combined_sentence

# def location2nl(subgoals, obj_dict_sim2nl, id2node):
#     survey_nls = []
#     for subgoal_key, subgoal_info in subgoals.items():
#         elements = subgoal_key.split('_')
#         if len(elements) == 3:
#             from_obj_name = subgoal_info['from_obj_name']
#             to_obj_name = subgoal_info['to_obj_name']
#             from_obj_locations = subgoal_info['from_obj_locations']
#             to_obj_location = subgoal_info['to_obj_location']
            
#             to_obj_location_tuple = extract_location_tuple(to_obj_location, id2node, obj_dict_sim2nl)
#             to_obj_location_nl = location_tuple2nl(to_obj_location_tuple, obj_dict_sim2nl[to_obj_name])
                
#             from_obj_location_nls = []
#             for from_obj_id, from_obj_location in from_obj_locations.items():
#                 from_obj_location_tuple = extract_location_tuple(from_obj_location, id2node, obj_dict_sim2nl)
#                 from_obj_location_nl = location_tuple2nl(from_obj_location_tuple, obj_dict_sim2nl[from_obj_name])
#                 from_obj_location_nls.append(from_obj_location_nl)
        
#             x = combine_location_sentences(from_obj_location_nls)
            
            
            
#             if len(from_obj_location_nls) == 1:
#                 survey_nl = f"There is {len(from_obj_location_nls)} {obj_dict_sim2nl[from_obj_name]} in the house: {combine_location_sentences(from_obj_location_nls)}"
#             elif len(from_obj_location_nls) > 1:
#                 survey_nl = f"There are {len(from_obj_location_nls)} {plural_dict[from_obj_name]} in the house: {combine_location_sentences(from_obj_location_nls)}"
#             survey_nls.append(survey_nl)
#         else:
#             continue
#     survey_nls.append(f"The {to_obj_location_nl}")
    
#     # final_nl = ""
#     # for id, survey_nl in enumerate(survey_nls):
#     #     if id < len(survey_nls)-1:
#     #         final_nl += f"{survey_nl}\n"
#     #     else:
#     #         final_nl += survey_nl
#     return survey_nls


# ##### OpenAI API
# def count_tokens(text, model_name="gpt-3.5-turbo", debug=False):
#     """
#     Count the number of tokens in a given text string without using the OpenAI API.
    
#     This function tries three methods in the following order:
#     1. tiktoken (preferred): Accurate token counting similar to the OpenAI API.
#     2. nltk: Token counting using the Natural Language Toolkit library.
#     3. split: Simple whitespace-based token counting as a fallback.
    
#     Usage:
#     ------
#     text = "Your text here"
#     result = count_tokens(text, model_name="gpt-3.5-turbo", debug=True)
#     print(result)
#     Required libraries:
#     -------------------
#     - tiktoken: Install with 'pip install tiktoken'
#     - nltk: Install with 'pip install nltk'
#     Parameters:
#     -----------
#     text : str
#         The text string for which you want to count tokens.
#     model_name : str, optional
#         The OpenAI model for which you want to count tokens (default: "gpt-3.5-turbo").
#     debug : bool, optional
#         Set to True to print error messages (default: False).
#     Returns:
#     --------
#     result : dict
#         A dictionary containing the number of tokens and the method used for counting.
#     """

#     # Try using tiktoken
#     try:
#         import tiktoken
#         encoding = tiktoken.encoding_for_model(model_name)
#         num_tokens = len(encoding.encode(text))
#         result = {"n_tokens": num_tokens, "method": "tiktoken"}
#         return result
#     except Exception as e:
#         if debug:
#             print(f"Error using tiktoken: {e}")
#         pass

#     # Try using nltk
#     try:
#         import nltk
#         nltk.download("punkt")
#         tokens = nltk.word_tokenize(text)
#         result = {"n_tokens": len(tokens), "method": "nltk"}
#         return result
#     except Exception as e:
#         if debug:
#             print(f"Error using nltk: {e}")
#         pass

#     # If nltk and tiktoken fail, use a simple split-based method
#     tokens = text.split()
#     result = {"n_tokens": len(tokens), "method": "split"}
#     return result