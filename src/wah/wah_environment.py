import sys
import os
import copy
import json
import wah.wah_utils as utils

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(curr_dir, '../..'))
from virtualhome.simulation.environment.unity_environment import UnityEnvironment as BaseUnityEnvironment
from virtualhome.simulation.evolving_graph import utils as utils_env

import pdb

class WahEnv(BaseUnityEnvironment):
    def __init__(self, cfg):
        super(WahEnv, self).__init__(num_agents=1,
                                     observation_types=cfg.environment.observation_types,
                                     use_editor=cfg.environment.use_editor,                                    
                                     base_port=cfg.environment.base_port,
                                     port_id=cfg.environment.port_id,
                                     executable_args=cfg.environment.executable_args,
                                    
                                     recording_options=cfg.environment.recording_options
                                     )
        print("Environmet is initialized")
        self.full_graph = None
        with open(cfg.dataset.obj_dict_sim2nl, 'r') as file:
            self.obj_dict_sim2nl = json.load(file)
        with open(cfg.dataset.obj_dict_nl2sim, 'r') as file:
            self.obj_dict_nl2sim = json.load(file)
        
    def reset(self, task_d):
        # Make sure that characters are out of graph, and ids are ok
        self.task_id = task_d['task_id']
        self.init_graph = copy.deepcopy(task_d['init_graph'])
        self.init_room = task_d['init_room']
        self.task_goal = task_d['task_goal']
        self.task_name = task_d['task_name']
        self.env_id = task_d['env_id']
        print("Resetting... Envid: {}. Taskid: {}. Taskname: {}".format(self.env_id, self.task_id, self.task_name))
                
        # comm & expand scene
        self.comm.reset(self.env_id)
        s,g = self.comm.environment_graph()
        edge_ids = set([edge['to_id'] for edge in g['edges']] + [edge['from_id'] for edge in g['edges']])
        node_ids = set([node['id'] for node in g['nodes']])
        if len(edge_ids - node_ids) > 0:
            pdb.set_trace()
        if self.env_id not in self.max_ids.keys():
            max_id = max([node['id'] for node in g['nodes']])
            self.max_ids[self.env_id] = max_id
        max_id = self.max_ids[self.env_id]

        updated_graph = self.init_graph
        s, g = self.comm.environment_graph()
        updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
        success, m = self.comm.expand_scene(updated_graph)

        if not success:
            print("Error expanding scene")
            pdb.set_trace()
            return None

        self.offset_cameras = self.comm.camera_count()[1]

        self.comm.add_character(self.agent_info[0], initial_room=self.init_room)
        
        _, self.init_unity_graph = self.comm.environment_graph()
        
        self.changed_graph = True
        graph = self.get_graph()
        self.rooms = [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms']
        self.id2node = {node['id']: node for node in graph['nodes']}
        self.steps = 0
        self.cur_recep = None

    def step(self, step, step_form='sim', instance=False):
        if step_form == 'nl' and instance == False:
            if 'find' in step:
                obj1_w_article = step.split("find ")[1]
                if "a " in obj1_w_article:
                    obj1_name = obj1_w_article.split("a ")[1]
                elif "an " in obj1_w_article:
                    obj1_name = obj1_w_article.split("an ")[1]
                else:
                    raise NotImplementedError
                self.cur_recep = obj1_name
            step_sim = utils.step_nl2sim(step, self.obj_dict_nl2sim, self.cur_recep)
            step = self.assign_id(step_sim, method="distance")
        else:
            ### TODO: self.cur_recep Update for step_form 'sim' Is Not Implemented
            pass
        
        ##### TODO: NL Feedback Is Not Implemented
        possible, feedback = self.check_step(step)
        
        if possible:
            script_list = [step]
            if self.recording_options['recording']:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        processing_time_limit=60,
                                                        recording=True,
                                                        skip_animation=False,
                                                        camera_mode=list(self.recording_options['cameras']),
                                                        output_folder=self.recording_options['output_folder'],
                                                        file_name_prefix=self.recording_options['file_name_prefix'])
            else:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        recording=False,
                                                        skip_animation=True)
            self.changed_graph = True
        else:
            self.changed_graph = False
            
        return possible, feedback
            # if not success:
            #     print(message) 
            #     # print(script_list)
            #     is_execution_success = False
            # else:
            #     self.changed_graph = True
            #     is_execution_success = True
    
    def assign_id(self, step_sim, method="distance"):
        graph = self.get_graph()
        elements = utils.split_step_sim(step_sim)
        if len(elements) == 2:
            obj1_name = elements[1]
            obj1_id = utils.select_obj_id(graph, utils.get_ids_by_class_name(graph, obj1_name), method=method)
            obj_ids = [obj1_id]
        elif len(elements) == 3:
            obj1_name, obj2_name = elements[1], elements[2]
            obj1_id = utils.select_obj_id(graph, utils.get_ids_by_class_name(graph, obj1_name), method=method)
            obj2_id = utils.select_obj_id(graph, utils.get_ids_by_class_name(graph, obj2_name), method=method)
            obj_ids = [obj1_id, obj2_id]
        else:
            raise NotImplementedError
        step = utils.change_step_sim_obj_ids(step_sim, obj_ids)
        return step


    def check_step(self, step):
        elements = utils.split_step_sim(step, with_ids=True)
        
        graph = self.get_graph()
        agent_ids = utils.get_ids_by_class_name(graph, 'character')
        if len(agent_ids) == 1:
            agent_id = agent_ids[0]
        else:
            raise NotImplementedError
        id2nodes = {node['id']: node for node in graph['nodes']}
        
        ### TODO: NL feedback is not implemented
        if elements[0] == 'walk':
            possible, feedback = self.check_walk(elements)
        elif elements[0] == 'grab':
            possible, feedback = self.check_grab(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'open':
            possible, feedback = self.check_open(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'close':
            possible, feedback = self.check_close(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'switchon':
            possible, feedback = self.check_switchon(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'switchoff':
            possible, feedback = self.check_switchoff(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'putin':
            possible, feedback = self.check_putin(elements, graph, agent_id, id2nodes)
        elif elements[0] == 'putback':
            possible, feedback = self.check_putback(elements, graph, agent_id, id2nodes)
        else:
            NotImplementedError
        return possible, feedback
        
    def check_walk(self, elements):
        return True, None
    def check_grab(self, elements, graph, agent_id, id2nodes):
        ### Case 1. obj grabbable? 2. obj close? 3. obj inside open rec? 4. agent has free hand?
        obj_name, obj_id = elements[1], elements[2]
        is_obj_grabbable = 'GRABBABLE' in id2nodes[obj_id]['properties']
        is_obj_close = utils.check_node_is_close_to_agent(graph, agent_id, obj_id)
        is_obj_in_open_recep = utils.check_in_recep_is_open(graph, obj_id)[0]
        is_agent_free_hand = utils.check_free_hand(graph, agent_id)
        # return True, None
        return is_obj_grabbable and is_obj_close and is_obj_in_open_recep and is_agent_free_hand, None
    def check_open(self, elements, graph, agent_id, id2nodes):
        ### Case 1. obj opennable? 2. obj close? 3. agent has free hand? 4. obj open?
        obj_name, obj_id = elements[1], elements[2]
        is_obj_opennable = 'CAN_OPEN' in id2nodes[obj_id]['properties']
        is_obj_closed = 'CLOSED' in id2nodes[obj_id]['states']
        is_obj_close = utils.check_node_is_close_to_agent(graph, agent_id, obj_id)
        is_agent_free_hand = utils.check_free_hand(graph, agent_id)
        return is_obj_opennable and is_obj_closed and is_obj_close and is_agent_free_hand, None
    def check_close(self, elements, graph, agent_id, id2nodes):
        ### Case 1. obj opennable? 2. obj close? 3. agent has free hand? 4. obj closed?
        obj_name, obj_id = elements[1], elements[2]
        is_obj_opennable = 'CAN_OPEN' in id2nodes[obj_id]['properties']
        is_obj_open = 'OPEN' in id2nodes[obj_id]['states']
        is_obj_close = utils.check_node_is_close_to_agent(graph, agent_id, obj_id)
        is_agent_free_hand = utils.check_free_hand(graph, agent_id)
        return is_obj_opennable and is_obj_open and is_obj_close and is_agent_free_hand, None
    def check_switchon(self, elements, graph, agent_id, id2nodes):
        ### Case 1. obj has_switch? 2. obj close? 3. agent has free hand? 4. obj off?
        obj_name, obj_id = elements[1], elements[2]
        is_obj_hasswitch = 'HAS_SWITCH' in id2nodes[obj_id]['properties']
        is_obj_close = utils.check_node_is_close_to_agent(graph, agent_id, obj_id)
        is_agent_free_hand = utils.check_free_hand(graph, agent_id)
        is_obj_off = 'OFF' in id2nodes[obj_id]['states']
        is_obj_closed = 'CLOSED' in id2nodes[obj_id]['states']
        return is_obj_hasswitch and is_obj_close and is_agent_free_hand and is_obj_off and is_obj_closed, None
    def check_switchoff(self, elements, graph, agent_id, id2nodes):
        ### Case 1. obj has_switch? 2. obj close? 3. agent has free hand? 4. obj on?
        obj_name, obj_id = elements[1], elements[2]
        is_obj_hasswitch = 'HAS_SWITCH' in id2nodes[obj_id]['properties']
        is_obj_close = utils.check_node_is_close_to_agent(graph, agent_id, obj_id)
        is_agent_free_hand = utils.check_free_hand(graph, agent_id)
        is_obj_on = 'ON' in id2nodes[obj_id]['states']
        return is_obj_hasswitch and is_obj_close and is_agent_free_hand and is_obj_on, None
    def check_putin(self, elements, graph, agent_id, id2nodes):
        ### Case 1. agent holding obj1? 2. obj2 close? 3. obj2 container?
        obj1_name, obj1_id, obj2_name, obj2_id = elements[1], elements[2], elements[3], elements[4]
        is_agent_holing_obj1 = utils.check_holding_obj(graph, agent_id, obj1_id)
        is_obj2_close = utils.check_node_is_close_to_agent(graph, agent_id, obj2_id)
        is_obj2_container = 'CONTAINERS' in id2nodes[obj2_id]['properties']
        return is_agent_holing_obj1 and is_obj2_close and is_obj2_container, None
    def check_putback(self, elements, graph, agent_id, id2nodes):
        ### Case 1. agent holding obj1? 2. obj2 close? 3. obj2 surface?
        obj1_name, obj1_id, obj2_name, obj2_id = elements[1], elements[2], elements[3], elements[4]
        is_agent_holing_obj1 = utils.check_holding_obj(graph, agent_id, obj1_id)
        is_obj2_close = utils.check_node_is_close_to_agent(graph, agent_id, obj2_id)
        is_obj2_surface = 'SURFACES' in id2nodes[obj2_id]['properties']
        return is_agent_holing_obj1 and is_obj2_close and is_obj2_surface, None