import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from torch.nn import CrossEntropyLoss


class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.planner.device
        self.max_steps = cfg.planner.max_steps
        self.model_name = cfg.planner.model_name
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.fast_mode = cfg.planner.fast_mode
        self.use_action_failure_msg = cfg.planner.use_action_failure_msg

        # Load pre-trained model
        print(f"LLM and tokenizer loading: {self.model_name}")

        model_args = {'pretrained_model_name_or_path': self.model_name, 'trust_remote_code': True,
                      'torch_dtype': torch.float16}
        if cfg.planner.use_accelerate_device_map:
            model_args['device_map'] = "auto"
            if cfg.planner.load_in_8bit:
                model_args['load_in_8bit'] = True
        if "starcode" in self.model_name:
            model_args['use_auth_token'] = cfg.planner.hf_auth_token

        if "alpaca" in self.model_name or "llama" in self.model_name:
            self.model = LlamaForCausalLM.from_pretrained(**model_args)
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(**model_args)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if not cfg.planner.use_accelerate_device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token_id = 0
        print(f"End loading\n")

        # Load prompt
        self.prompt = self.init_prompt(cfg.prompt.prefix, cfg.prompt.example_file_path,
                                       num_examples=cfg.prompt.num_examples, splitter=cfg.prompt.splitter)

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, prefix, example_file_path, num_examples="all", splitter=""):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()

    def score(self, prompt, skill_set):
        scores = {}
        batch_skill_set_list = [skill_set[chunk:chunk + self.scoring_batch_size] for chunk in
                                range(0, len(skill_set), self.scoring_batch_size)]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True)
        if not self.cfg.planner.use_accelerate_device_map:
            prompt_tokens = prompt_tokens.to(self.device)
        prompt_len = prompt_tokens.attention_mask[0].sum().item()

        for batch_skill_set in batch_skill_set_list:
            batch_sentence = [f"{prompt} {skill}" for skill in batch_skill_set]
            size_B = len(batch_skill_set)
            if "alpaca" in self.model_name or "llama" in self.model_name:
                batch_skill_set_for_model = batch_skill_set
            else:
                batch_skill_set_for_model = [f" {skill}" for skill in batch_skill_set]

            if self.fast_mode:
                with torch.no_grad():
                    prompt_output = self.model(**prompt_tokens, use_cache=True)
                    skill_tokens = self.tokenizer(batch_skill_set_for_model, add_special_tokens=False,
                                                  return_tensors="pt", padding=True)
                    if not self.cfg.planner.use_accelerate_device_map:
                        skill_tokens = skill_tokens.to(self.device)
                    concat_attention_mask = torch.cat(
                        (prompt_tokens.attention_mask.repeat(size_B, 1), skill_tokens.attention_mask), dim=1)
                    batch_past_key_values = self.duplicate_past_key_values(prompt_output.past_key_values, size_B)

                    output = self.model(input_ids=skill_tokens.input_ids,
                                        attention_mask=concat_attention_mask,
                                        past_key_values=batch_past_key_values,
                                        return_dict=True)
                    prompt_last_logits = prompt_output.logits[:, -1:, :].repeat(size_B, 1, 1)  # [B, 1, C]
                    logits = torch.cat((prompt_last_logits, output.logits[:, :-1, :]), dim=1)
                    labels = skill_tokens.input_ids
                    attention_mask = skill_tokens.attention_mask
            else:
                with torch.no_grad():
                    sentence_tokens = self.tokenizer(batch_sentence, add_special_tokens=False, return_tensors="pt",
                                                     padding=True)
                    if not self.cfg.planner.use_accelerate_device_map:
                        sentence_tokens = sentence_tokens.to(self.device)
                    output = self.model(sentence_tokens.input_ids, attention_mask=sentence_tokens.attention_mask,
                                        return_dict=True)
                    logits = output.logits[:, prompt_len - 1:-1]
                    labels = sentence_tokens.input_ids[:, prompt_len:]
                    attention_mask = sentence_tokens.attention_mask[:, prompt_len:]

            with torch.no_grad():
                size_B, size_L, size_C = logits.shape
                logits = logits.reshape([size_B * size_L, size_C])
                labels = labels.reshape([size_B * size_L])
                loss_fn = CrossEntropyLoss(reduction='none')
                loss = loss_fn(logits.float(), labels.long())
                loss = loss.reshape([size_B, size_L])
                skill_len = attention_mask.count_nonzero(axis=1)
                if self.score_function == 'sum':
                    score = -(loss * attention_mask).sum(axis=1)
                elif self.score_function == 'avg':
                    score = -(loss * attention_mask).sum(axis=1) / skill_len

            for skill_id, skill in enumerate(batch_skill_set):
                scores[skill] = score[skill_id].item()
        return scores

    def plan(self, query):
        step_seq = []
        skill_set_size_seq = []
        prompt = self.prompt + f'Human: {query}\nRobot: 1.'
        print(f"Input query: {query}")

        # Scoring
        for step in range(self.max_steps):
            # Make batch
            skill_set = self.skill_set

            # Save skill set size
            skill_set_size_seq.append(len(skill_set))

            # Scoring
            scores = self.score(prompt, skill_set)

            # Select best skill
            results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_step = results[0][0]
            step_seq.append(best_step.strip())
            print(f'{step + 1}. {best_step}')

            # Stop criteria
            if best_step in ['done', 'done.', 'done.\n']:
                prompt += f" {best_step}."
                break

            # Update skill set
            self.update_skill_set(best_step, self.nl_obj_list)

            # Update prompt
            prompt += f" {best_step}, {step + 2}."

        return step_seq, skill_set_size_seq

    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=()):
        if len(prev_steps) >= self.max_steps:
            return None, None

        prompt = self.prompt + f'Human: {query.strip()}\nRobot: 1.'

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_action_failure_msg and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        # score
        scores = self.score(prompt, self.skill_set)

        # find the best step
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_step = results[0][0]
        best_step = best_step.strip()

        return best_step, prompt

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
