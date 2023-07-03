import torch, gc
from torch.nn import CrossEntropyLoss
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import GPTNeoXForCausalLM, OPTForCausalLM, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


class TaskPlanner:
    def __init__(self, cfg):
        model_name = cfg.planner.model
        self.device = 'cuda'
        # self.device = 0
        self.max_steps = cfg.planner.max_steps
        self.batch_size = cfg.planner.batch_size
        self.use_action_failure_msg = cfg.planner.use_action_failure_msg

        # print_gpu_usage('before model loading')

        if cfg.planner.load_in_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

        # print_gpu_usage('model loaded')

        if 'llama' in model_name or 'alpaca' in model_name:
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = 0

        self.skills = self.init_skill_set()

        if cfg.planner.use_action_failure_msg:
            self.prompt = self.load_prompt(cfg)
        else:
            self.prompt = self.init_prompt(cfg)

        print('--- prompt start ---')
        print(self.prompt)
        print('--- prompt end ---')

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def load_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self):
        raise NotImplementedError()

    def model_forward(self, input_tensor, past_key_val=None, attention_mask=None):
        if type(self.model) is GPTNeoXForCausalLM:
            outputs = self.model.gpt_neox(input_tensor, past_key_values=past_key_val, attention_mask=attention_mask)
        elif type(self.model) is OPTForCausalLM:
            outputs = self.model.model.decoder(input_tensor, past_key_values=past_key_val, attention_mask=attention_mask)
        else:
            outputs = self.model.transformer(input_tensor, past_key_values=past_key_val, attention_mask=attention_mask)

        return outputs

    def model_headnet(self, hidden_states):
        if type(self.model) is GPTNeoXForCausalLM:
            lm_logits = self.model.embed_out(hidden_states)
        elif type(self.model) is OPTForCausalLM:
            lm_logits = self.model.lm_head(hidden_states).contiguous()
        else:
            lm_logits = self.model.lm_head(hidden_states)

        return lm_logits


    def run_for_prompt(self, prompt):
        past_tokenize_input = self.tokenizer.tokenize(prompt)
        past_tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(past_tokenize_input)])
        past_tensor_input = past_tensor_input.to(self.device)

        transformer_outputs = self.model_forward(past_tensor_input)

        return past_tensor_input, transformer_outputs

    def score(self, sentence, past_key_values, past_last_hidden_state, past_tensor_input):
        with torch.no_grad():
            tokenize_input = self.tokenizer.tokenize(sentence, )
            tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
            tensor_input = tensor_input.to(self.device)

            ###the following code is slightly modified from https://github.com/huggingface/transformers/blob/09a2f40684f77e62d0fd8485fe9d2d610390453f/src/transformers/modeling_gpt2.py#L604
            ###now we calculate the right part of the sentence with the already calculated past
            transformer_outputs = self.model_forward(tensor_input, past_key_values)

            ###and concatenate the output of with the hidden_state of the left part of the sentence
            hidden_states = torch.cat((past_last_hidden_state[:, -1:, :], transformer_outputs[0]), dim=1)
            lm_logits = self.model_headnet(hidden_states)        

            labels_input = tensor_input
            # print(labels_input.shape)

            shift_logits = lm_logits[..., :-1,
                        :].contiguous()  # last prediction is not necessary (the next word of the current candidate word)
            shift_labels = labels_input.contiguous()
            loss_fn = CrossEntropyLoss()
            # loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)).to(self.device), 
            #                shift_labels.view(-1).to(self.device))  # add .to(device) to make sure both tensors are in the same device
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)).cpu().float(), 
                        shift_labels.view(-1).cpu().long())

            ret = -loss.item()

            # clear (is this really necessary?)
            del transformer_outputs, tensor_input, hidden_states, lm_logits, labels_input, shift_logits, shift_labels, loss
            gc.collect()
            torch.cuda.empty_cache()

        return ret

    def batch_score(self, batch_sentences, past_key_values, past_last_hidden_state, past_tensor_input):
        with torch.no_grad():
            ### Padding
            self.tokenizer.pad_token_id = 0
            batch_size = len(batch_sentences)
            padded_batch_sequence = self.tokenizer(batch_sentences, padding=True)
            padded_tokens = padded_batch_sequence['input_ids']
            pad_mask = padded_batch_sequence['attention_mask']
            tensor_input = torch.tensor(padded_tokens)
            pad_mask = torch.tensor(pad_mask)
            tensor_input = tensor_input.to(self.device)
            pad_mask = pad_mask.to(self.device)
            ### Make batch of pask_key_values
            batch_past_key_values = []
            for layer in range(len(past_key_values)):
                batch_past_key_values_layer = []
                for kv in range(len(past_key_values[layer])):
                    batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
                batch_past_key_values_layer = tuple(batch_past_key_values_layer)
                batch_past_key_values.append(batch_past_key_values_layer)
            batch_past_key_values = tuple(batch_past_key_values)
            ### Make attn_mask
            prompt_len = past_key_values[0][0].shape[2]
            prompt_attn_mask = torch.ones([batch_size, prompt_len]).to(self.device)
            attn_mask = torch.concat([prompt_attn_mask, pad_mask], axis=1)
            ### LLM feedforward

            transformer_outputs = self.model_forward(tensor_input, batch_past_key_values, attn_mask)

            max_token_len = pad_mask.shape[1]
            hidden_states = torch.cat(
                (past_last_hidden_state[:, -1:, :].repeat(batch_size, 1, 1), transformer_outputs[0]), dim=1)
            hidden_states = hidden_states.reshape([batch_size * (max_token_len + 1), -1])
            lm_logits = self.model_headnet(hidden_states)    

            lm_logits = lm_logits.reshape([batch_size, max_token_len + 1, -1])
            labels_input = tensor_input
            shift_logits = lm_logits[..., :-1,
                           :].contiguous()  # last prediction is not necessary (the next word of the current candidate word)
            shift_labels = labels_input
            loss_fn = CrossEntropyLoss(reduction='none')
            # loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)).to(self.device), 
            #                shift_labels.view(-1).to(self.device))  # add .to(device) to make sure both tensors are in the same device
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))  # add .to(device) to make sure both tensors are in the same device
            loss = loss.reshape(batch_size, max_token_len)
            loss = loss * pad_mask
            ret = -(loss.sum(axis=1) / torch.count_nonzero(pad_mask, dim=1))

            del transformer_outputs, tensor_input, hidden_states, lm_logits, labels_input, shift_logits, shift_labels, loss, batch_past_key_values, attn_mask, batch_past_key_values_layer
            torch.cuda.empty_cache()
        return ret

    def score_without_past_values_old(self, sentence):
        with torch.no_grad():
            tokenize_input = self.tokenizer.tokenize(sentence, )
            tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
            tensor_input = tensor_input.to(self.device)

            ret = self.model(tensor_input, labels=tensor_input, return_dict=True)
            score = -ret['loss'].item()

        return score

    def score_without_past_values(self, batch_sentence, prompt_len):
        with torch.no_grad():
            padded_batch_sequence = self.tokenizer(batch_sentence, padding=True, return_tensors="pt")
            for key in padded_batch_sequence.keys():
                padded_batch_sequence[key] = padded_batch_sequence[key].to(self.device)
            outputs = self.model(**padded_batch_sequence)
            logits = outputs['logits'][:, prompt_len - 1:-1]  # [B, L, C]
            labels = padded_batch_sequence['input_ids'][:, prompt_len:]  # [B, L]
            attention_mask = padded_batch_sequence['attention_mask'][:, prompt_len:]
            # Size convert
            # loss input size: [BL, C], [BL] & loss output size: [BL] & converted loss size: [B, L] -> (average on L) final loss size: [B]
            size_B, size_L, size_C = logits.shape
            logits = logits.reshape([size_B * size_L, size_C])
            labels = labels.reshape([size_B * size_L])
            loss_fn = CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            loss = loss.reshape([size_B, size_L])
            sentence_L = attention_mask.count_nonzero(axis=1)
            # score = -(loss*attention_mask).sum(axis=1)/sentence_L
            score = -(loss * attention_mask).sum(axis=1)
        return score

    def plan(self, query):
        prompt = self.prompt + f'Human: {query} \nRobot: 1.'
        outputs = []

        for i in range(self.max_steps):
            use_past_key_values = False
            if use_past_key_values:
                # show_gpu('before run prompt')
                past_tensor_input, transformer_outputs = self.run_for_prompt(prompt)
                # show_gpu('after run prompt')
                past_last_hidden_state, past_key_values = transformer_outputs[0], transformer_outputs[1]

                if self.batch_size > 1:
                    # Batch Scoring
                    chunk_size = self.batch_size
                    batch_sentence_list = [self.skills[chunk:chunk + chunk_size] for chunk in range(0, len(self.skills),
                                                                                                    chunk_size)]
                    results = {}
                    for batch_sentence in batch_sentence_list:
                        batch_results = self.batch_score(batch_sentence, past_key_values, past_last_hidden_state,
                                                         past_tensor_input)
                        for candidate_id, candidate in enumerate(batch_sentence):
                            results[candidate] = batch_results[candidate_id].item()
                else:
                    # Single Scoring
                    results = {candidate: self.score(candidate, past_key_values, past_last_hidden_state, past_tensor_input)
                               for
                               candidate in self.skills}

                # show_gpu('after run score')

                # free gpu memory
                del past_tensor_input, transformer_outputs, past_last_hidden_state, past_key_values
                gc.collect()
                torch.cuda.empty_cache()

                # show_gpu('after free')
            else:
                prompt_len = self.tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
                results = {candidate: self.score_without_past_values([prompt + candidate], prompt_len) for candidate in self.skills}
                #results = {candidate: self.score_without_past_values(prompt + candidate, prompt_len) for candidate in self.skills}

            results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(results)

            best_step = results[0][0]
            outputs.append(best_step.strip())
            print(f'{i + 1}.{best_step}')

            # update prompt
            prompt += best_step + f', {i + 2}.'
            # print(prompt)

            # stop criteria
            if best_step == ' done':
                break

        return outputs

    def plan_step(self, query, prev_steps=(), prev_msgs=()):
        if len(prev_steps) >= self.max_steps:
            return None

        prompt = self.prompt + f'Human: {query.strip()} \nRobot: 1. '

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_action_failure_msg and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        # find next step
        prompt_len = self.tokenizer(prompt, return_tensors="pt")['input_ids'].shape[1]
        if self.batch_size > 1:
            chunk_size = self.batch_size
            input_sentences = [prompt + candidate for candidate in self.skills]
            batch_sentence_list = [input_sentences[chunk:chunk + chunk_size] for chunk in
                                   range(0, len(input_sentences), chunk_size)]
            batch_skill_list = [self.skills[chunk:chunk + chunk_size] for chunk in range(0, len(self.skills), chunk_size)]
            results = {}
            for (batch_id, batch_sentence) in enumerate(batch_sentence_list):
                batch_score = self.score_without_past_values(batch_sentence, prompt_len)
                batch_skill = batch_skill_list[batch_id]
                for candidate_id, candidate in enumerate(batch_skill):
                    results[candidate] = batch_score[candidate_id].item()
        else:
            results = {candidate: self.score_without_past_values([prompt + candidate], prompt_len) for candidate in
                       self.skills}
            #results = {candidate: self.score_without_past_values(self.prompt + prompt + candidate) for candidate in self.skills}

        results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        best_step = results[0][0]
        output = best_step.strip()

        print('---------------------------------------')
        print(prompt)
        print(f'{len(prev_steps) + 1}.{best_step}')
        print('---------------------------------------')

        return output
