import torch
import time
import numpy as np
from torch.nn import CrossEntropyLoss
import pprint
import subprocess

from transformers import GPTNeoForCausalLM, AutoModelForCausalLM, GPT2Tokenizer, GPTNeoXForCausalLM, OPTForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

prompt_sentences = [
    "Robot: Hi there, I’m a robot operating in an office kitchen. \n",
    "Robot: You can ask me to do various tasks and I’ll tell you the sequence of actions I would do to accomplish your task. \n",
    "Human: How would you hold the snickers? \n",
    "Robot: 1. pick up the snickers, 2. done. \n",
    "Human: How would you put the trailmix here? \n",
    "Robot: 1. put down the trailmix, 2. done. \n",
    "Human: How would you put a water bottle and an oatmeal next to the microwave? \n",
    "Robot: 1. find a water bottle, 2. pick up the water bottle, 3. go to microwave, 4. put down the water bottle, 5. find an oatmeal, 6. pick up the oatmeal, 7. go to the microwave, 8. put down the oatmeal, 9. done. \n",
    "Human: How would you put a grapefruit from the table into the bowl? \n",
    "Robot: 1. find a grapefruit, 2. pick up the grapefruit, 3. go to the bowl, 4. put down the grapefruit, 5. done. \n",
    "Human: How would you get a sponge from the counter and put it in the sink? \n",
    "Robot: 1. find a sponge, 2. pick up the sponge, 3. go to the sink, 4. put down the sponge, 5. done. \n",
    "Human: How would you move the water bottle from the table to the counter? \n",
    "Robot: 1. find a water bottle, 2. pick up the water bottle, 3. go to the counter, 4. put down the water bottle, 5. done. \n",
    "Human: How would you bring me some snacks? \n",
    "Robot: 1. find a twix, 2. pick up the twix, 3. bring it to you, 4. put down the twix, 5. find an dried fruit, 6. pick up the dried fruit, 7. bring it to you, 8. put down the dried fruit, 9. done. \n",
    "Human: How would you bring me an orange? \n",
    "Robot: 1. find an orange, 2. pick up the orange, 3. bring it to you, 4. put down the orange, 5. done. \n",
    "Human: How would you deliver a can of coke to me? \n",
    "Robot: 1. find a coke can, 2. pick up the coke can, 3. bring it to you, 4. put down the coke can, 5. done. \n",
    "Human: I finished my drink, can you throw it away and bring me a mountain dew? \n",
    "Robot: 1. find a sprite can, 2. pick up the sprite can, 3. go to the trash can, 4. put down the sprite can, 5. find a mountain dew can, 6. pick up the mountain dew can, 7. bring it to you, 8. put down the mountain dew can, 9. done. \n",
    "Human: How would you let go of the banana? \n",
    "Robot: 1. put down the banana, 2. done. \n",
    "Human: How would you put the grapes in the bowl and then move the cheese to the table? \n",
    "Robot: 1. find the grapes, 2. pick up the grapes, 3. go to the bowl, 4. put down the grapes, 5. pick up the cheese, 6. go to the table, 7. put down the cheese, 8. tell you I’m done. \n",
    "Human: How would you pick up the peanuts? \n",
    "Robot: 1. pick up the peanuts, 2. done. \n",
    "Human: How would you bring me the peanuts? \n",
    "Robot: 1. pick up the peanuts, 2. bring it to you, 3. put down the peanuts, 4. done. \n",
    "Human: How would you throw away a coffee cup? \n",
    "Robot: 1. find a coffee cup, 2. pick up the coffee cup, 3. go to trash can, 4. put down the coffee cup, 5. done. \n",
    # "Human: How would you place a knife and a banana to the table? \n",
    # "Robot: 1. find a knife, 2. pick up the knife, 3. go to the table, 4. put down the knife, 5. find a banana, 6. pick up the banana, 7. go to the table, 8. put down the banana, 9. done. \n"
]

# prompt_sentences = [
#     "Robot: Hi there, I’m a robot operating in an office kitchen. \n",
#     "Robot: You can ask me to do various tasks and I’ll tell you the sequence of actions I would do to accomplish your task. \n",
#     "Human: How would you hold the snickers? \n",
#     "Robot: 1. pick up the snickers, 2. done. \n",
# ]

inital_prompt = ''.join((s for s in prompt_sentences))

# from https://stackoverflow.com/questions/62703391/estimate-token-probability-logits-given-a-sentence-without-computing-the-entire


def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


show_gpu('init gpu')

# model_name = "EleutherAI/gpt-neo-125M"
# model_name = "EleutherAI/gpt-neo-1.3B"
# model_name = "EleutherAI/gpt-neo-2.7B"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "EleutherAI/gpt-neox-20b"
model_name = "facebook/opt-350m"
# model_name = "facebook/opt-66b"
# model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16)
# model = model.to(device)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)  # max_memory={0: "30GiB", "cpu": "60GiB"}
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

show_gpu('after loading gpu')

#####################################################################
# prompt
# query = 'Human: Could you bring me an apple?'
# query = 'Human: Could you bring me a coke can in the refrigerator?'
# query = 'Human: Could you put the coke in the drawer and clean up the table with a sponge?'
# query = 'Human: I am hungry. Could you help me?'
# query = 'Human: I am thirsty. Could you help me?'
# query = 'Human: Put this water bottle on the trash can and bring me an apple.'

query = 'Human: Move the book from the table to the sofa.'

query += ' \nRobot: 1.'
prompt = inital_prompt + query

#####################################################################
# skill sets

candidates = ['done', 'bring it to you']
actions = ['find', 'pick up', 'put down', 'go to', 'open', 'close']  # place?
objects = ['apple', 'coke can', 'pepsi can', 'lime soda', 'tea', 'rice chips', 'water bottle', 'energy bar', 'sponge',
           'cup', 'mobilephone', 'tissue box', 'book']
interactive_objects = ['microwave', 'refrigerator', 'drawer']
places = ['table', 'trash can', 'close counter', 'far counter', 'sink', 'sofa']

for a in actions:
    if a == 'go to':
        for o in places:
            new_candidate = f'{a} the {o}'
            candidates.append(new_candidate)
        for o in interactive_objects:
            new_candidate = f'{a} the {o}'
            candidates.append(new_candidate)
    elif a == 'open' or a == 'close':
        for o in interactive_objects:
            new_candidate = f'{a} the {o}'
            candidates.append(new_candidate)
    else:
        for o in objects:
            if a == 'find':
                if o[0] == 'a' or o[0] == 'e':
                    new_candidate = f'{a} an {o}'
                else:
                    new_candidate = f'{a} a {o}'
            else:
                new_candidate = f'{a} the {o}'

            candidates.append(new_candidate)

# add a leading whitespace
candidates = [' ' + c for c in candidates]

# print
print('# of skills:', len(candidates))
print(candidates)
# print(prompt)
print(query)


#####################################################################
# prediction

def run_for_prompt(prompt):
    past_tokenize_input = tokenizer.tokenize(prompt)
    past_tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(past_tokenize_input)])
    past_tensor_input = past_tensor_input.to(device)

    if isinstance(model, GPTNeoXForCausalLM):
        o = model(past_tensor_input, output_hidden_states=True)
        # o = model(past_tensor_input)
        transformer_outputs = [o.hidden_states[-1], o.past_key_values]
    elif isinstance(model, OPTForCausalLM):
        transformer_outputs = model.model.decoder(past_tensor_input)
    else:
        transformer_outputs = model.transformer(past_tensor_input)

    return past_tensor_input, transformer_outputs


def score(sentence, past_key_values, past_last_hidden_state, past_tensor_input):
    tokenize_input = tokenizer.tokenize(sentence, )
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    tensor_input = tensor_input.to(device)

    ###the following code is slightly modified from https://github.com/huggingface/transformers/blob/09a2f40684f77e62d0fd8485fe9d2d610390453f/src/transformers/modeling_gpt2.py#L604
    ###now we calculate the right part of the sentence with the already calculated past
    if isinstance(model, GPTNeoXForCausalLM):
        output_attentions = model.config.output_attentions
        output_hidden_states = (model.config.output_hidden_states)
        o = model(tensor_input, past_key_values=past_key_values,
                  output_attentions=output_attentions,
                  output_hidden_states=output_hidden_states)
        transformer_outputs = [o.hidden_states[-1]]
    elif isinstance(model, OPTForCausalLM):
        o = model.model.decoder(input_ids=tensor_input, past_key_values=past_key_values, output_hidden_states=True)
        transformer_outputs = [o.hidden_states[-1]]
    else:
        transformer_outputs = model.transformer(tensor_input, past_key_values=past_key_values)

    ###and concatenate the output of with the hidden_state of the left part of the sentence
    hidden_states = torch.cat((past_last_hidden_state[:, -1:, :], transformer_outputs[0]), dim=1)

    ###the following part is exactly the same as https://github.com/huggingface/transformers/blob/09a2f40684f77e62d0fd8485fe9d2d610390453f/src/transformers/modeling_gpt2.py#L604
    if isinstance(model, GPTNeoXForCausalLM):
        lm_logits = model.embed_out(hidden_states)
    elif isinstance(model, OPTForCausalLM):
        lm_logits = model.lm_head(hidden_states).contiguous()
    else:
        lm_logits = model.lm_head(hidden_states)

    labels_input = tensor_input
    # print(labels_input.shape)

    shift_logits = lm_logits[..., :-1,
                   :].contiguous().cpu()  # last prediction is not necessary (the next word of the current candidate word)
    shift_labels = labels_input.contiguous().cpu()
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1).long())

    ret = -loss.item()

    # clear (is this really necessary?)
    del transformer_outputs, tensor_input, hidden_states, lm_logits, labels_input, shift_logits, shift_labels, loss
    torch.cuda.empty_cache()

    return ret


def score_without_past_values(sentence):
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence, )
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        tensor_input = tensor_input.to(device)

        # outputs = model.generate(tensor_input, max_new_tokens=3, use_cache=True, temperature=0, return_dict_in_generate=True, output_scores=True)
        ret = model(tensor_input, labels=tensor_input, return_dict=True)

        # todo: using logit values doesn't work. why?
        # logits = ret['logits']
        # min_val = torch.min(logits, dim=-1)[0]
        # score = -torch.sum(min_val, dim=-1).item()

        score = -ret['loss'].item()

    return score

start = time.time()
max_steps = 10

for i in range(max_steps):
    if True:
        # show_gpu('before run prompt')
        past_tensor_input, transformer_outputs = run_for_prompt(prompt)
        # show_gpu('after run prompt')
        past_last_hidden_state, past_key_values = transformer_outputs[0], transformer_outputs[1]
        results = {candidate: score(candidate, past_key_values, past_last_hidden_state, past_tensor_input) for candidate in
                   candidates}

        # free gpu memory
        del past_tensor_input, transformer_outputs
        torch.cuda.empty_cache()
    else:
        results = {candidate: score_without_past_values(prompt + candidate) for candidate in
                   candidates}
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    # show_gpu('after run score')

    # show_gpu('after free')

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(results)

    best_step = results[0][0]
    print(f'{i + 1}.{best_step}')

    # update prompt
    prompt += best_step + f', {i + 2}.'
    # print(prompt)

    # stop criteria
    if best_step == ' done':
        break

show_gpu('gpu usage')
end = time.time()
print('time elapsed: ', end - start)
