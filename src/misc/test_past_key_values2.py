import math

import torch
import time
import numpy as np
from torch.nn import CrossEntropyLoss
import pprint
import subprocess

from transformers import AutoModelForCausalLM, AutoTokenizer

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

inital_prompt = ''.join((s for s in prompt_sentences))

# model_name = "EleutherAI/gpt-neo-125M"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "EleutherAI/gpt-neox-20b"
# model_name = "facebook/opt-350m"
model_name = "facebook/opt-13b"
# model_name = "facebook/opt-66b"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)  # max_memory={0: "30GiB", "cpu": "60GiB"}
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

query = 'Human: Move the book from the table to the sofa.'
# query = 'Human: How would you put the grapes in the bowl and then move the cheese to the table?'

query += ' \nRobot: 1'
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

# add a leading dot and whitespace
candidates = ['. ' + c for c in candidates]

# print
print('# of skills:', len(candidates))
print(candidates)
# print(prompt)
print(query)


#####################################################################
# prediction

def score_skills_fast(model_name, prompt_common, candidates):
    prompt_encoding = tokenizer(prompt_common, add_special_tokens=False, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        prompt_out = model(**prompt_encoding, use_cache=True)
    prompt_offsets = prompt_encoding.attention_mask.sum(dim=1)

    scores = {}

    for candidate in candidates:
        skill_encoding = tokenizer(candidate, add_special_tokens=False, return_tensors="pt", padding=True).to(device)
        num_completion_tokens = skill_encoding.input_ids.shape[1]
        skill_position_ids = (torch.arange(0, num_completion_tokens).to(device) + prompt_offsets[:, None])
        attention_mask = torch.cat((prompt_encoding.attention_mask,
                                    skill_encoding.attention_mask),
                                    dim=1)

        with torch.no_grad():
            if 'facebook/opt' in model_name:
                out = model(input_ids=skill_encoding.input_ids, labels=skill_encoding.input_ids,
                            attention_mask=attention_mask,
                            past_key_values=prompt_out.past_key_values,
                            return_dict=True)
            else:
                out = model(input_ids=skill_encoding.input_ids,
                            attention_mask=attention_mask,
                            past_key_values=prompt_out.past_key_values,
                            position_ids=skill_position_ids,
                            return_dict=True)

        logits = out['logits'][:, :-1]  # [B, L, C]
        labels = skill_encoding['input_ids'][:, 1:]  # [B, L]
        size_B, size_L, size_C = logits.shape
        logits = logits.reshape([size_B * size_L, size_C])
        labels = labels.reshape([size_B * size_L])
        loss_fn = CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels)
        loss = loss.reshape([size_B, size_L])
        score = -loss.sum(axis=1)

        scores[candidate] = score
        if math.isnan(scores[candidate]):
            scores[candidate] = -1e+10

    return scores


def score_text(sentence, prompt_len):
    with torch.no_grad():
        tensor_input = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding=True)
        tensor_input = tensor_input.to(device)
        out = model(tensor_input.input_ids, labels=tensor_input.input_ids, return_dict=True)
        # score = -out['loss'].item()

        logits = out['logits'][:, prompt_len-1:-1]  # [B, L, C]
        labels = tensor_input['input_ids'][:, prompt_len:]  # [B, L]
        size_B, size_L, size_C = logits.shape
        logits = logits.reshape([size_B * size_L, size_C])
        labels = labels.reshape([size_B * size_L])
        loss_fn = CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels)
        loss = loss.reshape([size_B, size_L])
        score = -loss.sum(axis=1)

    return score

max_steps = 5

for j in [0, 1]:
    start = time.time()
    prompt_updated = prompt

    for i in range(max_steps):
        if j == 0:
            results = score_skills_fast(model_name, prompt_updated, candidates)
        else:
            prompt_len = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")['input_ids'].shape[1]
            results = {candidate: score_text(prompt_updated + candidate, prompt_len) for candidate in candidates}
        results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        best_step = results[0][0]
        print(f'{i + 1}{best_step}')

        # update prompt
        prompt_updated += best_step + f', {i + 2}'

        # stop criteria
        if best_step == '. done':
            break

    end = time.time()
    print('time elapsed: ', end - start)

# print(prompt_updated)
