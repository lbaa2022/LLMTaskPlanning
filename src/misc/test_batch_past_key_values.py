import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from torch.nn import CrossEntropyLoss

import pdb

import json

def load_pretrained(model_name):
    if model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b", "facebook/opt-350m", "facebook/opt-2.7b", "facebook/opt-30b", "facebook/opt-66b"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_name in ["chainyo/alpaca-lora-7b", "decapoda-research/llama-13b-hf"]:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        NotImplementedError
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = model.to("cuda")
    model.eval()
    return model, tokenizer
    
def make_prompt_skill_set_sentence(prompt_init, skill_set_init, query, type=3):
    prompt_default = prompt_init + f'Human: {query}\nRobot:'
    if type == 0:
        prompt = prompt_default
        skill_set = [f" 1. {skill}" for skill in skill_set_init]
    elif type == 1:
        prompt = f'{prompt_default} '
        skill_set = [f"1. {skill}" for skill in skill_set_init]
    elif type == 2:
        prompt = f'{prompt_default} 1'
        skill_set = [f". {skill}" for skill in skill_set_init]
    elif type == 3:
        prompt = f'{prompt_default} 1.'
        skill_set = [f" {skill}" for skill in skill_set_init]
    elif type == 4: 
        prompt = f'{prompt_default} 1. '
        skill_set = [f"{skill}" for skill in skill_set_init]
    sentences = [f"{prompt}{skill}" for skill in skill_set]
    return prompt, skill_set, sentences

def duplicate_past_key_values(past_key_values, batch_size):
    batch_past_key_values = []
    for layer in range(len(past_key_values)):
        batch_past_key_values_layer = []
        for kv in range(len(past_key_values[layer])):
            batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
        batch_past_key_values_layer = tuple(batch_past_key_values_layer)
        batch_past_key_values.append(batch_past_key_values_layer)
    batch_past_key_values = tuple(batch_past_key_values) 
    return batch_past_key_values

def sentence_single_score(model, model_name, tokenizer, prompt, skill_set, sentences):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    prompt_len = prompt_tokens.attention_mask[0].sum().item()
    scores = []
    for idx, sentence in enumerate(sentences):
        sentence_tokens = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = model(sentence_tokens.input_ids, return_dict=True)
            # output = model(sentence_tokens.input_ids, labels=sentence_tokens.input_ids, return_dict=True)
            logits = output.logits[:, prompt_len-1:-1]
            labels = sentence_tokens.input_ids[:, prompt_len:]
            size_B, size_L, size_C = logits.shape
            logits = logits.reshape([size_B * size_L, size_C])
            labels = labels.reshape([size_B * size_L])
            loss_fn = CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            loss = loss.reshape([size_B, size_L])
            score = -loss.sum(axis=1)
        # scores[idx] = score
        scores.append(score.item())
    return torch.Tensor(scores)

def sentence_batch_score(model, model_name, tokenizer, prompt, skill_set, sentences, batch_size):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    prompt_len = prompt_tokens.attention_mask[0].sum().item()
    scores = []
    for idx in range(0, len(sentences), batch_size):
        sentence = sentences[idx:idx+batch_size]
        sentence_tokens = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            output = model(sentence_tokens.input_ids, attention_mask=sentence_tokens.attention_mask, return_dict=True)
            # output = model(sentence_tokens.input_ids, attention_mask=sentence_tokens.attention_mask,labels=sentence_tokens.input_ids, return_dict=True)
            logits = output.logits[:, prompt_len-1:-1]
            labels = sentence_tokens.input_ids[:, prompt_len:]
            attention_mask = sentence_tokens.attention_mask[:, prompt_len:]
            
            size_B, size_L, size_C = logits.shape
            logits = logits.reshape([size_B * size_L, size_C])
            labels = labels.reshape([size_B * size_L])
            loss_fn = CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            loss = loss.reshape([size_B, size_L])
            score = -(loss*attention_mask).sum(axis=1)
        for score_id in range(score.shape[0]):
            scores.append(score[score_id].item())
    return torch.Tensor(scores)
    
def prompt_skill_set_single_score(model, model_name, tokenizer, prompt, skill_set, sentences):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    # prompt_len = prompt_tokens.attention_mask[0].sum().item()
    prompt_offset = prompt_tokens.attention_mask.sum(dim=1)
    
    with torch.no_grad():
        prompt_output = model(**prompt_tokens, use_cache=True)
    scores = []
    for idx, skill in enumerate(skill_set):
        skill_tokens = tokenizer(skill, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        skill_len = skill_tokens.input_ids.shape[1]
        skill_position_ids = (torch.arange(0, skill_len).to("cuda") + prompt_offset[:, None])
        concat_attention_mask = torch.cat((prompt_tokens.attention_mask, skill_tokens.attention_mask), dim=1)
        with torch.no_grad():
            if 'facebook/opt' in model_name:
                output = model(input_ids=skill_tokens.input_ids,
                            attention_mask=concat_attention_mask,
                            past_key_values=prompt_output.past_key_values,
                            return_dict=True)
            else:
                # output = model(input_ids=skill_tokens.input_ids,
                #             attention_mask=concat_attention_mask,
                #             past_key_values=prompt_output.past_key_values,
                #             position_ids=skill_position_ids,
                #             return_dict=True)
                output = model(input_ids=skill_tokens.input_ids,
                            attention_mask=concat_attention_mask,
                            past_key_values=prompt_output.past_key_values,
                            return_dict=True)
                
            prompt_last_logits = prompt_output.logits[:, -1:, :] # [1, 1, C]
            logits = torch.cat((prompt_last_logits, output.logits[:, :-1, :]), dim=1)
            labels = skill_tokens.input_ids
            attention_mask = skill_tokens.attention_mask
            size_B, size_L, size_C = logits.shape
            logits = logits.reshape([size_B*size_L, size_C])
            labels = labels.reshape([size_B*size_L])
            loss_fn = CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            loss = loss.reshape([size_B, size_L])
            score = -(loss*attention_mask).sum(axis=1)
        scores.append(score.item())
    return torch.Tensor(scores)
    

def prompt_skill_set_batch_score(model, model_name, tokenizer, prompt, skill_set, sentences, batch_size):
    prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    # prompt_len = prompt_tokens.attention_mask[0].sum().item()
    prompt_offset = prompt_tokens.attention_mask.sum(dim=1)
    
    with torch.no_grad():
        prompt_output = model(**prompt_tokens, use_cache=True)
    scores = []
    for idx in range(0, len(skill_set), batch_size):
        skill = skill_set[idx:idx+batch_size]
        size_B = skill.__len__()
        skill_tokens = tokenizer(skill, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        
        skill_len = skill_tokens.input_ids.shape[1]
        skill_position_ids = (torch.arange(0, skill_len).to("cuda") + prompt_offset[:, None])
        concat_attention_mask = torch.cat((prompt_tokens.attention_mask.repeat(size_B, 1), skill_tokens.attention_mask), dim=1)
        batch_past_key_values = duplicate_past_key_values(prompt_output.past_key_values, size_B)
        
        with torch.no_grad():
            if 'facebook/opt' in model_name:
                output = model(input_ids=skill_tokens.input_ids,
                            attention_mask=concat_attention_mask,
                            past_key_values=batch_past_key_values,
                            return_dict=True)
            else:
                # output = model(input_ids=skill_tokens.input_ids,
                #             attention_mask=concat_attention_mask,
                #             past_key_values=batch_past_key_values,
                #             position_ids=skill_position_ids,
                #             return_dict=True)
                output = model(input_ids=skill_tokens.input_ids,
                            attention_mask=concat_attention_mask,
                            past_key_values=batch_past_key_values,
                            return_dict=True)
            
            prompt_last_logits = prompt_output.logits[:, -1:, :].repeat(size_B, 1, 1) # [B, 1, C]
            logits = torch.cat((prompt_last_logits, output.logits[:, :-1, :]), dim=1)
            labels = skill_tokens.input_ids
            attention_mask = skill_tokens.attention_mask
            size_B, size_L, size_C = logits.shape
            logits = logits.reshape([size_B*size_L, size_C])
            labels = labels.reshape([size_B*size_L])
            loss_fn = CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits, labels)
            loss = loss.reshape([size_B, size_L])
            score = -(loss*attention_mask).sum(axis=1)
        for score_id in range(score.shape[0]):
            scores.append(score[score_id].item())
    return torch.Tensor(scores)

prompt_init = 'Robot: Hi there, I’m a robot operating in a home.\nRobot: You can ask me to do various tasks and I’ll tell you the sequence of actions I would do to accomplish your task.\nHuman: Please put 1 cutlery fork, 1 water glass and 1 plate in the dishwasher and turn the dishwasher on\nRobot: 1. walk to dishwasher, 2. open dishwasher, 3. walk to cutlery fork, 4. grab cutlery fork, 5. walk to dishwasher, 6. put cutlery fork in dishwasher, 7. walk to water glass, 8. grab water glass, 9. walk to dishwasher, 10. put water glass in dishwasher, 11. walk to plate, 12. grab plate, 13. walk to dishwasher, 14. put plate in dishwasher, 15. close dishwasher, 16. switch on dishwasher, 17. done.\nHuman: store the pancake, the glass of wine, the cupcake and the glass of juice in the fridge\nRobot: 1. walk to fridge, 2. open fridge, 3. walk to pancake, 4. grab pancake, 5. walk to fridge, 6. put pancake in fridge, 7. walk to wine, 8. grab wine, 9. walk to fridge, 10. put wine in fridge, 11. walk to cupcake, 12. grab cupcake, 13. walk to fridge, 14. put cupcake in fridge, 15. walk to juice, 16. grab juice, 17. walk to fridge, 18. put juice in fridge, 19. close fridge, 20. done.\nHuman: Take a pudding, an apple, a cupcake, and the juice and put them on the coffee table.\nRobot: 1. walk to pudding, 2. grab pudding, 3. walk to coffee table, 4. put pudding on coffee table, 5. walk to apple, 6. grab apple, 7. walk to coffee table, 8. put apple on coffee table, 9. walk to cupcake, 10. grab cupcake, 11. walk to coffee table, 12. put cupcake on coffee table, 13. walk to juice, 14. grab juice, 15. walk to coffee table, 16. put juice on coffee table, 17. done.\nHuman: Find 1 coffee pot, 1 cupcake, 1 pudding and put on the kitchen table.\nRobot: 1. walk to coffee pot, 2. grab coffee pot, 3. walk to kitchen table, 4. put coffee pot on kitchen table, 5. walk to cupcake, 6. grab cupcake, 7. walk to kitchen table, 8. put cupcake on kitchen table, 9. walk to pudding, 10. grab pudding, 11. walk to kitchen table, 12. put pudding on kitchen table, 13. done.\nHuman: I need a wine glass, water glass and a plate on the kitchen table\nRobot: 1. walk to wine glass, 2. grab wine glass, 3. walk to kitchen table, 4. put wine glass on kitchen table, 5. walk to water glass, 6. grab water glass, 7. walk to kitchen table, 8. put water glass on kitchen table, 9. walk to plate, 10. grab plate, 11. walk to kitchen table, 12. put plate on kitchen table, 13. done.\n'
skill_set_init = ['done', 'done.', 'done.\n', 'walk to kitchen counter', 'walk to water glass', 'walk to cutlery fork', 'walk to peach', 'walk to bedroom', 'walk to plate', 'walk to toiletpaper', 'walk to door', 'walk to folder', 'walk to notes', 'walk to faucet', 'walk to toaster', 'walk to chocolate syrup', 'walk to hanger', 'walk to bathroom cabinet', 'walk to cabinet', 'walk to toothbrush', 'walk to washing sponge', 'walk to cereal', 'walk to bed', 'walk to dishwasher', 'walk to bucket', 'walk to closet drawer', 'walk to candle', 'walk to condiment bottle', 'walk to pile of clothes', 'walk to book', 'walk to radio', 'walk to keyboard', 'walk to mug', 'walk to toilet', 'walk to painkillers', 'walk to window', 'walk to mouse mat', 'walk to perfume', 'walk to kitchen counter drawer', 'walk to tv stand', 'walk to doorjamb', 'walk to bookshelf', 'walk to face cream', 'walk to pillow', 'walk to desk', 'walk to box', 'walk to tv', 'walk to stove fan', 'walk to dishwashing liquid', 'walk to coffee pot', 'walk to rug', 'walk to coffee table', 'walk to nightstand', 'walk to apple', 'walk to slice of bread', 'walk to cutlets', 'walk to fridge', 'walk to living room', 'walk to stove', 'walk to cutlery knife', 'walk to closet', 'walk to microwave oven', 'walk to wall shelf', 'walk to mouse', 'walk to milk', 'walk to curtains', 'walk to coffee maker', 'walk to wall phone', 'walk to creamy buns', 'walk to cooking pot', 'walk to pudding', 'walk to crackers', 'walk to lime', 'walk to washing machine', 'walk to wine', 'walk to deodorant', 'walk to plum', 'walk to frying pan', 'walk to table lamp', 'walk to wall picture frame', 'walk to kitchen cabinet', 'walk to hair product', 'walk to ceiling', 'walk to condiment shaker', 'walk to wall lamp', 'walk to juice', 'walk to kitchen', 'walk to toothpaste', 'walk to stall', 'walk to bathroom', 'walk to pear', 'walk to bar soap', 'walk to cpu screen', 'walk to orchid', 'walk to computer', 'walk to bananas', 'walk to chips', 'walk to knife block', 'walk to ceilinglamp', 'walk to wall', 'walk to pancake', 'walk to sofa', 'walk to power socket', 'walk to light switch', 'walk to cupcake', 'walk to bowl', 'walk to cell phone', 'walk to candy bar', 'walk to chair', 'walk to floor', 'walk to sink', 'walk to kitchen table', 'walk to bathroom counter', 'walk to photo frame', 'walk to oven tray', 'grab water glass', 'grab cutlery fork', 'grab plate', 'grab toiletpaper', 'grab folder', 'grab notes', 'grab chocolate syrup', 'grab hanger', 'grab toothbrush', 'grab washing sponge', 'grab cereal', 'grab bucket', 'grab candle', 'grab condiment bottle', 'grab pile of clothes', 'grab book', 'grab radio', 'grab keyboard', 'grab mug', 'grab face cream', 'grab pillow', 'grab box', 'grab dishwashing liquid', 'grab coffee pot', 'grab rug', 'grab apple', 'grab slice of bread', 'grab cutlets', 'grab cutlery knife', 'grab mouse', 'grab milk', 'grab wall phone', 'grab creamy buns', 'grab cooking pot', 'grab pudding', 'grab crackers', 'grab lime', 'grab wine', 'grab plum', 'grab frying pan', 'grab wall picture frame', 'grab hair product', 'grab condiment shaker', 'grab juice', 'grab toothpaste', 'grab pear', 'grab bar soap', 'grab bananas', 'grab chips', 'grab pancake', 'grab cupcake', 'grab bowl', 'grab cell phone', 'grab candy bar', 'grab chair', 'grab oven tray', 'open door', 'open folder', 'open bathroom cabinet', 'open cabinet', 'open dishwasher', 'open pile of clothes', 'open book', 'open radio', 'open toilet', 'open window', 'open bookshelf', 'open desk', 'open box', 'open coffee pot', 'open nightstand', 'open fridge', 'open stove', 'open closet', 'open microwave oven', 'open milk', 'open curtains', 'open coffee maker', 'open cooking pot', 'open washing machine', 'open kitchen cabinet', 'open hair product', 'open toothpaste', 'close door', 'close folder', 'close bathroom cabinet', 'close cabinet', 'close dishwasher', 'close pile of clothes', 'close book', 'close radio', 'close toilet', 'close window', 'close bookshelf', 'close desk', 'close box', 'close coffee pot', 'close nightstand', 'close fridge', 'close stove', 'close closet', 'close microwave oven', 'close milk', 'close curtains', 'close coffee maker', 'close cooking pot', 'close washing machine', 'close kitchen cabinet', 'close hair product', 'close toothpaste', 'switch on faucet', 'switch on toaster', 'switch on dishwasher', 'switch on candle', 'switch on radio', 'switch on tv', 'switch on fridge', 'switch on stove', 'switch on microwave oven', 'switch on coffee maker', 'switch on wall phone', 'switch on washing machine', 'switch on computer', 'switch on light switch', 'switch on cell phone']
# query = ['Put one cupcake and one apple on the coffee table']
query = 'Put one cupcake and one apple on the coffee table'

if __name__ == '__main__':
    for model_name in ["chainyo/alpaca-lora-7b", "decapoda-research/llama-13b-hf"]:
    # for model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-j-6B", "facebook/opt-2.7b"]:
        model, tokenizer = load_pretrained(model_name)
        prompt, skill_set, sentences = make_prompt_skill_set_sentence(prompt_init, skill_set_init, query, type=3)
        if model_name in ["chainyo/alpaca-lora-7b", "decapoda-research/llama-13b-hf"]:
            skill_set = [skill[1:] for skill in skill_set]
        
        # ##### DEBUG: LLaMMa
        # if model_name in ["chainyo/alpaca-lora-7b", "decapoda-research/llama-13b-hf"]:
        #     skill_set = [skill[1:] for skill in skill_set]
        #     test_b_size = 4
        #     sentence_tokens = tokenizer(sentences[:test_b_size], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        #     prompt_tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        #     skill_tokens = tokenizer(skill_set[:test_b_size], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            
        #     prompt_len = prompt_tokens.input_ids.shape[1]
        #     skill_len = skill_tokens.input_ids.shape[1]
            
        #     concat_tokens_input_ids = torch.cat((prompt_tokens.input_ids.repeat(test_b_size, 1), skill_tokens.input_ids), dim=1)
        #     concat_tokens_attention_mask = torch.cat((prompt_tokens.attention_mask.repeat(test_b_size, 1), skill_tokens.attention_mask), dim=1)
            
            
            
            
        #     pdb.set_trace()
        
        
        sentence_single_scores = sentence_single_score(model, model_name, tokenizer, prompt, skill_set, sentences)
        sentence_batch_scores = sentence_batch_score(model, model_name, tokenizer, prompt, skill_set, sentences, batch_size=4)
        # sentence_single_tokens_all = sentence_single_tokenize(tokenizer, sentences)
        # sentence_batch_tokens_all = sentence_batch_tokenize(tokenizer, sentences)
        check1 = torch.argsort(sentence_single_scores) == torch.argsort(sentence_batch_scores)
        
        prompt_skill_set_single_scores = prompt_skill_set_single_score(model, model_name, tokenizer, prompt, skill_set, sentences)
        check2 = torch.argsort(sentence_single_scores) == torch.argsort(prompt_skill_set_single_scores)
        
        prompt_skill_set_batch_scores = prompt_skill_set_batch_score(model, model_name, tokenizer, prompt, skill_set, sentences, batch_size=4)
        check3 = torch.argsort(prompt_skill_set_single_scores) == torch.argsort(prompt_skill_set_batch_scores)
        
        print(model_name)
        print("Check 1: sentence single vs sentence batch")
        print(check1[:20])
        print("Check 2: sentence single vs prompt skill set single")
        print(check2[:20])
        print("Check 3: prompt skill set single vs prompt skill set batch")
        print(check3[:20])
        print()
    pdb.set_trace()
