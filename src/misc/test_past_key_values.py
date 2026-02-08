'''
https://github.com/huggingface/transformers/issues/18104
https://github.com/microsoft/guidance/issues/116
'''

# change transformers cache directory (https://stackoverflow.com/questions/63312859/how-to-change-huggingface-transformers-default-cache-directory)
import os
os.environ['TRANSFORMERS_CACHE'] = '/media/yw/work/.cache/'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

model_name = 'EleutherAI/gpt-neo-125M'  # ok with position ids
# model_name = 'EleutherAI/gpt-j-6B'  # ok with position ids
# model_name = 'EleutherAI/gpt-neox-20b'  # ok with position ids
# model_name = 'facebook/opt-350m'  # ok
# model_name = 'openlm-research/open_llama_3b'  # failed
# model_name = 'chainyo/alpaca-lora-7b'  # ?
# model_name = 'tiiuae/falcon-7b'  # failed. attn mask size mismatch?
# model_name = 'bigscience/bloom-560m'  # ?


if 'llama' in model_name or 'alpaca' in model_name:
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# if not tokenizer.padding_side == 'right':
#     raise ValueError('Gotta use right padding to ensure position IDs are '
#                      'correct.')


# prefixes = ['a b c',
#             'l m n o p q r s t u v w x y']
# suffixes = [' d e',
#             ' z']

prefixes = ['a b c']
suffixes = [' d e']


# Batch inference prefixes
prefixes_encoding = tokenizer(prefixes, add_special_tokens=False, return_tensors='pt', padding=True)
with torch.no_grad():
    prefixes_out = model(**prefixes_encoding, use_cache=True)
# Need offsets so that position_ids for future tokens are set correctly
offsets = prefixes_encoding.attention_mask.sum(dim=1)


# Batch inference suffixes
suffixes_encoding = tokenizer(suffixes, add_special_tokens=False, return_tensors='pt', padding=True)
num_completion_tokens = suffixes_encoding.input_ids.shape[1]

# Set position_ids to what they were had we fed each prefix + suffix
# together w/ right-padding (right-padding b/c GPT-2 uses absolute position ids)
suffixes_position_ids = (torch.arange(0, num_completion_tokens) +
                         offsets[:, None]) # broadcast

# Need attention_mask to include the prefixes since it could have padding
attention_mask = torch.cat((prefixes_encoding.attention_mask,
                            suffixes_encoding.attention_mask),
                            dim=1)


# Everything should now be aligned 🤞 🙏
with torch.no_grad():
    if True:
        suffixes_out = model(input_ids=suffixes_encoding.input_ids,
                             attention_mask=attention_mask,
                             past_key_values=prefixes_out.past_key_values,
                             position_ids=suffixes_position_ids)
    else:
        suffixes_out = model(input_ids=suffixes_encoding.input_ids,
                             attention_mask=attention_mask,
                             past_key_values=prefixes_out.past_key_values)

# Expected output
full = [prefix + suffix for prefix, suffix in zip(prefixes, suffixes)]
full_encoding = tokenizer(full, add_special_tokens=False, return_tensors='pt', padding=True)
with torch.no_grad():
    full_out = model(**full_encoding)


# Test shape
assert suffixes_out.logits.shape[0]  == full_out.logits.shape[0]
assert suffixes_out.logits.shape[-1] == full_out.logits.shape[-1]


# Test that every non-pad token's logits are close.
# (in the comments, the token in parentheses is the one whose logits we're
#  acessing)
# torch.set_printoptions(profile="full")
# torch.set_printoptions(linewidth=200)
torch.set_printoptions(edgeitems=100)
print(suffixes_out.logits[0, 0])
print(full_out.logits[0, 3])
# torch.set_printoptions(profile="default") # reset
close_1 = torch.allclose(suffixes_out.logits[0, 0], # (d), e
                          full_out.logits[0, 3]) # a, b, c, (d), e, rest are <PAD>
print(torch.sum(torch.abs(suffixes_out.logits[0, 0] - full_out.logits[0, 3])))

close_2 = torch.allclose(suffixes_out.logits[0, 1], # d, (e)
                          full_out.logits[0, 4]) # a, b, c, d, (e), rest are <PAD>
print(torch.sum(torch.abs(suffixes_out.logits[0, 1] - full_out.logits[0, 4])))
print(torch.sum(torch.abs(suffixes_out.logits[0, 1] - full_out.logits[0, 3])))

# close_3 = torch.allclose(suffixes_out.logits[1,  0], # (z), <PAD>
#                           full_out.logits[1, -1]) # l m n o p q r s t u v w x y (z)
# print(torch.sum(torch.abs(suffixes_out.logits[1, 0] - full_out.logits[1, -1])))
#
# print(torch.sum(torch.abs(suffixes_out.logits[1, 0] - full_out.logits[1, -2])))

print(close_1)
print(close_2)
# print(close_3)
