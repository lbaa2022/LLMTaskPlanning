def test_transformers():
    from importlib.metadata import version
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    n_gpus = torch.cuda.device_count()
    print('# of gpus:', n_gpus)

    # model_name = 'EleutherAI/gpt-neo-125M'
    # model_name = 'facebook/opt-30b'
    model_name = 'facebook/opt-66b'
    sentence = 'Hello, nice to meet you. How are'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # max_memory = {i: '38GB' for i in range(n_gpus)}

    # # cpu
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # with torch.no_grad():
    #     tokenize_input = tokenizer.tokenize(sentence)
    #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    #     gen_tokens = model.generate(tensor_input, max_length=32)
    #     generated = tokenizer.batch_decode(gen_tokens)[0]

    # print(generated)
    # print('-------------------------------------------')

    # # on the gpu 0
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model = model.to('cuda:0')

    # with torch.no_grad():
    #     tokenize_input = tokenizer.tokenize(sentence)
    #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    #     tensor_input = tensor_input.to('cuda:0')
    #     gen_tokens = model.generate(tensor_input, max_length=32)
    #     generated = tokenizer.batch_decode(gen_tokens)[0]

    # print(generated)
    # print('-------------------------------------------')

    # # on the gpu 1
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # model = model.to('cuda:1')

    # with torch.no_grad():
    #     tokenize_input = tokenizer.tokenize(sentence)
    #     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    #     tensor_input = tensor_input.to('cuda:1')
    #     gen_tokens = model.generate(tensor_input, max_length=32)
    #     generated = tokenizer.batch_decode(gen_tokens)[0]

    # print(generated)
    # print('-------------------------------------------')

    # with device_map=auto
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True, max_memory=max_memory)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    print('hf_device_map output:', model.hf_device_map)

    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        tensor_input = tensor_input.to('cuda:0')
        gen_tokens = model.generate(tensor_input, max_length=32)
        generated = tokenizer.batch_decode(gen_tokens)[0]

    print(generated)


def test_gpt3():
    My_OpenAI_key = ''

    import openai
    openai.api_key = My_OpenAI_key
    completion = openai.Completion()

    # chatbot test
    question = 'What is your name?'
    prompt_initial = f'Human: %s\nAI: ' % (question)

    prompt = prompt_initial

    response = completion.create(
        prompt=prompt,
        engine="text-curie-001",
        max_tokens=0,
        logprobs=1,
        temperature=0,
        echo=True
    )
    print(response)
    answer = response.choices[0].text.strip()
    print(prompt, answer)


if __name__ == '__main__':
    # test_transformers()
    test_gpt3()
