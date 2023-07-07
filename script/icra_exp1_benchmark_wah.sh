### Planning
  ### EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-j-6B
  ### chainyo/alpaca-lora-7b, decapoda-research/llama-13b-hf
  ### facebook/opt-350m, facebook/opt-2.7b, facebook/opt-30b, facebook/opt-66b


### DEBUG EXP 5 -> Super
python src/planning.py experiment.exp_name=debugging_230703_super planner.model_name=EleutherAI/gpt-neo-1.3B planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=10
python src/planning.py experiment.exp_name=debugging_230703_super planner.model_name=EleutherAI/gpt-j-6B planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=10
python src/planning.py experiment.exp_name=debugging_230703_super planner.model_name=chainyo/alpaca-lora-7b planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=10
python src/planning.py experiment.exp_name=debugging_230703_super planner.model_name=facebook/opt-2.7b planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=10
python src/planning.py experiment.exp_name=debugging_230703_super planner.model_name=decapoda-research/llama-13b-hf planner.score_function='sum' planner.fast_mode=True planner.scoring_batch_size=4
