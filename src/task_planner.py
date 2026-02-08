"""
Task Planner base class using abstracted LLM providers.
"""
import logging
from typing import List, Tuple, Optional

from llm import LLMProviderFactory, LLMProvider
from prompts import get_prompt_loader


class TaskPlanner:
    """
    Base class for LLM-based task planning.

    Uses the Strategy Pattern via LLMProvider to support OpenAI API compatible backends:
    - OpenAI (GPT-3.5, GPT-4, GPT-5, o1, o3)
    - vLLM (locally served models with OpenAI-compatible API)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_steps = cfg.planner.max_steps
        self.model_name = cfg.planner.model_name
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt

        # Initialize LLM provider using factory
        print(f"Loading LLM: {self.model_name}")
        self.llm: LLMProvider = LLMProviderFactory.from_config(cfg)
        print(f"Provider: {type(self.llm).__name__}")

        self.prompt_loader = get_prompt_loader()

        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        # Load prompt
        self.prompt = self.init_prompt(cfg)

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()

    def score(self, prompt: str, skill_set: List[str]) -> dict:
        """
        Score skills by selecting the best action using the LLM provider.

        Args:
            prompt: Current context/prompt
            skill_set: List of candidate actions

        Returns:
            Dict mapping skills to scores (selected=1.0, others=0.0)
        """
        # Use LLM to select best action
        selected_action = self.llm.select_action(prompt, skill_set)

        # Build scores dict: selected action gets 1.0, others get 0.0
        scores = {}
        matched = False

        for skill in skill_set:
            if skill.strip().lower() == selected_action.lower() or selected_action.lower() in skill.strip().lower():
                scores[skill] = 1.0
                matched = True
            else:
                scores[skill] = 0.0

        # Fallback: if no exact match, try fuzzy matching
        if not matched:
            for skill in skill_set:
                if selected_action.lower() in skill.lower() or skill.lower() in selected_action.lower():
                    scores[skill] = 1.0
                    matched = True
                    break

            # Last resort: assign score to first candidate
            if not matched and skill_set:
                scores[skill_set[0]] = 0.5

        return scores

    def plan_whole(self, query: str) -> Tuple[List[str], List[int]]:
        """
        Generate a complete action plan in one shot.

        Args:
            query: User instruction/query

        Returns:
            Tuple of (action_sequence, skill_set_sizes)
        """
        step_seq = []
        skill_set_size_seq = []
        print(f"Input query: {query}")

        prompt_lines = self.prompt.split('\n')
        prompt_examples = prompt_lines[2:]
        example_text = '\n'.join(prompt_examples)
        skills_text = ', '.join([x.strip() for x in self.skill_set])

        # Use LLM provider's helper to build messages
        messages = self.llm._build_plan_messages(example_text, skills_text, query)

        # Generate response
        answer = self.llm.chat_completion(messages, temperature=0, max_tokens=500)
        print(answer)

        # Parse response to list
        answer = answer.replace('Robot: ', '')
        actions = [action.strip(' 1234567890.') for action in answer.split(',')]
        step_seq = actions

        return step_seq, skill_set_size_seq

    def plan_step_by_step(self, query: str, prev_steps: tuple = (), prev_msgs: tuple = ()) -> Tuple[Optional[str], Optional[str]]:
        """
        Plan one step at a time, selecting the best action for current context.

        Args:
            query: User instruction/query
            prev_steps: Previous steps taken
            prev_msgs: Previous feedback messages

        Returns:
            Tuple of (best_step, prompt) or (None, None) if max steps reached
        """
        if len(prev_steps) >= self.max_steps:
            return None, None

        prompt = self.prompt + self.prompt_loader.format_step_by_step_start(query)

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += self.prompt_loader.format_step_with_failure(step, msg, i + 2)
            else:
                prompt += self.prompt_loader.format_step_continuation(step, i + 2)

        # Score candidates
        scores = self.score(prompt, self.skill_set)

        # Find best step
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_step = results[0][0]
        best_step = best_step.strip()

        return best_step, prompt
