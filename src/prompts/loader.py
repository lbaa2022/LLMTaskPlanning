"""
Prompt loader for externalized prompt templates.
Loads prompt templates from text files and provides formatting utilities.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptTemplates:
    """Container for loaded prompt templates."""
    action_selection_system: str
    action_selection_user: str
    plan_generation_system: str
    plan_generation_user: str
    step_by_step_format: str
    step_continuation: str
    step_with_failure: str


class PromptLoader:
    """
    Loads and manages prompt templates from external files.

    Templates are loaded from src/prompts/templates/ by default,
    but can be overridden by specifying a custom templates directory.
    """

    DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"

    # Template file names
    TEMPLATE_FILES = {
        "action_selection_system": "action_selection_system.txt",
        "action_selection_user": "action_selection_user.txt",
        "plan_generation_system": "plan_generation_system.txt",
        "plan_generation_user": "plan_generation_user.txt",
        "step_by_step_format": "step_by_step_format.txt",
        "step_continuation": "step_continuation.txt",
        "step_with_failure": "step_with_failure.txt",
    }

    _instance: Optional['PromptLoader'] = None
    _templates: Optional[PromptTemplates] = None

    def __new__(cls, templates_dir: Optional[Path] = None):
        """Singleton pattern to avoid reloading templates."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize the prompt loader.

        Args:
            templates_dir: Optional custom templates directory.
                          Defaults to src/prompts/templates/
        """
        if self._initialized:
            return

        self.templates_dir = templates_dir or self.DEFAULT_TEMPLATES_DIR
        self._load_templates()
        self._initialized = True

    def _load_templates(self) -> None:
        """Load all template files from disk."""
        templates = {}

        for name, filename in self.TEMPLATE_FILES.items():
            filepath = self.templates_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Template file not found: {filepath}\n"
                    f"Please ensure all template files exist in {self.templates_dir}"
                )

            with open(filepath, 'r', encoding='utf-8') as f:
                templates[name] = f.read().strip()

        self._templates = PromptTemplates(**templates)

    @property
    def templates(self) -> PromptTemplates:
        """Get loaded templates."""
        if self._templates is None:
            self._load_templates()
        return self._templates

    def get_action_selection_messages(
        self,
        prompt: str,
        candidates: List[str]
    ) -> List[Dict[str, str]]:
        """
        Build messages for action selection task.

        Args:
            prompt: Current context/prompt
            candidates: List of candidate actions

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        candidates_text = '\n'.join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

        user_content = self.templates.action_selection_user.format(
            prompt=prompt,
            candidates=candidates_text
        )

        return [
            {"role": "system", "content": self.templates.action_selection_system},
            {"role": "user", "content": user_content}
        ]

    def get_plan_generation_messages(
        self,
        examples: str,
        skills: str,
        query: str
    ) -> List[Dict[str, str]]:
        """
        Build messages for whole-plan generation task.

        Args:
            examples: Example text with human-robot interactions
            skills: Comma-separated list of available skills
            query: User's instruction/query

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        user_content = self.templates.plan_generation_user.format(
            examples=examples,
            skills=skills,
            query=query
        )

        return [
            {"role": "system", "content": self.templates.plan_generation_system},
            {"role": "user", "content": user_content}
        ]

    def format_step_by_step_start(self, query: str) -> str:
        """
        Format the initial step-by-step planning prompt.

        Args:
            query: User's instruction/query

        Returns:
            Formatted prompt string
        """
        return self.templates.step_by_step_format.format(query=query.strip())

    def format_step_continuation(self, step: str, next_step_num: int) -> str:
        """
        Format a step continuation.

        Args:
            step: The completed step
            next_step_num: The next step number

        Returns:
            Formatted continuation string
        """
        return self.templates.step_continuation.format(
            step=step,
            next_step_num=next_step_num
        )

    def format_step_with_failure(
        self,
        step: str,
        failure_message: str,
        next_step_num: int
    ) -> str:
        """
        Format a step that failed with an error message.

        Args:
            step: The failed step
            failure_message: The failure message
            next_step_num: The next step number

        Returns:
            Formatted step with failure string
        """
        return self.templates.step_with_failure.format(
            step=step,
            failure_message=failure_message.lower(),
            next_step_num=next_step_num
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._templates = None


# Global convenience function
def get_prompt_loader(templates_dir: Optional[Path] = None) -> PromptLoader:
    """Get the prompt loader instance."""
    return PromptLoader(templates_dir)
