"""
Test suite for prompt loading functionality.
Tests PromptLoader and template formatting.
"""
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts import PromptLoader, PromptTemplates, get_prompt_loader


class TestPromptLoader:
    """Test PromptLoader functionality."""

    @pytest.fixture(autouse=True)
    def reset_loader(self):
        """Reset singleton before each test."""
        PromptLoader.reset()
        yield
        PromptLoader.reset()

    def test_singleton_pattern(self):
        """PromptLoader should be a singleton."""
        loader1 = PromptLoader()
        loader2 = PromptLoader()
        assert loader1 is loader2

    def test_templates_loaded(self):
        """All templates should be loaded."""
        loader = PromptLoader()
        templates = loader.templates

        assert isinstance(templates, PromptTemplates)
        assert templates.action_selection_system
        assert templates.action_selection_user
        assert templates.plan_generation_system
        assert templates.plan_generation_user
        assert templates.step_by_step_format
        assert templates.step_continuation
        assert templates.step_with_failure

    def test_get_prompt_loader_function(self):
        """get_prompt_loader should return the singleton."""
        loader1 = get_prompt_loader()
        loader2 = get_prompt_loader()
        assert loader1 is loader2


class TestActionSelectionMessages:
    """Test action selection message building."""

    @pytest.fixture(autouse=True)
    def reset_loader(self):
        """Reset singleton before each test."""
        PromptLoader.reset()
        yield
        PromptLoader.reset()

    def test_returns_two_messages(self):
        """Should return system and user messages."""
        loader = PromptLoader()
        messages = loader.get_action_selection_messages(
            "Test prompt",
            ["action1", "action2"]
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_candidates_formatted_with_numbers(self):
        """Candidates should be numbered in user message."""
        loader = PromptLoader()
        messages = loader.get_action_selection_messages(
            "Test prompt",
            ["action1", "action2", "action3"]
        )

        user_content = messages[1]["content"]
        assert "1. action1" in user_content
        assert "2. action2" in user_content
        assert "3. action3" in user_content

    def test_prompt_included_in_user_message(self):
        """The prompt should be in user message."""
        loader = PromptLoader()
        messages = loader.get_action_selection_messages(
            "Pick up the apple",
            ["action1"]
        )

        assert "Pick up the apple" in messages[1]["content"]


class TestPlanGenerationMessages:
    """Test plan generation message building."""

    @pytest.fixture(autouse=True)
    def reset_loader(self):
        """Reset singleton before each test."""
        PromptLoader.reset()
        yield
        PromptLoader.reset()

    def test_returns_two_messages(self):
        """Should return system and user messages."""
        loader = PromptLoader()
        messages = loader.get_plan_generation_messages(
            "Example text",
            "skill1, skill2",
            "User query"
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_examples_in_user_message(self):
        """Examples should be in user message."""
        loader = PromptLoader()
        messages = loader.get_plan_generation_messages(
            "Human: Do X\nRobot: 1. action",
            "skill1, skill2",
            "User query"
        )

        assert "Human: Do X" in messages[1]["content"]
        assert "Robot: 1. action" in messages[1]["content"]

    def test_skills_in_user_message(self):
        """Skills should be in user message."""
        loader = PromptLoader()
        messages = loader.get_plan_generation_messages(
            "Examples",
            "pick up, put down, open",
            "Query"
        )

        assert "pick up, put down, open" in messages[1]["content"]

    def test_query_in_user_message(self):
        """Query should be in user message."""
        loader = PromptLoader()
        messages = loader.get_plan_generation_messages(
            "Examples",
            "skills",
            "Put the apple on the table"
        )

        assert "Put the apple on the table" in messages[1]["content"]


class TestStepFormatting:
    """Test step-by-step formatting methods."""

    @pytest.fixture(autouse=True)
    def reset_loader(self):
        """Reset singleton before each test."""
        PromptLoader.reset()
        yield
        PromptLoader.reset()

    def test_format_step_by_step_start(self):
        """Should format initial step prompt."""
        loader = PromptLoader()
        result = loader.format_step_by_step_start("Put apple on table")

        assert "Put apple on table" in result
        assert "Human:" in result or "Robot:" in result

    def test_format_step_continuation(self):
        """Should format step continuation."""
        loader = PromptLoader()
        result = loader.format_step_continuation("pick up apple", 2)

        assert "pick up apple" in result
        assert "2" in result

    def test_format_step_with_failure(self):
        """Should format failed step with error message."""
        loader = PromptLoader()
        result = loader.format_step_with_failure(
            "pick up apple",
            "Object not visible",
            3
        )

        assert "pick up apple" in result
        assert "object not visible" in result.lower()  # Should be lowercased
        assert "3" in result


class TestTemplateFilesExist:
    """Test that all template files exist."""

    def test_all_template_files_exist(self):
        """All required template files should exist."""
        templates_dir = Path(__file__).parent.parent / "src" / "prompts" / "templates"

        required_files = [
            "action_selection_system.txt",
            "action_selection_user.txt",
            "plan_generation_system.txt",
            "plan_generation_user.txt",
            "step_by_step_format.txt",
            "step_continuation.txt",
            "step_with_failure.txt",
        ]

        for filename in required_files:
            filepath = templates_dir / filename
            assert filepath.exists(), f"Missing template file: {filepath}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
