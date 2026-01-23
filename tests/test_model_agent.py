"""
Tests for the ModelAgent module.

Run with: pytest -v
"""

from __future__ import annotations

import pytest

from tests.models import Place

from django_model_agent import ModelAgent
from django_model_agent.base import ModelAgentContext
from django_model_agent.tools import (
    ModelTool,
    ReadOnlyTool,
    ToolResult,
    UpdateTool,
)


# -----------------------------------------------------------------------------
# ModelAgent Base Tests
# -----------------------------------------------------------------------------


class TestModelAgent:
    """Tests for the ModelAgent base class."""

    def test_init_with_instance(self, place, simple_agent_class):
        """Test agent initialization with a model instance."""
        agent = simple_agent_class(place)

        assert agent.instance == place
        assert agent.field_set is None
        assert agent._schema is None  # Lazy loaded

    def test_schema_generation(self, place, simple_agent_class):
        """Test that schema is generated correctly from model fields."""
        agent = simple_agent_class(place)
        schema = agent.schema

        assert schema.__name__ == "PlaceAgentSchema"
        assert "name" in schema.model_fields
        assert "address" in schema.model_fields
        assert "phone" in schema.model_fields
        # Excluded fields should not be present
        assert "doordash_url" not in schema.model_fields

    def test_schema_caching(self, place, simple_agent_class):
        """Test that schema is cached after first access."""
        agent = simple_agent_class(place)

        schema1 = agent.schema
        schema2 = agent.schema

        assert schema1 is schema2

    def test_field_sets(self, place, agent_with_field_sets):
        """Test field set selection."""
        # Public field set
        public_agent = agent_with_field_sets(place, field_set="public")
        public_schema = public_agent.schema

        assert "name" in public_schema.model_fields
        assert "address" in public_schema.model_fields
        assert "phone" not in public_schema.model_fields

        # Staff field set
        staff_agent = agent_with_field_sets(place, field_set="staff")
        staff_schema = staff_agent.schema

        assert "name" in staff_schema.model_fields
        assert "address" in staff_schema.model_fields
        assert "phone" in staff_schema.model_fields
        assert "notes" in staff_schema.model_fields

    def test_get_system_prompt(self, place, simple_agent_class):
        """Test system prompt retrieval."""
        agent = simple_agent_class(place)

        assert agent.get_system_prompt() == "You are a test agent."

    def test_get_current_values(self, place, simple_agent_class):
        """Test getting current field values."""
        agent = simple_agent_class(place)
        values = agent.get_current_values()

        assert values["name"] == "Test Restaurant"
        assert values["address"] == "123 Main St"
        assert values["phone"] == "785-555-1234"

    def test_get_schema_description(self, place, simple_agent_class):
        """Test schema description generation."""
        agent = simple_agent_class(place)
        description = agent.get_schema_description()

        assert "Place" in description
        assert "name" in description
        assert "address" in description

    def test_context_property(self, place, simple_agent_class):
        """Test context object creation."""
        agent = simple_agent_class(place)
        context = agent.context

        assert isinstance(context, ModelAgentContext)
        assert context.instance == place
        assert context.agent == agent

    def test_repr(self, place, simple_agent_class):
        """Test agent string representation."""
        agent = simple_agent_class(place)

        assert "SimpleAgent" in repr(agent)
        assert "Place" in repr(agent)
        assert str(place.pk) in repr(agent)


class TestModelAgentContext:
    """Tests for ModelAgentContext."""

    def test_refresh_instance(self, place, simple_agent_class):
        """Test instance refresh from database."""
        agent = simple_agent_class(place)
        context = agent.context

        # Modify in database directly
        Place.objects.filter(pk=place.pk).update(name="Updated Name")

        # Instance still has old value
        assert context.instance.name == "Test Restaurant"

        # Refresh loads new value
        context.instance.refresh_from_db(fields=["name"])
        assert context.instance.name == "Updated Name"


class TestModelAgentDecorators:
    """Tests for ModelAgent decorator functionality."""

    def test_system_prompt_decorator(self, place):
        """Test @ModelAgent.system_prompt decorator."""

        class DecoratedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            base_prompt = "Base prompt."

            @ModelAgent.system_prompt
            def dynamic_prompt(self) -> str:
                return f"Working with: {self.instance.name}"

        agent = DecoratedAgent(place)
        prompt = agent.get_system_prompt()

        assert "Base prompt." in prompt
        assert "Working with: Test Restaurant" in prompt

    def test_multiple_system_prompt_decorators(self, place):
        """Test multiple @ModelAgent.system_prompt decorators."""

        class MultiPromptAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.system_prompt
            def first_prompt(self) -> str:
                return "First part."

            @ModelAgent.system_prompt
            def second_prompt(self) -> str:
                return "Second part."

        agent = MultiPromptAgent(place)
        prompt = agent.get_system_prompt()

        assert "First part." in prompt
        assert "Second part." in prompt

    def test_instructions_decorator(self, place):
        """Test @ModelAgent.instructions decorator."""

        class InstructedAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.instructions
            def state_instructions(self) -> str:
                return f"Current state: {self.instance.state}"

        agent = InstructedAgent(place)
        instructions = agent.get_instructions()

        assert instructions is not None
        assert "Current state: public" in instructions

    def test_tool_decorator(self, place):
        """Test @ModelAgent.tool decorator."""

        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

            @ModelAgent.tool
            def get_address(self) -> str:
                """Get the place address."""
                return self.instance.address or "No address"

        agent = ToolAgent(place)
        tools = agent.get_tools()

        assert len(tools) == 2

        # Tools should be callable and return expected values
        results = [t() for t in tools]
        assert "Test Restaurant" in results
        assert "123 Main St" in results


# -----------------------------------------------------------------------------
# Tool Tests
# -----------------------------------------------------------------------------


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(success=True, message="Done", data={"key": "value"})

        assert result.success is True
        assert result.message == "Done"
        assert result.data == {"key": "value"}

    def test_failure_result(self):
        """Test failed tool result."""
        result = ToolResult(success=False, message="Error occurred")

        assert result.success is False
        assert result.message == "Error occurred"
        assert result.data is None


class TestReadOnlyTool:
    """Tests for ReadOnlyTool."""

    def test_read_only_tool_execute(self, place, simple_agent_class):
        """Test that ReadOnlyTool.execute calls read method."""
        agent = simple_agent_class(place)
        context = agent.context

        class TestReadTool(ReadOnlyTool):
            name = "test_read"
            description = "Test read tool"

            def read(self, **kwargs):
                return {"name": self.instance.name}

        tool = TestReadTool(context)
        result = tool.execute()

        assert result.success is True
        assert result.data == {"name": "Test Restaurant"}


class TestUpdateTool:
    """Tests for UpdateTool."""

    def test_update_tool_execute(self, place, simple_agent_class):
        """Test that UpdateTool.execute calls update method and saves."""
        agent = simple_agent_class(place)
        context = agent.context

        class TestUpdateTool(UpdateTool):
            name = "test_update"
            description = "Test update tool"

            def update(self, **kwargs):
                self.instance.name = kwargs.get("new_name", self.instance.name)

        tool = TestUpdateTool(context)
        result = tool.execute(new_name="New Name")

        assert result.success is True
        place.refresh_from_db()
        assert place.name == "New Name"
