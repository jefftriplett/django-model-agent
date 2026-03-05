"""
Tests for the ModelAgent module.

Run with: pytest -v
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tests.models import Place

from django_model_agent import ModelAgent
from django_model_agent.base import ModelAgentContext
from django_model_agent.memory import AgentMemory
from django_model_agent.tools import (
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

    def test_get_system_prompts(self, place, simple_agent_class):
        """Test system prompt retrieval."""
        agent = simple_agent_class(place)

        assert agent.get_system_prompts() == "You are a test agent."

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

    def test_system_prompts_decorator(self, place):
        """Test @ModelAgent.system_prompt decorator."""

        class DecoratedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "Base prompt."

            @ModelAgent.system_prompt
            def dynamic_prompt(self) -> str:
                return f"Working with: {self.instance.name}"

        agent = DecoratedAgent(place)
        prompt = agent.get_system_prompts()

        assert "Base prompt." in prompt
        assert "Working with: Test Restaurant" in prompt

    def test_multiple_system_prompts_decorators(self, place):
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
        prompt = agent.get_system_prompts()

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


# -----------------------------------------------------------------------------
# Template Error Handling Tests
# -----------------------------------------------------------------------------


class TestTemplateErrorHandling:
    """Tests for template rendering error handling."""

    def test_render_missing_template_returns_empty(self, place, simple_agent_class):
        """Test that a missing template returns empty string instead of crashing."""
        agent = simple_agent_class(place)
        result = agent._render_template(
            "nonexistent/template.html", {"instance": place}
        )
        assert result == ""

    def test_instructions_with_missing_template(self, place):
        """Test that get_instructions handles missing template gracefully."""

        class TemplateAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _instructions_template = "does_not_exist.html"

        agent = TemplateAgent(place)
        instructions = agent.get_instructions()
        assert instructions is None or instructions == ""


# -----------------------------------------------------------------------------
# Field Type Mapping Tests
# -----------------------------------------------------------------------------


class TestFieldTypeMapping:
    """Tests for Django field to Python type mapping."""

    def test_decimal_field_maps_to_decimal(self, place, simple_agent_class):
        """Test that DecimalField maps to Decimal, not float."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.DecimalField(max_digits=10, decimal_places=2)
        python_type = agent._get_field_type(field)
        assert python_type is Decimal

    def test_file_field_maps_to_str(self, place, simple_agent_class):
        """Test that FileField maps to str."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.FileField()
        python_type = agent._get_field_type(field)
        assert python_type is str

    def test_image_field_maps_to_str(self, place, simple_agent_class):
        """Test that ImageField maps to str."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.ImageField()
        python_type = agent._get_field_type(field)
        assert python_type is str

    def test_duration_field_maps_to_str(self, place, simple_agent_class):
        """Test that DurationField maps to str."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.DurationField()
        python_type = agent._get_field_type(field)
        assert python_type is str

    def test_binary_field_maps_to_bytes(self, place, simple_agent_class):
        """Test that BinaryField maps to bytes."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.BinaryField()
        python_type = agent._get_field_type(field)
        assert python_type is bytes

    def test_generic_ip_field_maps_to_str(self, place, simple_agent_class):
        """Test that GenericIPAddressField maps to str."""
        agent = simple_agent_class(place)
        from django.db import models

        field = models.GenericIPAddressField()
        python_type = agent._get_field_type(field)
        assert python_type is str


# -----------------------------------------------------------------------------
# AgentMemory Unsaved Instance Tests
# -----------------------------------------------------------------------------


class TestAgentMemoryUnsavedInstance:
    """Tests for AgentMemory with unsaved model instances."""

    def test_get_for_unsaved_instance_raises(self, db):
        """Test that get_for raises ValueError for unsaved instance."""
        unsaved = Place(name="Unsaved", slug="unsaved")
        with pytest.raises(ValueError, match="unsaved model instance"):
            AgentMemory.objects.get_for(unsaved)

    def test_get_or_create_for_unsaved_instance_raises(self, db):
        """Test that get_or_create_for raises ValueError for unsaved instance."""
        unsaved = Place(name="Unsaved", slug="unsaved")
        with pytest.raises(ValueError, match="unsaved model instance"):
            AgentMemory.objects.get_or_create_for(unsaved)


# -----------------------------------------------------------------------------
# URL Validation Tests
# -----------------------------------------------------------------------------


class TestProposeDeliveryUrlValidation:
    """Tests for URL validation in ProposeDeliveryUrlTool."""

    def test_valid_url_accepted(self, place, simple_agent_class):
        """Test that a valid URL is accepted."""
        from django_model_agent.examples import ProposeDeliveryUrlTool

        agent = simple_agent_class(place)
        tool = ProposeDeliveryUrlTool(agent.context)
        result = tool.execute(service="doordash", url="https://doordash.com/store/123")
        assert result.success is True

    def test_invalid_url_rejected(self, place, simple_agent_class):
        """Test that an invalid URL is rejected."""
        from django_model_agent.examples import ProposeDeliveryUrlTool

        agent = simple_agent_class(place)
        tool = ProposeDeliveryUrlTool(agent.context)
        result = tool.execute(service="doordash", url="not-a-url")
        assert result.success is False
        assert "Invalid URL" in result.message

    def test_javascript_url_rejected(self, place, simple_agent_class):
        """Test that javascript: URLs are rejected."""
        from django_model_agent.examples import ProposeDeliveryUrlTool

        agent = simple_agent_class(place)
        tool = ProposeDeliveryUrlTool(agent.context)
        result = tool.execute(service="doordash", url="javascript:alert(1)")
        assert result.success is False
        assert "Invalid URL" in result.message


class TestExamplesModule:
    """Tests for packaged example imports."""

    def test_examples_module_imports_without_test_model_dependency(self):
        """Importing examples should not require tests.models at module import time."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.path.insert(0, 'src'); "
                    "import django_model_agent.examples as examples; "
                    "print(examples.PlaceAgent.model is None)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert result.stdout.strip() == "True"
