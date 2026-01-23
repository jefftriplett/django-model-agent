"""
Tests for the ModelAgent experimental module.

Run with: just test experimental/model_agent/tests.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from places.models import Place

# Import base and tools directly, avoid importing memory model at module level
from .base import ModelAgent, ModelAgentContext
from .tools import (
    DiffAwareUpdateTool,
    ModelTool,
    ProposedChange,
    ReadOnlyTool,
    ToolResult,
    UpdateTool,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def place(db):
    """Create a test Place instance."""
    return Place.objects.create(
        name="Test Restaurant",
        slug="test-restaurant",
        address="123 Main St",
        locality="Lawrence",
        region="KS",
        phone="785-555-1234",
        website="https://example.com",
        description="A test restaurant",
        state=Place.STATE_PUBLIC,
        delivery=True,
        takeout=True,
        dinein=True,
        doordash_url="https://doordash.com/test",
        grubhub_url="https://grubhub.com/test",
    )


@pytest.fixture
def draft_place(db):
    """Create a draft Place instance."""
    return Place.objects.create(
        name="Draft Restaurant",
        slug="draft-restaurant",
        state=Place.STATE_DRAFT,
    )


@pytest.fixture
def simple_agent_class():
    """Create a simple ModelAgent subclass for testing."""

    class SimpleAgent(ModelAgent):
        model = Place
        fields = ["name", "address", "phone"]
        base_prompt = "You are a test agent."

    return SimpleAgent


@pytest.fixture
def agent_with_field_sets():
    """Create a ModelAgent with field sets."""

    class FieldSetAgent(ModelAgent):
        model = Place
        field_sets = {
            "public": ["name", "address"],
            "staff": ["name", "address", "phone", "notes"],
        }
        base_prompt = "Agent with field sets."

    return FieldSetAgent


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

        # Modify a non-FSM field in database directly
        Place.objects.filter(pk=place.pk).update(name="Updated Name")

        # Instance still has old value
        assert context.instance.name == "Test Restaurant"

        # Refresh loads new value - use fields= to avoid FSM state field issue
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

    def test_combined_class_and_decorator_tools(self, place):
        """Test combining class-level tools with decorated tools."""
        from .tools import ReadOnlyTool

        class ClassTool(ReadOnlyTool):
            name = "class_tool"
            description = "A class-based tool"

            def read(self, **kwargs):
                return {"source": "class"}

        class CombinedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [ClassTool]

            @ModelAgent.tool
            def decorated_tool(self) -> str:
                """A decorated tool."""
                return "from decorator"

        agent = CombinedAgent(place)
        tools = agent.get_tools()

        # Should have both the class tool and the decorated tool
        assert len(tools) == 2

    def test_decorator_with_no_base_prompt(self, place):
        """Test that decorators work without class-level base_prompt."""

        class NoBasePromptAgent(ModelAgent):
            model = Place
            fields = ["name"]
            # No base_prompt class attribute

            @ModelAgent.system_prompt
            def only_prompt(self) -> str:
                return "The only prompt."

        agent = NoBasePromptAgent(place)
        prompt = agent.get_system_prompt()

        assert prompt == "The only prompt."

    def test_empty_instructions(self, place):
        """Test get_instructions returns None when no instructions defined."""

        class NoInstructionsAgent(ModelAgent):
            model = Place
            fields = ["name"]

        agent = NoInstructionsAgent(place)
        instructions = agent.get_instructions()

        assert instructions is None


# -----------------------------------------------------------------------------
# Tool Tests
# -----------------------------------------------------------------------------


class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result creation."""
        result = ToolResult(success=True, message="Done")

        assert result.success is True
        assert result.message == "Done"
        assert result.data is None
        assert result.changes == {}
        assert "✓" in str(result)

    def test_failure_result(self):
        """Test failure result creation."""
        result = ToolResult(success=False, message="Failed")

        assert result.success is False
        assert "✗" in str(result)

    def test_result_with_data(self):
        """Test result with data."""
        result = ToolResult(
            success=True,
            message="Data retrieved",
            data={"key": "value"},
            changes={"field": {"before": "a", "after": "b"}},
        )

        assert result.data == {"key": "value"}
        assert result.changes["field"]["before"] == "a"


class TestModelTool:
    """Tests for ModelTool base class."""

    def test_tool_initialization(self, place, simple_agent_class):
        """Test tool initialization with context."""
        agent = simple_agent_class(place)

        class TestTool(ModelTool):
            name = "test"
            description = "Test tool"

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = TestTool(agent.context)

        assert tool.instance == place
        assert tool.agent == agent

    def test_check_allowed_no_restrictions(self, place, simple_agent_class):
        """Test check_allowed with no state restrictions."""
        agent = simple_agent_class(place)

        class UnrestrictedTool(ModelTool):
            name = "unrestricted"
            description = "No restrictions"
            allowed_states = None

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = UnrestrictedTool(agent.context)
        allowed, reason = tool.check_allowed()

        assert allowed is True
        assert reason is None

    def test_check_allowed_state_restriction(self, place, simple_agent_class):
        """Test check_allowed with state restrictions."""
        agent = simple_agent_class(place)

        class DraftOnlyTool(ModelTool):
            name = "draft_only"
            description = "Only for draft"
            allowed_states = ["draft"]

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = DraftOnlyTool(agent.context)
        allowed, reason = tool.check_allowed()

        # Place is in 'public' state, not 'draft'
        assert allowed is False
        assert "draft_only" in reason
        assert "public" in reason

    def test_check_allowed_in_allowed_state(self, draft_place, simple_agent_class):
        """Test check_allowed when in an allowed state."""
        agent = simple_agent_class(draft_place)

        class DraftOnlyTool(ModelTool):
            name = "draft_only"
            description = "Only for draft"
            allowed_states = ["draft"]

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = DraftOnlyTool(agent.context)
        allowed, reason = tool.check_allowed()

        assert allowed is True

    def test_tool_call_checks_state(self, place, simple_agent_class):
        """Test that __call__ checks state before executing."""
        agent = simple_agent_class(place)

        class DraftOnlyTool(ModelTool):
            name = "draft_only"
            description = "Only for draft"
            allowed_states = ["draft"]

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Should not reach")

        tool = DraftOnlyTool(agent.context)
        result = tool()

        assert result.success is False
        assert "not allowed" in result.message


class TestReadOnlyTool:
    """Tests for ReadOnlyTool."""

    def test_read_only_tool(self, place, simple_agent_class):
        """Test read-only tool execution."""
        agent = simple_agent_class(place)

        class GetNameTool(ReadOnlyTool):
            name = "get_name"
            description = "Get the name"

            def read(self, **kwargs):
                return {"name": self.instance.name}

        tool = GetNameTool(agent.context)
        result = tool.execute()

        assert result.success is True
        assert result.data["name"] == "Test Restaurant"


class TestUpdateTool:
    """Tests for UpdateTool."""

    def test_update_tool(self, place, simple_agent_class):
        """Test update tool execution."""
        agent = simple_agent_class(place)

        class UpdateNameTool(UpdateTool):
            name = "update_name"
            description = "Update the name"

            def update(self, *, new_name: str, **kwargs):
                self.instance.name = new_name

        tool = UpdateNameTool(agent.context)
        result = tool.execute(new_name="New Name")

        assert result.success is True
        assert "name" in result.changes
        assert result.changes["name"]["before"] == "Test Restaurant"
        assert result.changes["name"]["after"] == "New Name"

        # Verify saved to database
        place.refresh_from_db(fields=["name"])
        assert place.name == "New Name"

    def test_update_tool_preview(self, place, simple_agent_class):
        """Test update tool in preview mode (no save)."""
        agent = simple_agent_class(place)

        class UpdateNameTool(UpdateTool):
            name = "update_name"
            description = "Update the name"

            def update(self, *, new_name: str, **kwargs):
                self.instance.name = new_name

        tool = UpdateNameTool(agent.context)
        result = tool.execute(new_name="Preview Name", preview=True)

        assert result.success is True

        # Not saved to database
        place.refresh_from_db(fields=["name"])
        assert place.name == "Test Restaurant"


class TestProposedChange:
    """Tests for ProposedChange."""

    def test_proposed_change_creation(self, place):
        """Test creating a proposed change."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Test Restaurant",
            new_value="New Restaurant",
            reason="Updating name",
        )

        assert change.field_name == "name"
        assert change.old_value == "Test Restaurant"
        assert change.new_value == "New Restaurant"
        assert change.approved is None

    def test_approve_change(self, place):
        """Test approving a change."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Test Restaurant",
            new_value="New Restaurant",
        )

        change.approve()
        assert change.approved is True

    def test_reject_change(self, place):
        """Test rejecting a change."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Test Restaurant",
            new_value="New Restaurant",
        )

        change.reject()
        assert change.approved is False

    def test_apply_approved_change(self, place):
        """Test applying an approved change."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Test Restaurant",
            new_value="New Restaurant",
        )

        change.approve()
        change.apply()

        assert place.name == "New Restaurant"

    def test_apply_unapproved_change_raises(self, place):
        """Test that applying unapproved change raises error."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Test Restaurant",
            new_value="New Restaurant",
        )

        with pytest.raises(ValueError, match="unapproved"):
            change.apply()

    def test_repr(self, place):
        """Test change string representation."""
        change = ProposedChange(
            instance=place,
            field_name="name",
            old_value="Old",
            new_value="New",
        )

        assert "?" in repr(change)  # Pending
        change.approve()
        assert "✓" in repr(change)


class TestDiffAwareUpdateTool:
    """Tests for DiffAwareUpdateTool."""

    def test_propose_change(self, place, simple_agent_class):
        """Test proposing a change."""
        agent = simple_agent_class(place)

        class ProposeTool(DiffAwareUpdateTool):
            name = "propose"
            description = "Propose changes"

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = ProposeTool(agent.context)
        change = tool.propose_change("name", "New Name", "Testing")

        assert len(tool.proposed_changes) == 1
        assert change.field_name == "name"
        assert change.new_value == "New Name"
        assert change.reason == "Testing"

    def test_get_pending_changes(self, place, simple_agent_class):
        """Test getting pending changes."""
        agent = simple_agent_class(place)

        class ProposeTool(DiffAwareUpdateTool):
            name = "propose"
            description = "Propose changes"

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = ProposeTool(agent.context)
        tool.propose_change("name", "New Name")
        tool.propose_change("address", "New Address")

        # Approve one
        tool.proposed_changes[0].approve()

        pending = tool.get_pending_changes()
        assert len(pending) == 1
        assert pending[0].field_name == "address"

    def test_apply_approved_changes(self, place, simple_agent_class):
        """Test applying approved changes."""
        agent = simple_agent_class(place)

        class ProposeTool(DiffAwareUpdateTool):
            name = "propose"
            description = "Propose changes"

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = ProposeTool(agent.context)
        tool.propose_change("name", "New Name")
        tool.propose_change("address", "New Address")

        # Approve only name change
        tool.proposed_changes[0].approve()

        applied = tool.apply_approved_changes()

        assert applied == 1
        place.refresh_from_db(fields=["name", "address"])
        assert place.name == "New Name"
        assert place.address == "123 Main St"  # Unchanged

    def test_get_diff_summary(self, place, simple_agent_class):
        """Test diff summary generation."""
        agent = simple_agent_class(place)

        class ProposeTool(DiffAwareUpdateTool):
            name = "propose"
            description = "Propose changes"

            def execute(self, **kwargs):
                return ToolResult(success=True, message="Done")

        tool = ProposeTool(agent.context)
        tool.propose_change("name", "New Name", "Reason 1")

        summary = tool.get_diff_summary()

        assert "name" in summary
        assert "New Name" in summary
        assert "Reason 1" in summary


# -----------------------------------------------------------------------------
# Integration Tests with Examples
# -----------------------------------------------------------------------------


class TestPlaceAgentExamples:
    """Integration tests using the example PlaceAgent."""

    def test_place_agent_initialization(self, place):
        """Test PlaceAgent can be initialized."""
        from .examples import PlaceAgent

        agent = PlaceAgent(place)

        assert agent.instance == place
        assert agent.model == Place

    def test_place_agent_field_sets(self, place):
        """Test PlaceAgent field sets."""
        from .examples import PlaceAgent

        public_agent = PlaceAgent(place, field_set="public")
        staff_agent = PlaceAgent(place, field_set="staff")

        public_schema = public_agent.schema
        staff_schema = staff_agent.schema

        # Public has fewer fields
        assert "notes" not in public_schema.model_fields
        assert "notes" in staff_schema.model_fields

    def test_place_agent_system_prompt(self, place):
        """Test PlaceAgent enhanced system prompt."""
        from .examples import PlaceAgent

        agent = PlaceAgent(place)
        prompt = agent.get_system_prompt()

        assert "Test Restaurant" in prompt
        assert "public" in prompt  # Current state

    def test_get_place_info_tool(self, place):
        """Test GetPlaceInfoTool."""
        from .examples import GetPlaceInfoTool, PlaceAgent

        agent = PlaceAgent(place)
        tool = GetPlaceInfoTool(agent.context)
        result = tool.execute()

        assert result.success is True
        assert result.data["name"] == "Test Restaurant"
        assert result.data["phone"] == "785-555-1234"
        assert result.data["has_delivery"] is True

    def test_get_delivery_options_tool(self, place):
        """Test GetDeliveryOptionsTool."""
        from .examples import GetDeliveryOptionsTool, PlaceAgent

        agent = PlaceAgent(place)
        tool = GetDeliveryOptionsTool(agent.context)
        result = tool.execute()

        assert result.success is True
        assert result.data["doordash"] == "https://doordash.com/test"
        assert result.data["grubhub"] == "https://grubhub.com/test"

    def test_update_description_tool(self, place):
        """Test UpdateDescriptionTool."""
        from .examples import PlaceAgent, UpdateDescriptionTool

        agent = PlaceAgent(place)
        tool = UpdateDescriptionTool(agent.context)
        result = tool.execute(description="New description")

        assert result.success is True
        place.refresh_from_db(fields=["description"])
        assert place.description == "New description"

    def test_update_description_tool_state_check(self, db):
        """Test UpdateDescriptionTool respects state restrictions."""
        from .examples import PlaceAgent, UpdateDescriptionTool

        # Create a closed place
        closed_place = Place.objects.create(
            name="Closed Place",
            slug="closed-place",
            state=Place.STATE_CLOSED,
        )

        agent = PlaceAgent(closed_place)
        tool = UpdateDescriptionTool(agent.context)
        result = tool()

        # Should fail because place is closed
        assert result.success is False
        assert "not allowed" in result.message

    def test_propose_delivery_url_tool(self, place):
        """Test ProposeDeliveryUrlTool."""
        from .examples import PlaceAgent, ProposeDeliveryUrlTool

        agent = PlaceAgent(place)
        tool = ProposeDeliveryUrlTool(agent.context)
        result = tool.execute(
            service="ubereats",
            url="https://ubereats.com/new",
            reason="Adding UberEats",
        )

        assert result.success is True
        assert len(tool.proposed_changes) == 1
        assert tool.proposed_changes[0].new_value == "https://ubereats.com/new"

    def test_propose_delivery_url_invalid_service(self, place):
        """Test ProposeDeliveryUrlTool with invalid service."""
        from .examples import PlaceAgent, ProposeDeliveryUrlTool

        agent = PlaceAgent(place)
        tool = ProposeDeliveryUrlTool(agent.context)
        result = tool.execute(
            service="invalid_service",
            url="https://example.com",
        )

        assert result.success is False
        assert "Unknown service" in result.message

    def test_change_state_tool(self, draft_place):
        """Test ChangeStateTool."""
        from .examples import ChangeStateTool, PlaceAgent

        agent = PlaceAgent(draft_place)
        tool = ChangeStateTool(agent.context)
        result = tool.execute(action="publish")

        assert result.success is True
        assert result.changes["state"]["before"] == "draft"
        assert result.changes["state"]["after"] == "public"

        # Verify the state was changed (can't use refresh_from_db on FSM field)
        # Re-fetch from database to verify
        updated_place = Place.objects.get(pk=draft_place.pk)
        assert updated_place.state == Place.STATE_PUBLIC

    def test_change_state_tool_invalid_transition(self, place):
        """Test ChangeStateTool with invalid transition."""
        from .examples import ChangeStateTool, PlaceAgent

        # Place is already public, can't publish again
        agent = PlaceAgent(place)
        tool = ChangeStateTool(agent.context)
        result = tool.execute(action="publish")

        assert result.success is False
        assert "Cannot perform" in result.message

    def test_flag_for_review_tool(self, place):
        """Test FlagForReviewTool."""
        from .examples import FlagForReviewTool, PlaceAgent

        agent = PlaceAgent(place)
        tool = FlagForReviewTool(agent.context)
        result = tool.execute(reason="Hours seem incorrect")

        assert result.success is True

        place.refresh_from_db(fields=["notes"])
        assert "[FLAGGED FOR REVIEW]" in place.notes
        assert "Hours seem incorrect" in place.notes


# -----------------------------------------------------------------------------
# Memory Tests (using mocks to avoid needing migrations)
# -----------------------------------------------------------------------------


class TestAgentMemoryMixin:
    """Tests for AgentMemoryMixin using mocks."""

    def test_mixin_get_memory_context_empty(self, place):
        """Test getting formatted memory context when empty."""
        from .memory import AgentMemoryMixin

        class MemoryAgent(AgentMemoryMixin, ModelAgent):
            model = Place
            fields = ["name"]
            base_prompt = "Test"

        agent = MemoryAgent(place)

        # Mock the memory property
        mock_memory = MagicMock()
        mock_memory.data = {}
        agent._memory = mock_memory

        context = agent.get_memory_context()
        assert context == ""

    def test_mixin_get_memory_context_with_data(self, place):
        """Test getting formatted memory context with data."""
        from .memory import AgentMemoryMixin

        class MemoryAgent(AgentMemoryMixin, ModelAgent):
            model = Place
            fields = ["name"]
            base_prompt = "Test"

        agent = MemoryAgent(place)

        # Mock the memory property
        mock_memory = MagicMock()
        mock_memory.data = {
            "last_topic": "hours",
            "visits": 3,
            "history": [{"role": "user", "content": "test"}],
        }
        agent._memory = mock_memory

        context = agent.get_memory_context()

        assert "last_topic" in context
        assert "hours" in context
        assert "visits" in context
        assert "history" in context.lower()
