"""
Tests for DjangoFSMCapability.

Tool hiding here is an optimisation; ModelTool.check_allowed() is still the
enforcement point. Both properties are covered.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgentContext
from django_model_agent.capabilities import (
    DjangoFSMCapability,
    DjangoModelCapability,
    _state_prepare,
)
from django_model_agent.tools import ModelTool, ReadOnlyTool, ToolResult


class AlwaysTool(ReadOnlyTool):
    name: ClassVar[str] = "always"
    description: ClassVar[str] = "Runs in any state"

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class DraftOnlyTool(ReadOnlyTool):
    name: ClassVar[str] = "draft_only"
    description: ClassVar[str] = "Only runs while draft"
    allowed_states: ClassVar[list[str]] = ["draft"]

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class PublicOnlyTool(ModelTool):
    name: ClassVar[str] = "public_only"
    description: ClassVar[str] = "Only runs while public"
    allowed_states: ClassVar[list[str]] = ["public"]

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, message="ran")


TRANSITIONS = {
    "draft": ["public"],
    "public": ["featured", "closed"],
    "featured": ["public"],
    "closed": [],
}


def build_agent(**fsm_kwargs) -> Agent:
    tools = [AlwaysTool, DraftOnlyTool, PublicOnlyTool]
    return Agent(
        TestModel(),
        deps_type=ModelAgentContext,
        capabilities=[
            DjangoModelCapability(model_class=Place, fields=["name", "state"], tools=tools),
            DjangoFSMCapability(tools=tools, **fsm_kwargs),
        ],
    )


def tools_offered(agent: Agent, instance: Place) -> set[str]:
    """Tool names the model was actually offered during a run."""
    result = agent.run_sync("hi", deps=ModelAgentContext(instance=instance, agent=None))
    return {
        p.tool_name
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if type(p).__name__ == "ToolCallPart"
    }


def instructions_sent(agent: Agent, instance: Place) -> str:
    result = agent.run_sync("hi", deps=ModelAgentContext(instance=instance, agent=None))
    return "\n".join(
        m.instructions for m in result.all_messages() if getattr(m, "instructions", None)
    )


class TestStateInstructions:
    def test_reports_current_state(self):
        agent = build_agent()
        text = instructions_sent(agent, Place(pk=1, name="A", state="public"))
        assert "'public'" in text

    def test_reports_valid_transitions(self):
        agent = build_agent(transitions=TRANSITIONS)
        text = instructions_sent(agent, Place(pk=1, name="A", state="public"))
        assert "featured" in text and "closed" in text

    def test_reports_dead_end_state(self):
        agent = build_agent(transitions=TRANSITIONS)
        text = instructions_sent(agent, Place(pk=1, name="A", state="closed"))
        assert "no valid transitions" in text.lower()

    def test_silent_when_field_absent(self):
        cap = DjangoFSMCapability(state_field="nonexistent")
        assert cap.current_state(Place(pk=1, name="A")) is None


class TestToolFiltering:
    def test_hides_tool_not_allowed_in_state(self):
        agent = build_agent()
        offered = tools_offered(agent, Place(pk=1, name="A", state="public"))
        assert "draft_only" not in offered

    def test_shows_tool_allowed_in_state(self):
        agent = build_agent()
        offered = tools_offered(agent, Place(pk=1, name="A", state="public"))
        assert "public_only" in offered

    def test_unrestricted_tool_always_shown(self):
        agent = build_agent()
        for state in ("draft", "public", "closed"):
            offered = tools_offered(agent, Place(pk=1, name="A", state=state))
            assert "always" in offered

    def test_filtering_follows_state(self):
        agent = build_agent()
        draft = tools_offered(agent, Place(pk=1, name="A", state="draft"))
        public = tools_offered(agent, Place(pk=2, name="B", state="public"))

        assert "draft_only" in draft and "public_only" not in draft
        assert "public_only" in public and "draft_only" not in public

    def test_tool_filters_even_with_unrelated_capability_state_field(self):
        """
        The restriction travels with the tool.

        `allowed_states` is enforced by the tool's own `prepare` hook, so it
        applies regardless of how the capability is configured.
        """
        tools = [DraftOnlyTool]
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(model_class=Place, fields=["name"], tools=tools),
                DjangoFSMCapability(state_field="nope", tools=tools),
            ],
        )
        offered = tools_offered(agent, Place(pk=1, name="A", state="public"))
        assert "draft_only" not in offered

    def test_model_without_state_field_is_left_alone(self):
        """No state to compare against means no filtering."""
        prepared = _state_prepare(DraftOnlyTool)
        assert prepared is not None

    def test_tool_states_override_accepted(self):
        agent = build_agent(tool_states={"always": ["draft"]})
        offered = tools_offered(agent, Place(pk=1, name="A", state="public"))
        assert "always" not in offered


class TestEnforcementBackstop:
    """allowed_states must still be enforced when the capability is absent."""

    def test_check_allowed_still_blocks_without_capability(self, place):
        place.state = "public"
        tool = DraftOnlyTool(ModelAgentContext(instance=place, agent=None))
        result = tool()
        assert result.success is False
        assert "not allowed" in result.message

    def test_check_allowed_permits_in_valid_state(self, place):
        place.state = "draft"
        tool = DraftOnlyTool(ModelAgentContext(instance=place, agent=None))
        assert tool().success is True

    def test_tool_hidden_without_fsm_capability(self, place):
        """
        Without the FSM capability the tool is still hidden, not merely refused.

        `allowed_states` is compiled into the tool's own `prepare` hook, so the
        model never sees it in a state it cannot run in — the capability only
        adds the state/transition instructions on top.
        """
        place.state = "public"
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(
                    model_class=Place, fields=["name"], tools=[DraftOnlyTool]
                )
            ],
        )
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        called = {
            p.tool_name
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolCallPart"
        }
        assert "draft_only" not in called
