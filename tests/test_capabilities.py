"""
Tests for DjangoModelCapability.

The capability is configured with a model class and never holds an instance,
so these tests lean on that: one agent is built and then run against several
different rows.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from tests.models import Place

from django_model_agent import ModelAgent, ModelAgentContext
from django_model_agent.capabilities import (
    DjangoModelCapability,
    agent_fields,
    field_python_type,
    model_tools_to_toolset,
)
from django_model_agent.tools import ReadOnlyTool


class NameTool(ReadOnlyTool):
    name: ClassVar[str] = "get_name"
    description: ClassVar[str] = "Get the place name"

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"name": self.instance.name}


def instructions_sent(result) -> str:
    """The instructions the model actually received across a run."""
    return "\n".join(
        m.instructions for m in result.all_messages() if getattr(m, "instructions", None)
    )


class TestFieldHelpers:
    def test_field_python_type_maps_char_to_str(self):
        field = Place._meta.get_field("name")
        assert field_python_type(field) is str

    def test_nullable_field_is_optional(self):
        field = Place._meta.get_field("phone")
        if field.null:
            assert field_python_type(field) is not str

    def test_agent_fields_respects_fields(self):
        names = [f.name for f in agent_fields(Place, fields=["name", "address"])]
        assert names == ["name", "address"]

    def test_agent_fields_respects_exclude(self):
        names = [f.name for f in agent_fields(Place, exclude=["name"])]
        assert "name" not in names

    def test_agent_fields_needs_no_instance(self):
        assert agent_fields(Place, fields=["name"])


class TestToolset:
    def test_model_tools_to_toolset_returns_toolset(self):
        assert isinstance(model_tools_to_toolset([NameTool]), FunctionToolset)

    def test_empty_toolset_when_no_tools(self):
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        assert cap.get_toolset() is None

    def test_toolset_present_when_tools_given(self):
        cap = DjangoModelCapability(model_class=Place, fields=["name"], tools=[NameTool])
        assert cap.get_toolset() is not None


class TestSchemaAndValues:
    def test_schema_description_lists_fields(self):
        cap = DjangoModelCapability(model_class=Place, fields=["name", "address"])
        desc = cap.schema_description()
        assert "name" in desc
        assert "address" in desc

    def test_current_values_reads_instance(self, place):
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        assert cap.current_values(place)["name"] == place.name

    def test_current_values_differ_per_instance(self):
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        a = Place(pk=1, name="Alpha")
        b = Place(pk=2, name="Beta")
        assert cap.current_values(a)["name"] == "Alpha"
        assert cap.current_values(b)["name"] == "Beta"


class TestStandaloneAgent:
    """The capability works with a plain pydantic_ai.Agent, no ModelAgent."""

    def build(self, **kwargs) -> Agent:
        cap = DjangoModelCapability(model_class=Place, fields=["name", "address"], **kwargs)
        return Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])

    def test_runs_without_model_agent(self, place):
        agent = self.build()
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        assert result.output is not None

    def test_instructions_include_current_values(self, place):
        agent = self.build()
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        assert place.name in instructions_sent(result)

    def test_static_instructions_included(self, place):
        agent = self.build(instructions="You help with places.")
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        assert "You help with places." in instructions_sent(result)

    def test_model_tool_is_callable(self, place):
        agent = self.build(tools=[NameTool])
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        returns = [
            p.content
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolReturnPart"
        ]
        assert returns


class TestInstanceIndependence:
    """One agent must serve many instances with correct per-instance values."""

    def test_one_agent_two_instances(self):
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        agent = Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])

        a = Place(pk=1, name="Alpha Cafe")
        b = Place(pk=2, name="Beta Diner")

        first = instructions_sent(
            agent.run_sync("hi", deps=ModelAgentContext(instance=a, agent=None))
        )
        second = instructions_sent(
            agent.run_sync("hi", deps=ModelAgentContext(instance=b, agent=None))
        )

        assert "Alpha Cafe" in first and "Beta Diner" not in first
        assert "Beta Diner" in second and "Alpha Cafe" not in second

    def test_values_refresh_between_runs(self, place):
        """A field changed between runs must be reflected on the next run."""
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        agent = Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])
        deps = ModelAgentContext(instance=place, agent=None)

        place.name = "Before Rename"
        first = instructions_sent(agent.run_sync("hi", deps=deps))

        place.name = "After Rename"
        second = instructions_sent(agent.run_sync("hi", deps=deps))

        assert "Before Rename" in first
        assert "After Rename" in second
        assert "Before Rename" not in second

    def test_instructions_stay_out_of_history(self, place):
        """
        Instructions must not be persisted as message parts.

        This is why current values go through instructions rather than a system
        prompt: a system prompt would leave turn-one values sitting in history.
        """
        cap = DjangoModelCapability(model_class=Place, fields=["name"])
        agent = Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])

        place.name = "Historic Name"
        result = agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))

        parts_text = "".join(
            str(p) for m in result.all_messages() for p in getattr(m, "parts", [])
        )
        assert "Historic Name" not in parts_text


class TestDecoratedToolFuncs:
    def test_decorated_tool_uses_agent_from_deps(self, place):
        class DecAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

        model_agent = DecAgent(place)
        cap = DjangoModelCapability(
            model_class=Place,
            fields=["name"],
            tool_funcs=model_agent._tool_funcs,
        )
        agent = Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])

        result = agent.run_sync(
            "hi", deps=ModelAgentContext(instance=place, agent=model_agent)
        )
        returns = [
            p.content
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolReturnPart"
        ]
        assert place.name in "".join(returns)

    def test_decorated_tool_errors_without_agent(self, place):
        class DecAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

        cap = DjangoModelCapability(
            model_class=Place,
            fields=["name"],
            tool_funcs=DecAgent(place)._tool_funcs,
        )
        agent = Agent(TestModel(), deps_type=ModelAgentContext, capabilities=[cap])

        with pytest.raises(ValueError, match="needs ModelAgentContext.agent to be set"):
            agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
