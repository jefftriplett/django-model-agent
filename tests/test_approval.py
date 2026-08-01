"""
Tests for requires_confirmation mapping onto pydantic-ai's approval flow.

A tool marked requires_confirmation suspends the run and returns a
DeferredToolRequests; approving and resuming then runs it for real.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgent, ModelAgentContext
from django_model_agent.capabilities import DjangoModelCapability, tools_need_approval
from django_model_agent.tools import ToolResult, UpdateTool

pytestmark = pytest.mark.django_db(transaction=True)


class GatedTool(UpdateTool):
    name: ClassVar[str] = "gated_rename"
    description: ClassVar[str] = "Rename, with approval"
    requires_confirmation: ClassVar[bool] = True

    def update(self, **kwargs: Any) -> None:
        self.instance.name = "Approved Rename"


class OpenTool(UpdateTool):
    name: ClassVar[str] = "open_rename"
    description: ClassVar[str] = "Rename, no approval"

    def update(self, **kwargs: Any) -> None:
        self.instance.name = "Direct Rename"


class TestNeedsApproval:
    def test_detects_gated_tool(self):
        assert tools_need_approval([GatedTool]) is True

    def test_ignores_ungated_tools(self):
        assert tools_need_approval([OpenTool]) is False

    def test_update_tool_no_longer_gated_by_default(self):
        """
        The default used to be True while nothing read it. Making the attribute
        real without changing the default would have gated every update tool.
        """
        assert OpenTool.requires_confirmation is False


class TestApprovalFlow:
    def build(self, tools):
        return Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(model_class=Place, fields=["name"], tools=tools)
            ],
            output_type=[str, DeferredToolRequests],
        )

    def test_gated_tool_suspends_the_run(self, place):
        agent = self.build([GatedTool])
        result = agent.run_sync("go", deps=ModelAgentContext(instance=place, agent=None))
        assert isinstance(result.output, DeferredToolRequests)
        assert [c.tool_name for c in result.output.approvals] == ["gated_rename"]

    def test_gated_tool_does_not_run_before_approval(self, place):
        original = place.name
        agent = self.build([GatedTool])
        agent.run_sync("go", deps=ModelAgentContext(instance=place, agent=None))
        place.refresh_from_db()
        assert place.name == original

    def test_approving_resumes_and_runs_the_tool(self, place):
        agent = self.build([GatedTool])
        deps = ModelAgentContext(instance=place, agent=None)
        first = agent.run_sync("go", deps=deps)

        results = DeferredToolResults()
        for call in first.output.approvals:
            results.approvals[call.tool_call_id] = True

        agent.run_sync(
            message_history=first.all_messages(),
            deferred_tool_results=results,
            deps=deps,
        )
        place.refresh_from_db()
        assert place.name == "Approved Rename"

    def test_ungated_tool_runs_immediately(self, place):
        agent = self.build([OpenTool])
        agent.run_sync("go", deps=ModelAgentContext(instance=place, agent=None))
        place.refresh_from_db()
        assert place.name == "Direct Rename"


class TestModelAgentIntegration:
    def test_deferred_output_added_automatically(self, place):
        """
        Without DeferredToolRequests among the outputs pydantic-ai raises a
        UserError, so build_agent() must add it for gated tools.
        """

        class GatedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [GatedTool]

        agent = GatedAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = agent.run_sync("go")

        assert isinstance(result.output, DeferredToolRequests)

    def test_no_deferred_output_without_gated_tools(self, place):
        class PlainAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [OpenTool]

        agent = PlainAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = agent.run_sync("go")

        assert isinstance(result.output, str)
