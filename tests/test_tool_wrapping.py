"""
Tests for what tools hand back to the model, and how they are called.

Each test runs a real agent with TestModel, which calls every tool, then reads
the tool returns out of the message history -- what the model actually saw.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.test import TestModel

from django_model_agent import ModelAgent, ModelAgentContext, model_tools_to_toolset
from django_model_agent.tools import ModelTool, ReadOnlyTool, ToolResult
from tests.models import Place


def tool_returns(result) -> dict[str, str]:
    """Map tool name to the content the model received from it."""
    return {
        part.tool_name: part.content
        for message in result.all_messages()
        for part in message.parts
        if type(part).__name__ == "ToolReturnPart"
    }


def agent_with(place, tools=(), **attrs):
    attrs = {"model": Place, "fields": ["name"], "tools": list(tools), **attrs}
    agent_cls = type("WrapAgent", (ModelAgent,), attrs)
    return agent_cls(place, ai_model=TestModel())


class ReadPhone(ReadOnlyTool):
    name = "read_phone"
    description = "Read the phone number."

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"phone": self.instance.phone}


class AsyncNote(ModelTool):
    name = "async_note"
    description = "An async tool."

    async def execute(self, note: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, message="async ran")


class NoArgs(ModelTool):
    name = "no_args"
    description = "Takes nothing."

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, message="no args ran")


class CountPlaces(ModelTool):
    name = "count_places"
    description = "Count places with a database query."

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, message=f"{Place.objects.count()} places")


@pytest.mark.django_db
class TestToolOutput:
    def test_read_only_data_reaches_model(self, place):
        result = agent_with(place, [ReadPhone]).run_sync("go")
        content = tool_returns(result)["read_phone"]
        assert "785-555-1234" in content

    def test_result_without_data_is_status_line(self, place):
        result = agent_with(place, [NoArgs]).run_sync("go")
        assert tool_returns(result)["no_args"] == "✓ no args ran"


@pytest.mark.django_db
class TestNoArgumentTools:
    def test_schema_takes_no_arguments(self):
        tool = model_tools_to_toolset([NoArgs]).tools["no_args"]
        schema = tool.function_schema.json_schema
        assert schema.get("properties", {}) == {}
        assert schema.get("additionalProperties") is not True

    def test_runs_when_model_calls_it(self, place):
        result = agent_with(place, [NoArgs]).run_sync("go")
        assert "no args ran" in tool_returns(result)["no_args"]


@pytest.mark.django_db
class TestAsyncTools:
    def test_async_model_tool_is_awaited(self, place):
        result = agent_with(place, [AsyncNote]).run_sync("go")
        assert tool_returns(result)["async_note"] == "✓ async ran"

    def test_async_decorated_method_is_awaited(self, place):
        async def get_name(self) -> str:
            """Get the name."""
            return self.instance.name

        result = agent_with(place, get_name=ModelAgent.tool(get_name)).run_sync("go")
        assert tool_returns(result)["get_name"] == place.name

    @pytest.mark.anyio
    async def test_execute_async_awaits_async_execute(self, place):
        tool = AsyncNote(ModelAgentContext(instance=place, agent=None))
        result = await tool.call_async()
        assert result.message == "async ran"


@pytest.mark.django_db(transaction=True)
class TestSyncToolsStayOffTheEventLoop:
    @pytest.mark.anyio
    async def test_sync_tool_can_query_orm_in_async_run(self, place):
        # Django raises SynchronousOnlyOperation for ORM calls made on the
        # event loop, so this passes only if sync tools run in a thread.
        result = await agent_with(place, [CountPlaces]).run("go")
        assert tool_returns(result)["count_places"] == "✓ 1 places"
