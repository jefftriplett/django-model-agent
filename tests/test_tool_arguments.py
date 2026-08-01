"""
Tests that tool arguments reach the model.

Both failure modes here were silent or build-time: a ModelTool advertised no
parameters at all, and a decorated tool with arguments could not be built.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgent
from django_model_agent.capabilities import model_tools_to_toolset
from django_model_agent.tools import ModelTool, ReadOnlyTool, ToolResult, UpdateTool


def properties(tool) -> dict[str, Any]:
    return tool.function_schema.json_schema.get("properties", {})


def schema_for(tool_cls) -> dict[str, Any]:
    toolset = model_tools_to_toolset([tool_cls])
    return properties(next(iter(toolset.tools.values())))


class TestModelToolArguments:
    def test_execute_arguments_advertised(self):
        class ArgTool(ModelTool):
            name: ClassVar[str] = "with_args"
            description: ClassVar[str] = "Takes arguments"

            def execute(self, *, max_words: int = 50, **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, message="ok")

        assert "max_words" in schema_for(ArgTool)

    def test_read_arguments_advertised(self):
        """ReadOnlyTool implements read(), not execute()."""

        class ReadArgTool(ReadOnlyTool):
            name: ClassVar[str] = "read_args"
            description: ClassVar[str] = "Reads with arguments"

            def read(self, *, verbose: bool = False, **kwargs: Any) -> dict[str, Any]:
                return {"verbose": verbose}

        assert "verbose" in schema_for(ReadArgTool)

    def test_update_arguments_advertised(self):
        """UpdateTool implements update(), not execute()."""

        class UpdateArgTool(UpdateTool):
            name: ClassVar[str] = "update_args"
            description: ClassVar[str] = "Updates with arguments"

            def update(self, *, new_name: str = "x", **kwargs: Any) -> None:
                self.instance.name = new_name

        assert "new_name" in schema_for(UpdateArgTool)

    def test_argument_types_preserved(self):
        class TypedTool(ModelTool):
            name: ClassVar[str] = "typed"
            description: ClassVar[str] = "Typed arguments"

            def execute(self, *, count: int = 1, label: str = "a", **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, message="ok")

        props = schema_for(TypedTool)
        assert props["count"]["type"] == "integer"
        assert props["label"]["type"] == "string"

    def test_tool_without_arguments_still_works(self):
        class BareTool(ReadOnlyTool):
            name: ClassVar[str] = "bare"
            description: ClassVar[str] = "No arguments"

            def read(self, **kwargs: Any) -> dict[str, Any]:
                return {"ok": True}

        assert schema_for(BareTool) == {}


class TestDecoratedToolArguments:
    def test_decorated_tool_with_arguments_builds(self, place):
        """This used to raise a UserError about the RunContext annotation."""

        class ArgAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def shorten(self, max_words: int = 20) -> str:
                """Shorten the description."""
                return f"cut to {max_words}"

        agent = ArgAgent(place)
        toolset = model_tools_to_toolset([], agent._tool_funcs)
        assert "max_words" in properties(next(iter(toolset.tools.values())))

    def test_decorated_tool_with_arguments_runs(self, place):
        class ArgAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def shorten(self, max_words: int = 20) -> str:
                """Shorten the description."""
                return f"cut to {max_words}"

        agent = ArgAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()):
            result = agent.run_sync("go")

        returns = "".join(
            str(p.content)
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolReturnPart"
        )
        assert "cut to" in returns


class TestArgumentsActuallyArrive:
    @pytest.mark.django_db(transaction=True)
    def test_model_receives_and_passes_arguments(self, place):
        seen: list[int] = []

        class RecordingTool(ModelTool):
            name: ClassVar[str] = "recording"
            description: ClassVar[str] = "Records what it was passed"

            def execute(self, *, amount: int = 7, **kwargs: Any) -> ToolResult:
                seen.append(amount)
                return ToolResult(success=True, message="ok")

        class RecordAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [RecordingTool]

        agent = RecordAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()):
            agent.run_sync("go")

        assert seen, "tool never ran"
        assert isinstance(seen[0], int)


class TestDiffAwareDeprecation:
    """DiffAwareUpdateTool is deprecated in favour of requires_confirmation."""

    def test_subclassing_warns(self):
        import warnings

        from django_model_agent.tools import DiffAwareUpdateTool

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class LegacyProposeTool(DiffAwareUpdateTool):
                name: ClassVar[str] = "legacy"
                description: ClassVar[str] = "Legacy proposal tool"

                def execute(self, **kwargs: Any) -> ToolResult:
                    return ToolResult(success=True, message="ok")

        messages = [str(w.message) for w in caught]
        assert any("requires_confirmation" in m for m in messages)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_still_functional(self, place):
        """Deprecated, not removed — existing code must keep working."""
        import warnings

        from django_model_agent import ModelAgentContext
        from django_model_agent.tools import DiffAwareUpdateTool

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            class LegacyTool(DiffAwareUpdateTool):
                name: ClassVar[str] = "legacy2"
                description: ClassVar[str] = "Legacy"

                def execute(self, **kwargs: Any) -> ToolResult:
                    self.propose_change("name", "Proposed")
                    return ToolResult(success=True, message="proposed")

        tool = LegacyTool(ModelAgentContext(instance=place, agent=None))
        tool.execute()
        assert len(tool.proposed_changes) == 1
