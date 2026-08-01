"""
Tests for pydantic-ai integration in ModelAgent.

These tests verify that build_agent() produces a properly configured
pydantic-ai Agent and that run/run_sync work with TestModel.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgent, ModelAgentContext
from django_model_agent.tools import ModelTool, ReadOnlyTool, ToolResult


class TestBuildAgent:
    """Tests for ModelAgent.build_agent()."""

    def test_build_agent_returns_pydantic_agent(self, place, simple_agent_class):
        agent = simple_agent_class(place)
        pai = agent.build_agent()
        assert isinstance(pai, Agent)

    def test_build_agent_includes_system_prompts(self, place):
        class PromptAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "You are a helpful assistant."

        agent = PromptAgent(place)
        pai = agent.build_agent()
        assert pai.name == "PromptAgent(Place)"

    def test_build_agent_with_instructions(self, place):
        class InstructedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _instructions = "Always be concise."

        agent = InstructedAgent(place)
        pai = agent.build_agent()
        assert isinstance(pai, Agent)

    def test_build_agent_with_decorated_tools(self, place):
        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

        agent = ToolAgent(place)
        pai = agent.build_agent()
        assert isinstance(pai, Agent)

    def test_build_agent_with_model_tools(self, place):
        class NameTool(ReadOnlyTool):
            name = "get_name"
            description = "Get the place name"

            def read(self, **kwargs: Any) -> dict[str, Any]:
                return {"name": self.instance.name}

        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [NameTool]

        agent = ToolAgent(place)
        pai = agent.build_agent()
        assert isinstance(pai, Agent)

    def test_build_agent_ai_model_override(self, place, simple_agent_class):
        agent = simple_agent_class(place, ai_model="test")
        pai = agent.build_agent()
        assert isinstance(pai, Agent)

    def test_build_agent_class_ai_model(self, place):
        class TestAgent(ModelAgent):
            model = Place
            fields = ["name"]
            ai_model = "test"

        agent = TestAgent(place)
        pai = agent.build_agent()
        assert isinstance(pai, Agent)


class TestRunSync:
    """Tests for ModelAgent.run_sync() using pydantic-ai's TestModel."""

    def test_run_sync_basic(self, place):
        class SimpleAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "You help with places."

        agent = SimpleAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = agent.run_sync("What is this place?")
            assert result.output is not None

    def test_run_sync_with_decorated_tool(self, place):
        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

        agent = ToolAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = agent.run_sync("What is the name?")
            assert result.output is not None


class TestRunAsync:
    """Tests for ModelAgent.run() (async) using pydantic-ai's TestModel."""

    @pytest.mark.anyio
    async def test_run_async_basic(self, place):
        class SimpleAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "You help with places."

        agent = SimpleAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = await agent.run("What is this place?")
            assert result.output is not None

    @pytest.mark.anyio
    async def test_run_async_with_instructions(self, place):
        class InstructedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _instructions = "Be brief."

            @ModelAgent.instructions
            def state_info(self) -> str:
                return f"State: {self.instance.state}"

        agent = InstructedAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            result = await agent.run("Tell me about this place.")
            assert result.output is not None


class TestPydanticAiToolConversion:
    """Tests that ModelTool subclasses are properly converted to pydantic-ai tools."""

    def test_model_tool_converted(self, place):
        class InfoTool(ReadOnlyTool):
            name: ClassVar[str] = "get_info"
            description: ClassVar[str] = "Get place info"

            def read(self, **kwargs: Any) -> dict[str, Any]:
                return {"name": self.instance.name}

        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [InfoTool]

        agent = ToolAgent(place)
        pydantic_tools = agent._build_pydantic_ai_tools()
        assert len(pydantic_tools) == 1

    def test_decorated_tool_converted(self, place):
        class DecAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get name."""
                return self.instance.name

            @ModelAgent.tool
            def get_address(self) -> str:
                """Get address."""
                return self.instance.address or ""

        agent = DecAgent(place)
        pydantic_tools = agent._build_pydantic_ai_tools()
        assert len(pydantic_tools) == 2

    def test_mixed_tools_converted(self, place):
        class InfoTool(ReadOnlyTool):
            name: ClassVar[str] = "get_info"
            description: ClassVar[str] = "Get info"

            def read(self, **kwargs: Any) -> dict[str, Any]:
                return {"name": self.instance.name}

        class MixedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [InfoTool]

            @ModelAgent.tool
            def get_address(self) -> str:
                """Get address."""
                return self.instance.address or ""

        agent = MixedAgent(place)
        pydantic_tools = agent._build_pydantic_ai_tools()
        assert len(pydantic_tools) == 2
