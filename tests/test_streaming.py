"""
Tests for ModelAgent streaming.

Both streaming entry points are context managers rather than bare iterables,
so the run is torn down properly when a consumer stops early.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimitExceeded, UsageLimits

from tests.models import Place

from django_model_agent import ModelAgent


class StreamAgent(ModelAgent):
    model = Place
    fields = ["name"]


def prepared(place, **kwargs):
    agent = StreamAgent(place, **kwargs)
    agent._pydantic_agent = agent.build_agent()
    return agent


class TestRunStream:
    @pytest.mark.anyio
    async def test_streams_text(self, place):
        agent = prepared(place)
        with agent._pydantic_agent.override(model=TestModel()):
            async with agent.run_stream("go") as stream:
                chunks = [chunk async for chunk in stream.stream_text(delta=True)]
        assert chunks
        assert "".join(chunks)

    @pytest.mark.anyio
    async def test_builds_agent_lazily(self, place):
        agent = StreamAgent(place)
        assert agent._pydantic_agent is None
        agent._pydantic_agent = agent.build_agent()
        with agent._pydantic_agent.override(model=TestModel()):
            async with agent.run_stream("go"):
                pass
        assert agent._pydantic_agent is not None

    @pytest.mark.anyio
    async def test_instance_values_reach_the_model(self, place):
        place.name = "Streamed Cafe"
        agent = prepared(place)
        with agent._pydantic_agent.override(model=TestModel()):
            async with agent.run_stream("go") as stream:
                async for _ in stream.stream_text(delta=True):
                    pass
                messages = stream.all_messages()

        instructions = "\n".join(
            m.instructions for m in messages if getattr(m, "instructions", None)
        )
        assert "Streamed Cafe" in instructions


class TestRunStreamEvents:
    @pytest.mark.anyio
    async def test_yields_events(self, place):
        agent = prepared(place)
        with agent._pydantic_agent.override(model=TestModel()):
            async with agent.run_stream_events("go") as events:
                seen = [type(event).__name__ async for event in events]
        assert seen
        assert any("Part" in name for name in seen)

    @pytest.mark.anyio
    async def test_consumer_can_stop_early(self, place):
        """Bailing out mid-stream must not leave the run dangling."""
        agent = prepared(place)
        with agent._pydantic_agent.override(model=TestModel()):
            async with agent.run_stream_events("go") as events:
                async for _ in events:
                    break


class TestStreamingHonoursUsageLimits:
    @pytest.mark.anyio
    async def test_limits_applied_to_stream(self, place):
        agent = prepared(place, usage_limits=UsageLimits(request_limit=0))
        with agent._pydantic_agent.override(model=TestModel()):
            with pytest.raises(UsageLimitExceeded):
                async with agent.run_stream("go") as stream:
                    async for _ in stream.stream_text(delta=True):
                        pass
