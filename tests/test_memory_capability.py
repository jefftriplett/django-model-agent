"""
Tests for DjangoMemoryCapability.

Memory is persisted to the database, so these tests need db access. They cover
the load/save cycle, per-run isolation, trimming, and the unsaved-instance case.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgentContext
from django_model_agent.capabilities import (
    DjangoMemoryCapability,
    DjangoModelCapability,
)
from django_model_agent.memory import AgentMemory, AgentMemoryMixin
from django_model_agent.tools import ReadOnlyTool

# transaction=True is required, not incidental: the capability reaches the ORM
# through sync_to_async, which runs on a different thread than the test. The
# default transaction-wrapped fixture would leave that thread locked out.
pytestmark = pytest.mark.django_db(transaction=True)


class InfoTool(ReadOnlyTool):
    name = "get_info"
    description = "Get information about this place"

    def read(self, **kwargs):
        return {"name": self.instance.name}


def build_agent(tools=(), **kwargs) -> Agent:
    return Agent(
        TestModel(),
        deps_type=ModelAgentContext,
        capabilities=[
            DjangoModelCapability(
                model_class=Place, fields=["name"], tools=list(tools)
            ),
            DjangoMemoryCapability(**kwargs),
        ],
    )


def run(agent: Agent, instance: Place, prompt: str = "hello"):
    return agent.run_sync(prompt, deps=ModelAgentContext(instance=instance, agent=None))


def message_text(result) -> str:
    """All string content across a run's messages."""
    return " | ".join(
        part.content
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if isinstance(getattr(part, "content", None), str)
    )


def instructions_sent(result) -> str:
    return "\n".join(
        m.instructions for m in result.all_messages() if getattr(m, "instructions", None)
    )


class TestPersistence:
    def test_creates_memory_record(self, place):
        assert AgentMemory.objects.get_for(place) is None
        run(build_agent(), place)
        assert AgentMemory.objects.get_for(place) is not None

    def test_records_prompt_and_response(self, place):
        run(build_agent(), place, "what are the hours?")
        history = AgentMemory.objects.get_for(place).get_history()
        roles = [turn["role"] for turn in history]
        assert "user" in roles and "assistant" in roles
        assert any("what are the hours?" in t["content"] for t in history)

    def test_history_accumulates_across_runs(self, place):
        agent = build_agent()
        run(agent, place, "first question")
        run(agent, place, "second question")

        history = AgentMemory.objects.get_for(place).get_history()
        contents = " ".join(t["content"] for t in history)
        assert "first question" in contents
        assert "second question" in contents

    def test_prior_turn_replayed_as_messages(self, place):
        agent = build_agent()
        run(agent, place, "remember the alamo")
        second = run(agent, place, "and now?")
        assert "remember the alamo" in message_text(second)

    def test_history_not_duplicated(self, place):
        """before_model_request fires per request; history must inject once."""
        agent = build_agent()
        run(agent, place, "unique marker phrase")
        second = run(agent, place, "and now?")
        assert message_text(second).count("unique marker phrase") == 1

    def test_first_run_has_no_prior_messages(self, place):
        result = run(build_agent(), place, "opening line")
        assert message_text(result).count("opening line") == 1

    def test_messages_stored_in_structured_form(self, place):
        run(build_agent(), place, "hello")
        stored = AgentMemory.objects.get_for(place).data["messages"]
        assert isinstance(stored, list) and stored
        assert all(isinstance(entry, dict) for entry in stored)

    def test_tool_calls_survive_into_next_turn(self, place):
        """
        The reason for using real messages rather than a flattened transcript:
        a text replay loses that a tool ran and what it returned.
        """
        agent = build_agent(tools=[InfoTool])
        run(agent, place, "call the tool")
        second = run(agent, place, "second turn")

        kinds = {
            type(part).__name__
            for message in second.all_messages()
            for part in getattr(message, "parts", [])
        }
        assert "ToolCallPart" in kinds
        assert "ToolReturnPart" in kinds

    def test_memory_is_per_instance(self, place, draft_place):
        agent = build_agent()
        run(agent, place, "about the first")
        run(agent, draft_place, "about the second")

        first = " ".join(
            t["content"] for t in AgentMemory.objects.get_for(place).get_history()
        )
        second = " ".join(
            t["content"] for t in AgentMemory.objects.get_for(draft_place).get_history()
        )
        assert "about the first" in first and "about the second" not in first
        assert "about the second" in second and "about the first" not in second


class TestOptions:
    def test_include_history_false_suppresses_replay(self, place):
        agent = build_agent(include_history=False)
        run(agent, place, "hidden line")
        second = run(agent, place, "next")
        assert "hidden line" not in message_text(second)

    def test_include_history_false_still_persists(self, place):
        agent = build_agent(include_history=False)
        run(agent, place, "still stored")
        contents = " ".join(
            t["content"] for t in AgentMemory.objects.get_for(place).get_history()
        )
        assert "still stored" in contents

    def test_max_history_trims_oldest(self, place):
        agent = build_agent(max_history=2)
        run(agent, place, "turn one")
        run(agent, place, "turn two")
        stored = AgentMemory.objects.get_for(place).data["messages"]
        assert len(stored) <= 2

    def test_trim_does_not_start_mid_exchange(self, place):
        """
        A blind cut can orphan a tool result from its call, which some providers
        reject. Trimming lands on a request boundary instead.
        """
        agent = build_agent(tools=[InfoTool], max_history=3)
        run(agent, place, "one")
        run(agent, place, "two")
        run(agent, place, "three")

        stored = AgentMemory.objects.get_for(place).data["messages"]
        if stored:
            assert stored[0].get("kind") == "request"


class TestPerRunIsolation:
    def test_for_run_returns_new_instance(self):
        cap = DjangoMemoryCapability(max_history=7, include_history=False)
        import asyncio

        fresh = asyncio.run(cap.for_run(None))
        assert fresh is not cap
        assert fresh.max_history == 7
        assert fresh.include_history is False

    def test_loaded_memory_does_not_leak_to_config_instance(self, place):
        cap = DjangoMemoryCapability()
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(model_class=Place, fields=["name"]),
                cap,
            ],
        )
        agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        # The configured capability stays clean; only the per-run copy loads.
        assert cap._memory is None


class TestUnsavedInstance:
    def test_unsaved_instance_is_skipped_not_raised(self):
        agent = build_agent()
        unsaved = Place(name="Never Saved")
        assert unsaved.pk is None
        result = agent.run_sync(
            "hi", deps=ModelAgentContext(instance=unsaved, agent=None)
        )
        assert result.output is not None

    def test_unsaved_instance_writes_no_memory(self):
        run(build_agent(), Place(name="Never Saved"))
        assert AgentMemory.objects.count() == 0


class TestMixinDeprecation:
    def test_subclassing_mixin_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class LegacyAgent(AgentMemoryMixin):
                pass

        messages = [str(w.message) for w in caught]
        assert any("DjangoMemoryCapability" in m for m in messages)

    def test_warning_is_deprecation_category(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            class AnotherLegacyAgent(AgentMemoryMixin):
                pass

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
