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

# transaction=True is required, not incidental: the capability reaches the ORM
# through sync_to_async, which runs on a different thread than the test. The
# default transaction-wrapped fixture would leave that thread locked out.
pytestmark = pytest.mark.django_db(transaction=True)


def build_agent(**kwargs) -> Agent:
    return Agent(
        TestModel(),
        deps_type=ModelAgentContext,
        capabilities=[
            DjangoModelCapability(model_class=Place, fields=["name"]),
            DjangoMemoryCapability(**kwargs),
        ],
    )


def run(agent: Agent, instance: Place, prompt: str = "hello"):
    return agent.run_sync(prompt, deps=ModelAgentContext(instance=instance, agent=None))


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

    def test_prior_history_fed_back_as_instructions(self, place):
        agent = build_agent()
        run(agent, place, "remember the alamo")
        second = run(agent, place, "and now?")
        assert "remember the alamo" in instructions_sent(second)

    def test_first_run_has_no_history(self, place):
        result = run(build_agent(), place, "opening line")
        assert "Earlier in this conversation" not in instructions_sent(result)

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
    def test_include_history_false_suppresses_feedback(self, place):
        agent = build_agent(include_history=False)
        run(agent, place, "hidden line")
        second = run(agent, place, "next")
        assert "hidden line" not in instructions_sent(second)

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

        history = AgentMemory.objects.get_for(place).get_history()
        assert len(history) == 2
        assert "turn one" not in " ".join(t["content"] for t in history)


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
