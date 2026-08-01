"""
Tests for ModelAgent composing capabilities.

The declarative ModelAgent API is unchanged; what changed is that build_agent()
now assembles capabilities instead of doing the work inline. These tests pin
the composition and the behaviour that motivated it.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimitExceeded, UsageLimits

from tests.models import Place

from django_model_agent import ModelAgent
from django_model_agent.capabilities import (
    DjangoAuditCapability,
    DjangoFSMCapability,
    DjangoMemoryCapability,
    DjangoModelCapability,
)
from django_model_agent.tools import ReadOnlyTool


class Review(BaseModel):
    summary: str
    score: int


class Summary(BaseModel):
    text: str


class PlainTool(ReadOnlyTool):
    name: ClassVar[str] = "plain"
    description: ClassVar[str] = "No state restriction"

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


class DraftTool(ReadOnlyTool):
    name: ClassVar[str] = "draft_only"
    description: ClassVar[str] = "Draft only"
    allowed_states: ClassVar[list[str]] = ["draft"]

    def read(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True}


def kinds(capabilities) -> list[type]:
    return [type(c) for c in capabilities]


def run_with_test_model(agent: ModelAgent, prompt: str = "hi"):
    pai = agent.build_agent()
    agent._pydantic_agent = pai
    with pai.override(model=TestModel()):
        return agent.run_sync(prompt)


def instructions_sent(result) -> str:
    return "\n".join(
        m.instructions for m in result.all_messages() if getattr(m, "instructions", None)
    )


class TestComposition:
    def test_model_capability_always_present(self, place, simple_agent_class):
        caps = simple_agent_class(place)._build_capabilities()
        assert DjangoModelCapability in kinds(caps)

    def test_fsm_capability_added_for_state_restricted_tools(self, place):
        class StatefulAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [DraftTool]

        caps = StatefulAgent(place)._build_capabilities()
        assert DjangoFSMCapability in kinds(caps)

    def test_fsm_capability_skipped_without_restrictions(self, place):
        class PlainAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [PlainTool]

        caps = PlainAgent(place)._build_capabilities()
        assert DjangoFSMCapability not in kinds(caps)

    def test_extra_capabilities_appended(self, place):
        class AuditedAgent(ModelAgent):
            model = Place
            fields = ["name"]

            def get_extra_capabilities(self):
                return [DjangoAuditCapability(log_to="none")]

        caps = AuditedAgent(place)._build_capabilities()
        assert DjangoAuditCapability in kinds(caps)

    def test_memory_capability_added_for_mixin(self, place):
        import warnings

        from django_model_agent.memory import AgentMemoryMixin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            class MemoryAgent(AgentMemoryMixin, ModelAgent):
                model = Place
                fields = ["name"]

        caps = MemoryAgent(place)._build_capabilities()
        assert DjangoMemoryCapability in kinds(caps)

    def test_no_memory_capability_without_mixin(self, place, simple_agent_class):
        caps = simple_agent_class(place)._build_capabilities()
        assert DjangoMemoryCapability not in kinds(caps)


class TestAgentConstruction:
    def test_agent_receives_capabilities(self, place, simple_agent_class):
        pai = simple_agent_class(place).build_agent()
        assert pai is not None

    def test_name_preserved(self, place):
        class NamedAgent(ModelAgent):
            model = Place
            fields = ["name"]

        assert NamedAgent(place).build_agent().name == "NamedAgent(Place)"

    def test_no_system_prompt_passed(self, place):
        """
        System prompts must not reach the Agent constructor.

        They are folded into instructions instead; see test_no_stale_values.
        """

        class PromptAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "Marker system prompt."

        result = run_with_test_model(PromptAgent(place))
        parts = "".join(
            str(p) for m in result.all_messages() for p in getattr(m, "parts", [])
        )
        assert "Marker system prompt." not in parts
        assert "Marker system prompt." in instructions_sent(result)


class TestPromptsBecomeInstructions:
    def test_system_prompts_reach_the_model(self, place):
        class PromptAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "You are careful."

        assert "You are careful." in instructions_sent(
            run_with_test_model(PromptAgent(place))
        )

    def test_instructions_reach_the_model(self, place):
        class InstructedAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _instructions = "Be concise."

        assert "Be concise." in instructions_sent(
            run_with_test_model(InstructedAgent(place))
        )

    def test_both_reach_the_model(self, place):
        class BothAgent(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "Static prompt."
            _instructions = "Dynamic guidance."

        text = instructions_sent(run_with_test_model(BothAgent(place)))
        assert "Static prompt." in text and "Dynamic guidance." in text

    def test_decorated_prompt_methods_reach_the_model(self, place):
        class DecoratedAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.system_prompt
            def extra(self) -> str:
                return "Decorated prompt text."

        assert "Decorated prompt text." in instructions_sent(
            run_with_test_model(DecoratedAgent(place))
        )


class TestNoStaleValues:
    """
    The reason system prompts were dropped.

    A system prompt is written into message history and stays there. Because
    the agent injects the instance's current field values, turn one's values
    would still be in context on turn five, contradicting the current ones.
    """

    def test_values_not_persisted_in_history(self, place):
        class ValueAgent(ModelAgent):
            model = Place
            fields = ["name"]

        place.name = "Original Name"
        result = run_with_test_model(ValueAgent(place))

        parts = "".join(
            str(p) for m in result.all_messages() for p in getattr(m, "parts", [])
        )
        assert "Original Name" not in parts
        assert "Original Name" in instructions_sent(result)

    def test_changed_value_does_not_carry_forward(self, place):
        class ValueAgent(ModelAgent):
            model = Place
            fields = ["name"]

        agent = ValueAgent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            place.name = "Before Rename"
            first = instructions_sent(agent.run_sync("turn one"))

            place.name = "After Rename"
            second = instructions_sent(agent.run_sync("turn two"))

        assert "Before Rename" in first
        assert "After Rename" in second
        assert "Before Rename" not in second


class TestDecoratedTools:
    def test_decorated_tool_still_works(self, place):
        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.tool
            def get_name(self) -> str:
                """Get the place name."""
                return self.instance.name

        result = run_with_test_model(ToolAgent(place))
        returns = "".join(
            str(p.content)
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolReturnPart"
        )
        assert place.name in returns

    def test_model_tool_still_works(self, place):
        class ToolAgent(ModelAgent):
            model = Place
            fields = ["name"]
            tools = [PlainTool]

        result = run_with_test_model(ToolAgent(place))
        called = {
            p.tool_name
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
            if type(p).__name__ == "ToolCallPart"
        }
        assert "plain" in called


class TestModelResolution:
    """ai_model resolution, including the PYDANTIC_AI_MODEL convention."""

    def agent_class(self, ai_model=None):
        class Resolved(ModelAgent):
            model = Place
            fields = ["name"]

        Resolved.ai_model = ai_model
        return Resolved

    def test_init_argument_wins(self, place, monkeypatch):
        monkeypatch.setenv("PYDANTIC_AI_MODEL", "env:model")
        cls = self.agent_class(ai_model="class:model")
        assert cls(place, ai_model="init:model")._get_ai_model() == "init:model"

    def test_class_attribute_beats_env(self, place, monkeypatch):
        monkeypatch.setenv("PYDANTIC_AI_MODEL", "env:model")
        cls = self.agent_class(ai_model="class:model")
        assert cls(place)._get_ai_model() == "class:model"

    def test_env_used_as_fallback(self, place, monkeypatch):
        monkeypatch.setenv("PYDANTIC_AI_MODEL", "env:model")
        assert self.agent_class()(place)._get_ai_model() == "env:model"

    def test_none_when_nothing_set(self, place, monkeypatch):
        monkeypatch.delenv("PYDANTIC_AI_MODEL", raising=False)
        assert self.agent_class()(place)._get_ai_model() is None

    def test_empty_env_treated_as_unset(self, place, monkeypatch):
        monkeypatch.setenv("PYDANTIC_AI_MODEL", "")
        assert self.agent_class()(place)._get_ai_model() is None


class TestStructuredOutput:
    """output_type, declared on the class or passed per agent/run."""

    def run_agent(self, agent, prompt="go", **kwargs):
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()):
            return agent.run_sync(prompt, **kwargs)

    def test_default_output_is_str(self, place, simple_agent_class):
        result = self.run_agent(simple_agent_class(place))
        assert isinstance(result.output, str)

    def test_class_level_output_type(self, place):
        class ReviewAgent(ModelAgent):
            model = Place
            fields = ["name"]
            output_type = Review

        result = self.run_agent(ReviewAgent(place))
        assert isinstance(result.output, Review)

    def test_init_override(self, place, simple_agent_class):
        result = self.run_agent(simple_agent_class(place, output_type=Review))
        assert isinstance(result.output, Review)

    def test_init_beats_class(self, place):
        class ReviewAgent(ModelAgent):
            model = Place
            fields = ["name"]
            output_type = Review

        result = self.run_agent(ReviewAgent(place, output_type=Summary))
        assert isinstance(result.output, Summary)

    def test_per_run_output_type(self, place, simple_agent_class):
        result = self.run_agent(simple_agent_class(place), output_type=Review)
        assert isinstance(result.output, Review)

    def test_output_is_a_real_pydantic_model(self, place):
        class ReviewAgent(ModelAgent):
            model = Place
            fields = ["name"]
            output_type = Review

        output = self.run_agent(ReviewAgent(place)).output
        assert set(output.model_dump()) == {"summary", "score"}


class TestInstructionsAreFresh:
    """
    Decorated prompt methods and templates must resolve per request.

    They used to be flattened to text at build time, which left them
    contradicting the field values the capability injects live.
    """

    def test_decorated_instructions_reevaluated(self, place):
        calls = []

        class Probe(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.instructions
            def hint(self) -> str:
                calls.append(self.instance.name)
                return f"SAW:{self.instance.name}"

        agent = Probe(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            place.name = "First"
            first = instructions_sent(agent.run_sync("one"))
            place.name = "Second"
            second = instructions_sent(agent.run_sync("two"))

        assert len(calls) == 2, "decorated method should run once per request"
        assert "SAW:First" in first
        assert "SAW:Second" in second
        assert "SAW:First" not in second

    def test_decorated_system_prompt_reevaluated(self, place):
        class Probe(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.system_prompt
            def hint(self) -> str:
                return f"SAW:{self.instance.name}"

        agent = Probe(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            place.name = "Before"
            first = instructions_sent(agent.run_sync("one"))
            place.name = "After"
            second = instructions_sent(agent.run_sync("two"))

        assert "SAW:Before" in first
        assert "SAW:After" in second
        assert "SAW:Before" not in second

    def test_decorated_text_agrees_with_current_values(self, place):
        """The failure mode worth guarding: context contradicting itself."""

        class Probe(ModelAgent):
            model = Place
            fields = ["name"]

            @ModelAgent.instructions
            def hint(self) -> str:
                return f"The name is {self.instance.name}."

        agent = Probe(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            place.name = "Renamed"
            text = instructions_sent(agent.run_sync("go"))

        assert "The name is Renamed." in text
        assert "'name': 'Renamed'" in text

    def test_static_class_attributes_still_sent(self, place):
        class Probe(ModelAgent):
            model = Place
            fields = ["name"]
            _system_prompts = "Static prompt."
            _instructions = "Static guidance."

        agent = Probe(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai

        with pai.override(model=TestModel()):
            text = instructions_sent(agent.run_sync("go"))

        assert "Static prompt." in text
        assert "Static guidance." in text


class TestUsageLimits:
    """usage_limits resolution and enforcement."""

    def limited_agent(self, place, **kwargs):
        class Limited(ModelAgent):
            model = Place
            fields = ["name"]

        for key, value in kwargs.items():
            setattr(Limited, key, value)
        return Limited(place)

    def test_class_attribute_enforced(self, place):
        agent = self.limited_agent(place, usage_limits=UsageLimits(request_limit=0))
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()), pytest.raises(UsageLimitExceeded):
            agent.run_sync("go")

    def test_init_override_beats_class(self, place):
        agent = self.limited_agent(place, usage_limits=UsageLimits(request_limit=99))
        agent._usage_limits_override = UsageLimits(request_limit=0)
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()), pytest.raises(UsageLimitExceeded):
            agent.run_sync("go")

    def test_setting_used_as_fallback(self, place, settings):
        settings.DJANGO_MODEL_AGENT_USAGE_LIMITS = UsageLimits(request_limit=0)
        agent = self.limited_agent(place)
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()), pytest.raises(UsageLimitExceeded):
            agent.run_sync("go")

    def test_class_attribute_beats_setting(self, place, settings):
        settings.DJANGO_MODEL_AGENT_USAGE_LIMITS = UsageLimits(request_limit=0)
        agent = self.limited_agent(place, usage_limits=UsageLimits(request_limit=99))
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()):
            assert agent.run_sync("go").output is not None

    def test_no_limits_by_default(self, place, simple_agent_class):
        assert simple_agent_class(place)._get_usage_limits() is None

    def test_explicit_run_kwarg_wins(self, place):
        agent = self.limited_agent(place, usage_limits=UsageLimits(request_limit=0))
        pai = agent.build_agent()
        agent._pydantic_agent = pai
        with pai.override(model=TestModel()):
            result = agent.run_sync("go", usage_limits=UsageLimits(request_limit=99))
        assert result.output is not None
