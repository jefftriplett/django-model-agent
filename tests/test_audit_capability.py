"""
Tests for DjangoAuditCapability.

Records are collected through the callback, because each run gets its own copy
of the capability and the configured instance never sees the record.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgentContext
from django_model_agent.capabilities import (
    AuditRecord,
    DjangoAuditCapability,
    DjangoModelCapability,
)
from django_model_agent.tools import UpdateTool

# transaction=True is required, not incidental: UpdateTool saves the instance
# from the thread pydantic-ai runs sync tools on. The default transaction-
# wrapped fixture would leave that thread locked out.
pytestmark = pytest.mark.django_db(transaction=True)


class RenameTool(UpdateTool):
    name: ClassVar[str] = "rename"
    description: ClassVar[str] = "Rename the place"

    def update(self, **kwargs: Any) -> None:
        self.instance.name = "Renamed By Agent"


class NoopTool(UpdateTool):
    name: ClassVar[str] = "noop"
    description: ClassVar[str] = "Change nothing"

    def update(self, **kwargs: Any) -> None:
        return None


def build_agent(records: list[AuditRecord], tools=(), **kwargs) -> Agent:
    return Agent(
        TestModel(),
        deps_type=ModelAgentContext,
        capabilities=[
            DjangoModelCapability(model_class=Place, fields=["name"], tools=list(tools)),
            DjangoAuditCapability(
                log_to="callback", callback=records.append, **kwargs
            ),
        ],
    )


def run(agent: Agent, instance: Place, prompt: str = "do it"):
    return agent.run_sync(prompt, deps=ModelAgentContext(instance=instance, agent=None))


class TestAuditRecord:
    def test_changed_false_when_empty(self):
        record = AuditRecord(instance_pk=1, model_class="Place", prompt="p")
        assert record.changed is False

    def test_changed_true_with_changes(self):
        record = AuditRecord(
            instance_pk=1,
            model_class="Place",
            prompt="p",
            field_changes={"name": {"before": "a", "after": "b"}},
        )
        assert record.changed is True

    def test_summary_mentions_field_and_values(self):
        record = AuditRecord(
            instance_pk=7,
            model_class="Place",
            prompt="p",
            field_changes={"name": {"before": "Old", "after": "New"}},
        )
        summary = record.summary()
        assert "Place#7" in summary and "Old" in summary and "New" in summary

    def test_summary_when_nothing_changed(self):
        record = AuditRecord(instance_pk=7, model_class="Place", prompt="p")
        assert "no field changes" in record.summary()


class TestFieldDiffing:
    def test_detects_field_change(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records, tools=[RenameTool]), place)

        assert records
        assert "name" in records[0].field_changes
        assert records[0].field_changes["name"]["after"] == "Renamed By Agent"

    def test_records_original_value(self, place):
        records: list[AuditRecord] = []
        original = place.name
        run(build_agent(records, tools=[RenameTool]), place)
        assert records[0].field_changes["name"]["before"] == original

    def test_no_changes_when_nothing_touched(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place)
        assert records[0].field_changes == {}
        assert records[0].changed is False

    def test_tool_that_changes_nothing_reports_nothing(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records, tools=[NoopTool]), place)
        assert records[0].changed is False

    def test_track_fields_limits_scope(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records, tools=[RenameTool], track_fields=["address"]), place)
        assert "name" not in records[0].field_changes


class TestRecordContents:
    def test_captures_prompt(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place, "please rename it")
        assert records[0].prompt == "please rename it"

    def test_captures_instance_identity(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place)
        assert records[0].instance_pk == place.pk
        assert records[0].model_class == "Place"

    def test_captures_tool_calls(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records, tools=[RenameTool]), place)
        assert any(call["name"] == "rename" for call in records[0].tool_calls)

    def test_no_tool_calls_when_no_tools(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place)
        assert records[0].tool_calls == []


class TestPerRunIsolation:
    def test_each_run_produces_its_own_record(self, place, draft_place):
        records: list[AuditRecord] = []
        agent = build_agent(records)
        run(agent, place, "first")
        run(agent, draft_place, "second")

        assert len(records) == 2
        assert {r.prompt for r in records} == {"first", "second"}
        assert {r.instance_pk for r in records} == {place.pk, draft_place.pk}

    def test_configured_instance_holds_no_record(self, place):
        cap = DjangoAuditCapability(log_to="none")
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(model_class=Place, fields=["name"]),
                cap,
            ],
        )
        agent.run_sync("hi", deps=ModelAgentContext(instance=place, agent=None))
        assert cap.record is None

    def test_snapshot_does_not_leak_between_runs(self, place):
        records: list[AuditRecord] = []
        agent = build_agent(records, tools=[RenameTool])
        run(agent, place)
        run(agent, place)
        # Second run starts from the already-renamed value, so nothing changes.
        assert records[0].changed is True
        assert records[1].changed is False


class TestLogTargets:
    def test_logger_mode_emits(self, place, caplog):
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(
                    model_class=Place, fields=["name"], tools=[RenameTool]
                ),
                DjangoAuditCapability(log_to="logger"),
            ],
        )
        with caplog.at_level(logging.INFO, logger="django_model_agent.capabilities"):
            run(agent, place)
        assert "audit" in caplog.text.lower()

    def test_none_mode_is_silent(self, place, caplog):
        agent = Agent(
            TestModel(),
            deps_type=ModelAgentContext,
            capabilities=[
                DjangoModelCapability(model_class=Place, fields=["name"]),
                DjangoAuditCapability(log_to="none"),
            ],
        )
        with caplog.at_level(logging.INFO, logger="django_model_agent.capabilities"):
            run(agent, place)
        assert "audit" not in caplog.text.lower()

    def test_callback_mode_requires_callback(self):
        with pytest.raises(ValueError, match="requires a callback"):
            DjangoAuditCapability(log_to="callback")


class TestUsageCapture:
    """Token and request counts land on the audit record."""

    def test_usage_recorded(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place)
        usage = records[0].usage
        assert usage["requests"] >= 1
        assert usage["input_tokens"] > 0

    def test_total_tokens_sums_input_and_output(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records), place)
        record = records[0]
        assert record.total_tokens == (
            record.usage["input_tokens"] + record.usage["output_tokens"]
        )

    def test_tool_calls_counted(self, place):
        records: list[AuditRecord] = []
        run(build_agent(records, tools=[RenameTool]), place)
        assert records[0].usage["tool_calls"] >= 1

    def test_usage_empty_when_result_has_none(self):
        cap = DjangoAuditCapability(log_to="none")
        assert cap._usage(object()) == {}

    def test_total_tokens_zero_without_usage(self):
        assert AuditRecord(instance_pk=1, model_class="Place", prompt="p").total_tokens == 0
