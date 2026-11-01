"""
Tests that sync tools close the database connections they open.

pydantic-ai runs sync tools in reused worker threads, where a connection the
tool opened would otherwise stay open. The test database is in-memory SQLite,
whose close() Django turns into a no-op, so the unit tests below use stand-in
connections to see what gets closed.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from django.db import connections
from pydantic_ai.models.test import TestModel

from tests.models import Place

from django_model_agent import ModelAgent, capabilities
from django_model_agent.tools import ModelTool, ToolResult


class FakeConnection:
    def __init__(self, alias: str, *, open: bool = False, atomic: bool = False):
        self.alias = alias
        self.connection = object() if open else None
        self.in_atomic_block = atomic
        self.closed = False

    def connect(self) -> None:
        self.connection = object()

    def close(self) -> None:
        self.closed = True
        self.connection = None


@pytest.fixture
def fake_connections(monkeypatch):
    fakes: list[FakeConnection] = []
    monkeypatch.setattr(connections, "all", lambda: list(fakes))
    return fakes


class TestClosingNewConnections:
    def test_closes_connection_opened_during_call(self, fake_connections):
        conn = FakeConnection("default")
        fake_connections.append(conn)

        with capabilities._closing_new_connections():
            conn.connect()

        assert conn.closed

    def test_leaves_connection_that_was_already_open(self, fake_connections):
        conn = FakeConnection("default", open=True)
        fake_connections.append(conn)

        with capabilities._closing_new_connections():
            pass

        assert not conn.closed

    def test_leaves_connection_inside_atomic_block(self, fake_connections):
        conn = FakeConnection("default")
        fake_connections.append(conn)

        with capabilities._closing_new_connections():
            conn.connect()
            conn.in_atomic_block = True

        assert not conn.closed

    def test_closes_even_when_tool_raises(self, fake_connections):
        conn = FakeConnection("default")
        fake_connections.append(conn)

        with pytest.raises(RuntimeError):
            with capabilities._closing_new_connections():
                conn.connect()
                raise RuntimeError("tool failed")

        assert conn.closed


class CountPlaces(ModelTool):
    name = "count_places"
    description = "Count places."

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, message=f"{Place.objects.count()} places")


@pytest.mark.django_db(transaction=True)
class TestSyncToolsUseIt:
    @pytest.mark.anyio
    async def test_sync_tool_runs_inside_wrapper_in_worker_thread(
        self, place, monkeypatch
    ):
        entered: list[int] = []
        original = capabilities._closing_new_connections

        def spy():
            entered.append(threading.get_ident())
            return original()

        monkeypatch.setattr(capabilities, "_closing_new_connections", spy)

        agent_cls = type(
            "CountAgent",
            (ModelAgent,),
            {"model": Place, "fields": ["name"], "tools": [CountPlaces]},
        )
        await agent_cls(place, ai_model=TestModel()).run("go")

        assert entered
        assert threading.get_ident() not in entered
