"""
Pydantic AI capabilities for Django models.

Capabilities are the composable extension mechanism in pydantic-ai. Each one
contributes instructions, tools, and lifecycle hooks to an agent, so behaviour
that used to be hardcoded in ``ModelAgent.build_agent()`` can now be mixed and
matched -- and used with a plain ``pydantic_ai.Agent`` without ``ModelAgent``
at all.

``DjangoModelCapability`` is configured with a model *class* and never holds a
model *instance*. The instance arrives per-run through ``ctx.deps.instance``,
so a single agent serves every row of the table:

    agent = Agent(
        "openai:gpt-4o",
        deps_type=ModelAgentContext,
        capabilities=[DjangoModelCapability(model_class=Place, fields=["name"])],
    )
    await agent.run("...", deps=ModelAgentContext(instance=place, agent=None))
    await agent.run("...", deps=ModelAgentContext(instance=other, agent=None))

Note: this module deliberately does NOT use ``from __future__ import
annotations``. Tool functions are introspected with ``get_type_hints()``, which
cannot resolve ``RunContext`` when annotations are strings.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from decimal import Decimal
from typing import Any, Optional

from django.db import models
from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from .base import ModelAgentContext

logger = logging.getLogger(__name__)


# Django field internal type -> Python type. Dates and times become strings
# because models talk to the AI in ISO format, not datetime objects.
FIELD_TYPE_MAP: dict[str, type] = {
    "AutoField": int,
    "BigAutoField": int,
    "SmallAutoField": int,
    "IntegerField": int,
    "SmallIntegerField": int,
    "BigIntegerField": int,
    "PositiveIntegerField": int,
    "PositiveSmallIntegerField": int,
    "PositiveBigIntegerField": int,
    "FloatField": float,
    "DecimalField": Decimal,
    "CharField": str,
    "TextField": str,
    "EmailField": str,
    "URLField": str,
    "SlugField": str,
    "UUIDField": str,
    "FilePathField": str,
    "FileField": str,
    "ImageField": str,
    "GenericIPAddressField": str,
    "IPAddressField": str,
    "BooleanField": bool,
    "DateField": str,
    "DateTimeField": str,
    "TimeField": str,
    "DurationField": str,
    "BinaryField": bytes,
    "JSONField": dict,
    "ForeignKey": int,
}


def field_python_type(field: models.Field) -> Any:
    """Map a Django field to the Python type the AI should see."""
    try:
        python_type = FIELD_TYPE_MAP.get(field.get_internal_type(), str)
    except AttributeError:
        return str

    if getattr(field, "null", False):
        return Optional[python_type]
    return python_type


def agent_fields(
    model_class: type[models.Model],
    fields: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[models.Field]:
    """
    The model fields an agent may see, honouring ``fields`` and ``exclude``.

    Works from the model class alone -- no instance required -- so it can run
    at agent construction time.
    """
    exclude = set(exclude or ())
    selected = []
    for field in model_class._meta.fields:
        if fields and field.name not in fields:
            continue
        if field.name in exclude:
            continue
        selected.append(field)
    return selected


def field_value(instance: models.Model, name: str) -> Any:
    """Read a field off an instance, collapsing related objects to their pk."""
    value = getattr(instance, name, None)
    if hasattr(value, "pk"):
        return value.pk
    return value


def model_tools_to_toolset(
    tool_classes: Sequence[type] = (),
    tool_funcs: Sequence[Callable] = (),
) -> FunctionToolset:
    """
    Build a toolset from ``ModelTool`` subclasses and decorated methods.

    Every tool takes the run context and resolves the model instance from
    ``ctx.deps`` at call time, so the resulting toolset is bound to no
    particular instance and can be reused across runs.
    """
    toolset: FunctionToolset = FunctionToolset()

    for tool_cls in tool_classes:
        toolset.add_tool(_tool_from_class(tool_cls))

    for func in tool_funcs:
        toolset.add_tool(_tool_from_func(func))

    return toolset


def _tool_from_class(tool_cls: type) -> Tool:
    """Wrap a ``ModelTool`` subclass as a context-taking pydantic-ai tool."""

    def tool_func(ctx: RunContext[ModelAgentContext], **kwargs: Any) -> str:
        # Constructed per call so the tool sees this run's instance.
        return str(tool_cls(ctx.deps)(**kwargs))

    tool_func.__name__ = tool_cls.name
    tool_func.__qualname__ = tool_cls.name
    tool_func.__doc__ = tool_cls.description

    return Tool(tool_func, name=tool_cls.name, takes_ctx=True)


def _tool_from_func(func: Callable) -> Tool:
    """
    Wrap an ``@ModelAgent.tool`` decorated method as a pydantic-ai tool.

    The method is unbound: it is called with the ``ModelAgent`` from
    ``ctx.deps.agent``, so the tool is not tied to one agent instance.
    """
    import inspect

    sig = inspect.signature(func)
    # Drop `self` -- the agent is supplied from the run context instead.
    params = [p for name, p in sig.parameters.items() if name != "self"]
    param_names = [p.name for p in params]

    def tool_func(ctx: RunContext[ModelAgentContext], **kwargs: Any) -> str:
        agent = ctx.deps.agent
        if agent is None:
            raise ValueError(
                f"Tool {func.__name__!r} needs ModelAgentContext.agent to be set"
            )
        filtered = {k: v for k, v in kwargs.items() if k in param_names}
        return str(func(agent, **filtered))

    tool_func.__name__ = func.__name__
    tool_func.__qualname__ = func.__qualname__
    tool_func.__doc__ = func.__doc__
    tool_func.__signature__ = sig.replace(parameters=params)
    tool_func.__annotations__ = {
        k: v for k, v in getattr(func, "__annotations__", {}).items() if k != "return"
    } | {"return": str}

    return Tool(tool_func, name=func.__name__, takes_ctx=True)


class DjangoModelCapability(AbstractCapability[ModelAgentContext]):
    """
    Bridges a Django model to a pydantic-ai agent.

    Contributes the field schema, the instance's current values, and the
    model's tools. Configured with a model class only; the instance is read
    from ``ctx.deps.instance`` on every request, so values are never stale and
    one agent can serve many instances.

    Args:
        model_class: The Django model this agent operates on
        fields: Field names to expose (None exposes all)
        exclude: Field names to hide
        tools: ``ModelTool`` subclasses to expose
        tool_funcs: Unbound ``@ModelAgent.tool`` methods to expose
        instructions: Static instruction text
        dynamic_instructions: Callable returning text to resolve per request
    """

    def __init__(
        self,
        *,
        model_class: type[models.Model],
        fields: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        tools: Sequence[type] = (),
        tool_funcs: Sequence[Callable] = (),
        instructions: str | Sequence[str] = "",
        dynamic_instructions: Callable[[], str] | None = None,
        id: str | None = None,
    ) -> None:
        self.model_class = model_class
        self.fields = list(fields) if fields else None
        self.exclude = list(exclude or ())
        self.tools = list(tools)
        self.tool_funcs = list(tool_funcs)
        self.instructions = instructions
        self.dynamic_instructions = dynamic_instructions
        self.id = id

    def _fields(self) -> list[models.Field]:
        return agent_fields(self.model_class, self.fields, self.exclude)

    def _static_instructions(self) -> list[str]:
        if not self.instructions:
            return []
        if isinstance(self.instructions, str):
            return [self.instructions.strip()]
        return [text.strip() for text in self.instructions if text]

    def schema_description(self) -> str:
        """Human-readable listing of the fields the agent can see."""
        lines = [f"You have access to the following {self.model_class.__name__} fields:"]
        for field in self._fields():
            lines.append(f"  - {field.name}: {field_python_type(field)}")
        return "\n".join(lines)

    def current_values(self, instance: models.Model) -> dict[str, Any]:
        """The instance's values for every exposed field."""
        return {f.name: field_value(instance, f.name) for f in self._fields()}

    def get_instructions(self):
        """
        Instructions, not a system prompt, on purpose.

        Instructions are re-sent fresh on every request and kept out of message
        history. A system prompt would be baked into the history, so the field
        values captured on turn one would linger and contradict the current
        values on turn five.
        """
        static = self._static_instructions()
        # Computed once on purpose: which fields exist does not change between
        # runs, only their values do.
        schema = self.schema_description()

        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            instance = ctx.deps.instance
            parts = [*static]
            if self.dynamic_instructions is not None:
                text = self.dynamic_instructions()
                if text:
                    parts.append(text)
            parts.append(schema)
            parts.append(f"Current values: {self.current_values(instance)}")
            return "\n\n".join(parts)

        return _instructions

    def get_toolset(self) -> AbstractToolset[ModelAgentContext] | None:
        if not self.tools and not self.tool_funcs:
            return None
        return model_tools_to_toolset(self.tools, self.tool_funcs)


class DjangoFSMCapability(AbstractCapability[ModelAgentContext]):
    """
    Makes an agent aware of a model's state machine.

    Tells the agent which state the instance is in and which transitions are
    legal, and hides tools that cannot run in that state.

    Hiding is an optimisation, not the enforcement point: ``ModelTool`` still
    checks ``allowed_states`` in ``check_allowed()`` when it runs. Filtering
    here just means the model never sees a tool it cannot use, so it does not
    spend tokens calling one or get a refusal back mid-conversation.

    Args:
        state_field: Name of the field holding the state
        tools: ``ModelTool`` subclasses whose ``allowed_states`` should apply
        tool_states: Extra ``{tool_name: [states]}`` mapping, for tools that
            are not ``ModelTool`` subclasses
        transitions: ``{from_state: [to_states]}`` used to describe legal moves
    """

    def __init__(
        self,
        *,
        state_field: str = "state",
        tools: Sequence[type] = (),
        tool_states: dict[str, Sequence[str]] | None = None,
        transitions: dict[str, Sequence[str]] | None = None,
        id: str | None = None,
    ) -> None:
        self.state_field = state_field
        self.tools = list(tools)
        self.transitions = transitions or {}
        self.id = id

        # Tool name -> states it is allowed in. Tools with no restriction are
        # left out entirely so they are never filtered.
        self.tool_states: dict[str, list[str]] = {}
        for tool_cls in self.tools:
            allowed = getattr(tool_cls, "allowed_states", None)
            if allowed:
                self.tool_states[tool_cls.name] = list(allowed)
        for name, states in (tool_states or {}).items():
            self.tool_states[name] = list(states)

    def current_state(self, instance: models.Model) -> Any:
        """The instance's state, or None if it has no such field."""
        return getattr(instance, self.state_field, None)

    def available_transitions(self, state: Any) -> list[str]:
        return list(self.transitions.get(state, ()))

    def get_instructions(self):
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            state = self.current_state(ctx.deps.instance)
            if state is None:
                return ""

            lines = [f"The current {self.state_field} is '{state}'."]
            if self.transitions:
                moves = self.available_transitions(state)
                if moves:
                    lines.append(f"Valid transitions from here: {', '.join(moves)}.")
                else:
                    lines.append("There are no valid transitions from here.")
            return "\n".join(lines)

        return _instructions

    async def prepare_tools(
        self,
        ctx: RunContext[ModelAgentContext],
        tool_defs: list[Any],
    ) -> list[Any]:
        """Drop tools that cannot run in the instance's current state."""
        if not self.tool_states:
            return tool_defs

        state = self.current_state(ctx.deps.instance)
        if state is None:
            return tool_defs

        kept = []
        for tool_def in tool_defs:
            allowed = self.tool_states.get(tool_def.name)
            if allowed is not None and state not in allowed:
                logger.debug(
                    "Hiding tool %r: state %r not in %r", tool_def.name, state, allowed
                )
                continue
            kept.append(tool_def)
        return kept


def _load_messages(raw: Any) -> list[Any]:
    """Deserialise stored messages, tolerating rows written before this format."""
    if not raw:
        return []
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    try:
        return list(ModelMessagesTypeAdapter.validate_python(raw))
    except Exception:
        logger.warning("Ignoring unreadable stored messages", exc_info=True)
        return []


def _dump_messages(messages: Sequence[Any]) -> Any:
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")


def _trim(messages: list[Any], max_messages: int) -> list[Any]:
    """
    Keep the most recent messages, without orphaning a tool result.

    Cutting blindly can leave a tool return whose originating call was dropped,
    which some providers reject outright. Trimming to a request boundary avoids
    starting the history mid-exchange.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages

    from pydantic_ai.messages import ModelRequest

    kept = messages[-max_messages:]
    for index, message in enumerate(kept):
        if isinstance(message, ModelRequest):
            return kept[index:]
    return []


class DjangoMemoryCapability(AbstractCapability[ModelAgentContext]):
    """
    Gives an agent memory that persists across runs, keyed to the instance.

    Backed by the existing ``AgentMemory`` model. Replaces ``AgentMemoryMixin``,
    which required multiple inheritance on the agent class; as a capability it
    composes instead and works with a plain ``pydantic_ai.Agent``.

    All database work happens in ``before_run``/``after_run``, wrapped in
    ``sync_to_async`` because Django's ORM is synchronous and these hooks are
    not. The instructions callable only reads what was already loaded, so it
    never touches the database from a sync context.

    Instances with no primary key are skipped rather than raising: an unsaved
    model has nothing to key memory against.

    Args:
        max_history: Messages to retain before trimming oldest
        include_history: Whether past turns are fed back as instructions
    """

    def __init__(
        self,
        *,
        max_history: int = 100,
        include_history: bool = True,
        id: str | None = None,
    ) -> None:
        self.max_history = max_history
        self.include_history = include_history
        self.id = id
        self._memory: Any = None
        self._history: list[Any] = []
        self._injected = False

    async def for_run(self, ctx: RunContext[ModelAgentContext]):
        """Fresh instance per run so loaded memory never leaks between runs."""
        return DjangoMemoryCapability(
            max_history=self.max_history,
            include_history=self.include_history,
            id=self.id,
        )

    async def before_run(self, ctx: RunContext[ModelAgentContext]) -> None:
        from asgiref.sync import sync_to_async

        from .memory import AgentMemory

        instance = ctx.deps.instance
        if instance.pk is None:
            logger.debug("Skipping memory for unsaved %s", type(instance).__name__)
            return

        memory, _ = await sync_to_async(AgentMemory.objects.get_or_create_for)(instance)
        self._memory = memory
        self._history = _load_messages(memory.data.get("messages"))
        self._injected = False

    async def before_model_request(self, ctx: RunContext[ModelAgentContext], request_context: Any) -> Any:
        """
        Prepend the stored conversation to this run's messages.

        Done here rather than as instructions so the model receives real
        messages -- tool calls, their results, and structured output survive,
        where a flattened transcript would lose them.

        Injected once per run. This hook fires before every model request, and
        the messages it returns are carried forward, so prepending each time
        would duplicate the history.
        """
        if not self.include_history or self._injected or not self._history:
            return request_context

        request_context.messages = [*self._history, *request_context.messages]
        self._injected = True
        return request_context

    async def after_run(self, ctx: RunContext[ModelAgentContext], *, result: Any) -> Any:
        from asgiref.sync import sync_to_async

        if self._memory is None:
            return result

        messages = list(result.all_messages())
        self._memory.data["messages"] = _dump_messages(_trim(messages, self.max_history))

        # Kept for anything still reading the old text shape.
        prompt = ctx.prompt
        if isinstance(prompt, str) and prompt:
            self._memory.append_to_history("user", prompt, max_history=self.max_history)
        output = getattr(result, "output", None)
        if output is not None:
            self._memory.append_to_history(
                "assistant", str(output), max_history=self.max_history
            )

        await sync_to_async(self._memory.save)()
        return result


@dataclass
class AuditRecord:
    """What an agent run did to a model instance."""

    instance_pk: Any
    model_class: str
    prompt: str
    field_changes: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = dataclass_field(default_factory=list)
    usage: dict[str, int] = dataclass_field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.field_changes)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0)

    def summary(self) -> str:
        if not self.field_changes:
            return f"{self.model_class}#{self.instance_pk}: no field changes"
        parts = [
            f"{name}: {change['before']!r} -> {change['after']!r}"
            for name, change in self.field_changes.items()
        ]
        return f"{self.model_class}#{self.instance_pk}: " + ", ".join(parts)


class DjangoAuditCapability(AbstractCapability[ModelAgentContext]):
    """
    Records what an agent run changed on the model instance.

    Snapshots the instance's fields before the run, diffs them after, and
    reports the result. Tool calls are collected from the run's messages.

    The snapshot is taken from the in-memory instance rather than the database,
    so it reflects what the agent actually altered rather than any concurrent
    writes from elsewhere.

    Note that each run gets its own copy of the capability, so the ``record``
    attribute on the instance you constructed stays ``None``. Use
    ``log_to="callback"`` to receive records -- that is the supported way to
    get them out, and the route to persisting them in your own audit table.

    Args:
        log_to: ``"logger"`` writes to the module logger, ``"callback"`` hands
            the record to ``callback``, ``"none"`` only collects it
        callback: Receives the ``AuditRecord`` when ``log_to="callback"``
        track_fields: Field names to watch (None watches all editable fields)
    """

    def __init__(
        self,
        *,
        log_to: str = "logger",
        callback: Callable[[AuditRecord], None] | None = None,
        track_fields: Sequence[str] | None = None,
        id: str | None = None,
    ) -> None:
        if log_to == "callback" and callback is None:
            raise ValueError('log_to="callback" requires a callback')

        self.log_to = log_to
        self.callback = callback
        self.track_fields = list(track_fields) if track_fields else None
        self.id = id
        self._before: dict[str, Any] | None = None
        self.record: AuditRecord | None = None

    async def for_run(self, ctx: RunContext[ModelAgentContext]):
        """Fresh instance per run so snapshots never bleed between runs."""
        return DjangoAuditCapability(
            log_to=self.log_to,
            callback=self.callback,
            track_fields=self.track_fields,
            id=self.id,
        )

    def _snapshot(self, instance: models.Model) -> dict[str, Any]:
        names = self.track_fields or [
            f.name for f in instance._meta.fields if f.editable
        ]
        return {name: field_value(instance, name) for name in names}

    async def before_run(self, ctx: RunContext[ModelAgentContext]) -> None:
        self._before = self._snapshot(ctx.deps.instance)

    def _tool_calls(self, result: Any) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for message in getattr(result, "all_messages", lambda: [])():
            for part in getattr(message, "parts", []):
                if type(part).__name__ == "ToolCallPart":
                    calls.append({"name": part.tool_name, "args": part.args})
        return calls

    def _usage(self, result: Any) -> dict[str, int]:
        """Token and request counts for the run, if the result carries them."""
        usage = getattr(result, "usage", None)
        if usage is None:
            return {}
        if callable(usage):
            usage = usage()

        fields = (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "requests",
            "tool_calls",
        )
        return {
            name: value
            for name in fields
            if isinstance(value := getattr(usage, name, None), int)
        }

    async def after_run(self, ctx: RunContext[ModelAgentContext], *, result: Any) -> Any:
        if self._before is None:
            return result

        instance = ctx.deps.instance
        after = self._snapshot(instance)

        changes = {
            name: {"before": before, "after": after[name]}
            for name, before in self._before.items()
            if before != after.get(name)
        }

        prompt = ctx.prompt if isinstance(ctx.prompt, str) else ""
        self.record = AuditRecord(
            instance_pk=instance.pk,
            model_class=type(instance).__name__,
            prompt=prompt,
            field_changes=changes,
            tool_calls=self._tool_calls(result),
            usage=self._usage(result),
        )

        if self.log_to == "logger":
            logger.info("Agent run audit -- %s", self.record.summary())
        elif self.log_to == "callback" and self.callback is not None:
            self.callback(self.record)

        return result
