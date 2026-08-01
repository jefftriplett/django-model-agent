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
        id: str | None = None,
    ) -> None:
        self.model_class = model_class
        self.fields = list(fields) if fields else None
        self.exclude = list(exclude or ())
        self.tools = list(tools)
        self.tool_funcs = list(tool_funcs)
        self.instructions = instructions
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
        schema = self.schema_description()

        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            instance = ctx.deps.instance
            parts = [*static, schema]
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

    def get_instructions(self):
        def _instructions(ctx: RunContext[ModelAgentContext]) -> str:
            if self._memory is None or not self.include_history:
                return ""

            history = self._memory.get_history()
            if not history:
                return ""

            lines = ["Earlier in this conversation:"]
            lines.extend(f"  {turn['role']}: {turn['content']}" for turn in history)
            return "\n".join(lines)

        return _instructions

    async def after_run(self, ctx: RunContext[ModelAgentContext], *, result: Any) -> Any:
        from asgiref.sync import sync_to_async

        if self._memory is None:
            return result

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
