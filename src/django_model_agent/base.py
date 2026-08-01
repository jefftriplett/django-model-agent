"""
ModelAgent: Django Ninja-style abstraction for binding Django models to Pydantic AI Agents.

This module provides a declarative way to create AI agents that understand and operate
on Django model instances, similar to how Django Ninja's ModelSchema provides a
declarative way to serialize models.

Example using class attributes:
    class RestaurantAgent(ModelAgent):
        model = Restaurant
        fields = ["name", "address", "hours", "neighborhood"]

        _system_prompts = '''
        You are an assistant that helps reason about restaurant information.
        Use the provided model fields as your source of truth.
        '''

        tools = [UpdateHoursTool, FlagForReviewTool]

Example using decorators (pydantic-ai style):
    class RestaurantAgent(ModelAgent):
        model = Restaurant
        fields = ["name", "address", "hours"]

        @ModelAgent.system_prompt
        def context_prompt(self) -> str:
            return "You help with restaurant information."

        @ModelAgent.instructions
        def dynamic_instructions(self) -> str:
            return f"Current restaurant: {self.instance.name}"

        @ModelAgent.tool
        def get_hours(self) -> str:
            '''Get the restaurant hours.'''
            return str(self.instance.hours)

    # Usage
    restaurant = Restaurant.objects.get(pk=123)
    agent = RestaurantAgent(restaurant)
    result = await agent.run("Are we open on Christmas Day?")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from decimal import Decimal
from functools import wraps
from typing import Any, ClassVar, Optional

from django.db import models
from django.template import TemplateDoesNotExist, TemplateSyntaxError, engines
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


# Sentinel for marking decorated methods
class _DecoratorMarker:
    """Marker to identify decorated methods and their type."""

    def __init__(self, func: Callable, decorator_type: str) -> None:
        self.func = func
        self.decorator_type = decorator_type
        # Preserve function metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModelAgentContext:
    """Context object passed to tools, providing access to the model instance."""

    def __init__(self, instance: models.Model, agent: ModelAgent) -> None:
        self.instance = instance
        self.agent = agent

    def refresh_instance(self) -> None:
        """Reload the instance from the database."""
        self.instance.refresh_from_db()


class ModelAgent:
    """
    Base class for creating AI agents bound to Django models.

    Class Attributes:
        model: The Django model class this agent operates on
        fields: List of field names to expose to the agent (None = all fields)
        exclude: List of field names to exclude from the schema
        _system_prompts: System prompt string or list of strings (combined with @system_prompt decorators)
        _instructions: Instructions string or list of strings (combined with @instructions decorators)
        _instructions_template: Path to a Django/Jinja template for instructions
        tools: List of tool classes available to the agent
        _field_sets: Named groups of fields for role-based exposure
        ai_model: The pydantic-ai model name to use (e.g. 'openai:gpt-4o')
        output_type: Structured output type; None gives plain string output
        usage_limits: Default UsageLimits applied to every run

    Decorators:
        @ModelAgent.system_prompt - Register a method as a system prompt provider
        @ModelAgent.instructions - Register a method as an instructions provider
        @ModelAgent.tool - Register a method as a tool available to the agent
    """

    model: ClassVar[type[models.Model]]
    fields: ClassVar[list[str] | None] = None
    exclude: ClassVar[list[str]] = []

    _system_prompts: str | list[str] = ""
    _instructions: str | list[str] = ""
    _instructions_template: str | None = None
    tools: Sequence[Any] = []

    _field_sets: dict[str, list[str]] = {}

    ai_model: ClassVar[str | None] = None
    output_type: ClassVar[Any] = None
    usage_limits: ClassVar[Any] = None

    # -------------------------------------------------------------------------
    # Decorators for pydantic-ai style registration
    # -------------------------------------------------------------------------

    @staticmethod
    def system_prompt(func: Callable) -> _DecoratorMarker:
        """
        Decorator to register a method as a system prompt provider.

        The decorated method will be called to generate part of the system prompt.
        Multiple methods can be decorated; their outputs will be combined with
        the _system_prompts class attribute.

        Example:
            @ModelAgent.system_prompt
            def context_prompt(self) -> str:
                return "You are a helpful assistant."

            @ModelAgent.system_prompt
            def instance_context(self) -> str:
                return f"Working with: {self.instance.name}"
        """
        return _DecoratorMarker(func, "system_prompt")

    @staticmethod
    def instructions(func: Callable) -> _DecoratorMarker:
        """
        Decorator to register a method as an instructions provider.

        Instructions are dynamic guidance that can change per-run.
        Multiple methods can be decorated; their outputs will be combined.

        Example:
            @ModelAgent.instructions
            def current_state(self) -> str:
                return f"The current state is: {self.instance.state}"
        """
        return _DecoratorMarker(func, "instructions")

    @staticmethod
    def tool(func: Callable) -> _DecoratorMarker:
        """
        Decorator to register a method as a tool available to the agent.

        The method's docstring becomes the tool description.
        The method signature defines the tool's parameters.

        Example:
            @ModelAgent.tool
            def get_hours(self) -> str:
                '''Get the operating hours for this place.'''
                return str(self.instance.hours)

            @ModelAgent.tool
            def update_description(self, new_description: str) -> str:
                '''Update the place description.'''
                self.instance.description = new_description
                self.instance.save()
                return "Description updated."
        """
        return _DecoratorMarker(func, "tool")

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        instance: models.Model,
        *,
        system_prompt: str | list[str] | None = None,
        instructions: str | list[str] | None = None,
        field_set: str | None = None,
        ai_model: str | None = None,
        output_type: Any = None,
        usage_limits: Any = None,
    ) -> None:
        """
        Initialize a ModelAgent for a specific model instance.

        Args:
            instance: The Django model instance to operate on
            system_prompt: Override or extend the class-level system prompts
            instructions: Override or extend the class-level instructions
            field_set: Optional name of a field set to use for schema generation
            ai_model: Override the pydantic-ai model to use (e.g. 'openai:gpt-4o')
            output_type: Override the structured output type for this agent
            usage_limits: Override the default UsageLimits for this agent
        """
        self.instance = instance
        self.field_set = field_set
        self._schema: type[BaseModel] | None = None
        self._pydantic_agent: Any = None
        self._ai_model_override = ai_model
        self._output_type_override = output_type
        self._usage_limits_override = usage_limits

        # Override class-level prompts/instructions if provided at init
        if system_prompt is not None:
            self._system_prompts = system_prompt
        if instructions is not None:
            self._instructions = instructions

        # Collect decorated methods from the class
        self._system_prompt_funcs: list[Callable] = []
        self._instructions_funcs: list[Callable] = []
        self._tool_funcs: list[Callable] = []
        self._collect_decorated_methods()

    def _collect_decorated_methods(self) -> None:
        """Scan the class for decorated methods and collect them by type."""
        for name in dir(self.__class__):
            if name.startswith("_"):
                continue
            attr = getattr(self.__class__, name, None)
            if isinstance(attr, _DecoratorMarker):
                if attr.decorator_type == "system_prompt":
                    self._system_prompt_funcs.append(attr.func)
                elif attr.decorator_type == "instructions":
                    self._instructions_funcs.append(attr.func)
                elif attr.decorator_type == "tool":
                    self._tool_funcs.append(attr.func)

    @property
    def schema(self) -> type[BaseModel]:
        """Lazily build and cache the Pydantic schema."""
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    @property
    def context(self) -> ModelAgentContext:
        """Get the context object for tool access."""
        return ModelAgentContext(instance=self.instance, agent=self)

    def _get_active_fields(self) -> list[str] | None:
        """Determine which fields to include based on field_set or fields."""
        if self.field_set and self.field_set in self._field_sets:
            return self._field_sets[self.field_set]
        return self.fields

    def _build_schema(self) -> type[BaseModel]:
        """
        Create a Pydantic model dynamically from the Django model.

        Returns:
            A dynamically created Pydantic BaseModel class
        """
        model_fields: dict[str, tuple[type, Any]] = {}
        active_fields = self._get_active_fields()

        for field in self.model._meta.fields:
            # Skip if not in active fields (when specified)
            if active_fields and field.name not in active_fields:
                continue

            # Skip excluded fields
            if field.name in self.exclude:
                continue

            # Get the Python type for this field
            python_type = self._get_field_type(field)
            default_value = self._get_field_default(field)

            model_fields[field.name] = (python_type, default_value)

        return create_model(
            f"{self.model.__name__}AgentSchema",
            **model_fields,
        )

    def _get_field_type(self, field: models.Field) -> type:
        """
        Map a Django field to its Python type, respecting nullability.

        Args:
            field: A Django model field

        Returns:
            The appropriate Python type (potentially Optional)
        """
        try:
            base_type = field.get_internal_type()
            type_mapping = {
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
                "NullBooleanField": Optional[bool],
                "DateField": str,  # ISO format strings for AI
                "DateTimeField": str,
                "TimeField": str,
                "DurationField": str,
                "BinaryField": bytes,
                "JSONField": dict,
                "ForeignKey": int,  # Return the ID
            }
            python_type = type_mapping.get(base_type, str)
        except AttributeError:
            python_type = str

        # Handle nullability
        if getattr(field, "null", False) and python_type is not type(None):
            return Optional[python_type]

        return python_type

    def _get_field_default(self, field: models.Field) -> Any:
        """
        Get the current value from the instance as the default.

        Args:
            field: A Django model field

        Returns:
            The current value on the instance, or ... if required
        """
        try:
            value = getattr(self.instance, field.name)
            # Handle related fields - get the ID
            if hasattr(value, "pk"):
                return value.pk
            return value
        except AttributeError:
            return ...

    def get_system_prompts(self) -> str:
        """
        Get the combined system prompt for this agent.

        Combines:
        1. The class-level _system_prompts (string or list of strings)
        2. All @ModelAgent.system_prompt decorated methods

        Override this method to customize system prompt generation.
        """
        parts = []

        # Add class-level prompt(s) if defined
        if self._system_prompts:
            if isinstance(self._system_prompts, list):
                parts.extend(s.strip() for s in self._system_prompts if s)
            else:
                parts.append(self._system_prompts.strip())

        # Add prompts from decorated methods
        for func in self._system_prompt_funcs:
            result = func(self)
            if result:
                parts.append(str(result).strip())

        return "\n\n".join(parts)

    def get_instructions(self) -> str | None:
        """
        Get the combined instructions for this agent.

        Combines:
        1. The class-level _instructions (string or list of strings)
        2. Rendered instructions_template (if provided)
        3. All @ModelAgent.instructions decorated methods

        Returns:
            Combined instructions string or None
        """
        parts = []

        # Add class-level instructions if defined
        if self._instructions:
            if isinstance(self._instructions, list):
                parts.extend(s.strip() for s in self._instructions if s)
            else:
                parts.append(self._instructions.strip())

        # Add template-based instructions if defined
        if self._instructions_template:
            rendered = self._render_template(
                self._instructions_template,
                context={"instance": self.instance, "schema": self.schema},
            )
            if rendered:
                parts.append(rendered.strip())

        # Add instructions from decorated methods
        for func in self._instructions_funcs:
            result = func(self)
            if result:
                parts.append(str(result).strip())

        return "\n\n".join(parts) if parts else None

    def get_tools(self) -> list[Callable]:
        """
        Get all tools available to this agent.

        Combines:
        1. The class-level tools list (ModelTool classes)
        2. All @ModelAgent.tool decorated methods

        Returns:
            List of tool functions/classes
        """
        alltools: list[Any] = list(self.tools)

        # Add decorated tool methods (bound to self)
        for func in self._tool_funcs:
            # Create a bound method
            bound_method = func.__get__(self, self.__class__)
            alltools.append(bound_method)

        return alltools

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render a Django template with the given context.

        Args:
            template_name: Path to the template
            context: Template context dict

        Returns:
            Rendered template string, or empty string if the template
            is missing or contains syntax errors.
        """
        try:
            engine = engines["django"]
            template = engine.get_template(template_name)
            return template.render(context)
        except TemplateDoesNotExist:
            logger.warning("Template not found: %s", template_name)
            return ""
        except TemplateSyntaxError:
            logger.warning("Template syntax error in: %s", template_name)
            return ""

    def get_schema_description(self) -> str:
        """
        Generate a human-readable description of the schema for the agent.

        Returns:
            A formatted string describing the available fields
        """
        lines = [f"You have access to the following {self.model.__name__} fields:"]
        for name, field_info in self.schema.model_fields.items():
            annotation = field_info.annotation
            lines.append(f"  - {name}: {annotation}")
        return "\n".join(lines)

    def get_current_values(self) -> dict[str, Any]:
        """
        Get the current values of all schema fields from the instance.

        Returns:
            Dict mapping field names to their current values
        """
        values = {}
        for field_name in self.schema.model_fields:
            value = getattr(self.instance, field_name, None)
            if hasattr(value, "pk"):
                value = value.pk
            values[field_name] = value
        return values

    def _get_ai_model(self) -> Any:
        """
        Resolve the AI model to use.

        Checked in order: the value passed to ``__init__``, the class-level
        ``ai_model``, then the ``PYDANTIC_AI_MODEL`` environment variable.

        The environment fallback follows the convention pydantic-ai uses in its
        own examples, which read ``PYDANTIC_AI_MODEL`` to switch providers
        without editing code. pydantic-ai itself does not read the variable, so
        honouring it here is what makes that convention work for these agents.
        """
        if self._ai_model_override:
            return self._ai_model_override
        if self.__class__.ai_model:
            return self.__class__.ai_model
        return os.environ.get("PYDANTIC_AI_MODEL") or None

    def _get_output_type(self) -> Any:
        """Resolve the structured output type, init argument beating the class."""
        if self._output_type_override is not None:
            return self._output_type_override
        return self.__class__.output_type

    def _get_usage_limits(self) -> Any:
        """
        Resolve the usage limits for a run.

        Checked in order: the value passed to ``__init__``, the class-level
        ``usage_limits``, then the ``DJANGO_MODEL_AGENT_USAGE_LIMITS`` Django
        setting. The setting lets a project cap every agent at once rather than
        remembering to do it per class.
        """
        if self._usage_limits_override is not None:
            return self._usage_limits_override
        if self.__class__.usage_limits is not None:
            return self.__class__.usage_limits

        from django.conf import settings

        return getattr(settings, "DJANGO_MODEL_AGENT_USAGE_LIMITS", None)

    def _run_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Fill in defaults an explicit call did not already provide."""
        limits = self._get_usage_limits()
        if limits is not None:
            kwargs.setdefault("usage_limits", limits)
        return kwargs

    def _build_pydantic_ai_tools(self) -> list[Any]:
        """
        Convert this agent's tools into pydantic-ai Tool objects.

        Deprecated: tools are now supplied by ``DjangoModelCapability``. Kept
        because it was part of the public surface; it returns the same tools
        the capability builds.
        """
        from .capabilities import model_tools_to_toolset

        toolset = model_tools_to_toolset(self.tools, self._tool_funcs)
        return list(toolset.tools.values())

    def get_extra_capabilities(self) -> list[Any]:
        """
        Additional capabilities to compose into the agent.

        Override to add capabilities such as ``DjangoAuditCapability`` or
        ``DjangoMemoryCapability``:

            class PlaceAgent(ModelAgent):
                model = Place

                def get_extra_capabilities(self):
                    return [DjangoAuditCapability(log_to="logger")]
        """
        return []

    def _static_instructions(self) -> list[str]:
        """
        Instruction text that cannot change between runs.

        Only the class-level attributes qualify. Decorated methods and the
        template are resolved per request by ``_dynamic_instructions``.
        """
        parts: list[str] = []
        for source in (self._system_prompts, self._instructions):
            if not source:
                continue
            if isinstance(source, list):
                parts.extend(text.strip() for text in source if text)
            else:
                parts.append(source.strip())
        return parts

    def _dynamic_instructions(self) -> str:
        """
        Instruction text resolved fresh on every request.

        Decorated ``@system_prompt`` / ``@instructions`` methods and the
        rendered template all read from ``self.instance``, so evaluating them
        once at build time would leave them contradicting the field values the
        capability injects live.
        """
        parts: list[str] = []

        for func in self._system_prompt_funcs:
            result = func(self)
            if result:
                parts.append(str(result).strip())

        if self._instructions_template:
            rendered = self._render_template(
                self._instructions_template,
                context={"instance": self.instance, "schema": self.schema},
            )
            if rendered:
                parts.append(rendered.strip())

        for func in self._instructions_funcs:
            result = func(self)
            if result:
                parts.append(str(result).strip())

        return "\n\n".join(parts)

    def _build_capabilities(self) -> list[Any]:
        """
        Compose the capabilities backing this agent.

        The class-level ``_system_prompts`` are folded in with ``_instructions``
        rather than passed as a system prompt. Both describe the model, and the
        model's field values change between turns -- see ``build_agent``.
        """
        from .capabilities import (
            DjangoFSMCapability,
            DjangoMemoryCapability,
            DjangoModelCapability,
        )
        from .memory import AgentMemoryMixin

        capabilities: list[Any] = [
            DjangoModelCapability(
                model_class=self.model,
                fields=self._get_active_fields(),
                exclude=self.exclude,
                tools=self.tools,
                tool_funcs=self._tool_funcs,
                instructions=self._static_instructions(),
                dynamic_instructions=self._dynamic_instructions,
            )
        ]

        # Only worth adding when some tool actually restricts by state.
        if any(getattr(tool, "allowed_states", None) for tool in self.tools):
            capabilities.append(DjangoFSMCapability(tools=self.tools))

        if isinstance(self, AgentMemoryMixin):
            capabilities.append(DjangoMemoryCapability())

        capabilities.extend(self.get_extra_capabilities())
        return capabilities

    def build_agent(self) -> Any:
        """
        Build the Pydantic AI Agent from this agent's capabilities.

        Everything the agent knows about the model now arrives through
        capabilities rather than being assembled here.

        Note that the schema and current values go through instructions, not
        ``system_prompt``. A system prompt is written into message history and
        stays there, so the field values captured on the first turn would still
        be sitting in the context on the fifth, contradicting the current ones.
        Instructions are re-sent fresh each request and kept out of history.

        Returns:
            A configured pydantic_ai.Agent with deps_type=ModelAgentContext
        """
        from pydantic_ai import Agent

        from .capabilities import tools_need_approval

        kwargs: dict[str, Any] = {}
        output_type = self._get_output_type()

        if tools_need_approval(self.tools):
            # A tool marked requires_approval suspends the run and returns a
            # DeferredToolRequests. pydantic-ai raises a UserError unless that
            # type is among the outputs, so add it rather than let the run fail.
            from pydantic_ai import DeferredToolRequests

            base = output_type if output_type is not None else str
            output_type = [base, DeferredToolRequests]

        if output_type is not None:
            # Only passed when set; pydantic-ai's default is plain str output
            # and handing it None would override that.
            kwargs["output_type"] = output_type

        return Agent(
            self._get_ai_model(),
            deps_type=ModelAgentContext,
            capabilities=self._build_capabilities(),
            name=f"{self.__class__.__name__}({self.model.__name__})",
            **kwargs,
        )

    async def run(self, prompt: str | None = None, **kwargs: Any) -> Any:
        """
        Run the agent with a prompt.

        ``prompt`` may be omitted when resuming a run that suspended for tool
        approval -- pass ``message_history`` and ``deferred_tool_results``
        instead, and the run continues from where it stopped.

        Args:
            prompt: The user prompt, or None when resuming a suspended run
            **kwargs: Additional keyword arguments passed to pydantic-ai's Agent.run()

        Returns:
            The AgentRunResult from pydantic-ai
        """
        if self._pydantic_agent is None:
            self._pydantic_agent = self.build_agent()
        return await self._pydantic_agent.run(
            prompt,
            deps=self.context,
            **self._run_kwargs(kwargs),
        )

    def run_sync(self, prompt: str | None = None, **kwargs: Any) -> Any:
        """
        Run the agent with a prompt synchronously.

        Args:
            prompt: The user prompt, or None when resuming a suspended run
            **kwargs: Additional keyword arguments passed to pydantic-ai's Agent.run_sync()

        Returns:
            The AgentRunResult from pydantic-ai
        """
        if self._pydantic_agent is None:
            self._pydantic_agent = self.build_agent()
        return self._pydantic_agent.run_sync(
            prompt,
            deps=self.context,
            **self._run_kwargs(kwargs),
        )

    def run_stream(self, prompt: str | None = None, **kwargs: Any) -> Any:
        """
        Stream a run, yielding output as the model produces it.

        Returns pydantic-ai's async context manager, so use it directly rather
        than awaiting it:

            async with agent.run_stream("Summarise this.") as stream:
                async for chunk in stream.stream_text(delta=True):
                    ...

        Args:
            prompt: The user prompt, or None when resuming a suspended run
            **kwargs: Forwarded to pydantic-ai's Agent.run_stream()

        Returns:
            The streaming context manager from pydantic-ai
        """
        if self._pydantic_agent is None:
            self._pydantic_agent = self.build_agent()
        return self._pydantic_agent.run_stream(
            prompt,
            deps=self.context,
            **self._run_kwargs(kwargs),
        )

    def run_stream_events(self, prompt: str | None = None, **kwargs: Any) -> Any:
        """
        Stream structured events from a run.

        Yields events for tool calls, output deltas, and completion, rather than
        text alone -- useful when the UI should show that a tool is running.

        Like ``run_stream``, this is a context manager rather than a bare
        iterable, so the run is torn down properly if the consumer stops early:

            async with agent.run_stream_events("Tidy this up.") as events:
                async for event in events:
                    ...

        Args:
            prompt: The user prompt, or None when resuming a suspended run
            **kwargs: Forwarded to pydantic-ai's Agent.run_stream_events()

        Returns:
            An async iterable of run events
        """
        if self._pydantic_agent is None:
            self._pydantic_agent = self.build_agent()
        return self._pydantic_agent.run_stream_events(
            prompt,
            deps=self.context,
            **self._run_kwargs(kwargs),
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.model.__name__}:{self.instance.pk})>"
