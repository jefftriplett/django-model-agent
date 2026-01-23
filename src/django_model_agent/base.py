"""
ModelAgent: Django Ninja-style abstraction for binding Django models to Pydantic AI Agents.

This module provides a declarative way to create AI agents that understand and operate
on Django model instances, similar to how Django Ninja's ModelSchema provides a
declarative way to serialize models.

Example using class attributes:
    class RestaurantAgent(ModelAgent):
        model = Restaurant
        fields = ["name", "address", "hours", "neighborhood"]

        base_prompt = '''
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

from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, ClassVar

from django.db import models
from django.template import engines
from pydantic import BaseModel, create_model


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
        base_prompt: Base system prompt for the agent (combined with @system_prompt decorators)
        instructions_template: Path to a Django/Jinja template for instructions
        tools: List of tool classes available to the agent
        field_sets: Named groups of fields for role-based exposure

    Decorators:
        @ModelAgent.system_prompt - Register a method as a system prompt provider
        @ModelAgent.instructions - Register a method as an instructions provider
        @ModelAgent.tool - Register a method as a tool available to the agent
    """

    model: ClassVar[type[models.Model]]
    fields: ClassVar[list[str] | None] = None
    exclude: ClassVar[list[str]] = []

    base_prompt: ClassVar[str] = ""
    instructions_template: ClassVar[str | None] = None
    tools: ClassVar[Sequence[Any]] = []

    field_sets: ClassVar[dict[str, list[str]]] = {}

    # -------------------------------------------------------------------------
    # Decorators for pydantic-ai style registration
    # -------------------------------------------------------------------------

    @staticmethod
    def system_prompt(func: Callable) -> _DecoratorMarker:
        """
        Decorator to register a method as a system prompt provider.

        The decorated method will be called to generate part of the system prompt.
        Multiple methods can be decorated; their outputs will be combined with
        the base_prompt class attribute.

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
        field_set: str | None = None,
    ) -> None:
        """
        Initialize a ModelAgent for a specific model instance.

        Args:
            instance: The Django model instance to operate on
            field_set: Optional name of a field set to use for schema generation
        """
        self.instance = instance
        self.field_set = field_set
        self._schema: type[BaseModel] | None = None
        self._pydantic_agent: Any = None

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
        if self.field_set and self.field_set in self.field_sets:
            return self.field_sets[self.field_set]
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
        from typing import Optional

        # Get the base Python type
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
                "DecimalField": float,
                "CharField": str,
                "TextField": str,
                "EmailField": str,
                "URLField": str,
                "SlugField": str,
                "UUIDField": str,
                "BooleanField": bool,
                "NullBooleanField": Optional[bool],
                "DateField": str,  # ISO format strings for AI
                "DateTimeField": str,
                "TimeField": str,
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

    def get_system_prompt(self) -> str:
        """
        Get the combined system prompt for this agent.

        Combines:
        1. The class-level base_prompt string
        2. All @ModelAgent.system_prompt decorated methods

        Override this method to customize system prompt generation.
        """
        parts = []

        # Add class-level base prompt if defined
        if self.base_prompt:
            parts.append(self.base_prompt.strip())

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
        1. Rendered instructions_template (if provided)
        2. All @ModelAgent.instruction decorated methods

        Returns:
            Combined instructions string or None
        """
        parts = []

        # Add template-based instructions if defined
        if self.instructions_template:
            rendered = self._render_template(
                self.instructions_template,
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
        all_tools: list[Any] = list(self.tools)

        # Add decorated tool methods (bound to self)
        for func in self._tool_funcs:
            # Create a bound method
            bound_method = func.__get__(self, self.__class__)
            all_tools.append(bound_method)

        return all_tools

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render a Django template with the given context.

        Args:
            template_name: Path to the template
            context: Template context dict

        Returns:
            Rendered template string
        """
        engine = engines["django"]
        template = engine.get_template(template_name)
        return template.render(context)

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

    def build_agent(self) -> Any:
        """
        Build the Pydantic AI Agent.

        This is where you would integrate with pydantic-ai.
        Override this method to customize agent construction.

        Returns:
            A configured Pydantic AI Agent instance
        """
        # Placeholder - actual implementation depends on pydantic-ai API
        # from pydantic_ai import Agent
        # return Agent(
        #     model=self.schema,
        #     system_prompt=self.get_system_prompt(),
        #     instructions=self.get_instructions(),
        #     tools=self.tools,
        # )
        raise NotImplementedError("build_agent() must be implemented once pydantic-ai is integrated")

    async def run(self, prompt: str) -> Any:
        """
        Run the agent with a prompt.

        Args:
            prompt: The user prompt to process

        Returns:
            The agent's response
        """
        if self._pydantic_agent is None:
            self._pydantic_agent = self.build_agent()
        return await self._pydantic_agent.run(prompt)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.model.__name__}:{self.instance.pk})>"
