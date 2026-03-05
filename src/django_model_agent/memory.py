"""
Model-backed memory for agents.

Provides persistent memory storage tied to Django model instances via GenericForeignKey.
This allows agents to remember context across conversations about specific entities.

Example:
    # Store memory for a restaurant
    memory = AgentMemory.objects.get_or_create_for(restaurant)
    memory.data["last_question"] = "hours"
    memory.data["conversation_count"] = 1
    memory.save()

    # Later, retrieve it
    memory = AgentMemory.objects.get_for(restaurant)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

if TYPE_CHECKING:
    from django.db.models import QuerySet


class AgentMemoryManager(models.Manager):
    """Custom manager for AgentMemory with convenience methods."""

    def get_for(self, instance: models.Model) -> AgentMemory | None:
        """
        Get the memory record for a model instance.

        Args:
            instance: Any Django model instance

        Returns:
            AgentMemory instance or None if not found

        Raises:
            ValueError: If the instance has not been saved (pk is None)
        """
        if instance.pk is None:
            raise ValueError(
                "Cannot get memory for an unsaved model instance (pk is None). "
                "Save the instance first."
            )
        content_type = ContentType.objects.get_for_model(instance)
        try:
            return self.get(content_type=content_type, object_id=instance.pk)
        except self.model.DoesNotExist:
            return None

    def get_or_create_for(
        self,
        instance: models.Model,
        defaults: dict[str, Any] | None = None,
    ) -> tuple[AgentMemory, bool]:
        """
        Get or create memory for a model instance.

        Args:
            instance: Any Django model instance
            defaults: Default values for the memory data

        Returns:
            Tuple of (AgentMemory instance, created boolean)

        Raises:
            ValueError: If the instance has not been saved (pk is None)
        """
        if instance.pk is None:
            raise ValueError(
                "Cannot get or create memory for an unsaved model instance (pk is None). "
                "Save the instance first."
            )
        content_type = ContentType.objects.get_for_model(instance)
        return self.get_or_create(
            content_type=content_type,
            object_id=instance.pk,
            defaults={"data": defaults or {}},
        )

    def filter_for_model(
        self, model_class: type[models.Model]
    ) -> QuerySet[AgentMemory]:
        """
        Get all memory records for a specific model type.

        Args:
            model_class: A Django model class

        Returns:
            QuerySet of AgentMemory records
        """
        content_type = ContentType.objects.get_for_model(model_class)
        return self.filter(content_type=content_type)


class AgentMemory(models.Model):
    """
    Stores agent memory/state tied to any Django model instance.

    Uses Django's contenttypes framework for generic relations,
    allowing memory to be attached to any model.

    The `data` field is a JSONField that can store arbitrary
    structured data like conversation history, learned facts,
    user preferences, etc.
    """

    content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        help_text="The type of model this memory is attached to",
    )
    object_id = models.PositiveIntegerField(
        help_text="The primary key of the model instance",
    )
    content_object = GenericForeignKey("content_type", "object_id")

    data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Arbitrary JSON data storing agent memory/state",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = AgentMemoryManager()

    class Meta:
        app_label = "django_model_agent"
        verbose_name = "Agent Memory"
        verbose_name_plural = "Agent Memories"
        unique_together = [["content_type", "object_id"]]
        indexes = [
            models.Index(fields=["content_type", "object_id"]),
        ]

    def __str__(self) -> str:
        return f"AgentMemory for {self.content_type}:{self.object_id}"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory data."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in memory data (doesn't save)."""
        self.data[key] = value

    def update(self, **kwargs: Any) -> None:
        """Update multiple values in memory data (doesn't save)."""
        self.data.update(kwargs)

    def clear(self) -> None:
        """Clear all memory data (doesn't save)."""
        self.data = {}

    def append_to_history(
        self,
        role: str,
        content: str,
        *,
        max_history: int = 100,
    ) -> None:
        """
        Append a message to conversation history.

        Args:
            role: The role (user, assistant, system)
            content: The message content
            max_history: Maximum number of messages to keep
        """
        if "history" not in self.data:
            self.data["history"] = []

        self.data["history"].append({"role": role, "content": content})

        # Trim history if needed
        if len(self.data["history"]) > max_history:
            self.data["history"] = self.data["history"][-max_history:]

    def get_history(self) -> list[dict[str, str]]:
        """Get the conversation history."""
        return self.data.get("history", [])


class AgentMemoryMixin:
    """
    Mixin for ModelAgent to add memory support.

    Add this to your ModelAgent subclass to enable persistent memory.

    Example:
        class RestaurantAgent(AgentMemoryMixin, ModelAgent):
            model = Restaurant
            ...

        agent = RestaurantAgent(restaurant)
        agent.memory.set("last_topic", "hours")
        agent.save_memory()
    """

    _memory: AgentMemory | None = None

    @property
    def memory(self) -> AgentMemory:
        """
        Get or create memory for this agent's instance.

        Returns:
            AgentMemory instance
        """
        if self._memory is None:
            self._memory, _ = AgentMemory.objects.get_or_create_for(self.instance)
        return self._memory

    def load_memory(self) -> dict[str, Any]:
        """Load and return the memory data."""
        return self.memory.data

    def save_memory(self) -> None:
        """Save the current memory state."""
        self.memory.save()

    def get_memory_context(self) -> str:
        """
        Get memory as a formatted string for inclusion in prompts.

        Returns:
            Formatted memory context string
        """
        if not self.memory.data:
            return ""

        lines = ["Previous context:"]
        for key, value in self.memory.data.items():
            if key == "history":
                lines.append(f"  Conversation history: {len(value)} messages")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
