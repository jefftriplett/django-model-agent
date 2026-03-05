"""
Tool base classes for ModelAgent.

Provides a structured way to define tools that can operate on Django model instances
with proper context injection and type safety.

Example:
    class UpdateHoursTool(ModelTool):
        name = "update_hours"
        description = "Update the restaurant's operating hours"

        async def execute(self, hours: str) -> str:
            self.instance.hours = hours
            self.instance.save()
            return f"Hours updated to: {hours}"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from django.db import models

from .base import ModelAgent, ModelAgentContext


@dataclass
class ToolResult:
    """
    Structured result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully
        message: Human-readable result message
        data: Optional structured data to return
        changes: Dict of field changes made (for audit/diff)
    """

    success: bool
    message: str
    data: dict[str, Any] | None = None
    changes: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.message}"


class ModelTool(ABC):
    """
    Base class for tools that operate on Django model instances.

    Subclass this to create tools that can read or modify the model
    instance bound to a ModelAgent.

    Class Attributes:
        name: Unique identifier for the tool
        description: Human-readable description for the AI
        requires_confirmation: If True, changes require approval
        allowed_states: List of FSM states where this tool is allowed
    """

    name: ClassVar[str]
    description: ClassVar[str]
    requires_confirmation: ClassVar[bool] = False
    allowed_states: ClassVar[list[str] | None] = None

    def __init__(self, context: ModelAgentContext) -> None:
        """
        Initialize the tool with agent context.

        Args:
            context: The ModelAgentContext providing access to instance and agent
        """
        self._context = context

    @property
    def instance(self) -> models.Model:
        """Get the model instance."""
        return self._context.instance

    @property
    def agent(self) -> ModelAgent:
        """Get the parent agent."""
        return self._context.agent

    def check_allowed(self) -> tuple[bool, str | None]:
        """
        Check if this tool is allowed in the current state.

        Returns:
            Tuple of (allowed, reason if not allowed)
        """
        if self.allowed_states is None:
            return True, None

        # Check for FSM state if the model uses django-fsm
        if hasattr(self.instance, "state"):
            current_state = self.instance.state
            if current_state not in self.allowed_states:
                return (
                    False,
                    f"Tool '{self.name}' not allowed in state '{current_state}'",
                )

        return True, None

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool's action (synchronous).

        Override this method to implement the tool's functionality.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult indicating success/failure and any data
        """
        ...

    async def execute_async(self, **kwargs: Any) -> ToolResult:
        """
        Async wrapper for execute. Override for true async implementations.

        By default, just calls the sync execute method.
        """
        return self.execute(**kwargs)

    def __call__(self, **kwargs: Any) -> ToolResult:
        """
        Invoke the tool with state checking (synchronous).

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult from execution or error result
        """
        allowed, reason = self.check_allowed()
        if not allowed:
            return ToolResult(success=False, message=reason or "Tool not allowed")

        return self.execute(**kwargs)

    async def call_async(self, **kwargs: Any) -> ToolResult:
        """
        Invoke the tool with state checking (async version).

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult from execution or error result
        """
        allowed, reason = self.check_allowed()
        if not allowed:
            return ToolResult(success=False, message=reason or "Tool not allowed")

        return await self.execute_async(**kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name})>"


class ReadOnlyTool(ModelTool):
    """Base class for tools that only read data, never modify."""

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a read-only operation."""
        data = self.read(**kwargs)
        return ToolResult(success=True, message="Data retrieved", data=data)

    @abstractmethod
    def read(self, **kwargs: Any) -> dict[str, Any]:
        """
        Read data from the model instance.

        Override this method to implement the read operation.

        Returns:
            Dict of data read from the instance
        """
        ...


class UpdateTool(ModelTool):
    """Base class for tools that update model fields."""

    requires_confirmation: ClassVar[bool] = True

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute an update operation."""
        # Capture before state
        before = self._capture_state()

        # Perform the update
        self.update(**kwargs)

        # Capture after state and compute diff
        after = self._capture_state()
        changes = self._compute_diff(before, after)

        # Save unless this is a preview
        if not kwargs.get("preview", False):
            self.instance.save()

        return ToolResult(
            success=True,
            message=f"Updated {len(changes)} field(s)",
            changes=changes,
        )

    def _capture_state(self) -> dict[str, Any]:
        """Capture current field values for diffing."""
        # Get all editable fields
        state = {}
        for model_field in self.instance._meta.fields:
            if not model_field.editable:
                continue
            state[model_field.name] = getattr(self.instance, model_field.name)
        return state

    def _compute_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute differences between two states."""
        changes = {}
        for key in before:
            if before[key] != after.get(key):
                changes[key] = {"before": before[key], "after": after[key]}
        return changes

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """
        Update the model instance.

        Override this method to implement the update logic.
        Do not call save() - the base class handles that.
        """
        ...


class ProposedChange:
    """
    Represents a proposed change that requires approval.

    Used for diff-aware writes where changes are proposed first,
    then approved by a human or another agent.
    """

    def __init__(
        self,
        instance: models.Model,
        field_name: str,
        old_value: Any,
        new_value: Any,
        reason: str = "",
    ) -> None:
        self.instance = instance
        self.field_name = field_name
        self.old_value = old_value
        self.new_value = new_value
        self.reason = reason
        self.approved: bool | None = None

    def approve(self) -> None:
        """Mark this change as approved."""
        self.approved = True

    def reject(self) -> None:
        """Mark this change as rejected."""
        self.approved = False

    def apply(self) -> None:
        """Apply the change if approved."""
        if self.approved is not True:
            raise ValueError("Cannot apply unapproved change")
        setattr(self.instance, self.field_name, self.new_value)

    def __repr__(self) -> str:
        status = "✓" if self.approved else ("✗" if self.approved is False else "?")
        return f"<ProposedChange {status} {self.field_name}: {self.old_value!r} -> {self.new_value!r}>"


class DiffAwareUpdateTool(ModelTool):
    """
    Tool that proposes changes instead of applying them directly.

    Changes are collected and can be reviewed before application.
    Useful for multi-agent workflows where one agent proposes and another approves.
    """

    def __init__(self, context: ModelAgentContext) -> None:
        super().__init__(context)
        self.proposed_changes: list[ProposedChange] = []

    def propose_change(
        self,
        field_name: str,
        new_value: Any,
        reason: str = "",
    ) -> ProposedChange:
        """
        Propose a change to a field.

        Args:
            field_name: Name of the field to change
            new_value: The proposed new value
            reason: Why this change is being proposed

        Returns:
            ProposedChange object
        """
        old_value = getattr(self.instance, field_name)
        change = ProposedChange(
            instance=self.instance,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
        )
        self.proposed_changes.append(change)
        return change

    def get_pending_changes(self) -> list[ProposedChange]:
        """Get all changes that haven't been approved or rejected."""
        return [c for c in self.proposed_changes if c.approved is None]

    def apply_approved_changes(self) -> int:
        """
        Apply all approved changes and save.

        Returns:
            Number of changes applied
        """
        applied = 0
        for change in self.proposed_changes:
            if change.approved is True:
                change.apply()
                applied += 1

        if applied > 0:
            self.instance.save()

        return applied

    def get_diff_summary(self) -> str:
        """Get a human-readable summary of proposed changes."""
        if not self.proposed_changes:
            return "No changes proposed"

        lines = ["Proposed changes:"]
        for change in self.proposed_changes:
            status = (
                "✓" if change.approved else ("✗" if change.approved is False else "?")
            )
            lines.append(
                f"  {status} {change.field_name}: {change.old_value!r} -> {change.new_value!r}"
            )
            if change.reason:
                lines.append(f"      Reason: {change.reason}")
        return "\n".join(lines)
